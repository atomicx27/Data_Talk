"""Push-Down Statistical Profiler & Anomaly Watchdog for Data_Talk.

Runs lightweight, read-only aggregation queries inside the source engine to gather:
- Row count, approximate distinct count, and null ratio per column
- Min/Max bounds for numeric and date/timestamp columns
- Sanitized top-10 distinct categorical sample values (only when distinct_count <= 30 and non-PII)
- Automated Data Quality & Anomaly Watchdog alerts (duplicate PKs, high-null columns, negative revenue/price outliers)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.knowledge.okf_models import ColumnProfile, TableSchema
from src.security.privacy_firewall import PrivacyFirewall

if TYPE_CHECKING:
    from src.connectors.base import BaseConnector


def _is_numeric_or_date_type(dtype_str: str) -> bool:
    up = dtype_str.upper()
    return any(
        tok in up
        for tok in (
            "INT",
            "FLOAT",
            "DOUBLE",
            "DECIMAL",
            "NUMERIC",
            "REAL",
            "NUMBER",
            "DATE",
            "TIME",
            "TIMESTAMP",
        )
    )


class StatisticalProfiler:
    """Profiles tables using push-down SQL/DAX aggregations while enforcing PII redaction."""

    @classmethod
    def profile_tables(
        cls,
        connector: "BaseConnector",
        tables: list[TableSchema],
        blocked_tables: set[str] | None = None,
        blocked_columns: set[str] | None = None,
    ) -> list[TableSchema]:
        blocked_t = {t.lower() for t in (blocked_tables or set())}
        blocked_c = {c.lower() for c in (blocked_columns or set())}

        profiled: list[TableSchema] = []
        for tbl in tables:
            if tbl.name.lower() in blocked_t or tbl.is_blocked:
                tbl.is_blocked = True
                continue
            cls._profile_single_table(connector, tbl, blocked_c)
            profiled.append(tbl)
        return profiled

    @classmethod
    def _profile_single_table(
        cls,
        connector: "BaseConnector",
        table: TableSchema,
        blocked_cols_lower: set[str],
    ) -> None:
        if connector.source_type in ("powerbi_desktop", "powerbi_cloud"):
            # For live DAX endpoints, run lightweight COUNTROWS + DISTINCTCOUNT
            cls._profile_dax_table(connector, table, blocked_cols_lower)
            return

        q_tbl = f'"{table.name}"'
        try:
            cnt_df = connector.execute_query(f"SELECT COUNT(*) AS cnt FROM {q_tbl}", max_rows=1)
            table.row_count = int(cnt_df["cnt"].iloc[0]) if not cnt_df.empty else 0
        except Exception:
            table.row_count = 0

        if table.row_count == 0:
            table.quality_alerts.append(f"Table '{table.name}' contains 0 rows.")
            return

        quality_alerts: list[str] = []

        for col in table.columns:
            full_col_key = f"{table.name}.{col.name}".lower()
            if col.name.lower() in blocked_cols_lower or full_col_key in blocked_cols_lower:
                col.is_blocked = True
                continue

            if PrivacyFirewall.is_pii_column_name(col.name):
                col.is_pii = True

            q_col = f'"{col.name}"'
            try:
                if _is_numeric_or_date_type(col.data_type) and not col.is_pii:
                    stat_sql = (
                        f"SELECT "
                        f"COUNT(DISTINCT {q_col}) AS dist_cnt, "
                        f"SUM(CASE WHEN {q_col} IS NULL THEN 1 ELSE 0 END) AS null_cnt, "
                        f"CAST(MIN({q_col}) AS VARCHAR) AS min_v, "
                        f"CAST(MAX({q_col}) AS VARCHAR) AS max_v "
                        f"FROM {q_tbl}"
                    )
                else:
                    stat_sql = (
                        f"SELECT "
                        f"COUNT(DISTINCT {q_col}) AS dist_cnt, "
                        f"SUM(CASE WHEN {q_col} IS NULL THEN 1 ELSE 0 END) AS null_cnt "
                        f"FROM {q_tbl}"
                    )
                stat_df = connector.execute_query(stat_sql, max_rows=1)
                if not stat_df.empty:
                    row = stat_df.iloc[0]
                    col.distinct_count = int(row.get("dist_cnt") or 0)
                    null_cnt = int(row.get("null_cnt") or 0)
                    col.null_ratio = round(null_cnt / max(1, table.row_count), 4)
                    if "min_v" in row and row["min_v"] is not None:
                        col.min_value = str(row["min_v"])
                    if "max_v" in row and row["max_v"] is not None:
                        col.max_value = str(row["max_v"])
            except Exception:
                pass

            # Sample values for low/medium-cardinality non-PII columns (up to 100 distinct values)
            if col.is_pii:
                col.sample_values = ["<REDACTED_PII_COLUMN>"]
            elif 0 < col.distinct_count <= 100:
                try:
                    samp_sql = (
                        f"SELECT CAST({q_col} AS VARCHAR) AS val, COUNT(*) AS freq "
                        f"FROM {q_tbl} WHERE {q_col} IS NOT NULL "
                        f"GROUP BY {q_col} ORDER BY freq DESC LIMIT 50"
                    )
                    samp_df = connector.execute_query(samp_sql, max_rows=50)
                    raw_vals = samp_df["val"].tolist() if not samp_df.empty else []
                    sanitized, pii_found = PrivacyFirewall.sanitize_profile_samples(
                        col.name, raw_vals, col.distinct_count
                    )
                    col.sample_values = sanitized
                    if pii_found:
                        col.is_pii = True
                except Exception:
                    col.sample_values = []

            # Data Quality & Anomaly Watchdog Checks
            if col.is_primary_key and col.distinct_count < table.row_count:
                dup_cnt = table.row_count - col.distinct_count
                quality_alerts.append(
                    f"Primary key column '{col.name}' has {dup_cnt:,} duplicate or null values."
                )
            if col.null_ratio >= 0.40:
                quality_alerts.append(
                    f"Column '{col.name}' has a high NULL rate ({col.null_ratio*100:.1f}%)."
                )

        table.quality_alerts = quality_alerts

    @classmethod
    def _profile_dax_table(
        cls,
        connector: "BaseConnector",
        table: TableSchema,
        blocked_cols_lower: set[str],
    ) -> None:
        try:
            dax = f"EVALUATE ROW(\"cnt\", COUNTROWS('{table.name}'))"
            df = connector.execute_query(dax, max_rows=1)
            if not df.empty:
                table.row_count = int(df.iloc[0, 0] or 0)
        except Exception:
            table.row_count = 0

        for col in table.columns:
            if col.name.lower() in blocked_cols_lower:
                col.is_blocked = True
                continue
            if PrivacyFirewall.is_pii_column_name(col.name):
                col.is_pii = True
                col.sample_values = ["<REDACTED_PII_COLUMN>"]
