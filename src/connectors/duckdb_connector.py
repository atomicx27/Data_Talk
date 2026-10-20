"""Embedded DuckDB Connector for CSV, Multi-Sheet Excel (.xlsx), Parquet, and Demo Data.

Key Hardening Features:
- Thread-safe per-workspace connection protected by `threading.RLock()`.
- Automatic `SET memory_limit = '4GB'` and disk spill directory to prevent OOM on 50M+ row files.
- 3-Layer Read-Only SQL AST validation via `PrivacyFirewall.validate_sql_read_only`.
- Built-in 1-Click 4-Table Enterprise E-Commerce Demo Dataset (`orders`, `customers`, `products`, `returns`)
  with intentional cryptic column names (`ord_stat`, `rev_gross_lcy`, `disc_amt`, `cust_tier`, `ret_reason_cd`)
  so the user can test Join Approval, Smart Business Interview, DAX Studio, and Chat in 30 seconds!
"""

from __future__ import annotations

import re
import threading
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from src.connectors.base import BaseConnector
from src.knowledge.okf_models import (
    CalculatedColumn,
    ColumnProfile,
    RelationshipEdge,
    SemanticMeasure,
    TableRole,
    TableSchema,
)
from src.security.privacy_firewall import PrivacyFirewall


def _sanitize_table_name(raw_name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_]+", "_", raw_name.strip()).strip("_").lower()
    if not cleaned or cleaned[0].isdigit():
        cleaned = f"tbl_{cleaned}"
    return cleaned


class DuckDBConnector(BaseConnector):
    """High-performance local OLAP connector backed by DuckDB."""

    dialect = "duckdb"
    source_type = "duckdb"

    def __init__(self, workspace_dir: str | Path, memory_limit: str = "4GB") -> None:
        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.workspace_dir / "warehouse.duckdb"
        self.temp_dir = self.workspace_dir / ".duckdb_tmp"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.read_only = False
        try:
            self._conn = duckdb.connect(str(self.db_path), read_only=False)
        except duckdb.IOException:
            # If warehouse.duckdb is locked by another process (e.g. parallel worker or background server instance),
            # attach in read_only mode or in-memory so queries, profiling and conversational BI continue smoothly.
            try:
                self._conn = duckdb.connect(str(self.db_path), read_only=True)
                self.read_only = True
            except duckdb.IOException:
                self._conn = duckdb.connect(":memory:")
                self.read_only = True
        self._configure_engine(memory_limit)

    def _configure_engine(self, memory_limit: str) -> None:
        with self._lock:
            self._conn.execute(f"SET memory_limit = '{memory_limit}';")
            self._conn.execute("SET preserve_insertion_order = false;")
            safe_tmp = str(self.temp_dir).replace("\\", "/")
            self._conn.execute(f"SET temp_directory = '{safe_tmp}';")

    def ingest_dataframe(self, table_name: str, df: pd.DataFrame) -> str:
        """Register and materialize a DataFrame as a persistent DuckDB table."""
        safe_name = _sanitize_table_name(table_name)
        with self._lock:
            self._conn.register("_tmp_ingest_df", df)
            self._conn.execute(f'CREATE OR REPLACE TABLE "{safe_name}" AS SELECT * FROM _tmp_ingest_df')
            self._conn.unregister("_tmp_ingest_df")
        return safe_name

    def ingest_file(self, file_path: str | Path, original_filename: str | None = None) -> list[str]:
        """Ingest CSV, Parquet, or multi-sheet Excel (.xlsx/.xls) into persistent DuckDB tables."""
        path = Path(file_path)
        fname = original_filename or path.name
        stem = Path(fname).stem
        suffix = Path(fname).suffix.lower()
        created_tables: list[str] = []

        if suffix == ".csv":
            t_name = _sanitize_table_name(stem)
            safe_p = str(path).replace("\\", "/")
            with self._lock:
                try:
                    self._conn.execute(
                        f'CREATE OR REPLACE TABLE "{t_name}" AS '
                        f"SELECT * FROM read_csv_auto('{safe_p}', ignore_errors=true)"
                    )
                except Exception:
                    df = pd.read_csv(path, encoding_errors="replace")
                    self.ingest_dataframe(t_name, df)
            created_tables.append(t_name)

        elif suffix in (".parquet", ".pq"):
            t_name = _sanitize_table_name(stem)
            safe_p = str(path).replace("\\", "/")
            with self._lock:
                self._conn.execute(
                    f'CREATE OR REPLACE TABLE "{t_name}" AS SELECT * FROM read_parquet(\'{safe_p}\')'
                )
            created_tables.append(t_name)

        elif suffix in (".xlsx", ".xls"):
            sheets: dict[str, pd.DataFrame] = pd.read_excel(path, sheet_name=None)
            for sheet_name, df in sheets.items():
                if df.empty and len(df.columns) == 0:
                    continue
                # Clean column names
                df.columns = [re.sub(r"[^a-zA-Z0-9_]+", "_", str(c).strip()).strip("_") or f"col_{i}" for i, c in enumerate(df.columns)]
                raw_t = sheet_name if len(sheets) > 1 else stem
                t_name = self.ingest_dataframe(raw_t, df)
                created_tables.append(t_name)
        else:
            raise ValueError(f"Unsupported file extension: {suffix}")

        return created_tables

    def load_builtin_demo_dataset(self) -> list[str]:
        """Create a realistic 4-table Enterprise E-Commerce dataset in DuckDB for instant testing.

        Tables:
        - `customers` (250 rows): customer_id, company_name, cust_tier ('ENT', 'MM', 'SMB'), region_code ('NA', 'EMEA', 'APAC', 'LATAM'), contact_email (PII!), acct_mgr
        - `products` (40 rows): product_id, sku_name, category ('Cloud Suite', 'AI Analytics', 'Data Security', 'Edge Hardware'), unit_cost_usd, list_price_usd
        - `orders` (1,500 rows): order_id, customer_id, product_id, order_date, ord_stat ('C', 'P', 'X', 'H'), qty, rev_gross_lcy, disc_amt, shipping_cost, sales_channel
        - `returns` (180 rows): return_id, order_id, customer_id, product_id, return_date, ret_reason_cd ('DEF', 'LATE', 'DUP', 'CHG'), refund_amt_usd
        """
        rng = np.random.default_rng(42)

        # 1. Customers (250 rows)
        n_cust = 250
        tiers = rng.choice(["ENT", "MM", "SMB"], size=n_cust, p=[0.20, 0.35, 0.45])
        regions = rng.choice(["NA", "EMEA", "APAC", "LATAM"], size=n_cust, p=[0.45, 0.28, 0.17, 0.10])
        customers_df = pd.DataFrame(
            {
                "customer_id": np.arange(1001, 1001 + n_cust),
                "company_name": [f"Acme Corp #{i}" for i in range(1, n_cust + 1)],
                "cust_tier": tiers,
                "region_code": regions,
                "contact_email": [f"finance.buyer{i}@enterprise-client.com" for i in range(1, n_cust + 1)],
                "signup_date": [
                    (date(2023, 1, 1) + timedelta(days=int(rng.integers(0, 700)))).isoformat()
                    for _ in range(n_cust)
                ],
            }
        )

        # 2. Products (40 rows)
        categories = ["Cloud Suite", "AI Analytics", "Data Security", "Edge Hardware"]
        prod_rows = []
        for pid in range(101, 141):
            cat = categories[(pid - 101) % len(categories)]
            base_price = {"Cloud Suite": 450.0, "AI Analytics": 800.0, "Data Security": 600.0, "Edge Hardware": 1200.0}[cat]
            list_price = round(base_price * float(rng.uniform(0.8, 1.4)), 2)
            unit_cost = round(list_price * float(rng.uniform(0.32, 0.58)), 2)
            prod_rows.append(
                {
                    "product_id": pid,
                    "sku_name": f"{cat.split()[0]}-Pro-{pid}",
                    "category": cat,
                    "unit_cost_usd": unit_cost,
                    "list_price_usd": list_price,
                }
            )
        products_df = pd.DataFrame(prod_rows)
        price_map = dict(zip(products_df["product_id"], products_df["list_price_usd"]))
        cost_map = dict(zip(products_df["product_id"], products_df["unit_cost_usd"]))

        # 3. Orders (1,500 rows across 2024-01-01 to 2025-12-31)
        n_orders = 1500
        ord_cust_ids = rng.choice(customers_df["customer_id"], size=n_orders)
        ord_prod_ids = rng.choice(products_df["product_id"], size=n_orders)
        ord_statuses = rng.choice(["C", "P", "X", "H"], size=n_orders, p=[0.74, 0.12, 0.09, 0.05])
        channels = rng.choice(["Direct Enterprise", "Partner Reseller", "Self-Serve Web"], size=n_orders, p=[0.45, 0.30, 0.25])
        qtys = rng.integers(1, 25, size=n_orders)

        order_rows = []
        base_dt = date(2024, 1, 1)
        for idx in range(n_orders):
            oid = 50001 + idx
            cid = int(ord_cust_ids[idx])
            pid = int(ord_prod_ids[idx])
            q = int(qtys[idx])
            dt = base_dt + timedelta(days=int(rng.integers(0, 730)))
            gross = round(q * price_map[pid], 2)
            disc = round(gross * float(rng.choice([0.0, 0.05, 0.10, 0.15, 0.20], p=[0.4, 0.25, 0.2, 0.1, 0.05])), 2)
            cogs = round(q * cost_map[pid], 2)
            net_rev = round(gross - disc, 2)
            profit = round(net_rev - cogs, 2)
            order_rows.append(
                {
                    "order_id": oid,
                    "customer_id": cid,
                    "product_id": pid,
                    "order_date": dt.isoformat(),
                    "ord_stat": str(ord_statuses[idx]),
                    "sales_channel": str(channels[idx]),
                    "qty": q,
                    "rev_gross_lcy": gross,
                    "disc_amt": disc,
                    "rev_net_usd": net_rev,
                    "cogs_usd": cogs,
                    "profit_usd": profit,
                }
            )
        orders_df = pd.DataFrame(order_rows)

        # 4. Returns (180 rows)
        sampled_orders = orders_df.sample(n=180, random_state=42).reset_index(drop=True)
        reasons = rng.choice(["DEF", "LATE", "DUP", "CHG"], size=180, p=[0.35, 0.25, 0.15, 0.25])
        return_rows = []
        for idx, orow in sampled_orders.iterrows():
            ret_dt = date.fromisoformat(str(orow["order_date"])) + timedelta(days=int(rng.integers(5, 45)))
            refund = round(float(orow["rev_net_usd"]) * float(rng.uniform(0.5, 1.0)), 2)
            return_rows.append(
                {
                    "return_id": 9001 + int(idx),
                    "order_id": int(orow["order_id"]),
                    "customer_id": int(orow["customer_id"]),
                    "product_id": int(orow["product_id"]),
                    "return_date": ret_dt.isoformat(),
                    "ret_reason_cd": str(reasons[idx]),
                    "refund_amt_usd": refund,
                }
            )
        returns_df = pd.DataFrame(return_rows)

        self.ingest_dataframe("customers", customers_df)
        self.ingest_dataframe("products", products_df)
        self.ingest_dataframe("orders", orders_df)
        self.ingest_dataframe("returns", returns_df)
        return ["customers", "products", "orders", "returns"]

    def apply_calculated_column(self, calc_col: CalculatedColumn) -> list[Any]:
        """Materialize or verify a CalculatedColumn inside DuckDB and return top 5 preview values."""
        if not calc_col.sql_expression:
            return []
        check_sql = (
            f'SELECT ({calc_col.sql_expression}) AS "{calc_col.name}" '
            f'FROM "{calc_col.table_name}" LIMIT 5'
        )
        df = self.execute_query(check_sql, max_rows=5)
        return df[calc_col.name].tolist() if not df.empty else []

    def extract_schema(
        self,
        blocked_tables: set[str] | None = None,
        blocked_columns: set[str] | None = None,
    ) -> tuple[list[TableSchema], list[RelationshipEdge], list[SemanticMeasure]]:
        blocked_t = {t.lower() for t in (blocked_tables or set())}
        blocked_c = {c.lower() for c in (blocked_columns or set())}
        tables: list[TableSchema] = []

        with self._lock:
            tbl_rows = self._conn.execute("SHOW TABLES").fetchall()
            table_names = [r[0] for r in tbl_rows if not str(r[0]).startswith("_")]

            for t_name in table_names:
                is_tbl_blocked = t_name.lower() in blocked_t
                desc_rows = self._conn.execute(f'DESCRIBE "{t_name}"').fetchall()
                cols: list[ColumnProfile] = []
                for idx, drow in enumerate(desc_rows):
                    c_name = str(drow[0])
                    c_type = str(drow[1])
                    is_col_blocked = (
                        c_name.lower() in blocked_c
                        or f"{t_name}.{c_name}".lower() in blocked_c
                    )
                    is_pk = idx == 0 and c_name.lower() in ("id", f"{t_name[:-1]}_id", f"{t_name}_id")
                    cols.append(
                        ColumnProfile(
                            name=c_name,
                            data_type=c_type,
                            is_primary_key=is_pk,
                            is_pii=PrivacyFirewall.is_pii_column_name(c_name),
                            is_blocked=is_col_blocked,
                        )
                    )
                tables.append(
                    TableSchema(
                        name=t_name,
                        is_blocked=is_tbl_blocked,
                        columns=cols,
                    )
                )

        return tables, [], []

    def execute_query(
        self,
        query: str,
        max_rows: int = 10000,
        blocked_tables: set[str] | None = None,
        blocked_columns: set[str] | None = None,
    ) -> pd.DataFrame:
        validated_sql = PrivacyFirewall.validate_sql_read_only(
            query,
            dialect="duckdb",
            blocked_tables=blocked_tables,
            blocked_columns=blocked_columns,
        )
        # Ensure outer row cap
        wrapped_sql = f"SELECT * FROM ({validated_sql.rstrip(';')}) AS _dt_sub LIMIT {int(max_rows)}"
        with self._lock:
            return self._conn.execute(wrapped_sql).df()

    def compute_column_overlap(
        self,
        src_table: str,
        src_col: str,
        tgt_table: str,
        tgt_col: str,
    ) -> float:
        sql = (
            f"SELECT "
            f"COUNT(DISTINCT s.\"{src_col}\") AS src_cnt, "
            f"COUNT(DISTINCT CASE WHEN t.\"{tgt_col}\" IS NOT NULL THEN s.\"{src_col}\" END) AS match_cnt "
            f"FROM (SELECT \"{src_col}\" FROM \"{src_table}\" WHERE \"{src_col}\" IS NOT NULL LIMIT 5000) s "
            f"LEFT JOIN (SELECT DISTINCT \"{tgt_col}\" FROM \"{tgt_table}\" WHERE \"{tgt_col}\" IS NOT NULL) t "
            f"ON s.\"{src_col}\" = t.\"{tgt_col}\""
        )
        with self._lock:
            df = self._conn.execute(sql).df()
        if df.empty or int(df["src_cnt"].iloc[0] or 0) == 0:
            return 0.0
        return float(df["match_cnt"].iloc[0]) / float(df["src_cnt"].iloc[0])
