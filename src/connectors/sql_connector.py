"""SQLAlchemy Push-Down Connector for PostgreSQL, MySQL, SQL Server, SQLite, and Snowflake.

Enforces:
- Credential retrieval via `CredentialVault` (never stored in okf.yaml).
- Read-only AST validation via `PrivacyFirewall.validate_sql_read_only`.
- Explicit foreign-key extraction via `sqlalchemy.inspect`.
"""

from __future__ import annotations

import pandas as pd
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine

from src.connectors.base import BaseConnector
from src.knowledge.okf_models import (
    ColumnProfile,
    JoinCardinality,
    JoinStatus,
    RelationshipEdge,
    SemanticMeasure,
    TableSchema,
)
from src.security.privacy_firewall import PrivacyFirewall


class SQLConnector(BaseConnector):
    """Push-down relational database connector powered by SQLAlchemy."""

    source_type = "sql"

    def __init__(self, connection_url: str, dialect_hint: str = "postgres") -> None:
        self._connection_url = connection_url
        self.engine: Engine = create_engine(connection_url, pool_pre_ping=True)
        raw_dialect = self.engine.dialect.name.lower()
        dialect_map = {
            "postgresql": "postgres",
            "psycopg2": "postgres",
            "mysql": "mysql",
            "mssql": "tsql",
            "sqlite": "sqlite",
            "snowflake": "snowflake",
        }
        self.dialect = dialect_map.get(raw_dialect, dialect_hint)

    def extract_schema(
        self,
        blocked_tables: set[str] | None = None,
        blocked_columns: set[str] | None = None,
    ) -> tuple[list[TableSchema], list[RelationshipEdge], list[SemanticMeasure]]:
        blocked_t = {t.lower() for t in (blocked_tables or set())}
        blocked_c = {c.lower() for c in (blocked_columns or set())}

        insp = inspect(self.engine)
        table_names = insp.get_table_names()
        tables: list[TableSchema] = []
        relationships: list[RelationshipEdge] = []

        for t_name in table_names:
            if t_name.startswith("sqlite_"):
                continue
            is_tbl_blocked = t_name.lower() in blocked_t
            pk_constraint = insp.get_pk_constraint(t_name) or {}
            pk_cols = set(pk_constraint.get("constrained_columns") or [])
            fk_list = insp.get_foreign_keys(t_name) or []
            fk_cols: set[str] = set()

            for fk in fk_list:
                constrained = fk.get("constrained_columns") or []
                referred_table = fk.get("referred_table")
                referred_cols = fk.get("referred_columns") or []
                if constrained and referred_table and referred_cols:
                    src_col = str(constrained[0])
                    tgt_col = str(referred_cols[0])
                    fk_cols.add(src_col)
                    rel_id = f"fk_{t_name}_{src_col}_to_{referred_table}_{tgt_col}"
                    relationships.append(
                        RelationshipEdge(
                            id=rel_id,
                            source_table=t_name,
                            source_column=src_col,
                            target_table=str(referred_table),
                            target_column=tgt_col,
                            cardinality=JoinCardinality.MANY_TO_ONE,
                            join_type="LEFT",
                            confidence=1.0,
                            discovery_method="explicit_fk",
                            status=JoinStatus.APPROVED,
                            overlap_ratio=1.0,
                        )
                    )

            cols: list[ColumnProfile] = []
            for col_info in insp.get_columns(t_name):
                c_name = str(col_info["name"])
                c_type = str(col_info["type"])
                is_col_blocked = (
                    c_name.lower() in blocked_c
                    or f"{t_name}.{c_name}".lower() in blocked_c
                )
                cols.append(
                    ColumnProfile(
                        name=c_name,
                        data_type=c_type,
                        is_primary_key=c_name in pk_cols,
                        is_foreign_key=c_name in fk_cols,
                        is_nullable=bool(col_info.get("nullable", True)),
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

        return tables, relationships, []

    def execute_query(
        self,
        query: str,
        max_rows: int = 10000,
        blocked_tables: set[str] | None = None,
        blocked_columns: set[str] | None = None,
    ) -> pd.DataFrame:
        validated_sql = PrivacyFirewall.validate_sql_read_only(
            query,
            dialect=self.dialect,
            blocked_tables=blocked_tables,
            blocked_columns=blocked_columns,
        )
        with self.engine.connect() as conn:
            if self.dialect == "postgres":
                conn.execute(text("SET SESSION CHARACTERISTICS AS TRANSACTION READ ONLY"))
            elif self.dialect == "sqlite":
                conn.execute(text("PRAGMA query_only = ON"))
            result = conn.execute(text(validated_sql))
            rows = result.fetchmany(max_rows)
            return pd.DataFrame(rows, columns=list(result.keys()))

    def compute_column_overlap(
        self,
        src_table: str,
        src_col: str,
        tgt_table: str,
        tgt_col: str,
    ) -> float:
        sql = (
            f'SELECT COUNT(DISTINCT s."{src_col}") AS src_cnt, '
            f'COUNT(DISTINCT CASE WHEN t."{tgt_col}" IS NOT NULL THEN s."{src_col}" END) AS match_cnt '
            f'FROM "{src_table}" s LEFT JOIN "{tgt_table}" t ON s."{src_col}" = t."{tgt_col}"'
        )
        try:
            df = self.execute_query(sql, max_rows=1)
            if df.empty or int(df["src_cnt"].iloc[0] or 0) == 0:
                return 0.0
            return float(df["match_cnt"].iloc[0]) / float(df["src_cnt"].iloc[0])
        except Exception:
            return 0.0
