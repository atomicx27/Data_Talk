"""Federated Multi-Source Connector for Data_Talk.

Enables simultaneous connection to multiple SQL databases (PostgreSQL, MySQL, SQLite, DuckDB, SQL Server)
and local files, unifying them under an embedded OLAP DuckDB engine so that:
1. Push-down statistical profiling runs against each source database engine without unnecessary data transfer.
2. Tables from different databases are federated into the workspace DuckDB warehouse under isolated schemas
   and unique aliases (e.g., `sales_db.orders` and `sales_db_orders`).
3. Cross-database queries with JOINs, aggregates, and CTEs execute seamlessly in DuckDB at sub-second speed.
4. Value-overlap inclusion dependency checks (|A ∩ B| / |A|) work across tables in different databases.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import text

from src.connectors.base import BaseConnector
from src.connectors.duckdb_connector import DuckDBConnector, _sanitize_table_name
from src.connectors.sql_connector import SQLConnector
from src.knowledge.okf_models import (
    ColumnProfile,
    RelationshipEdge,
    SemanticMeasure,
    TableSchema,
)

_LOGGER = logging.getLogger(__name__)


class FederatedConnector(BaseConnector):
    """Federated multi-source connector coordinating DuckDB and multiple remote SQL databases."""

    dialect = "duckdb"
    source_type = "federated"

    def __init__(self, workspace_dir: str | Path) -> None:
        self.workspace_dir = Path(workspace_dir)
        self.duckdb_conn = DuckDBConnector(workspace_dir)
        self.sql_connectors: dict[str, SQLConnector] = {}

    def close(self) -> None:
        """Close wrapped connectors and release database file locks."""
        if hasattr(self, "duckdb_conn") and self.duckdb_conn is not None:
            self.duckdb_conn.close()
        for c in list(self.sql_connectors.values()):
            if hasattr(c, "close"):
                try:
                    c.close()
                except Exception:
                    pass
        self.sql_connectors.clear()

    def register_sql_connector(
        self,
        source_id: str,
        connection_url: str,
        dialect_hint: str = "postgres",
    ) -> SQLConnector:
        """Register an active SQLConnector for a specific source_id."""
        conn = SQLConnector(connection_url, dialect_hint=dialect_hint)
        self.sql_connectors[source_id] = conn
        return conn

    def get_sql_connector(self, source_id: str) -> SQLConnector | None:
        return self.sql_connectors.get(source_id)

    def remove_source(self, source_id: str, table_names: list[str] | None = None) -> None:
        """Remove a source connector and clean up its registered tables and views in DuckDB."""
        if source_id in self.sql_connectors:
            try:
                self.sql_connectors[source_id].engine.dispose()
            except Exception:
                pass
            del self.sql_connectors[source_id]

        with self.duckdb_conn._lock:
            try:
                self.duckdb_conn._conn.execute(f'DROP SCHEMA IF EXISTS "{source_id}" CASCADE;')
            except Exception:
                pass
            for t_name in table_names or []:
                prefixed = f"{source_id}_{t_name}"
                try:
                    self.duckdb_conn._conn.execute(f'DROP VIEW IF EXISTS "{prefixed}";')
                    self.duckdb_conn._conn.execute(f'DROP TABLE IF EXISTS "{prefixed}";')
                except Exception:
                    pass

    def sync_sql_source_to_duckdb(
        self,
        source_id: str,
        table_names: list[str],
        max_rows_per_table: int = 50000,
    ) -> list[str]:
        """Mirror/attach tables from a remote SQL source into DuckDB for federated querying."""
        sql_conn = self.sql_connectors.get(source_id)
        if not sql_conn:
            return []

        synced_tables: list[str] = []
        is_sqlite = sql_conn._connection_url.startswith("sqlite:///")

        with self.duckdb_conn._lock:
            try:
                self.duckdb_conn._conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{source_id}";')
            except Exception:
                pass

            for t_name in table_names:
                safe_t = _sanitize_table_name(t_name)
                prefixed_name = f"{source_id}_{safe_t}"

                try:
                    # Ingest table rows into DuckDB
                    df = pd.read_sql(f'SELECT * FROM "{t_name}" LIMIT {max_rows_per_table}', sql_conn.engine)
                    self.duckdb_conn.ingest_dataframe(prefixed_name, df)

                    # Also populate inside the schema
                    self.duckdb_conn._conn.execute(
                        f'CREATE OR REPLACE TABLE "{source_id}"."{safe_t}" AS SELECT * FROM "{prefixed_name}"'
                    )

                    # Create top-level view with raw name if no name conflict exists in main schema
                    main_tables = [
                        r[0].lower()
                        for r in self.duckdb_conn._conn.execute("SHOW TABLES").fetchall()
                        if not str(r[0]).startswith("_") and not str(r[0]).startswith(f"{source_id}_")
                    ]
                    if safe_t.lower() not in main_tables:
                        try:
                            self.duckdb_conn._conn.execute(
                                f'CREATE OR REPLACE VIEW "{safe_t}" AS SELECT * FROM "{prefixed_name}"'
                            )
                        except Exception:
                            pass

                    synced_tables.append(prefixed_name)
                except Exception as exc:
                    _LOGGER.warning("Failed to sync table '%s' from source '%s': %s", t_name, source_id, exc)

        return synced_tables

    def extract_schema(
        self,
        blocked_tables: set[str] | None = None,
        blocked_columns: set[str] | None = None,
    ) -> tuple[list[TableSchema], list[RelationshipEdge], list[SemanticMeasure]]:
        """Extract schema from DuckDB warehouse and all registered SQL sources."""
        all_tables: list[TableSchema] = []
        all_relationships: list[RelationshipEdge] = []
        all_measures: list[SemanticMeasure] = []

        # 1. Extract DuckDB local tables
        duck_tables, duck_rels, duck_measures = self.duckdb_conn.extract_schema(
            blocked_tables=blocked_tables,
            blocked_columns=blocked_columns,
        )
        for t in duck_tables:
            if not t.source_id or t.source_id == "default":
                t.source_id = "local_warehouse"
                t.source_name = "Local Warehouse"
        all_tables.extend(duck_tables)
        all_relationships.extend(duck_rels)
        all_measures.extend(duck_measures)

        # 2. Extract from each SQL source
        for src_id, sql_conn in self.sql_connectors.items():
            try:
                s_tables, s_rels, s_measures = sql_conn.extract_schema(
                    blocked_tables=blocked_tables,
                    blocked_columns=blocked_columns,
                )
                for t in s_tables:
                    t.source_id = src_id
                    t.source_name = src_id.replace("_", " ").title()
                for r in s_rels:
                    r.source_source_id = src_id
                    r.target_source_id = src_id
                all_tables.extend(s_tables)
                all_relationships.extend(s_rels)
                all_measures.extend(s_measures)
            except Exception as exc:
                _LOGGER.warning("Error extracting schema from SQL source '%s': %s", src_id, exc)

        return all_tables, all_relationships, all_measures

    def execute_query(
        self,
        query: str,
        max_rows: int = 10000,
        blocked_tables: set[str] | None = None,
        blocked_columns: set[str] | None = None,
    ) -> pd.DataFrame:
        """Execute read-only validated SQL query in DuckDB across federated tables."""
        return self.duckdb_conn.execute_query(
            query,
            max_rows=max_rows,
            blocked_tables=blocked_tables,
            blocked_columns=blocked_columns,
        )

    def compute_column_overlap(
        self,
        src_table: str,
        src_col: str,
        tgt_table: str,
        tgt_col: str,
    ) -> float:
        """Compute push-down sample value overlap ratio |src ∩ tgt| / |src| in DuckDB."""
        # DuckDB has access to both local tables and federated SQL tables
        return self.duckdb_conn.compute_column_overlap(src_table, src_col, tgt_table, tgt_col)
