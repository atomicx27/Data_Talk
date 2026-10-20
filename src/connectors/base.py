"""Abstract Base Connector for all Data_Talk data sources."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from src.knowledge.okf_models import RelationshipEdge, SemanticMeasure, TableSchema


class BaseConnector(ABC):
    """Unified interface for SQL, DuckDB (Excel/CSV/Parquet), and Power BI connectors."""

    dialect: str = "duckdb"
    source_type: str = "duckdb"

    @abstractmethod
    def extract_schema(
        self,
        blocked_tables: set[str] | None = None,
        blocked_columns: set[str] | None = None,
    ) -> tuple[list[TableSchema], list[RelationshipEdge], list[SemanticMeasure]]:
        """Extract raw tables, explicit relationships, and any pre-existing measures."""

    @abstractmethod
    def execute_query(
        self,
        query: str,
        max_rows: int = 10000,
        blocked_tables: set[str] | None = None,
        blocked_columns: set[str] | None = None,
    ) -> pd.DataFrame:
        """Validate read-only safety and execute push-down query, returning capped DataFrame."""

    @abstractmethod
    def compute_column_overlap(
        self,
        src_table: str,
        src_col: str,
        tgt_table: str,
        tgt_col: str,
    ) -> float:
        """Compute push-down sample value inclusion ratio |src ∩ tgt| / |src| in [0.0, 1.0]."""

    def list_raw_tables_and_columns(self) -> dict[str, list[str]]:
        """Return a fast dictionary of {table_name: [col1, col2, ...]} for pre-OKF blocklist UI."""
        tables, _, _ = self.extract_schema()
        return {t.name: [c.name for c in t.columns] for t in tables}
