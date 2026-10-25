"""3-Signal Join Relationship Inference Engine for Data_Talk.

Discovers candidate joins across tables when explicit Foreign Keys are missing
(common in Excel/CSV workbooks, staging databases, and denormalized schemas):
- Signal 1 (Confidence = 1.00): Explicit database Foreign Keys or Power BI DMV relationships.
- Signal 2 (Confidence = 0.65 - 0.90): Lexical & Data-Type key matching (e.g., `orders.customer_id` <-> `customers.id` or `customers.customer_id`).
- Signal 3 (Confidence boost/penalty): Push-down sample value overlap check (|A ∩ B| / |A|) & cardinality detection (N:1 vs 1:1).
"""

from __future__ import annotations

import re
from itertools import permutations
from typing import TYPE_CHECKING

from src.knowledge.okf_models import (
    ColumnProfile,
    JoinCardinality,
    JoinStatus,
    RelationshipEdge,
    TableSchema,
)

if TYPE_CHECKING:
    from src.connectors.base import BaseConnector


def _singularize(word: str) -> str:
    w = word.lower().strip()
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith(("sses", "shes", "ches", "xes", "zes")) and len(w) > 4:
        return w[:-2]
    if w in ("campuses", "buses", "bonuses", "statuses", "syllabuses", "viruses"):
        return w[:-2]
    if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        return w[:-1]
    return w


def _normalize_table_stem(table_name: str) -> str:
    cleaned = re.sub(r"^(dim_|fact_|f_|d_|tbl_|stg_|raw_)", "", table_name.lower())
    return _singularize(cleaned)


def _compatible_types(type_a: str, type_b: str) -> bool:
    ta = type_a.upper()
    tb = type_b.upper()
    int_types = ("INT", "BIGINT", "SMALLINT", "INTEGER", "HUGEINT", "NUMBER", "NUMERIC", "DECIMAL")
    str_types = ("VARCHAR", "TEXT", "CHAR", "STRING", "NVARCHAR", "UUID")
    if any(t in ta for t in int_types) and any(t in tb for t in int_types):
        return True
    if any(t in ta for t in str_types) and any(t in tb for t in str_types):
        return True
    return ta == tb


def _is_candidate_key_column(col: ColumnProfile, table: TableSchema) -> bool:
    if col.is_primary_key:
        return True
    name_low = col.name.lower()
    stem = _normalize_table_stem(table.name)
    last_part = _singularize(stem.split("_")[-1]) if "_" in stem else stem
    candidate_names = {
        "id", "code", "key",
        f"{stem}_id", f"{stem}_key", f"{stem}_code", f"{stem}id",
        f"{last_part}_id", f"{last_part}_key", f"{last_part}_code", f"{last_part}id",
    }
    if name_low in candidate_names:
        if table.row_count > 0 and col.distinct_count >= int(table.row_count * 0.90):
            return True
        if table.row_count == 0:
            return True
    return False


class JoinInferenceEngine:
    """Infers foreign-key relationships across tables using schema, lexical, and value-overlap signals."""

    @classmethod
    def infer_relationships(
        cls,
        tables: list[TableSchema],
        existing_relationships: list[RelationshipEdge] | None = None,
        connector: "BaseConnector | None" = None,
    ) -> list[RelationshipEdge]:
        """Discover and score candidate relationships across active tables."""
        discovered: dict[tuple[str, str, str, str], RelationshipEdge] = {}

        # 1. Preserve any explicit FK / Power BI DMV relationships first
        tbl_by_name = {t.name: t for t in tables}
        for rel in existing_relationships or []:
            key = (rel.source_table, rel.source_column, rel.target_table, rel.target_column)
            discovered[key] = rel
            src_t = tbl_by_name.get(rel.source_table)
            if src_t:
                sc = src_t.get_column(rel.source_column)
                if sc:
                    sc.is_foreign_key = True

        active_tables = [t for t in tables if not t.is_blocked]

        # 2. Lexical + Type + Uniqueness discovery across ordered pairs (Fact/Source -> Dim/Target)
        for src_tbl, tgt_tbl in permutations(active_tables, 2):
            tgt_stem = _normalize_table_stem(tgt_tbl.name)
            tgt_last = _singularize(tgt_stem.split("_")[-1]) if "_" in tgt_stem else tgt_stem
            for tgt_col in tgt_tbl.columns:
                if tgt_col.is_blocked or not _is_candidate_key_column(tgt_col, tgt_tbl):
                    continue

                tgt_col_low = tgt_col.name.lower()
                for src_col in src_tbl.columns:
                    if src_col.is_blocked or not _compatible_types(src_col.data_type, tgt_col.data_type):
                        continue

                    src_col_low = src_col.name.lower()
                    # Avoid matching two generic 'id' columns unless table names align
                    lexical_score = 0.0
                    if src_col_low == tgt_col_low and src_col_low not in ("id", "key", "code", "name", "status", "type"):
                        # e.g., orders.customer_id == customers.customer_id or students.program_id == degree_programs.program_id
                        if tgt_stem in src_col_low or tgt_last in src_col_low or src_col_low.endswith(("_id", "_key", "_code", "id")):
                            lexical_score = 0.92 if (tgt_stem in src_col_low or tgt_last in src_col_low) else 0.88
                    elif tgt_col_low in (
                        "id", "key", "code",
                        f"{tgt_stem}_id", f"{tgt_stem}_key", f"{tgt_stem}_code",
                        f"{tgt_last}_id", f"{tgt_last}_key", f"{tgt_last}_code",
                    ):
                        # e.g., orders.customer_id -> customers.id or orders.cust_id -> customers.cust_id
                        expected_names = {
                            f"{tgt_stem}_id",
                            f"{tgt_stem}id",
                            f"{tgt_stem}_key",
                            f"{tgt_stem}_code",
                            f"{tgt_stem[:4]}_id",
                            f"{tgt_last}_id",
                            f"{tgt_last}_key",
                            f"{tgt_last}_code",
                        }
                        if src_col_low in expected_names:
                            lexical_score = 0.86

                    if lexical_score <= 0.0:
                        continue

                    # Prevent duplicate reverse edges if tgt_tbl is clearly the parent (unique key) and src_tbl is child
                    src_is_unique = (
                        src_tbl.row_count > 0
                        and src_col.distinct_count >= int(src_tbl.row_count * 0.98)
                    )
                    tgt_is_unique = (
                        tgt_tbl.row_count > 0
                        and tgt_col.distinct_count >= int(tgt_tbl.row_count * 0.95)
                    )
                    if src_is_unique and not tgt_is_unique:
                        continue
                    if (
                        src_is_unique
                        and tgt_is_unique
                        and src_tbl.row_count < tgt_tbl.row_count
                    ):
                        continue

                    key = (src_tbl.name, src_col.name, tgt_tbl.name, tgt_col.name)
                    rev_key = (tgt_tbl.name, tgt_col.name, src_tbl.name, src_col.name)
                    if key in discovered or rev_key in discovered:
                        continue

                    # 3. Signal 3: Push-down value overlap verification if connector is available
                    overlap_ratio = 0.0
                    final_conf = lexical_score
                    if connector is not None:
                        try:
                            overlap_ratio = connector.compute_column_overlap(
                                src_tbl.name, src_col.name, tgt_tbl.name, tgt_col.name
                            )
                            if overlap_ratio >= 0.80:
                                final_conf = min(0.98, lexical_score + 0.10)
                            elif overlap_ratio >= 0.40:
                                final_conf = lexical_score
                            elif overlap_ratio == 0.0 and src_tbl.row_count > 0 and tgt_tbl.row_count > 0:
                                # Zero matching values -> reject false lexical match
                                continue
                        except Exception:
                            overlap_ratio = 0.0

                    cardinality = (
                        JoinCardinality.ONE_TO_ONE
                        if (src_is_unique and tgt_is_unique)
                        else JoinCardinality.MANY_TO_ONE
                    )
                    rel_id = f"rel_{src_tbl.name}_{src_col.name}_to_{tgt_tbl.name}_{tgt_col.name}"
                    is_cross = bool(
                        src_tbl.source_id
                        and tgt_tbl.source_id
                        and src_tbl.source_id != tgt_tbl.source_id
                    )
                    disc_method = (
                        "cross_database_link"
                        if is_cross
                        else ("lexical_overlap" if overlap_ratio > 0 else "lexical_schema")
                    )
                    discovered[key] = RelationshipEdge(
                        id=rel_id,
                        source_table=src_tbl.name,
                        source_column=src_col.name,
                        target_table=tgt_tbl.name,
                        target_column=tgt_col.name,
                        cardinality=cardinality,
                        join_type="LEFT",
                        confidence=round(final_conf, 2),
                        discovery_method=disc_method,
                        status=JoinStatus.APPROVED if final_conf >= 0.92 else JoinStatus.PENDING,
                        overlap_ratio=round(overlap_ratio, 3),
                        is_cross_source=is_cross,
                        source_source_id=src_tbl.source_id,
                        target_source_id=tgt_tbl.source_id,
                    )
                    src_col.is_foreign_key = True

        return sorted(
            discovered.values(), key=lambda r: (r.confidence, r.overlap_ratio), reverse=True
        )
