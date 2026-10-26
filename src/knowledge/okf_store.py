"""Ontology & Knowledge Framework (OKF) Persistence & NetworkX Join Graph Engine.

Responsibilities:
- Persist OKFWorkspace to human-readable `workspaces/<workspace_id>/okf.yaml` (zero credentials stored).
- Classify tables into FACT, DIMENSION, or BRIDGE roles based on row counts & FK topology.
- Build a NetworkX graph of approved joins and solve Steiner-tree shortest join paths across N tables.
- Detect Multi-Fact Fan/Chasm Traps (when 2+ FACT tables join across shared DIMENSION tables).
- Generate Graphviz DOT specifications with a 1-Hop Focal Neighborhood filter for 100+ table schemas.
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import networkx as nx
import yaml

from src.knowledge.okf_models import (
    JoinCardinality,
    JoinStatus,
    OKFWorkspace,
    RelationshipEdge,
    TableRole,
    TableSchema,
)
from src.security.validators import ensure_within, normalize_workspace_id


class JoinPlan(dict):
    """Structured join resolution result returned by OKFStore.get_join_plan()."""

    def __init__(
        self,
        required_tables: list[str],
        bridge_tables: list[str],
        join_edges: list[RelationshipEdge],
        sql_join_clauses: list[str],
        has_fan_trap: bool,
        fact_tables_involved: list[str],
        shared_dimensions: list[str],
        fan_trap_guidance: str,
    ) -> None:
        super().__init__(
            required_tables=required_tables,
            bridge_tables=bridge_tables,
            join_edges=[e.model_dump() for e in join_edges],
            sql_join_clauses=sql_join_clauses,
            has_fan_trap=has_fan_trap,
            fact_tables_involved=fact_tables_involved,
            shared_dimensions=shared_dimensions,
            fan_trap_guidance=fan_trap_guidance,
        )
        self.required_tables = required_tables
        self.bridge_tables = bridge_tables
        self.join_edges = join_edges
        self.sql_join_clauses = sql_join_clauses
        self.has_fan_trap = has_fan_trap
        self.fact_tables_involved = fact_tables_involved
        self.shared_dimensions = shared_dimensions
        self.fan_trap_guidance = fan_trap_guidance


class OKFStore:
    """Thread-safe manager for reading, updating, and querying the OKF semantic model."""

    def __init__(self, base_dir: str | Path = "workspaces") -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def workspace_dir(self, workspace_id: str) -> Path:
        # Defence in depth: never allow a workspace id to escape the workspaces root.
        ws_id = normalize_workspace_id(workspace_id)
        path = self.base_dir / ws_id
        ensure_within(self.base_dir, path)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def okf_path(self, workspace_id: str) -> Path:
        return self.workspace_dir(workspace_id) / "okf.yaml"

    def save(self, workspace: OKFWorkspace) -> Path:
        """Atomically persist OKFWorkspace to okf.yaml after auto-classifying table roles."""
        with self._lock:
            self.classify_table_roles(workspace)
            workspace.updated_at = datetime.now(timezone.utc).isoformat()
            data = workspace.model_dump(mode="json")
            target = self.okf_path(workspace.workspace_id)
            tmp_target = target.with_suffix(".yaml.tmp")
            with open(tmp_target, "w", encoding="utf-8") as f:
                yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
            tmp_target.replace(target)
            return target

    def load(self, workspace_id: str) -> OKFWorkspace | None:
        """Load OKFWorkspace from disk if it exists, healing table roles and FK relationships."""
        with self._lock:
            target = self.okf_path(workspace_id)
            if not target.exists():
                return None
            with open(target, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
            ws = OKFWorkspace.model_validate(raw)
            from src.knowledge.join_inference import JoinInferenceEngine

            ws.relationships = JoinInferenceEngine.infer_relationships(
                ws.tables, existing_relationships=ws.relationships
            )
            self.classify_table_roles(ws)
            return ws

    def list_workspaces(self) -> list[str]:
        if not self.base_dir.exists():
            return []
        return sorted(
            [
                d.name
                for d in self.base_dir.iterdir()
                if d.is_dir() and (d / "okf.yaml").exists()
            ]
        )

    def delete_workspace(self, workspace_id: str) -> bool:
        """Completely and permanently delete a workspace and all its files from disk."""
        ws_id = normalize_workspace_id(workspace_id)
        path = self.base_dir / ws_id
        ensure_within(self.base_dir, path)
        with self._lock:
            if not path.exists():
                return False
            import gc
            import os
            import shutil
            import stat

            gc.collect()

            def _onerror(func, p, exc_info):
                try:
                    os.chmod(p, stat.S_IWRITE)
                    func(p)
                except Exception:
                    pass

            shutil.rmtree(path, onerror=_onerror)
            return True

    @staticmethod
    def classify_table_roles(workspace: OKFWorkspace) -> None:
        """Automatically classify tables as FACT, DIMENSION, or BRIDGE based on FK topology & names."""
        out_degree: dict[str, int] = {t.name: 0 for t in workspace.tables}
        in_degree: dict[str, int] = {t.name: 0 for t in workspace.tables}

        active_rels = [
            r for r in workspace.relationships if r.status != JoinStatus.REJECTED
        ]
        for rel in active_rels:
            if rel.source_table in out_degree:
                out_degree[rel.source_table] += 1
            if rel.target_table in in_degree:
                in_degree[rel.target_table] += 1

        for tbl in workspace.tables:
            name_low = tbl.name.lower()
            if name_low.startswith(("dim_", "d_", "lkp_", "ref_")) or name_low in {
                "customers",
                "products",
                "categories",
                "regions",
                "stores",
                "dates",
                "calendar",
                "employees",
                "suppliers",
                "colleges",
                "departments",
                "degree_programs",
                "programs",
                "courses",
                "academic_terms",
                "terms",
                "students",
                "faculty",
                "instructors",
                "teachers",
                "campuses",
                "branches",
                "accounts",
                "users",
                "patients",
                "doctors",
            }:
                tbl.table_role = TableRole.DIMENSION
            elif name_low.startswith(("fact_", "f_", "trn_", "txn_")) or any(
                k in name_low
                for k in (
                    "orders",
                    "order_items",
                    "sales",
                    "returns",
                    "transactions",
                    "invoices",
                    "fee_invoices",
                    "shipments",
                    "payments",
                    "enrollments",
                    "course_sections",
                    "marks",
                    "grades",
                    "attendance",
                    "claims",
                )
            ):
                tbl.table_role = TableRole.FACT
            elif tbl.table_role != TableRole.UNKNOWN:
                continue
            elif out_degree.get(tbl.name, 0) >= 1 and in_degree.get(tbl.name, 0) == 0:
                tbl.table_role = TableRole.FACT
            elif in_degree.get(tbl.name, 0) >= 1 and out_degree.get(tbl.name, 0) == 0:
                tbl.table_role = TableRole.DIMENSION
            elif out_degree.get(tbl.name, 0) >= 1 and in_degree.get(tbl.name, 0) >= 1:
                tbl.table_role = TableRole.DIMENSION
            else:
                tbl.table_role = TableRole.DIMENSION

    @staticmethod
    def build_graph(
        workspace: OKFWorkspace, include_pending: bool = False
    ) -> nx.Graph:
        """Build an undirected NetworkX graph of table relationships with edge metadata."""
        G = nx.Graph()
        for tbl in workspace.active_tables():
            G.add_node(tbl.name, role=tbl.table_role.value, row_count=tbl.row_count)

        allowed_statuses = {JoinStatus.APPROVED}
        if include_pending:
            allowed_statuses.add(JoinStatus.PENDING)

        for rel in workspace.relationships:
            if rel.status not in allowed_statuses:
                continue
            if G.has_node(rel.source_table) and G.has_node(rel.target_table):
                # Prefer higher-confidence or approved edges as lower weight for shortest path
                weight = 1.0 if rel.status == JoinStatus.APPROVED else 1.5
                G.add_edge(
                    rel.source_table,
                    rel.target_table,
                    edge_obj=rel,
                    weight=weight,
                )
        return G

    def get_join_plan(
        self,
        workspace: OKFWorkspace,
        target_tables: list[str],
        include_pending_fallback: bool = True,
    ) -> JoinPlan:
        """Compute the Steiner-tree shortest join path connecting `target_tables` and check for Fan Traps."""
        valid_targets = [
            t for t in dict.fromkeys(target_tables) if workspace.get_table(t) is not None
        ]
        if len(valid_targets) <= 1:
            return JoinPlan(
                required_tables=valid_targets,
                bridge_tables=[],
                join_edges=[],
                sql_join_clauses=[],
                has_fan_trap=False,
                fact_tables_involved=[],
                shared_dimensions=[],
                fan_trap_guidance="",
            )

        G = self.build_graph(workspace, include_pending=False)
        # If approved edges don't connect all target tables, fallback to including pending edges
        if include_pending_fallback:
            connected = True
            for u, v in combinations(valid_targets, 2):
                if not (G.has_node(u) and G.has_node(v) and nx.has_path(G, u, v)):
                    connected = False
                    break
            if not connected:
                G = self.build_graph(workspace, include_pending=True)

        # Chasm-aware, cycle-free Steiner tree construction
        # Order target tables so FACT tables and granular child dimensions are connected
        # before coarse parent dimensions (e.g., departments before colleges).
        def _target_priority(t_name: str) -> tuple[int, int, str]:
            t_obj = workspace.get_table(t_name)
            role_rank = 0 if (t_obj and t_obj.table_role == TableRole.FACT) else 1
            # Prefer tables that have outgoing N:1 FKs (granular child tables) before coarse parent tables
            out_fks = sum(
                1
                for r in workspace.relationships
                if r.source_table == t_name and r.status != JoinStatus.REJECTED
            )
            return (role_rank, -out_fks, t_name)

        ordered_targets = sorted(valid_targets, key=_target_priority)

        def _score_candidate_path(
            path: list[str], existing_edges: list[RelationshipEdge]
        ) -> float:
            """Score a path between nodes, penalizing 1:N fan-out reversals and N:1 <- 1:N chasm turns."""
            cost = 0.0
            edges_along: list[RelationshipEdge] = []
            # Nodes that are already reached as PK targets (N:1) in the existing join tree
            existing_pk_targets = {e.target_table for e in existing_edges}

            for i in range(len(path) - 1):
                u_node, v_node = path[i], path[i + 1]
                ed = G.get_edge_data(u_node, v_node)
                base_w = float(ed.get("weight", 1.0)) if ed else 1.0
                rel_obj: RelationshipEdge | None = ed.get("edge_obj") if ed else None
                if rel_obj is not None:
                    edges_along.append(rel_obj)
                    # Forward N:1 (source -> target) preserves grain; reverse 1:N risks fan-out
                    if rel_obj.source_table == u_node and rel_obj.target_table == v_node:
                        cost += base_w * 1.0
                    else:
                        cost += base_w * 1.35
                        # If u_node was already reached via N:1 (e.g. students -> colleges)
                        # and we now traverse 1:N out of u_node (colleges <- departments),
                        # that creates an N:1:N chasm fan-out at u_node!
                        if u_node in existing_pk_targets:
                            cost += 8.0
                else:
                    cost += base_w

            # Also penalize head-to-head chasm turns within the candidate path itself:
            # A --(N:1)--> C <--(1:N)-- B where both edges have target_table == C.
            for i in range(len(edges_along) - 1):
                mid_node = path[i + 1]
                e1, e2 = edges_along[i], edges_along[i + 1]
                if e1.target_table == mid_node and e2.target_table == mid_node:
                    cost += 8.0
            return cost

        path_nodes: set[str] = set()
        path_edges: list[RelationshipEdge] = []
        seen_edge_ids: set[str] = set()

        if ordered_targets:
            tree_nodes: set[str] = {ordered_targets[0]}
            path_nodes.add(ordered_targets[0])
            remaining = [t for t in ordered_targets[1:] if G.has_node(t)]

            while remaining:
                # Connect targets in specificity order (FACT -> granular child dim -> coarse parent dim)
                # while still picking the lowest-cost chasm-free path from the current tree.
                best_target = None
                best_path: list[str] | None = None
                best_score = float("inf")

                for idx, target in enumerate(remaining):
                    if target in tree_nodes:
                        best_target = target
                        best_path = [target]
                        best_score = -1.0
                        break
                    for src in tree_nodes:
                        if not (G.has_node(src) and nx.has_path(G, src, target)):
                            continue
                        try:
                            cand_paths = list(nx.all_simple_paths(G, source=src, target=target, cutoff=4))
                        except Exception:
                            cand_paths = [nx.shortest_path(G, source=src, target=target, weight="weight")]
                        if not cand_paths:
                            cand_paths = [nx.shortest_path(G, source=src, target=target, weight="weight")]

                        for cp in cand_paths:
                            # Priority penalty (idx * 2.5) ensures granular child dimensions (e.g. departments)
                            # are connected via N:1 hierarchy before coarse parent shortcuts (e.g. colleges).
                            score = _score_candidate_path(cp, path_edges) + (idx * 2.5)
                            if src != ordered_targets[0]:
                                score -= 0.02
                            if score < best_score:
                                best_score = score
                                best_path = cp
                                best_target = target

                if best_target is None or best_path is None:
                    # Remaining targets are in a disconnected component; add next target as a new seed
                    next_seed = remaining.pop(0)
                    tree_nodes.add(next_seed)
                    path_nodes.add(next_seed)
                    continue

                remaining.remove(best_target)
                path_nodes.update(best_path)
                tree_nodes.update(best_path)
                for i in range(len(best_path) - 1):
                    edge_data = G.get_edge_data(best_path[i], best_path[i + 1])
                    if edge_data and "edge_obj" in edge_data:
                        rel: RelationshipEdge = edge_data["edge_obj"]
                        if rel.id not in seen_edge_ids:
                            seen_edge_ids.add(rel.id)
                            path_edges.append(rel)

        if path_edges:
            connected_tables = {e.source_table for e in path_edges} | {e.target_table for e in path_edges}
            path_nodes = {t for t in path_nodes if t in connected_tables}

        bridge_tables = [t for t in path_nodes if t not in valid_targets]
        sql_clauses = [
            f'{e.join_type} JOIN "{e.target_table}" ON "{e.source_table}"."{e.source_column}" = "{e.target_table}"."{e.target_column}"'
            for e in path_edges
        ]

        # Multi-Fact Fan / Chasm Trap Detection
        involved_table_objs = [
            workspace.get_table(t) for t in path_nodes if workspace.get_table(t)
        ]
        fact_tables = [
            t.name for t in involved_table_objs if t and t.table_role == TableRole.FACT
        ]
        dim_tables = [
            t.name
            for t in involved_table_objs
            if t and t.table_role == TableRole.DIMENSION
        ]

        # Check if 2+ fact tables DO NOT have a direct PK-FK parent-child header-detail relationship
        # (e.g., orders <-> order_items is parent-child header-detail; orders <-> returns via customers/products is a chasm/fan trap)
        has_fan_trap = False
        fan_trap_guidance = ""
        if len(fact_tables) >= 2:
            # Check if every pair of fact tables is only connected via a dimension table or both have N:1 into the same table
            for f1, f2 in combinations(fact_tables, 2):
                direct_edge = G.get_edge_data(f1, f2)
                if not direct_edge:
                    has_fan_trap = True
                else:
                    # If both f1 and f2 are transaction tables joining on a non-unique key, flag CTE split
                    if any(
                        w in f1.lower() or w in f2.lower()
                        for w in ("return", "refund", "budget", "target", "payment")
                    ):
                        has_fan_trap = True

        if has_fan_trap:
            fan_trap_guidance = (
                f"CRITICAL FAN/CHASM TRAP WARNING: Multiple fact tables ({', '.join(fact_tables)}) "
                f"are involved. Do NOT join {', '.join(fact_tables)} directly in a single SELECT before aggregating, "
                "as this causes a Cartesian fan-out that multiplies SUM() totals. "
                "Instead, write a separate CTE for each fact table pre-aggregated by the shared dimension key(s) "
                f"({', '.join(dim_tables) or 'grouping keys'}), and then JOIN the pre-aggregated CTEs!"
            )

        return JoinPlan(
            required_tables=sorted(path_nodes),
            bridge_tables=sorted(bridge_tables),
            join_edges=path_edges,
            sql_join_clauses=sql_clauses,
            has_fan_trap=has_fan_trap,
            fact_tables_involved=fact_tables,
            shared_dimensions=dim_tables,
            fan_trap_guidance=fan_trap_guidance,
        )

    def generate_er_graphviz(
        self,
        workspace: OKFWorkspace,
        focal_tables: list[str] | None = None,
        max_tables: int = 15,
    ) -> str:
        """Generate a clean Graphviz DOT diagram with 1-hop neighborhood filtering for large schemas."""
        active_tables = workspace.active_tables()
        G = self.build_graph(workspace, include_pending=True)

        if focal_tables:
            visible_names: set[str] = set()
            for ft in focal_tables:
                if G.has_node(ft):
                    visible_names.add(ft)
                    visible_names.update(G.neighbors(ft))
            selected_tables = [t for t in active_tables if t.name in visible_names]
        else:
            # Sort by degree so most connected tables appear first, capped at max_tables
            ranked = sorted(
                active_tables,
                key=lambda t: G.degree(t.name) if G.has_node(t.name) else 0,
                reverse=True,
            )
            selected_tables = ranked[:max_tables]

        visible_set = {t.name for t in selected_tables}

        lines = [
            "digraph OKF_ER {",
            '  rankdir="LR";',
            '  graph [bgcolor="transparent", pad="0.3", nodesep="0.5", ranksep="0.9", fontname="Helvetica"];',
            '  node [shape=plaintext, fontname="Helvetica"];',
            '  edge [fontname="Helvetica", fontsize=10];',
        ]

        for tbl in selected_tables:
            header_bg = (
                "#1E3A8A"
                if tbl.table_role == TableRole.FACT
                else "#065F46"
                if tbl.table_role == TableRole.DIMENSION
                else "#374151"
            )
            role_badge = tbl.table_role.value
            col_rows = []
            for col in tbl.columns[:10]:
                if col.is_blocked:
                    continue
                icon = "🔑 " if col.is_primary_key else ("🔗 " if col.is_foreign_key else "• ")
                pii_tag = " [PII]" if col.is_pii else ""
                col_rows.append(
                    f'<tr><td align="left" cellpadding="4"><font color="#E5E7EB">{icon}{col.name}</font></td>'
                    f'<td align="right" cellpadding="4"><font color="#9CA3AF">{col.data_type}{pii_tag}</font></td></tr>'
                )
            if len(tbl.columns) > 10:
                extra = len(tbl.columns) - 10
                col_rows.append(
                    f'<tr><td colspan="2" align="left" cellpadding="4"><font color="#9CA3AF">... +{extra} more columns</font></td></tr>'
                )

            src_badge = f"[{tbl.source_name or tbl.source_id}] " if tbl.source_id and tbl.source_id != "default" else ""
            table_html = (
                f'<<table border="0" cellborder="1" cellspacing="0" cellpadding="4" bgcolor="#111827" color="#374151">'
                f'<tr><td colspan="2" bgcolor="{header_bg}" cellpadding="6">'
                f'<font color="#FFFFFF"><b>{src_badge}{tbl.name}</b> ({role_badge} • {tbl.row_count:,} rows)</font></td></tr>'
                f'{"".join(col_rows)}'
                f"</table>>"
            )
            lines.append(f'  "{tbl.name}" [label={table_html}];')

        for rel in workspace.relationships:
            if rel.status == JoinStatus.REJECTED:
                continue
            if rel.source_table in visible_set and rel.target_table in visible_set:
                if rel.is_cross_source:
                    color = "#8B5CF6" if rel.status == JoinStatus.APPROVED else "#A78BFA"
                    prefix = "🌐 Cross-DB: "
                else:
                    color = "#10B981" if rel.status == JoinStatus.APPROVED else "#F59E0B"
                    prefix = ""
                style = "solid" if rel.status == JoinStatus.APPROVED else "dashed"
                status_tag = "✓" if rel.status == JoinStatus.APPROVED else "?"
                label = f"{prefix}{rel.source_column} → {rel.target_column}\\n({rel.cardinality.value} {status_tag} {int(rel.confidence*100)}%)"
                penwidth = 2.5 if rel.is_cross_source else 2.0
                lines.append(
                    f'  "{rel.source_table}" -> "{rel.target_table}" '
                    f'[label="{label}", color="{color}", fontcolor="{color}", style="{style}", penwidth={penwidth}];'
                )

        lines.append("}")
        return "\n".join(lines)
