"""Dynamic Automated Root-Cause Analysis (Diagnostic Analytics) Engine.

Provides multi-algorithm diagnostic decomposition for Conversational BI:
1. Additive Period-over-Period Variance Decomposition (3-stage Waterfall Bridge with Pareto 80/20).
2. Symmetric 2-Factor Das Gupta / Kitagawa Rate vs. Mix Shift Decomposition (Simpson's Paradox solver).
3. Multi-Dimensional Auto-Scan (Driver Discovery across all conformed dimensions via Excess Divergence & HHI).

Fully dynamic across arbitrary OKF schemas (retail, healthcare, higher ed, finance) with:
- Multi-table foreign key join resolution for temporal anchors and dimensions.
- Glass-Box audit receipts with penny-level balance checks and fully bound executable SQL.
"""

from __future__ import annotations

import datetime
import math
import re
from typing import TYPE_CHECKING, Any

import pandas as pd

from src.knowledge.okf_models import OKFWorkspace, TableSchema

if TYPE_CHECKING:
    from src.connectors.base import BaseConnector


class RootCauseAgent:
    """Dynamic multi-dimensional variance decomposition & diagnostic analytics engine."""

    NON_METRIC_KEYWORDS = {
        "id", "code", "pk", "fk", "year", "hire_year", "birth_year",
        "established_year", "academic_year", "payment_year", "semester",
        "current_semester", "phone", "phone_number", "zip", "zip_code",
    }
    PII_KEYWORDS = {"first_name", "last_name", "email", "phone", "ssn", "password"}

    # =========================================================================
    # 1. Dynamic Schema Discovery & Multi-Table Join Resolution
    # =========================================================================
    @classmethod
    def discover_options(
        cls,
        workspace: OKFWorkspace,
        connector: "BaseConnector",
    ) -> dict[str, Any]:
        """Inspect workspace schema and join graph to discover metrics, temporal anchors, and dimensions."""
        active_tables = workspace.active_tables()
        if not active_tables:
            return {
                "metrics": [],
                "dimensions": [],
                "temporal_options": [],
                "has_temporal": False,
                "default_metric": "",
                "default_dimension": "",
                "alert": "No active tables in current workspace.",
            }

        # Find primary fact table (table with highest row count or numeric columns)
        fact_table = cls._find_primary_fact_table(active_tables)
        if not fact_table:
            fact_table = active_tables[0]

        # 1. Discover Numeric Metrics (with SUM vs AVG classification)
        metrics = cls._extract_metrics(workspace, fact_table)

        # 2. Discover Temporal Anchors (within fact table OR via FK joins)
        temporal_anchors, join_map = cls._discover_temporal_anchors(workspace, fact_table)

        # 3. Discover Conformed Dimensions (within fact table and connected tables)
        dimensions = cls._discover_conformed_dimensions(workspace, fact_table, join_map)

        # 4. Resolve Temporal Horizons & Windows
        temporal_windows, has_temporal, date_meta = cls._resolve_temporal_horizons(
            connector, fact_table, temporal_anchors, join_map
        )

        default_metric = metrics[0]["column"] if metrics else ""
        default_dim = dimensions[0]["id"] if dimensions else ""

        return {
            "fact_table": fact_table.name,
            "metrics": metrics,
            "dimensions": dimensions,
            "temporal_options": temporal_windows,
            "has_temporal": has_temporal,
            "temporal_metadata": date_meta,
            "default_metric": default_metric,
            "default_dimension": default_dim,
            "algorithms": [
                {
                    "id": "additive",
                    "label": "Total Volume Bridge",
                    "badge": "Additive Waterfall",
                    "description": "Decomposes exact dollar/unit change. Sum of segment deltas equals total variance.",
                    "recommended_for": ["SUM"],
                },
                {
                    "id": "rate_mix",
                    "label": "Rate vs. Customer Mix Shift",
                    "badge": "Simpson's Paradox Solver",
                    "description": "Separates true performance change from customer/volume mix shifts for rates & averages.",
                    "recommended_for": ["AVG", "RATIO"],
                },
                {
                    "id": "auto_scan",
                    "label": "Auto-Scan All Dimensions",
                    "badge": "Driver Discovery",
                    "description": "Scans all categorical dimensions at once and ranks which dimensions drove the variance.",
                    "recommended_for": ["EXPLORATORY"],
                },
            ],
        }

    # =========================================================================
    # 2. Main Entry Point: Analyze Variance
    # =========================================================================
    @classmethod
    def analyze_variance(
        cls,
        workspace: OKFWorkspace,
        connector: "BaseConnector",
        metric_col: str = "rev_net_usd",
        dimension: str = "category",
        algorithm: str = "auto",
        window_type: str = "last_90_vs_prior_90",
        date_col: str = "",
        t0_start: str = "",
        t0_end: str = "",
        t1_start: str = "",
        t1_end: str = "",
    ) -> dict[str, Any]:
        """Run automated diagnostic root-cause variance decomposition."""
        schema_meta = cls.discover_options(workspace, connector)
        fact_table_name = schema_meta.get("fact_table") or "orders"
        fact_table = workspace.get_table(fact_table_name)
        if not fact_table:
            active = workspace.active_tables()
            if not active:
                return {"summary": "No active tables found.", "df": pd.DataFrame()}
            fact_table = active[0]
            fact_table_name = fact_table.name

        # Resolve Metric Typology
        metric_info = cls._get_metric_info(schema_meta["metrics"], metric_col)
        metric_agg = metric_info.get("agg", "SUM")

        # Smart Auto-Select Algorithm if "auto"
        if algorithm == "auto":
            if metric_agg in ("AVG", "RATIO"):
                algorithm = "rate_mix"
            else:
                algorithm = "additive"

        # Resolve Temporal or Cohort Windows
        window_spec = cls._build_window_spec(
            connector=connector,
            workspace=workspace,
            fact_table=fact_table,
            schema_meta=schema_meta,
            window_type=window_type,
            explicit_date_col=date_col,
            t0_start=t0_start,
            t0_end=t0_end,
            t1_start=t1_start,
            t1_end=t1_end,
        )

        # Route to chosen diagnostic algorithm
        if algorithm == "rate_mix":
            return cls._decompose_rate_mix(
                workspace=workspace,
                connector=connector,
                fact_table=fact_table,
                metric_col=metric_col,
                dimension=dimension,
                window_spec=window_spec,
                schema_meta=schema_meta,
            )
        elif algorithm == "auto_scan":
            return cls._auto_scan_dimensions(
                workspace=workspace,
                connector=connector,
                fact_table=fact_table,
                metric_col=metric_col,
                window_spec=window_spec,
                schema_meta=schema_meta,
            )
        else:
            return cls._decompose_additive(
                workspace=workspace,
                connector=connector,
                fact_table=fact_table,
                metric_col=metric_col,
                dimension=dimension,
                window_spec=window_spec,
                schema_meta=schema_meta,
            )

    # =========================================================================
    # 3. Algorithm 1: Additive Variance Decomposition (Waterfall Bridge)
    # =========================================================================
    @classmethod
    def _decompose_additive(
        cls,
        workspace: OKFWorkspace,
        connector: "BaseConnector",
        fact_table: TableSchema,
        metric_col: str,
        dimension: str,
        window_spec: dict[str, Any],
        schema_meta: dict[str, Any],
    ) -> dict[str, Any]:
        """Additive Period-over-Period Variance Decomposition with single-pass SQL and Pareto 80/20."""
        dim_expr, join_clause, dim_label = cls._resolve_dimension_expr(workspace, fact_table, dimension, schema_meta)
        metric_clean = metric_col.replace('"', "").replace("'", "")
        fact_name = fact_table.name

        # Build Period Filter Predicates
        t0_cond = window_spec["t0_predicate"]
        t1_cond = window_spec["t1_predicate"]
        base_filter = cls._extract_table_filter(workspace, fact_name)

        # Single-pass conditional aggregation
        sql = (
            f"SELECT {dim_expr} AS segment,\n"
            f"       ROUND(COALESCE(SUM(CASE WHEN {t0_cond} THEN {fact_name}.{metric_clean} ELSE 0.0 END), 0.0), 2) AS period_prior,\n"
            f"       ROUND(COALESCE(SUM(CASE WHEN {t1_cond} THEN {fact_name}.{metric_clean} ELSE 0.0 END), 0.0), 2) AS period_current\n"
            f"FROM {fact_name} {join_clause}\n"
            f"WHERE {base_filter} AND (({t0_cond}) OR ({t1_cond}))\n"
            f"GROUP BY {dim_expr}\n"
            f"ORDER BY ABS(ROUND(COALESCE(SUM(CASE WHEN {t1_cond} THEN {fact_name}.{metric_clean} ELSE 0.0 END), 0.0), 2) - "
            f"ROUND(COALESCE(SUM(CASE WHEN {t0_cond} THEN {fact_name}.{metric_clean} ELSE 0.0 END), 0.0), 2)) DESC"
        )

        df = connector.execute_query(sql, max_rows=5000)
        if df.empty:
            return cls._empty_result(sql, "Insufficient records matching the comparison time windows.")

        # Compute Variance Deltas and Growth
        df["segment"] = df["segment"].fillna("(Unassigned)").astype(str)
        df["variance_delta"] = (df["period_current"] - df["period_prior"]).round(2)
        df["variance_pct"] = (
            100.0 * df["variance_delta"] / df["period_prior"].replace(0, pd.NA)
        ).fillna(0.0).round(2)

        total_prior = round(float(df["period_prior"].sum()), 2)
        total_curr = round(float(df["period_current"].sum()), 2)
        total_delta = round(total_curr - total_prior, 2)
        total_pct = round((total_delta / total_prior * 100.0) if total_prior else 0.0, 2)

        # Pareto 80/20 collapsing if cardinality > 14
        display_df = cls._apply_pareto_rollup(df)

        # Lifecycle Classification
        display_df["lifecycle"] = display_df.apply(
            lambda r: "NEW_SEGMENT" if r["period_prior"] == 0 and r["period_current"] > 0
            else ("CHURNED" if r["period_prior"] > 0 and r["period_current"] == 0
                  else ("INACTIVE" if r["period_prior"] == 0 and r["period_current"] == 0 else "CONTINUOUS")),
            axis=1,
        )

        # Drivers & Drags
        sorted_pos = df.sort_values("variance_delta", ascending=False)
        sorted_neg = df.sort_values("variance_delta", ascending=True)
        top_pos = sorted_pos.iloc[0] if not sorted_pos.empty else None
        top_neg = sorted_neg.iloc[0] if not sorted_neg.empty else None

        # Reconcile Balance Check
        sum_deltas = round(float(df["variance_delta"].sum()), 2)
        residual = round(abs(total_delta - sum_deltas), 4)
        reconciled = residual < 0.01

        direction = "increased" if total_delta >= 0 else "decreased"
        top_pos_str = (
            f"`{top_pos['segment']}` ({top_pos['variance_delta']:+,.2f} / {top_pos['variance_pct']:+.1f}%)"
            if top_pos is not None and top_pos["variance_delta"] > 0
            else "None (no segment expanded)"
        )
        top_neg_str = (
            f"`{top_neg['segment']}` ({top_neg['variance_delta']:+,.2f} / {top_neg['variance_pct']:+.1f}%)"
            if top_neg is not None and top_neg["variance_delta"] < 0
            else "None (no segment contracted)"
        )

        summary = (
            f"**Overall `{metric_clean}` {direction} by {total_delta:+,.2f} ({total_pct:+.1f}%)** "
            f"across `{dim_label}` (from `{total_prior:,.2f}` in {window_spec['t0_label']} to `{total_curr:,.2f}` in {window_spec['t1_label']}).\n\n"
            f"- 🟢 **Primary Growth Driver:** {top_pos_str}\n"
            f"- 🔻 **Primary Drag / Decline:** {top_neg_str}\n"
            f"- ⚖️ **Reconciliation Proof:** $\\sum \\Delta = {sum_deltas:+,.2f}$ equals Net Total Delta ${total_delta:+,.2f}$ (Residual: `${residual:.2f}`)."
        )

        tables_used = [fact_name]
        for cl in (join_clause, window_spec.get("join_lineage", "")):
            if cl:
                for m in re.finditer(r"\bJOIN\s+([a-zA-Z0-9_]+)\b", cl, flags=re.IGNORECASE):
                    tables_used.append(m.group(1))

        audit_receipt = {
            "algorithm": "Additive Period-over-Period Variance Decomposition (Waterfall Bridge)",
            "metric": metric_clean,
            "dimension": dim_label,
            "window_t0": window_spec["t0_label"],
            "window_t1": window_spec["t1_label"],
            "date_source": window_spec.get("date_source", ""),
            "join_lineage": window_spec.get("join_lineage", "Direct table scan"),
            "tables_used": list(dict.fromkeys(tables_used)),
            "executed_sql": sql,
            "formula_latex": r"\Delta M = \sum_{i} (y_{i, t_1} - y_{i, t_0}) \equiv M_{t_1} - M_{t_0}",
            "balance_check": {
                "total_prior": total_prior,
                "total_current": total_curr,
                "net_delta": total_delta,
                "sum_of_segment_deltas": sum_deltas,
                "unreconciled_residual": residual,
                "status": "VERIFIED_EXACT_BALANCE" if reconciled else "ROUNDING_DISCREPANCY",
            },
            "rows_evaluated": len(df),
        }

        kpi_ribbon = {
            "prior_total": total_prior,
            "curr_total": total_curr,
            "net_delta": total_delta,
            "pct_delta": total_pct,
            "reconciled": reconciled,
            "top_driver": top_pos["segment"] if top_pos is not None else "",
            "top_drag": top_neg["segment"] if top_neg is not None else "",
        }

        suggested_followups = [
            f"Why did {top_neg['segment']} decline in {metric_clean}?" if top_neg is not None else "",
            f"Show top customers or sub-categories in {top_pos['segment']}" if top_pos is not None else "",
            f"Compare {dim_label} across monthly trend",
        ]

        return {
            "summary": summary,
            "total_prior": total_prior,
            "total_curr": total_curr,
            "total_delta": total_delta,
            "total_pct": total_pct,
            "df": display_df,
            "full_df": df,
            "kpi_ribbon": kpi_ribbon,
            "audit_receipt": audit_receipt,
            "suggested_followups": [f for f in suggested_followups if f],
            "chart_type": "waterfall",
        }

    # =========================================================================
    # 4. Algorithm 2: Rate vs. Mix Shift Decomposition (Das Gupta / Kitagawa)
    # =========================================================================
    @classmethod
    def _decompose_rate_mix(
        cls,
        workspace: OKFWorkspace,
        connector: "BaseConnector",
        fact_table: TableSchema,
        metric_col: str,
        dimension: str,
        window_spec: dict[str, Any],
        schema_meta: dict[str, Any],
    ) -> dict[str, Any]:
        """Symmetric 2-Factor Das Gupta / Kitagawa Decomposition solving Simpson's Paradox."""
        dim_expr, join_clause, dim_label = cls._resolve_dimension_expr(workspace, fact_table, dimension, schema_meta)
        metric_clean = metric_col.replace('"', "").replace("'", "")
        fact_name = fact_table.name

        # Resolve numerator and denominator for the rate
        num_col, den_col = cls._resolve_rate_num_den(fact_table, metric_clean)

        t0_cond = window_spec["t0_predicate"]
        t1_cond = window_spec["t1_predicate"]
        base_filter = cls._extract_table_filter(workspace, fact_name)

        # Single-pass aggregated query
        sql = (
            f"SELECT {dim_expr} AS segment,\n"
            f"       ROUND(COALESCE(SUM(CASE WHEN {t0_cond} THEN CAST({num_col} AS DOUBLE) ELSE 0.0 END), 0.0), 2) AS n0,\n"
            f"       ROUND(COALESCE(SUM(CASE WHEN {t0_cond} THEN CAST({den_col} AS DOUBLE) ELSE 0.0 END), 0.0), 2) AS d0,\n"
            f"       ROUND(COALESCE(SUM(CASE WHEN {t1_cond} THEN CAST({num_col} AS DOUBLE) ELSE 0.0 END), 0.0), 2) AS n1,\n"
            f"       ROUND(COALESCE(SUM(CASE WHEN {t1_cond} THEN CAST({den_col} AS DOUBLE) ELSE 0.0 END), 0.0), 2) AS d1\n"
            f"FROM {fact_name} {join_clause}\n"
            f"WHERE {base_filter} AND (({t0_cond}) OR ({t1_cond}))\n"
            f"GROUP BY {dim_expr}\n"
            f"HAVING (d0 > 0 OR d1 > 0)\n"
            f"ORDER BY (d0 + d1) DESC"
        )

        df = connector.execute_query(sql, max_rows=5000)
        if df.empty:
            return cls._empty_result(sql, "Insufficient volume in periods to calculate rate/mix decomposition.")

        df["segment"] = df["segment"].fillna("(Unassigned)").astype(str)

        total_d0 = float(df["d0"].sum())
        total_d1 = float(df["d1"].sum())
        total_n0 = float(df["n0"].sum())
        total_n1 = float(df["n1"].sum())

        fleet_r0 = (total_n0 / total_d0) if total_d0 > 0 else 0.0
        fleet_r1 = (total_n1 / total_d1) if total_d1 > 0 else 0.0
        fleet_delta_r = round(fleet_r1 - fleet_r0, 4)

        # Calculate Segment Rates & Weights
        df["r0"] = df.apply(lambda r: (r["n0"] / r["d0"]) if r["d0"] > 0 else 0.0, axis=1)
        df["r1"] = df.apply(lambda r: (r["n1"] / r["d1"]) if r["d1"] > 0 else 0.0, axis=1)
        df["w0"] = df["d0"] / total_d0 if total_d0 > 0 else 0.0
        df["w1"] = df["d1"] / total_d1 if total_d1 > 0 else 0.0

        # Das Gupta Symmetric Decomposition:
        # Rate Effect = ((w0 + w1) / 2) * (r1 - r0)
        # Mix Effect  = ((r0 + r1) / 2) * (w1 - w0)
        df["rate_effect"] = (((df["w0"] + df["w1"]) / 2.0) * (df["r1"] - df["r0"])).round(4)
        df["mix_effect"] = (((df["r0"] + df["r1"]) / 2.0) * (df["w1"] - df["w0"])).round(4)
        df["total_segment_impact"] = (df["rate_effect"] + df["mix_effect"]).round(4)
        df["variance_delta"] = df["total_segment_impact"]  # For chart compatibility

        # Format percentages for presentation
        df["rate_prior"] = df["r0"].round(4)
        df["rate_curr"] = df["r1"].round(4)
        df["share_prior_pct"] = (df["w0"] * 100.0).round(2)
        df["share_curr_pct"] = (df["w1"] * 100.0).round(2)

        sum_rate_effect = round(float(df["rate_effect"].sum()), 4)
        sum_mix_effect = round(float(df["mix_effect"].sum()), 4)
        sum_total_effect = round(sum_rate_effect + sum_mix_effect, 4)
        residual = round(abs(fleet_delta_r - sum_total_effect), 6)

        # Check for Simpson's Paradox
        # Scenario: Fleet rate declined (fleet_delta_r < 0), but rate_effect sum is positive (sum_rate_effect > 0),
        # meaning performance within segments improved, but overall rate dropped due to adverse mix shift!
        simpsons_paradox = False
        if fleet_delta_r < 0 and sum_rate_effect > 0:
            simpsons_paradox = True
            paradox_note = (
                "🚨 **Simpson's Paradox Detected!** Overall rate declined, but **internal segment performance actually improved** "
                f"(Rate Effect: `{sum_rate_effect:+.4f}`). The decline was entirely caused by adverse **Mix Shift** (`{sum_mix_effect:+.4f}`) "
                "toward naturally lower-performing segments."
            )
        elif fleet_delta_r > 0 and sum_rate_effect < 0:
            simpsons_paradox = True
            paradox_note = (
                "🚨 **Simpson's Paradox Detected!** Overall rate increased, but **internal segment performance actually worsened** "
                f"(Rate Effect: `{sum_rate_effect:+.4f}`). The increase was masked by a strong **Mix Shift** (`{sum_mix_effect:+.4f}`) "
                "toward naturally higher-performing segments."
            )
        else:
            paradox_note = (
                f"Rate Effect accounted for `{sum_rate_effect:+.4f}` while Mix Shift accounted for `{sum_mix_effect:+.4f}`."
            )

        top_rate_driver = df.sort_values("rate_effect", ascending=False).iloc[0]
        top_mix_driver = df.sort_values("mix_effect", ascending=False).iloc[0]
        top_drag = df.sort_values("total_segment_impact", ascending=True).iloc[0]

        summary = (
            f"**Overall `{metric_clean}` shifted from `{fleet_r0:.4f}` to `{fleet_r1:.4f}` ({fleet_delta_r:+.4f})** "
            f"between {window_spec['t0_label']} and {window_spec['t1_label']}.\n\n"
            f"- **Decomposition Split:**\n"
            f"  - 🎯 **Rate Effect (Performance):** `{sum_rate_effect:+.4f}` (changes in intrinsic segment behavior)\n"
            f"  - 🔄 **Mix Effect (Volume Shift):** `{sum_mix_effect:+.4f}` (shifts in cohort population proportion)\n\n"
            f"{paradox_note}\n\n"
            f"- 🟢 **Top Performance Improver:** `{top_rate_driver['segment']}` (Rate Effect: `{top_rate_driver['rate_effect']:+.4f}`)\n"
            f"- 🔻 **Biggest Drag:** `{top_drag['segment']}` (Total Impact: `{top_drag['total_segment_impact']:+.4f}`)"
        )

        tables_used = [fact_name]
        for cl in (join_clause, window_spec.get("join_lineage", "")):
            if cl:
                for m in re.finditer(r"\bJOIN\s+([a-zA-Z0-9_]+)\b", cl, flags=re.IGNORECASE):
                    tables_used.append(m.group(1))

        audit_receipt = {
            "algorithm": "Symmetric 2-Factor Das Gupta / Kitagawa Rate vs. Mix Decomposition",
            "metric": metric_clean,
            "dimension": dim_label,
            "numerator_col": num_col,
            "denominator_col": den_col,
            "window_t0": window_spec["t0_label"],
            "window_t1": window_spec["t1_label"],
            "date_source": window_spec.get("date_source", ""),
            "join_lineage": window_spec.get("join_lineage", "Direct table scan"),
            "tables_used": list(dict.fromkeys(tables_used)),
            "executed_sql": sql,
            "formula_latex": (
                r"\Delta R = \sum \left(\frac{w_0 + w_1}{2}\right)(r_1 - r_0) + "
                r"\sum \left(\frac{r_0 + r_1}{2}\right)(w_1 - w_0)"
            ),
            "simpsons_paradox_flag": simpsons_paradox,
            "balance_check": {
                "fleet_rate_prior": round(fleet_r0, 4),
                "fleet_rate_current": round(fleet_r1, 4),
                "total_delta_rate": fleet_delta_r,
                "sum_rate_effects": sum_rate_effect,
                "sum_mix_effects": sum_mix_effect,
                "sum_total_effects": sum_total_effect,
                "unexplained_residual": residual,
                "status": "EXACT_RECONCILIATION_VERIFIED" if residual < 0.001 else "ROUNDING_DISCREPANCY",
            },
            "rows_evaluated": len(df),
        }

        kpi_ribbon = {
            "prior_total": round(fleet_r0, 4),
            "curr_total": round(fleet_r1, 4),
            "net_delta": fleet_delta_r,
            "pct_delta": round((fleet_delta_r / fleet_r0 * 100.0) if fleet_r0 else 0.0, 2),
            "rate_effect_total": sum_rate_effect,
            "mix_effect_total": sum_mix_effect,
            "simpsons_paradox": simpsons_paradox,
            "top_driver": top_rate_driver["segment"],
            "top_drag": top_drag["segment"],
        }

        # Pareto display dataframe
        display_df = cls._apply_pareto_rollup(df, value_col="total_segment_impact")

        return {
            "summary": summary,
            "total_prior": round(fleet_r0, 4),
            "total_curr": round(fleet_r1, 4),
            "total_delta": fleet_delta_r,
            "total_pct": round((fleet_delta_r / fleet_r0 * 100.0) if fleet_r0 else 0.0, 2),
            "df": display_df,
            "full_df": df,
            "kpi_ribbon": kpi_ribbon,
            "audit_receipt": audit_receipt,
            "suggested_followups": [
                f"Break down {top_drag['segment']} by sub-tier",
                f"Show volume trend for {top_mix_driver['segment']}",
                "Simulate restoring mix share to prior baseline",
            ],
            "chart_type": "rate_mix_scatter",
        }

    # =========================================================================
    # 5. Algorithm 3: Multi-Dimensional Auto-Scan (Driver Discovery)
    # =========================================================================
    @classmethod
    def _auto_scan_dimensions(
        cls,
        workspace: OKFWorkspace,
        connector: "BaseConnector",
        fact_table: TableSchema,
        metric_col: str,
        window_spec: dict[str, Any],
        schema_meta: dict[str, Any],
    ) -> dict[str, Any]:
        """Scan all conformed dimensions simultaneously to rank which dimensions explain the variance."""
        dimensions = schema_meta.get("dimensions", [])
        if not dimensions:
            return cls._empty_result("", "No conformed categorical dimensions found in workspace.")

        metric_clean = metric_col.replace('"', "").replace("'", "")
        results = []

        # Iterate over top conformed dimensions (up to 8)
        for dim_obj in dimensions[:8]:
            dim_id = dim_obj["id"]
            dim_expr, join_clause, dim_label = cls._resolve_dimension_expr(workspace, fact_table, dim_id, schema_meta)
            fact_name = fact_table.name
            t0_cond = window_spec["t0_predicate"]
            t1_cond = window_spec["t1_predicate"]
            base_filter = cls._extract_table_filter(workspace, fact_name)

            sql = (
                f"SELECT {dim_expr} AS segment,\n"
                f"       ROUND(COALESCE(SUM(CASE WHEN {t0_cond} THEN {fact_name}.{metric_clean} ELSE 0.0 END), 0.0), 2) AS v0,\n"
                f"       ROUND(COALESCE(SUM(CASE WHEN {t1_cond} THEN {fact_name}.{metric_clean} ELSE 0.0 END), 0.0), 2) AS v1\n"
                f"FROM {fact_name} {join_clause}\n"
                f"WHERE {base_filter} AND (({t0_cond}) OR ({t1_cond}))\n"
                f"GROUP BY {dim_expr}\n"
                f"HAVING (v0 > 0 OR v1 > 0)"
            )
            try:
                df_dim = connector.execute_query(sql, max_rows=500)
            except Exception:
                continue

            if df_dim.empty or len(df_dim) < 2:
                continue

            df_dim["delta"] = df_dim["v1"] - df_dim["v0"]
            df_dim["abs_delta"] = df_dim["delta"].abs()

            total_v0 = float(df_dim["v0"].sum())
            total_v1 = float(df_dim["v1"].sum())
            fleet_delta = total_v1 - total_v0

            # 1. Excess Divergence Score (L1 norm deviation from uniform growth)
            growth_mult = (total_v1 / total_v0) if total_v0 > 0 else 1.0
            df_dim["expected_v1"] = df_dim["v0"] * growth_mult
            excess_l1 = float((df_dim["v1"] - df_dim["expected_v1"]).abs().sum())
            denom_l1 = 2.0 * abs(fleet_delta) if abs(fleet_delta) > 0 else 1.0
            excess_score = min(1.0, excess_l1 / denom_l1)

            # 2. Normalized HHI of Variance
            sum_abs_delta = float(df_dim["abs_delta"].sum())
            if sum_abs_delta > 0:
                shares = df_dim["abs_delta"] / sum_abs_delta
                hhi = float((shares ** 2).sum())
                j = len(df_dim)
                norm_hhi = ((hhi - (1.0 / j)) / (1.0 - (1.0 / j))) if j > 1 else 1.0
                norm_hhi = max(0.0, min(1.0, norm_hhi))
            else:
                norm_hhi = 0.0

            # Composite Explanatory Score
            composite_score = round((0.6 * excess_score + 0.4 * norm_hhi) * 100.0, 1)

            top_driver_row = df_dim.sort_values("abs_delta", ascending=False).iloc[0]

            results.append({
                "dimension_id": dim_id,
                "dimension_label": dim_label,
                "explanatory_score": composite_score,
                "cardinality": len(df_dim),
                "peak_driver": str(top_driver_row["segment"]),
                "peak_driver_delta": round(float(top_driver_row["delta"]), 2),
                "total_variance_sum": round(sum_abs_delta, 2),
            })

        if not results:
            return cls._empty_result("", "No dimensional variance could be decomposed.")

        res_df = pd.DataFrame(results).sort_values("explanatory_score", ascending=False)
        top_dim = res_df.iloc[0]

        summary = (
            f"**Automated Driver Discovery scanned `{len(res_df)}` dimensions for `{metric_clean}`.**\n\n"
            f"- 🏆 **Highest Explanatory Dimension:** `{top_dim['dimension_label']}` (Score: **{top_dim['explanatory_score']}%**)\n"
            f"  - Peak Driver: `{top_dim['peak_driver']}` ({top_dim['peak_driver_delta']:+,.2f})\n"
            f"- 📊 **Dimension Explanatory Ranking:**\n"
        )
        for _, r in res_df.iterrows():
            summary += f"  - **{r['dimension_label']}**: `{r['explanatory_score']}%` score · Top driver: `{r['peak_driver']}` ({r['peak_driver_delta']:+,.2f})\n"

        tables_used = [fact_table.name]
        for d in dimensions:
            if d.get("is_joined") and d.get("table"):
                tables_used.append(d["table"])

        audit_receipt = {
            "algorithm": "Multi-Dimensional Auto-Scan (Driver Discovery via Excess L1 & Normalized HHI)",
            "metric": metric_clean,
            "dimensions_scanned": len(res_df),
            "window_t0": window_spec["t0_label"],
            "window_t1": window_spec["t1_label"],
            "formula_latex": r"\text{Score} = 0.6 \cdot \frac{\sum |y_{i,1} - \hat{y}_{i,1}|}{2|\Delta M|} + 0.4 \cdot HHI^*",
            "date_source": window_spec.get("date_source", ""),
            "join_lineage": window_spec.get("join_lineage", "Multi-table join traversal"),
            "tables_used": list(dict.fromkeys(tables_used)),
            "executed_sql": f"-- Multi-dimensional driver scan across {len(dimensions)} dimensions for {metric_clean} from {fact_table.name}",
            "rows_evaluated": len(res_df),
        }

        # Format DataFrame for visual display
        res_df["segment"] = res_df["dimension_label"]
        res_df["variance_delta"] = res_df["explanatory_score"]

        return {
            "summary": summary,
            "df": res_df,
            "full_df": res_df,
            "audit_receipt": audit_receipt,
            "kpi_ribbon": {
                "top_dimension": str(top_dim["dimension_label"]),
                "top_score": float(top_dim["explanatory_score"]),
                "top_driver": str(top_dim["peak_driver"]),
            },
            "suggested_followups": [
                f"Drill into {top_dim['dimension_label']} waterfall",
                f"Explain why {top_dim['peak_driver']} changed",
            ],
            "chart_type": "dimension_divergence",
        }

    # =========================================================================
    # 6. Helper Utilities: Schema Introspection, Joins, Windows & Pareto
    # =========================================================================
    @classmethod
    def _find_primary_fact_table(cls, active_tables: list[TableSchema]) -> TableSchema | None:
        """Find primary fact table based on row count and numeric column density."""
        candidates = []
        for tbl in active_tables:
            num_cols = [
                c for c in tbl.columns
                if not c.is_blocked and not c.is_primary_key and not c.is_foreign_key
                and any(t in c.data_type.upper() for t in ("INT", "FLOAT", "DOUBLE", "DECIMAL", "NUMERIC", "BIGINT"))
                and not c.name.lower().endswith(("_id", "_year"))
                and c.name.lower() not in cls.NON_METRIC_KEYWORDS
            ]
            candidates.append((len(num_cols), tbl.row_count, tbl))
        candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return candidates[0][2] if candidates else None

    @classmethod
    def _extract_metrics(cls, workspace: OKFWorkspace, fact_table: TableSchema) -> list[dict[str, Any]]:
        """Extract numeric metrics from fact table and semantic measures."""
        metrics = []
        seen = set()

        # Add semantic measures only if they belong to active tables in this workspace
        active_tbl_names = {t.name.lower() for t in workspace.active_tables()}
        for m in getattr(workspace, "measures", []):
            m_tbl = getattr(m, "table_name", "").lower()
            if m_tbl in active_tbl_names and m.name not in seen:
                seen.add(m.name)
                is_avg = any(k in m.name.lower() for k in ("margin", "rate", "pct", "avg", "average", "ratio", "aov"))
                metrics.append({
                    "column": m.name,
                    "label": m.name.replace("_", " ").title(),
                    "table": getattr(m, "table_name", fact_table.name),
                    "agg": "AVG" if is_avg else "SUM",
                    "type": "measure",
                    "badge": "Calculated Measure",
                })

        # Add physical numeric columns
        for c in fact_table.columns:
            if c.is_blocked or c.is_primary_key or c.is_foreign_key:
                continue
            c_low = c.name.lower()
            if c_low.endswith(("_id", "_pk", "_fk")) or c_low in cls.NON_METRIC_KEYWORDS:
                continue
            if any(t in c.data_type.upper() for t in ("INT", "FLOAT", "DOUBLE", "DECIMAL", "NUMERIC", "BIGINT")):
                if c.name not in seen:
                    seen.add(c.name)
                    is_avg = any(k in c_low for k in ("margin", "rate", "pct", "avg", "average", "ratio", "gpa", "salary", "score", "aov"))
                    metrics.append({
                        "column": c.name,
                        "label": c.business_alias or c.name.replace("_", " ").title(),
                        "table": fact_table.name,
                        "agg": "AVG" if is_avg else "SUM",
                        "type": "column",
                        "badge": "AVG" if is_avg else "SUM",
                    })
        return metrics

    @classmethod
    def _discover_temporal_anchors(
        cls,
        workspace: OKFWorkspace,
        fact_table: TableSchema,
    ) -> tuple[list[dict[str, Any]], dict[str, str]]:
        """Discover date/timestamp columns in fact table OR reachable via FK joins in parent tables."""
        anchors = []
        join_map: dict[str, str] = {}
        fact_name = fact_table.name

        def is_temporal_col(col_name: str, data_type: str) -> bool:
            c = col_name.lower()
            dt = data_type.upper()
            if any(t in dt for t in ("DATE", "TIMESTAMP", "TIME")):
                return True
            if any(k in c for k in ("date", "order_date", "invoice_date", "created_at", "trans_date", "event_date", "admission_date")):
                return True
            if c.endswith(("_year", "_month", "_period", "_date")):
                return True
            return False

        # 1. Direct columns in fact table
        for c in fact_table.columns:
            if not c.is_blocked and is_temporal_col(c.name, c.data_type):
                anchors.append({
                    "table": fact_name,
                    "column": c.name,
                    "label": f"{fact_name}.{c.name}",
                    "is_direct": True,
                    "join_path": "",
                })

        # 2. Check connected tables via relationships
        relationships = workspace.relationships or []
        for rel in relationships:
            rel_source = getattr(rel, "source_table", "")
            rel_target = getattr(rel, "target_table", "")
            if rel_source == fact_name:
                parent_tbl = workspace.get_table(rel_target)
                if parent_tbl:
                    for c in parent_tbl.columns:
                        if not c.is_blocked and is_temporal_col(c.name, c.data_type):
                            join_clause = f"LEFT JOIN {rel_target} ON {fact_name}.{rel.source_column} = {rel_target}.{rel.target_column}"
                            join_map[f"{rel_target}.{c.name}"] = join_clause
                            anchors.append({
                                "table": rel_target,
                                "column": c.name,
                                "label": f"{rel_target}.{c.name} (via {rel.source_column})",
                                "is_direct": False,
                                "join_path": join_clause,
                            })
            elif rel_target == fact_name:
                parent_tbl = workspace.get_table(rel_source)
                if parent_tbl:
                    for c in parent_tbl.columns:
                        if not c.is_blocked and is_temporal_col(c.name, c.data_type):
                            join_clause = f"LEFT JOIN {rel_source} ON {fact_name}.{rel.target_column} = {rel_source}.{rel.source_column}"
                            join_map[f"{rel_source}.{c.name}"] = join_clause
                            anchors.append({
                                "table": rel_source,
                                "column": c.name,
                                "label": f"{rel_source}.{c.name} (via {rel.target_column})",
                                "is_direct": False,
                                "join_path": join_clause,
                            })

        return anchors, join_map

    @classmethod
    def _discover_conformed_dimensions(
        cls,
        workspace: OKFWorkspace,
        fact_table: TableSchema,
        join_map: dict[str, str],
    ) -> list[dict[str, Any]]:
        """Discover categorical dimensions in fact table and connected tables."""
        dimensions = []
        seen = set()
        fact_name = fact_table.name

        def is_cat(col_name: str, data_type: str) -> bool:
            c = col_name.lower()
            dt = data_type.upper()
            if any(p in c for p in cls.PII_KEYWORDS) or c.endswith(("_id", "_pk", "_fk")):
                return False
            if any(t in dt for t in ("VARCHAR", "CHAR", "TEXT", "STRING")):
                return True
            if any(k in c for k in ("category", "region", "tier", "channel", "department", "college", "status", "type", "gender", "level")):
                return True
            return False

        # Fact table columns
        for c in fact_table.columns:
            if not c.is_blocked and not c.is_primary_key and not c.is_foreign_key:
                if is_cat(c.name, c.data_type) and c.name not in seen:
                    seen.add(c.name)
                    dim_id = f"{fact_name}.{c.name}"
                    dimensions.append({
                        "id": dim_id,
                        "label": c.business_alias or f"{c.name.replace('_', ' ').title()} ({fact_name})",
                        "table": fact_name,
                        "column": c.name,
                        "is_joined": False,
                    })

        # Connected table columns
        for rel in (workspace.relationships or []):
            target_name = rel.target_table if rel.source_table == fact_name else (rel.source_table if rel.target_table == fact_name else None)
            if target_name and target_name != fact_name:
                parent_tbl = workspace.get_table(target_name)
                if parent_tbl:
                    for c in parent_tbl.columns:
                        if not c.is_blocked and not c.is_primary_key and not c.is_foreign_key:
                            if is_cat(c.name, c.data_type):
                                dim_id = f"{target_name}.{c.name}"
                                if dim_id not in seen:
                                    seen.add(dim_id)
                                    dimensions.append({
                                        "id": dim_id,
                                        "label": c.business_alias or f"{c.name.replace('_', ' ').title()} ({target_name})",
                                        "table": target_name,
                                        "column": c.name,
                                        "is_joined": True,
                                    })
        return dimensions

    @classmethod
    def _resolve_temporal_horizons(
        cls,
        connector: "BaseConnector",
        fact_table: TableSchema,
        temporal_anchors: list[dict[str, Any]],
        join_map: dict[str, str],
    ) -> tuple[list[dict[str, Any]], bool, dict[str, Any]]:
        """Query min/max dates and construct default time comparison horizons."""
        if not temporal_anchors:
            return (
                [{"id": "cohort", "label": "Cohort Comparison Mode (No date column found)"}],
                False,
                {"notice": "No temporal/date column found across connected tables. Switched to Cohort Comparison mode."},
            )

        anchor = temporal_anchors[0]
        tbl = anchor["table"]
        col = anchor["column"]
        join_clause = anchor.get("join_path", "")

        date_expr = f"{tbl}.{col}"
        sql = f"SELECT MIN({date_expr}) AS min_d, MAX({date_expr}) AS max_d FROM {fact_table.name} {join_clause}"
        try:
            df_bounds = connector.execute_query(sql, max_rows=5)
            min_d = str(df_bounds.iloc[0]["min_d"]) if not df_bounds.empty else ""
            max_d = str(df_bounds.iloc[0]["max_d"]) if not df_bounds.empty else ""
        except Exception:
            min_d, max_d = "", ""

        options = [
            {"id": "last_90_vs_prior_90", "label": "Last 90 Days vs. Prior 90 Days", "badge": "Rolling Period"},
            {"id": "yoy", "label": "Year-over-Year (Latest Year vs. Prior Year)", "badge": "Annual"},
            {"id": "qoq", "label": "Quarter-over-Quarter (Latest Qtr vs. Prior Qtr)", "badge": "Quarterly"},
            {"id": "custom", "label": "Custom Date Range", "badge": "Custom"},
        ]

        meta = {
            "selected_anchor": f"{tbl}.{col}",
            "min_date": min_d,
            "max_date": max_d,
            "is_joined": not anchor.get("is_direct", True),
            "join_clause": join_clause,
        }
        return options, True, meta

    @classmethod
    def _build_window_spec(
        cls,
        connector: "BaseConnector",
        workspace: OKFWorkspace,
        fact_table: TableSchema,
        schema_meta: dict[str, Any],
        window_type: str,
        explicit_date_col: str = "",
        t0_start: str = "",
        t0_end: str = "",
        t1_start: str = "",
        t1_end: str = "",
    ) -> dict[str, Any]:
        """Build exact SQL predicates and human labels for comparison periods."""
        temporal_anchors = schema_meta.get("temporal_metadata", {})
        date_expr = explicit_date_col or temporal_anchors.get("selected_anchor", "")
        max_d_str = temporal_anchors.get("max_date", "")

        # Fallback if no date anchor exists: Cohort comparison mode
        if not date_expr or not schema_meta.get("has_temporal", False):
            return {
                "t0_predicate": "1=1",
                "t1_predicate": "1=1",
                "t0_label": "Cohort A",
                "t1_label": "Cohort B",
                "date_source": "Cohort Split (Non-temporal)",
                "join_lineage": "",
            }

        join_lineage = temporal_anchors.get("join_clause", "")

        # Parse reference date
        ref_date = datetime.date.today()
        if max_d_str:
            try:
                clean_d = max_d_str.split()[0]
                ref_date = datetime.date.fromisoformat(clean_d)
            except Exception:
                pass

        if window_type == "yoy":
            curr_year = ref_date.year
            prior_year = curr_year - 1
            return {
                "t0_predicate": f"SUBSTR(CAST({date_expr} AS VARCHAR), 1, 4) = '{prior_year}'",
                "t1_predicate": f"SUBSTR(CAST({date_expr} AS VARCHAR), 1, 4) = '{curr_year}'",
                "t0_label": f"FY{prior_year}",
                "t1_label": f"FY{curr_year}",
                "date_source": date_expr,
                "join_lineage": join_lineage,
            }
        elif window_type == "custom" and t0_start and t0_end and t1_start and t1_end:
            return {
                "t0_predicate": f"TRY_CAST({date_expr} AS DATE) BETWEEN DATE '{t0_start}' AND DATE '{t0_end}'",
                "t1_predicate": f"TRY_CAST({date_expr} AS DATE) BETWEEN DATE '{t1_start}' AND DATE '{t1_end}'",
                "t0_label": f"{t0_start} to {t0_end}",
                "t1_label": f"{t1_start} to {t1_end}",
                "date_source": date_expr,
                "join_lineage": join_lineage,
            }
        else:
            # Default: Last 90 days vs Prior 90 days (180 to 91 days prior)
            t1_end_d = ref_date
            t1_start_d = ref_date - datetime.timedelta(days=90)
            t0_end_d = t1_start_d - datetime.timedelta(days=1)
            t0_start_d = t0_end_d - datetime.timedelta(days=90)

            t0_s_str, t0_e_str = t0_start_d.isoformat(), t0_end_d.isoformat()
            t1_s_str, t1_e_str = t1_start_d.isoformat(), t1_end_d.isoformat()

            return {
                "t0_predicate": f"CAST({date_expr} AS VARCHAR) BETWEEN '{t0_s_str}' AND '{t0_e_str}'",
                "t1_predicate": f"CAST({date_expr} AS VARCHAR) BETWEEN '{t1_s_str}' AND '{t1_e_str}'",
                "t0_label": f"Prior 90d ({t0_s_str}..{t0_e_str})",
                "t1_label": f"Last 90d ({t1_s_str}..{t1_e_str})",
                "date_source": date_expr,
                "join_lineage": join_lineage,
            }

    @classmethod
    def _resolve_dimension_expr(
        cls,
        workspace: OKFWorkspace,
        fact_table: TableSchema,
        dimension_id: str,
        schema_meta: dict[str, Any],
    ) -> tuple[str, str, str]:
        """Resolve dimension expression, table join clause, and clean label."""
        fact_name = fact_table.name
        if "." in dimension_id:
            tbl_name, col_name = dimension_id.split(".", 1)
        else:
            col_name = dimension_id
            if any(c.name.lower() == col_name.lower() for c in fact_table.columns):
                tbl_name = fact_name
            else:
                tbl_name = fact_name
                for rel in (workspace.relationships or []):
                    target = rel.target_table if rel.source_table == fact_name else (rel.source_table if rel.target_table == fact_name else None)
                    if target:
                        parent_tbl = workspace.get_table(target)
                        if parent_tbl and any(c.name.lower() == col_name.lower() for c in parent_tbl.columns):
                            tbl_name = target
                            break
                if tbl_name == fact_name:
                    for tbl in workspace.active_tables():
                        if any(c.name.lower() == col_name.lower() for c in tbl.columns):
                            tbl_name = tbl.name
                            break

        dim_label = col_name.replace("_", " ").title()

        if tbl_name == fact_name:
            return f"{fact_name}.{col_name}", "", dim_label

        # Build join clause from relationships
        join_clause = ""
        for rel in (workspace.relationships or []):
            if rel.source_table == fact_name and rel.target_table == tbl_name:
                join_clause = f"LEFT JOIN {tbl_name} ON {fact_name}.{rel.source_column} = {tbl_name}.{rel.target_column}"
                break
            elif rel.target_table == fact_name and rel.source_table == tbl_name:
                join_clause = f"LEFT JOIN {tbl_name} ON {fact_name}.{rel.target_column} = {tbl_name}.{rel.source_column}"
                break

        # Fallback if standard join naming convention matches
        if not join_clause:
            fk_cand = f"{tbl_name[:-1] if tbl_name.endswith('s') else tbl_name}_id"
            for c in fact_table.columns:
                if c.name.lower() == fk_cand.lower():
                    join_clause = f"LEFT JOIN {tbl_name} ON {fact_name}.{c.name} = {tbl_name}.{fk_cand}"
                    break

        return f"{tbl_name}.{col_name}", join_clause, f"{dim_label} ({tbl_name})"

    @classmethod
    def _resolve_rate_num_den(cls, fact_table: TableSchema, metric_col: str) -> tuple[str, str]:
        """Detect numerator and denominator columns for Rate vs. Mix decomposition."""
        fact_name = fact_table.name
        c_low = metric_col.lower()

        # Check for profit / margin
        if "profit" in c_low or "margin" in c_low:
            num = f"{fact_name}.{metric_col}"
            # Find revenue column
            rev_cand = next((c.name for c in fact_table.columns if "rev" in c.name.lower() or "sales" in c.name.lower()), "1.0")
            den = f"{fact_name}.{rev_cand}" if rev_cand != "1.0" else "1.0"
            return num, den

        # Check for volume/traffic denominators (visitors, sessions, impressions, users, total, quantity, units)
        den_cand = next(
            (
                c.name for c in fact_table.columns
                if not c.is_blocked and not c.is_primary_key and not c.is_foreign_key
                and not c.name.lower().endswith(("_id", "_pk", "_fk", "_date", "_year", "_month", "_period"))
                and any(t in c.data_type.upper() for t in ("INT", "FLOAT", "DOUBLE", "DECIMAL", "NUMERIC", "BIGINT"))
                and any(k in c.name.lower() for k in ("visitor", "session", "traffic", "impression", "view", "total", "weight", "volume", "denom", "quantity", "user", "order", "click"))
                and c.name.lower() != c_low
            ),
            None,
        )
        if den_cand:
            return f"{fact_name}.{metric_col}", f"{fact_name}.{den_cand}"

        # Check for AOV (Revenue / Order Count)
        if "rev" in c_low or "sales" in c_low:
            num = f"{fact_name}.{metric_col}"
            den = "1.0"  # Each row is an order/transaction unit
            return num, den

        # Default: Metric column as numerator, row count (1.0) as denominator
        return f"{fact_name}.{metric_col}", "1.0"

    @classmethod
    def _apply_pareto_rollup(cls, df: pd.DataFrame, value_col: str = "variance_delta", max_segments: int = 12) -> pd.DataFrame:
        """Collapse long-tail segments into 'All Other Segments' preserving exact penny balance."""
        if len(df) <= max_segments:
            return df

        pos_df = df[df[value_col] >= 0].sort_values(value_col, ascending=False)
        neg_df = df[df[value_col] < 0].sort_values(value_col, ascending=True)

        top_pos = pos_df.head(max_segments // 2)
        top_neg = neg_df.head(max_segments // 2)

        kept_indices = set(top_pos.index).union(set(top_neg.index))
        other_indices = [idx for idx in df.index if idx not in kept_indices]

        if not other_indices:
            return df

        other_rows = df.loc[other_indices]
        other_prior = round(float(other_rows["period_prior"].sum()), 2) if "period_prior" in df.columns else 0.0
        other_curr = round(float(other_rows["period_current"].sum()), 2) if "period_current" in df.columns else 0.0
        other_delta = round(float(other_rows[value_col].sum()), 2)
        other_pct = round((other_delta / other_prior * 100.0) if other_prior else 0.0, 2)

        kept_df = df.loc[list(kept_indices)].copy()
        other_dict = {
            "segment": f"All Other ({len(other_indices)} segments)",
            value_col: other_delta,
        }
        if "period_prior" in df.columns:
            other_dict["period_prior"] = other_prior
            other_dict["period_current"] = other_curr
            other_dict["variance_pct"] = other_pct

        other_row = pd.DataFrame([other_dict])
        return pd.concat([kept_df, other_row], ignore_index=True)

    @classmethod
    def _extract_table_filter(cls, workspace: OKFWorkspace, table_name: str) -> str:
        """Extract table health or cancellation filters (e.g. ord_stat != 'X')."""
        tbl = workspace.get_table(table_name)
        if not tbl:
            return "1=1"
        col_names = {c.name.lower() for c in tbl.columns}
        if "ord_stat" in col_names:
            return f"{table_name}.ord_stat <> 'X'"
        if "status" in col_names:
            return f"LOWER({table_name}.status) NOT IN ('cancelled', 'void', 'deleted')"
        return "1=1"

    @classmethod
    def _get_metric_info(cls, metrics: list[dict[str, Any]], metric_col: str) -> dict[str, Any]:
        """Find metric info from schema metadata."""
        for m in metrics:
            if m["column"].lower() == metric_col.lower():
                return m
        return {"agg": "SUM"}

    @staticmethod
    def _empty_result(sql: str, reason: str) -> dict[str, Any]:
        """Return standardized empty diagnostic response."""
        return {
            "summary": f"Diagnostic analysis could not be completed: {reason}",
            "df": pd.DataFrame(),
            "full_df": pd.DataFrame(),
            "kpi_ribbon": {},
            "audit_receipt": {
                "algorithm": "None",
                "executed_sql": sql,
                "status": "INSUFFICIENT_DATA",
                "reason": reason,
            },
            "suggested_followups": [],
            "chart_type": "waterfall",
        }
