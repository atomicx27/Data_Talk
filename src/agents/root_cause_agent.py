"""Power BI-Style 'Why Did X Change?' Automated Root-Cause Variance Decomposition Engine.

Emulates Power BI's *Analyze -> Explain the Increase/Decrease* and Decomposition Tree:
- Compares a target metric (e.g., `rev_net_usd` or `profit_usd`) across two time periods
  (or top vs bottom segments) across multiple categorical dimensions (`category`, `region_code`,
  `cust_tier`, `sales_channel`).
- Quantifies the exact dollar and percentage variance contribution of each segment so users
  immediately see *why* a metric increased or decreased.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

from src.knowledge.okf_models import OKFWorkspace

if TYPE_CHECKING:
    from src.connectors.base import BaseConnector


class RootCauseAgent:
    """Multi-dimensional variance decomposition engine for 'Why did X change?' analysis."""

    @classmethod
    def analyze_variance(
        cls,
        workspace: OKFWorkspace,
        connector: "BaseConnector",
        metric_col: str = "rev_net_usd",
        dimension: str = "category",
    ) -> dict[str, Any]:
        """Decompose period-over-period variance of `metric_col` across `dimension`."""
        if not workspace.get_table("orders"):
            return cls._analyze_generic_table(workspace, connector, metric_col, dimension)

        dim_map = {
            "category": ("products.category", "LEFT JOIN products ON orders.product_id = products.product_id"),
            "region": ("customers.region_code", "LEFT JOIN customers ON orders.customer_id = customers.customer_id"),
            "customer_tier": ("customers.cust_tier", "LEFT JOIN customers ON orders.customer_id = customers.customer_id"),
            "sales_channel": ("orders.sales_channel", ""),
        }
        dim_expr, join_clause = dim_map.get(dimension, dim_map["category"])

        # Compare 2025 vs 2024 across the selected dimension
        sql = (
            f"SELECT {dim_expr} AS segment,\n"
            f"       ROUND(SUM(CASE WHEN SUBSTR(CAST(orders.order_date AS VARCHAR), 1, 4) = '2024' THEN orders.{metric_col} ELSE 0 END), 2) AS period_prior,\n"
            f"       ROUND(SUM(CASE WHEN SUBSTR(CAST(orders.order_date AS VARCHAR), 1, 4) = '2025' THEN orders.{metric_col} ELSE 0 END), 2) AS period_current\n"
            f"FROM orders {join_clause}\n"
            f"WHERE orders.ord_stat <> 'X'\n"
            f"GROUP BY {dim_expr}\n"
            f"ORDER BY (period_current - period_prior) DESC"
        )
        df = connector.execute_query(sql, max_rows=5000)
        if df.empty:
            return {"summary": "Insufficient time-series data for variance analysis.", "df": df}

        df["variance_delta"] = (df["period_current"] - df["period_prior"]).round(2)
        df["variance_pct"] = (
            100.0 * df["variance_delta"] / df["period_prior"].replace(0, pd.NA)
        ).fillna(0.0).round(2)

        total_prior = float(df["period_prior"].sum())
        total_curr = float(df["period_current"].sum())
        total_delta = round(total_curr - total_prior, 2)
        total_pct = round((total_delta / total_prior * 100.0) if total_prior else 0.0, 2)

        top_pos = df.sort_values("variance_delta", ascending=False).iloc[0]
        top_neg = df.sort_values("variance_delta", ascending=True).iloc[0]

        direction = "increased" if total_delta >= 0 else "decreased"
        summary = (
            f"Overall `{metric_col}` **{direction} by {total_delta:+,.2f} ({total_pct:+.1f}%)** "
            f"(from `{total_prior:,.2f}` in 2024 to `{total_curr:,.2f}` in 2025).\n"
            f"- **Biggest Positive Driver:** `{top_pos['segment']}` ({top_pos['variance_delta']:+,.2f} / {top_pos['variance_pct']:+.1f}%)\n"
            f"- **Biggest Drag / Decline:** `{top_neg['segment']}` ({top_neg['variance_delta']:+,.2f} / {top_neg['variance_pct']:+.1f}%)"
        )
        return {
            "summary": summary,
            "total_prior": total_prior,
            "total_curr": total_curr,
            "total_delta": total_delta,
            "total_pct": total_pct,
            "df": df,
        }

    @staticmethod
    def _analyze_generic_table(
        workspace: OKFWorkspace,
        connector: "BaseConnector",
        metric_col: str = "",
        dimension: str = "",
    ) -> dict[str, Any]:
        active = workspace.active_tables()
        if not active:
            return {"summary": "No active tables.", "df": pd.DataFrame()}

        NON_METRIC = {"year", "hire_year", "birth_year", "established_year", "academic_year", "payment_year", "id", "code", "semester", "current_semester"}
        chosen_tbl = None
        chosen_num = None
        chosen_cat = None

        # Prefer a table with business metrics (e.g. fee_invoices, faculty, enrollments_marks)
        for tbl in active:
            t_nums = [
                c.name
                for c in tbl.columns
                if not c.is_blocked
                and not c.is_primary_key
                and not c.is_foreign_key
                and any(t in c.data_type.upper() for t in ("INT", "FLOAT", "DOUBLE", "DECIMAL", "NUMERIC", "BIGINT"))
                and not c.name.lower().endswith(("_id", "_year"))
                and c.name.lower() not in NON_METRIC
            ]
            t_cats = [
                c.name
                for c in tbl.columns
                if not c.is_blocked
                and c.name not in t_nums
                and not c.is_primary_key
                and not c.is_foreign_key
                and not c.name.lower().endswith(("_id", "_year"))
                and c.name.lower() not in ("first_name", "last_name", "email", "phone", "phone_number")
            ]
            if t_nums and t_cats:
                if metric_col and metric_col in t_nums:
                    chosen_tbl, chosen_num = tbl, metric_col
                    chosen_cat = dimension if dimension in t_cats else t_cats[0]
                    break
                if chosen_tbl is None:
                    chosen_tbl, chosen_num, chosen_cat = tbl, t_nums[0], (dimension if dimension in t_cats else t_cats[0])

        if not (chosen_tbl and chosen_num and chosen_cat):
            return {"summary": "Table requires at least 1 categorical and 1 numeric business metric column.", "df": pd.DataFrame()}

        agg_fn = "AVG" if any(k in chosen_num.lower() for k in ("salary", "gpa", "rate", "pct", "score", "marks")) else "SUM"
        sql = (
            f'SELECT "{chosen_cat}" AS segment, ROUND({agg_fn}("{chosen_num}"), 2) AS metric_value '
            f'FROM "{chosen_tbl.name}" GROUP BY "{chosen_cat}" ORDER BY metric_value DESC'
        )
        df = connector.execute_query(sql, max_rows=500)
        return {"summary": f"Segment contribution breakdown of `{chosen_num}` ({agg_fn}) by `{chosen_cat}`.", "df": df}
