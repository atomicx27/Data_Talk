"""Plotly Auto-Chart Recommender & Interactive Visual Builder for Data_Talk.

Automatically inspects result DataFrame shape and column types to recommend and render:
- `kpi`: Single-row multi-metric KPI summary
- `bar` / `horizontal_bar`: Categorical comparison & rankings
- `combo`: Dual-axis Bar + Line chart (e.g., Net Revenue $ on left axis + Gross Margin % on right axis)
- `line` / `area`: Time-series trends (months, dates, years)
- `waterfall`: Root-cause variance decomposition ("Why did X change?")
- `treemap` / `pie`: Part-to-whole composition
- `scatter`: 2D numeric correlation

Uses the Okabe-Ito colorblind-safe high-contrast analytical palette.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

OKABE_ITO_PALETTE = [
    "#3B82F6",  # Blue
    "#10B981",  # Emerald
    "#F59E0B",  # Amber
    "#8B5CF6",  # Violet
    "#EC4899",  # Pink
    "#06B6D4",  # Cyan
    "#F97316",  # Orange
]


class ChartRecommender:
    """Recommends the best chart type for a DataFrame and builds interactive Plotly figures."""

    SUPPORTED_CHART_TYPES = [
        "bar",
        "horizontal_bar",
        "stacked_bar",
        "stacked_bar_100",
        "stacked_horizontal_bar",
        "combo",
        "line",
        "area",
        "donut",
        "pie",
        "waterfall",
        "rate_mix_scatter",
        "dimension_divergence",
        "treemap",
        "sunburst",
        "heatmap",
        "radar",
        "box",
        "funnel",
        "scatter",
        "table",
    ]

    @staticmethod
    def _is_temporal_dim(col_name: str) -> bool:
        c = str(col_name).lower()
        return any(k in c for k in ("date", "month", "year", "quarter", "week", "day", "time", "semester"))

    @staticmethod
    def _is_non_additive_metric(col_name: str) -> bool:
        c = str(col_name).lower()
        if any(c.startswith(p) for p in ("avg_", "average_", "mean_", "median_", "pct_", "rate_")):
            return True
        if any(k in c for k in ("_avg", "_mean", "_rate", "_pct", "_percent", "_ratio", "margin", "gpa", "cgpa", "score", "marks")):
            return True
        if c in ("salary", "base_salary", "gpa", "cgpa", "rate", "percentage", "ratio", "score", "marks", "averagegpa"):
            return True
        return False

    @staticmethod
    def _is_part_to_whole_share_col(col_name: str) -> bool:
        """Return True if column is a compositional share/ratio percentage that sums to 100% across a subgroup."""
        c = str(col_name).lower()
        return (
            c.endswith(("_ratio_pct", "_share_pct", "_ratio", "_share"))
            or c in ("ratio_pct", "share_pct", "gender_ratio_pct", "gender_ratio", "ratio", "share")
        )

    @classmethod
    def _select_preferred_metric(cls, num_cols: list[str], question: str = "") -> str:
        if not num_cols:
            return ""
        if len(num_cols) == 1:
            return num_cols[0]

        import re

        q_low = (question or "").lower()
        wants_count = any(k in q_low for k in ("count", "headcount", "how many", "number of", "volume", "num records", "total records"))

        def is_count_col(c: str) -> bool:
            cl = str(c).lower()
            return any(cl.endswith(p) for p in ("_count", "_cnt", "_qty", "_quantity", "_rows", "_records")) or cl in ("count", "cnt", "row_count", "total_count", "headcount", "qty", "volume")

        # Universal metric concept mappings for semantic prioritization
        KEYWORD_MAPPINGS = [
            (("ratio", "ritio", "raio", "proportion", "share"), ("ratio", "pct", "share", "prop")),
            (("rate", "rates"), ("rate", "pct")),
            (("margin", "margins"), ("margin", "pct")),
            (("grade", "grades", "mark", "marks", "score", "scores", "gpa", "cgpa"), ("mark", "grade", "score", "gpa", "cgpa")),
            (("salary", "salaries", "compensation", "wage", "wages", "pay"), ("salary", "pay", "comp")),
            (("revenue", "revenues", "sales", "turnover"), ("rev", "sales", "revenue")),
            (("profit", "profits"), ("profit", "margin")),
            (("cogs", "cost", "costs", "expense", "expenses"), ("cogs", "cost", "expense")),
            (("fee", "fees", "tuition"), ("fee", "tuition")),
            (("refund", "refunds"), ("refund",)),
            (("discount", "discounts"), ("discount",)),
            (("budget", "budgets", "expenditure"), ("budget", "expenditure")),
            (("earnings", "earning"), ("earning", "earnings")),
        ]

        scored: list[tuple[float, str]] = []
        for idx, col in enumerate(num_cols):
            c_str = str(col)
            c_low = c_str.lower()
            c_words = c_low.replace("_", " ")
            score = 0.0

            # 1. Exact phrase match: e.g. "gender ratio" in "compare the top2 colleges gender ratio..."
            if c_words in q_low:
                score += 150.0

            # 2. Concept keyword alignment
            for q_kws, c_kws in KEYWORD_MAPPINGS:
                q_has = any(re.search(rf"\b{re.escape(k)}\b", q_low) for k in q_kws)
                c_has = any(k in c_low for k in c_kws)
                if q_has and c_has:
                    score += 100.0

            # 3. Individual token overlap
            c_tokens = [t for t in c_low.split("_") if len(t) >= 3 and t not in ("total", "sum", "avg", "mean", "val", "all")]
            token_matches = sum(1 for t in c_tokens if re.search(rf"\b{re.escape(t)}\b", q_low))
            score += token_matches * 30.0

            # 4. Count column handling
            if is_count_col(col):
                if wants_count:
                    score += 80.0
                else:
                    score -= 40.0

            # 5. Non-count business metric baseline boost
            if not is_count_col(col):
                score += 10.0

            # Tie-break by original column order
            score -= (idx * 0.01)
            scored.append((score, col))

        scored.sort(key=lambda x: -x[0])
        return scored[0][1]

    @classmethod
    def recommend_chart_type(cls, df: pd.DataFrame, question: str = "") -> str:
        """Select the best default chart type based on DataFrame shape, column names, and user prompt."""
        if df is None or df.empty:
            return "none"
        if len(df) == 1:
            return "kpi"

        num_cols = list(df.select_dtypes(include=["number"]).columns)
        dim_cols = [c for c in df.columns if c not in num_cols]

        # Identify calendar/id columns that pandas loaded as numeric but are semantically dimensions
        calendar_id_dims = [
            c for c in num_cols
            if str(c).lower().endswith(("_year", "_id", "_code", "_tier", "_quarter", "_month", "_day"))
            or str(c).lower() in ("year", "payment_year", "hire_year", "academic_year", "calendar_year", "birth_year", "fiscal_year", "id", "code", "quarter", "month", "day")
        ]
        if calendar_id_dims:
            num_cols = [c for c in num_cols if c not in calendar_id_dims]
            dim_cols = [c for c in dim_cols if c not in calendar_id_dims] + calendar_id_dims

        q_low = (question or "").lower()
        if "waterfall" in q_low or "variance" in q_low or ("why" in q_low and "variance_delta" in df.columns):
            return "waterfall"
        if "sunburst" in q_low:
            return "sunburst"
        if "treemap" in q_low or "hierarchy" in q_low:
            return "treemap"
        if any(k in q_low for k in ("heatmap", "matrix", "cross tab", "crosstab", "intensity")):
            return "heatmap"
        if any(k in q_low for k in ("radar", "spider", "polar")):
            return "radar"
        if any(k in q_low for k in ("box", "whisker", "spread", "dispersion", "outlier")):
            return "box"
        if any(k in q_low for k in ("funnel", "pipeline", "conversion", "lifecycle stages")):
            return "funnel"
        if any(k in q_low for k in ("100%", "percent stack", "normalized stack", "proportional")):
            return "stacked_bar_100"
        if any(k in q_low for k in ("horizontal stack", "stacked horizontal")):
            return "stacked_horizontal_bar"
        if any(k in q_low for k in ("stacked bar", "stacked", "stack")):
            return "stacked_bar"
        if any(k in q_low for k in ("donut", "ring")):
            return "donut"
        if "scatter" in q_low or "correlation" in q_low:
            return "scatter"
        if "pie" in q_low or ("share" in q_low and len(dim_cols) <= 1 and len(df) <= 12):
            return "pie"

        if len(dim_cols) == 1:
            first_dim = str(dim_cols[0]).lower()
            if cls._is_temporal_dim(first_dim):
                return "area" if "area" in q_low else "line"

        # High cardinality guard: categorical charts with > 40 items become illegible walls of bars.
        # Fallback to interactive paginated table to preserve readability.
        # For 2-dimension breakdowns, cardinality applies to the primary X-axis dimension, not total rows.
        primary_dim = dim_cols[0] if dim_cols else None
        primary_categories = df[primary_dim].nunique() if (primary_dim and primary_dim in df.columns) else len(df)
        is_primary_time = cls._is_temporal_dim(primary_dim) if primary_dim else False

        if primary_categories > 40 and not is_primary_time:
            return "table"

        # Multi-metric breakdown / composition check (e.g. Male/Female/Other or New/Returning/Churned)
        total_cols = [
            c for c in num_cols
            if any(k in str(c).lower() for k in ("total_count", "row_count", "grand_total", "all_students", "total_students"))
        ]
        non_total_num = [c for c in num_cols if c not in total_cols]
        pct_breakdowns = [
            c for c in non_total_num
            if any(k in str(c).lower() for k in ("pct", "percent", "margin", "rate", "ratio"))
        ]
        raw_count_breakdowns = [c for c in non_total_num if c not in pct_breakdowns]
        # If there are non-count business metrics (e.g. tuition_fee, salary, marks), exclude auxiliary *_count columns
        # so a business metric is never stacked with a record count column.
        has_non_count_business = any(
            not str(c).lower().endswith("_count") and str(c).lower() != "count"
            for c in raw_count_breakdowns
        )
        if has_non_count_business:
            count_breakdowns = [
                c for c in raw_count_breakdowns
                if not str(c).lower().endswith("_count") and str(c).lower() != "count"
            ]
        else:
            count_breakdowns = raw_count_breakdowns

        is_breakdown_intent = any(
            k in q_low
            for k in (
                "ratio", "raio", "ritio", "gender", "breakdown", "demographic", "split",
                "proportion", "share", "composition", "distribution", "other", "wise"
            )
        )
        is_comparison_intent = any(
            k in q_low
            for k in ("comparison", "compare", "vs", "versus", "between")
        )

        preferred_metric = cls._select_preferred_metric(num_cols, question)
        is_non_additive = cls._is_non_additive_metric(preferred_metric)
        is_unstackable_rate = is_non_additive and not cls._is_part_to_whole_share_col(preferred_metric)

        # If 1D comparison across 2+ financial/value metrics (e.g. Expenditure vs Earning, Revenue vs Refund), use Grouped Bar Chart
        fin_comparison_cols = [
            c for c in count_breakdowns
            if not str(c).lower().endswith("_count") and str(c).lower() != "row_count"
        ]
        if len(dim_cols) == 1 and len(fin_comparison_cols) >= 2 and (
            is_comparison_intent
            or any("expenditure" in str(c).lower() or "earning" in str(c).lower() for c in fin_comparison_cols)
        ):
            return "bar"

        # If 2+ percentage breakdown columns exist that represent a demographic/composition split (e.g., Male Pct, Female Pct, Other Pct)
        if len(pct_breakdowns) >= 2 and (
            is_breakdown_intent
            or any(any(g in str(c).lower() for g in ("male", "female", "other", "share")) for c in pct_breakdowns)
        ):
            return "stacked_bar_100" if (is_breakdown_intent or "100" in q_low) else "stacked_bar"

        # If 2+ count/volume breakdown columns exist in a 1D wide-format table (e.g., Male Count, Female Count, Other Count)
        if (
            len(dim_cols) <= 1
            and len(count_breakdowns) >= 2
            and not is_non_additive
            and not is_comparison_intent
            and (
                is_breakdown_intent
                or any(any(g in str(c).lower() for g in ("male", "female", "other")) for c in count_breakdowns)
            )
        ):
            return "stacked_bar"

        # If 2 dimensions exist (e.g. COLLEGE NAME and GENDER/YEAR)
        if len(dim_cols) >= 2:
            has_time_dim = any(cls._is_temporal_dim(d) for d in dim_cols[:2])
            time_col = next((d for d in dim_cols[:2] if cls._is_temporal_dim(d)), None)
            time_nunique = df[time_col].nunique() if (time_col and time_col in df.columns) else 0
            has_explicit_time_query = any(k in q_low for k in ("year", "trend", "over time", "yoy", "annual", "yearly", "monthly", "timeline"))
            # Power BI / Data Analyst Rule:
            # 1. Non-additive rates/averages (AVG salary, GPA, margin) must NEVER stack!
            #    (Part-to-whole composition shares like *_ratio_pct / *_share_pct DO sum to 100% across the 2nd dimension and CAN stack.)
            # 2. Explicit time-trend queries ("year on year", "trend") or > 8 time periods should use a Line chart instead of stacking 19 years of rainbow bars!
            if is_unstackable_rate:
                if has_time_dim:
                    return "line"
                sec_nunique = df[dim_cols[1]].nunique() if dim_cols[1] in df.columns else 1
                if sec_nunique > 8:
                    return "heatmap"
                return "bar"

            if has_time_dim and (has_explicit_time_query or time_nunique > 8):
                return "line"

            if cls._is_part_to_whole_share_col(preferred_metric) and df[dim_cols[0]].nunique() <= 10:
                return "stacked_bar_100"

            if is_breakdown_intent or any(k in q_low for k in ("by", "breakdown", "per", "across", "wise", "student", "college", "grade", "trend")):
                if df[dim_cols[0]].nunique() > 5 or len(df) > 8:
                    return "stacked_horizontal_bar"
                return "stacked_bar"
            return "stacked_bar"

        # Dual-Axis Combo chart: ONLY when exactly 1 volume metric + 1 rate/percentage metric exist,
        # and NOT a multi-category demographic/composition breakdown
        if (
            len(pct_breakdowns) == 1
            and len(count_breakdowns) == 1
            and dim_cols
            and 2 <= len(df) <= 20
            and not is_breakdown_intent
        ):
            # Guard against flat / constant rate metrics (e.g. constant 100% or single value across all rows)
            p_col = pct_breakdowns[0]
            if df[p_col].nunique() <= 1:
                return "horizontal_bar" if len(df) > 8 else "bar"
            return "combo"

        if dim_cols:
            if len(df) > 10:
                return "horizontal_bar"
            if len(df) >= 6 and len(dim_cols) == 1 and not is_comparison_intent:
                avg_lbl_len = df[dim_cols[0]].astype(str).str.len().mean()
                if avg_lbl_len > 22:
                    return "horizontal_bar"

        return "bar"

    @classmethod
    def build_figure(
        cls,
        df: pd.DataFrame,
        chart_type: str | None = None,
        title: str = "",
        x_col: str | None = None,
        y_cols: list[str] | None = None,
    ) -> go.Figure | None:
        """Build a styled, interactive Plotly Figure for the given DataFrame."""
        if df is None or df.empty:
            return None

        chosen_type = chart_type or cls.recommend_chart_type(df, title)
        if chosen_type == "table":
            return None
        if chosen_type in ("none", "kpi") and len(df) == 1:
            return None

        num_cols = list(df.select_dtypes(include=["number"]).columns)
        dim_cols = [c for c in df.columns if c not in num_cols]

        calendar_id_dims = [
            c for c in num_cols
            if str(c).lower().endswith(("_year", "_id", "_code", "_tier", "_quarter", "_month", "_day"))
            or str(c).lower() in ("year", "payment_year", "hire_year", "academic_year", "calendar_year", "birth_year", "fiscal_year", "id", "code", "quarter", "month", "day")
        ]
        if calendar_id_dims:
            num_cols = [c for c in num_cols if c not in calendar_id_dims]
            dim_cols = [c for c in dim_cols if c not in calendar_id_dims] + calendar_id_dims

        if not num_cols:
            return None

        # When 2 dimensions exist and dim_cols[1] is temporal (e.g. department_name + hire_year),
        # place the temporal dimension on the primary X-axis and the category on the color legend for line/area charts.
        if (
            not x_col
            and len(dim_cols) >= 2
            and cls._is_temporal_dim(dim_cols[1])
            and not cls._is_temporal_dim(dim_cols[0])
            and chosen_type in ("line", "area")
        ):
            dim_cols = [dim_cols[1], dim_cols[0]] + dim_cols[2:]
            p_sort = cls._select_preferred_metric(num_cols, title) or num_cols[0]
            cat_series_col = dim_cols[1]
            if cat_series_col in df.columns and df[cat_series_col].nunique() > 10:
                agg_op = "mean" if cls._is_non_additive_metric(p_sort) else "sum"
                top_series = getattr(df.groupby(cat_series_col)[p_sort], agg_op)().sort_values(ascending=False).head(10).index
                df = df[df[cat_series_col].isin(top_series)]
            df = df.sort_values(by=[dim_cols[0], dim_cols[1]])

        # High cardinality guard in visual generation: categorical charts with > 40 items become illegible walls of bars.
        if chosen_type in ("bar", "horizontal_bar", "stacked_bar", "stacked_bar_100", "stacked_horizontal_bar", "pie", "donut", "radar", "funnel"):
            primary_dim = dim_cols[0] if dim_cols else None
            primary_categories = df[primary_dim].nunique() if (primary_dim and primary_dim in df.columns) else len(df)
            is_time = cls._is_temporal_dim(primary_dim) if primary_dim else False
            if primary_categories > 40 and not is_time:
                # If raw entity data (e.g. 10,000 raw students), refuse to render an unreadable blob
                if primary_categories > 100:
                    return None
                # Otherwise, gracefully truncate to Top 25 primary categories
                p_sort = cls._select_preferred_metric(num_cols, title) or (num_cols[0] if num_cols else df.columns[0])
                if len(dim_cols) >= 2 and primary_dim in df.columns:
                    agg_op = "mean" if cls._is_non_additive_metric(p_sort) else "sum"
                    top_cats = getattr(df.groupby(primary_dim)[p_sort], agg_op)().sort_values(ascending=False).head(25).index
                    df = df[df[primary_dim].isin(top_cats)]
                else:
                    df = df.sort_values(by=p_sort, ascending=False).head(25)

        x = x_col or (dim_cols[0] if dim_cols else df.columns[0])

        # Detect multi-metric breakdown candidates (e.g., Male/Female/Other) excluding grand totals and auxiliary row counts
        total_cols = [
            c for c in num_cols
            if any(k in str(c).lower() for k in ("total_count", "row_count", "grand_total", "all_students", "total_students"))
        ]
        non_total_num = [c for c in num_cols if c not in total_cols and c != x]
        pct_breakdowns = [
            c for c in non_total_num
            if any(k in str(c).lower() for k in ("pct", "percent", "margin", "rate", "ratio"))
        ]
        raw_count_breakdowns = [c for c in non_total_num if c not in pct_breakdowns]
        has_non_count_business = any(
            not str(c).lower().endswith("_count") and str(c).lower() != "count"
            for c in raw_count_breakdowns
        )
        if has_non_count_business:
            count_breakdowns = [
                c for c in raw_count_breakdowns
                if not str(c).lower().endswith("_count") and str(c).lower() != "count"
            ]
        else:
            count_breakdowns = raw_count_breakdowns

        preferred_single_metric = cls._select_preferred_metric(num_cols, title)
        is_demo_pct_split = len(pct_breakdowns) >= 2 and any(
            any(g in str(c).lower() for g in ("male", "female", "other", "share"))
            for c in pct_breakdowns
        )
        fin_comparison_cols = [
            c for c in count_breakdowns
            if not str(c).lower().endswith("_count")
            and str(c).lower() != "row_count"
            and not any(k in str(c).lower() for k in ("surplus", "deficit", "variance"))
        ]
        title_low = (title or "").lower()
        is_comparison_chart = (
            len(dim_cols) == 1
            and len(fin_comparison_cols) >= 2
            and chosen_type in ("bar", "horizontal_bar")
            and (
                any(k in title_low for k in ("comparison", "compare", "vs", "versus", "between"))
                or any("expenditure" in str(c).lower() or "earning" in str(c).lower() for c in fin_comparison_cols)
            )
        )

        if chosen_type in ("stacked_bar_100", "percent_stacked_bar") and len(pct_breakdowns) >= 2:
            breakdown_ys = pct_breakdowns
        elif is_comparison_chart:
            breakdown_ys = fin_comparison_cols[:2]
        elif len(count_breakdowns) >= 2 and chosen_type in ("stacked_bar", "stacked_horizontal_bar") and len(dim_cols) == 1:
            breakdown_ys = count_breakdowns
        elif is_demo_pct_split:
            breakdown_ys = pct_breakdowns
        else:
            breakdown_ys = [preferred_single_metric]

        ys = y_cols or breakdown_ys
        if not ys:
            ys = [preferred_single_metric]

        primary_y = ys[0]
        is_non_additive_primary = cls._is_non_additive_metric(primary_y)
        is_unstackable_primary = is_non_additive_primary and not cls._is_part_to_whole_share_col(primary_y)

        if chosen_type == "combo" and len(num_cols) >= 2:
            pct_cols = [
                c for c in num_cols
                if any(k in str(c).lower() for k in ("pct", "percent", "margin", "rate"))
                and not any(t in str(c).lower() for t in ("total", "sum"))
            ]
            non_pct_cols = [
                c for c in num_cols
                if c not in pct_cols
                and not any(t in str(c).lower() for t in ("total", "sum"))
            ]
            bar_col = non_pct_cols[0] if non_pct_cols else num_cols[0]
            line_col = pct_cols[0] if pct_cols else (num_cols[1] if len(num_cols) > 1 else num_cols[0])
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(
                go.Bar(
                    x=df[x],
                    y=df[bar_col],
                    name=str(bar_col),
                    marker_color=OKABE_ITO_PALETTE[0],
                ),
                secondary_y=False,
            )
            fig.add_trace(
                go.Scatter(
                    x=df[x],
                    y=df[line_col],
                    name=str(line_col),
                    mode="lines+markers",
                    line=dict(color=OKABE_ITO_PALETTE[1], width=3),
                    marker=dict(size=8),
                ),
                secondary_y=True,
            )
            fig.update_yaxes(title_text=str(bar_col), secondary_y=False)
            fig.update_yaxes(title_text=str(line_col), secondary_y=True)

        elif chosen_type == "horizontal_bar":
            color_col = dim_cols[1] if len(dim_cols) >= 2 else None
            if color_col:
                fig = px.bar(
                    df,
                    x=primary_y,
                    y=x,
                    color=color_col,
                    orientation="h",
                    barmode="group",
                    color_discrete_sequence=OKABE_ITO_PALETTE,
                )
            else:
                fig = px.bar(
                    df.sort_values(by=primary_y, ascending=True),
                    x=primary_y,
                    y=x,
                    orientation="h",
                    color_discrete_sequence=OKABE_ITO_PALETTE,
                    text_auto=".2s",
                )

        elif chosen_type == "stacked_bar":
            color_col = dim_cols[1] if len(dim_cols) >= 2 else None
            if color_col:
                is_comp_intent = any(k in title_low for k in ("compare", "comparison", "vs", "versus", "between"))
                effective_barmode = "group" if is_unstackable_primary or is_comp_intent else "stack"
                fig = px.bar(
                    df,
                    x=x,
                    y=primary_y,
                    color=color_col,
                    barmode=effective_barmode,
                    color_discrete_sequence=OKABE_ITO_PALETTE,
                )
            else:
                fig = px.bar(
                    df,
                    x=x,
                    y=ys,
                    barmode="stack",
                    color_discrete_sequence=OKABE_ITO_PALETTE,
                )
                for trace in fig.data:
                    clean = (
                        str(trace.name)
                        .replace("_", " ")
                        .replace(" Count", "")
                        .replace(" count", "")
                        .replace(" Pct", " %")
                        .replace(" pct", " %")
                        .strip()
                        .title()
                    )
                    if clean.endswith(" %"):
                        clean = clean[:-2] + " %"
                    trace.name = clean

        elif chosen_type in ("stacked_bar_100", "percent_stacked_bar"):
            color_col = dim_cols[1] if len(dim_cols) >= 2 else None
            if color_col:
                fig = px.bar(
                    df,
                    x=x,
                    y=primary_y,
                    color=color_col,
                    barmode="group" if is_unstackable_primary else "stack",
                    color_discrete_sequence=OKABE_ITO_PALETTE,
                )
            else:
                fig = px.bar(
                    df,
                    x=x,
                    y=ys,
                    barmode="stack",
                    color_discrete_sequence=OKABE_ITO_PALETTE,
                )
                for trace in fig.data:
                    clean = (
                        str(trace.name)
                        .replace("_", " ")
                        .replace(" Count", "")
                        .replace(" count", "")
                        .replace(" Pct", " %")
                        .replace(" pct", " %")
                        .strip()
                        .title()
                    )
                    if clean.endswith(" %"):
                        clean = clean[:-2] + " %"
                    trace.name = clean
            if not (color_col and is_unstackable_primary):
                fig.update_layout(barnorm="percent", yaxis=dict(ticksuffix="%"))

        elif chosen_type == "stacked_horizontal_bar":
            color_col = dim_cols[1] if len(dim_cols) >= 2 else None
            if color_col:
                fig = px.bar(
                    df,
                    x=primary_y,
                    y=x,
                    color=color_col,
                    orientation="h",
                    barmode="group" if is_unstackable_primary else "stack",
                    color_discrete_sequence=OKABE_ITO_PALETTE,
                )
            else:
                fig = px.bar(
                    df,
                    x=ys,
                    y=x,
                    orientation="h",
                    barmode="stack",
                    color_discrete_sequence=OKABE_ITO_PALETTE,
                )

        elif chosen_type == "line":
            line_x = x
            color_col = dim_cols[1] if (len(dim_cols) >= 2 and len(ys) == 1) else None
            plot_df = df
            if len(dim_cols) >= 2 and cls._is_temporal_dim(dim_cols[1]) and not cls._is_temporal_dim(dim_cols[0]):
                line_x = dim_cols[1]
                color_col = dim_cols[0]
                plot_df = df.sort_values(by=[line_x, color_col])
            elif cls._is_temporal_dim(line_x) and line_x in df.columns:
                plot_df = df.sort_values(by=line_x)
            fig = px.line(
                plot_df,
                x=line_x,
                y=ys if not color_col else ys[0],
                color=color_col,
                markers=True,
                color_discrete_sequence=OKABE_ITO_PALETTE,
            )

        elif chosen_type == "area":
            area_x = x
            color_col = dim_cols[1] if (len(dim_cols) >= 2 and len(ys) == 1) else None
            plot_df = df
            if len(dim_cols) >= 2 and cls._is_temporal_dim(dim_cols[1]) and not cls._is_temporal_dim(dim_cols[0]):
                area_x = dim_cols[1]
                color_col = dim_cols[0]
                plot_df = df.sort_values(by=[area_x, color_col])
            fig = px.area(
                plot_df,
                x=area_x,
                y=ys if not color_col else ys[0],
                color=color_col,
                color_discrete_sequence=OKABE_ITO_PALETTE,
            )

        elif chosen_type == "waterfall":
            val_col = "variance_delta" if "variance_delta" in df.columns else primary_y
            if "period_prior" in df.columns and "period_current" in df.columns:
                prior_tot = float(df["period_prior"].sum())
                wf_x = ["Total Prior"] + df[x].astype(str).tolist() + ["Total Current"]
                wf_y = [prior_tot] + df[val_col].tolist() + [0]
                wf_measure = ["absolute"] + (["relative"] * len(df)) + ["total"]
            else:
                wf_x = df[x].astype(str).tolist()
                wf_y = df[val_col].tolist()
                wf_measure = ["relative"] * len(df)

            fig = go.Figure(
                go.Waterfall(
                    name=str(val_col),
                    orientation="v",
                    measure=wf_measure,
                    x=wf_x,
                    y=wf_y,
                    connector={"line": {"color": "#6B7280"}},
                    increasing={"marker": {"color": "#10B981"}},
                    decreasing={"marker": {"color": "#EF4444"}},
                    totals={"marker": {"color": "#3B82F6"}},
                )
            )

        elif chosen_type == "rate_mix_scatter":
            x_col = "mix_effect" if "mix_effect" in df.columns else x
            y_col = "rate_effect" if "rate_effect" in df.columns else primary_y
            seg_col = "segment" if "segment" in df.columns else x
            fig = px.scatter(
                df,
                x=x_col,
                y=y_col,
                text=seg_col,
                color="total_segment_impact" if "total_segment_impact" in df.columns else None,
                color_continuous_scale="RdYlGn",
                labels={
                    x_col: "Mix Shift Effect (Volume/Share Shift)",
                    y_col: "Rate Effect (Intrinsic Behavior)",
                },
            )
            fig.update_traces(textposition="top center", marker=dict(size=12))
            fig.add_hline(y=0, line_dash="dash", line_color="#9CA3AF")
            fig.add_vline(x=0, line_dash="dash", line_color="#9CA3AF")

        elif chosen_type == "dimension_divergence":
            y_col = "segment" if "segment" in df.columns else x
            x_col = "explanatory_score" if "explanatory_score" in df.columns else primary_y
            sorted_df = df.sort_values(x_col, ascending=True)
            fig = px.bar(
                sorted_df,
                x=x_col,
                y=y_col,
                orientation="h",
                text=x_col,
                color=x_col,
                color_continuous_scale="Blues",
                labels={x_col: "Explanatory Score (%)", y_col: "Dimension"},
            )
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")

        elif chosen_type == "treemap":
            fig = px.treemap(
                df,
                path=[x],
                values=primary_y,
                color=primary_y,
                color_continuous_scale="Blues",
            )

        elif chosen_type == "sunburst":
            path_cols = dim_cols[:3] if len(dim_cols) >= 2 else [x]
            fig = px.sunburst(
                df,
                path=path_cols,
                values=primary_y,
                color=primary_y,
                color_continuous_scale="Blues",
            )

        elif chosen_type == "donut":
            fig = px.pie(
                df,
                names=x,
                values=primary_y,
                hole=0.55,
                color_discrete_sequence=OKABE_ITO_PALETTE,
            )
            center_val = df[primary_y].mean() if is_non_additive_primary else df[primary_y].sum()
            center_lbl = "Avg" if is_non_additive_primary else "Total"
            formatted_total = f"{center_val:,.1f}" if isinstance(center_val, float) else f"{center_val:,}"
            fig.update_layout(
                annotations=[
                    dict(
                        text=f"<b>{center_lbl}</b><br>{formatted_total}",
                        x=0.5,
                        y=0.5,
                        font=dict(size=13, color="#cbd5e1"),
                        showarrow=False,
                    )
                ]
            )

        elif chosen_type == "pie":
            fig = px.pie(
                df,
                names=x,
                values=primary_y,
                hole=0.42,
                color_discrete_sequence=OKABE_ITO_PALETTE,
            )

        elif chosen_type == "heatmap":
            if len(dim_cols) >= 2:
                pivot_df = df.pivot_table(
                    index=dim_cols[0],
                    columns=dim_cols[1],
                    values=primary_y,
                    aggfunc="mean",
                )
                if not is_non_additive_primary:
                    pivot_df = pivot_df.fillna(0)
                fig = go.Figure(
                    data=go.Heatmap(
                        z=pivot_df.values,
                        x=[str(c) for c in pivot_df.columns],
                        y=[str(r) for r in pivot_df.index],
                        colorscale="Blues",
                        hoverongaps=False,
                    )
                )
            elif len(num_cols) >= 2 and dim_cols:
                matrix_df = df.set_index(x)[num_cols[:8]].fillna(0)
                fig = go.Figure(
                    data=go.Heatmap(
                        z=matrix_df.values,
                        x=[str(c) for c in matrix_df.columns],
                        y=[str(r) for r in matrix_df.index],
                        colorscale="Blues",
                    )
                )
            else:
                fig = px.density_heatmap(
                    df,
                    x=x,
                    y=primary_y,
                    color_continuous_scale="Blues",
                )

        elif chosen_type == "radar":
            fig = go.Figure()
            if len(num_cols) >= 3:
                theta_cols = [str(c) for c in num_cols[:8]]
                for i, row in df.head(6).iterrows():
                    name = str(row[x]) if x in row else f"Item {i+1}"
                    r_vals = [float(row[c] or 0.0) for c in theta_cols]
                    r_vals.append(r_vals[0])
                    color = OKABE_ITO_PALETTE[i % len(OKABE_ITO_PALETTE)]
                    fig.add_trace(
                        go.Scatterpolar(
                            r=r_vals,
                            theta=theta_cols + [theta_cols[0]],
                            fill="toself",
                            name=name,
                            line=dict(color=color, width=2),
                            opacity=0.6,
                        )
                    )
            else:
                theta_vals = [str(v) for v in df[x]]
                r_vals = df[primary_y].tolist()
                fig.add_trace(
                    go.Scatterpolar(
                        r=r_vals + [r_vals[0]],
                        theta=theta_vals + [theta_vals[0]],
                        fill="toself",
                        name=str(primary_y),
                        line=dict(color=OKABE_ITO_PALETTE[0], width=2),
                    )
                )
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, showticklabels=True),
                    bgcolor="rgba(17,24,39,0.25)",
                )
            )

        elif chosen_type == "box":
            fig = px.box(
                df,
                x=x if len(df) <= 50 else None,
                y=primary_y,
                color=x if len(df) <= 20 else None,
                points="all" if len(df) <= 80 else "outliers",
                color_discrete_sequence=OKABE_ITO_PALETTE,
            )

        elif chosen_type == "funnel":
            fig = px.funnel(
                df.sort_values(by=primary_y, ascending=False),
                x=primary_y,
                y=x,
                color_discrete_sequence=OKABE_ITO_PALETTE,
            )

        elif chosen_type == "scatter":
            second_y = num_cols[1] if len(num_cols) >= 2 else primary_y
            fig = px.scatter(
                df,
                x=primary_y,
                y=second_y,
                hover_name=x if x in df.columns else None,
                size=primary_y if (df[primary_y] >= 0).all() else None,
                color=x if x in dim_cols and len(df) <= 15 else None,
                color_discrete_sequence=OKABE_ITO_PALETTE,
            )

        else:
            # Default grouped/single Bar chart
            color_col = dim_cols[1] if len(dim_cols) >= 2 else None
            if color_col:
                fig = px.bar(
                    df,
                    x=x,
                    y=primary_y,
                    color=color_col,
                    barmode="group",
                    color_discrete_sequence=OKABE_ITO_PALETTE,
                )
            else:
                fig = px.bar(
                    df,
                    x=x,
                    y=ys,
                    barmode="group",
                    color_discrete_sequence=OKABE_ITO_PALETTE,
                    text_auto=".2s" if len(ys) == 1 else False,
                )

        # Dynamic height scaling for high-category horizontal bar, heatmap, and funnel charts
        n_y_cats = df[x].nunique() if (x in df.columns) else len(df)
        if chosen_type in ("horizontal_bar", "stacked_horizontal_bar", "heatmap", "funnel") and n_y_cats > 10:
            chart_height = max(380, min(920, 110 + n_y_cats * 26))
        else:
            chart_height = 380

        fig.update_layout(
            title=dict(text="", font=dict(size=12, color="#94a3b8"), x=0.01)
            if not title
            else dict(text=title, font=dict(size=12, color="#94a3b8"), x=0.01, y=0.98),
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, -apple-system, BlinkMacSystemFont, sans-serif", size=12),
            margin=dict(l=16, r=24, t=44 if title else 20, b=32),
            legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="right", x=1),
            hoverlabel=dict(bgcolor="#0f172a", font_size=12, font_family="Inter, sans-serif"),
            bargap=0.28,
            height=chart_height,
        )
        if (
            chosen_type in ("bar", "stacked_bar", "stacked_bar_100", "combo", "line", "area", "waterfall")
            and x in df.columns
        ):
            max_lbl_len = max((len(str(v)) for v in df[x].dropna()), default=0)
            if max_lbl_len > 18 and n_y_cats >= 4:
                fig.update_xaxes(tickangle=-25, automargin=True)
            else:
                fig.update_xaxes(automargin=True)
        else:
            fig.update_xaxes(automargin=True)
        fig.update_yaxes(automargin=True)
        return fig
