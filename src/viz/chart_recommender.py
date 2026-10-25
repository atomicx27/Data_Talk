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
        "treemap",
        "sunburst",
        "heatmap",
        "radar",
        "box",
        "funnel",
        "scatter",
        "table",
    ]

    @classmethod
    def recommend_chart_type(cls, df: pd.DataFrame, question: str = "") -> str:
        """Select the best default chart type based on DataFrame shape, column names, and user prompt."""
        if df is None or df.empty:
            return "none"
        if len(df) == 1:
            return "kpi"

        q_low = (question or "").lower()
        if "waterfall" in q_low or "variance" in q_low or "why" in q_low:
            return "waterfall"
        if "sunburst" in q_low:
            return "sunburst"
        if "treemap" in q_low or "hierarchy" in q_low:
            return "treemap"
        if any(k in q_low for k in ("heatmap", "matrix", "cross tab", "crosstab", "intensity")):
            return "heatmap"
        if any(k in q_low for k in ("radar", "spider", "polar")):
            return "radar"
        if any(k in q_low for k in ("box", "whisker", "distribution", "spread", "dispersion", "outlier")):
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
        if "pie" in q_low or "share" in q_low:
            return "pie"

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

        if len(dim_cols) == 1:
            first_dim = str(dim_cols[0]).lower()
            if any(k in first_dim for k in ("date", "month", "year", "quarter", "week", "day", "time")):
                return "area" if "area" in q_low else "line"

        # High cardinality guard: categorical charts with > 40 items become illegible walls of bars.
        # Fallback to interactive paginated table to preserve readability.
        # For 2-dimension breakdowns, cardinality applies to the primary X-axis dimension, not total rows.
        primary_dim = dim_cols[0] if dim_cols else None
        primary_categories = df[primary_dim].nunique() if (primary_dim and primary_dim in df.columns) else len(df)
        is_primary_time = any(k in str(primary_dim).lower() for k in ("date", "month", "year", "quarter", "week", "day", "time")) if primary_dim else False

        if primary_categories > 40 and not is_primary_time:
            return "table"

        # Multi-metric breakdown / composition check (e.g. Male/Female/Other or New/Returning/Churned)
        total_cols = [
            c for c in num_cols
            if any(k in str(c).lower() for k in ("total_count", "grand_total", "all_students", "total_students"))
        ]
        non_total_num = [c for c in num_cols if c not in total_cols]
        pct_breakdowns = [
            c for c in non_total_num
            if any(k in str(c).lower() for k in ("pct", "percent", "margin", "rate", "ratio"))
        ]
        count_breakdowns = [c for c in non_total_num if c not in pct_breakdowns]

        is_breakdown_intent = any(
            k in q_low
            for k in (
                "ratio", "raio", "ritio", "rio", "gender", "breakdown", "demographic", "split",
                "proportion", "share", "composition", "distribution", "other", "wise"
            )
        )

        # If 2+ percentage breakdown columns exist (e.g., Male Pct, Female Pct, Other Pct)
        if len(pct_breakdowns) >= 2:
            return "stacked_bar_100" if (is_breakdown_intent or "100" in q_low) else "stacked_bar"

        # If 2+ count/volume breakdown columns exist (e.g., Male Count, Female Count, Other Count)
        if len(count_breakdowns) >= 2 and (
            is_breakdown_intent
            or any(k in q_low for k in ("by", "across", "per", "category", "status", "student", "college", "wise"))
        ):
            return "stacked_bar"

        # If 2 categorical dimensions exist (e.g. COLLEGE NAME and GENDER/YEAR), multi-series stacked chart is ideal
        if len(dim_cols) >= 2:
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

        if len(df) > 10 and dim_cols:
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

        # High cardinality guard in visual generation: categorical charts with > 40 items become illegible walls of bars.
        if chosen_type in ("bar", "horizontal_bar", "stacked_bar", "stacked_bar_100", "stacked_horizontal_bar", "pie", "donut", "radar", "funnel"):
            primary_dim = dim_cols[0] if dim_cols else None
            primary_categories = df[primary_dim].nunique() if (primary_dim and primary_dim in df.columns) else len(df)
            is_time = any(k in str(primary_dim).lower() for k in ("date", "month", "year", "quarter", "week", "day", "time")) if primary_dim else False
            if primary_categories > 40 and not is_time:
                # If raw entity data (e.g. 10,000 raw students), refuse to render an unreadable blob
                if primary_categories > 100:
                    return None
                # Otherwise, gracefully truncate to Top 25 primary categories
                p_sort = num_cols[0] if num_cols else df.columns[0]
                if len(dim_cols) >= 2 and primary_dim in df.columns:
                    top_cats = df.groupby(primary_dim)[p_sort].sum().sort_values(ascending=False).head(25).index
                    df = df[df[primary_dim].isin(top_cats)]
                else:
                    df = df.sort_values(by=p_sort, ascending=False).head(25)

        x = x_col or (dim_cols[0] if dim_cols else df.columns[0])

        # Detect multi-metric breakdown candidates (e.g., Male/Female/Other) excluding grand totals
        total_cols = [
            c for c in num_cols
            if any(k in str(c).lower() for k in ("total_count", "grand_total", "all_students", "total_students"))
        ]
        non_total_num = [c for c in num_cols if c not in total_cols and c != x]
        pct_breakdowns = [
            c for c in non_total_num
            if any(k in str(c).lower() for k in ("pct", "percent", "margin", "rate", "ratio"))
        ]
        count_breakdowns = [c for c in non_total_num if c not in pct_breakdowns]

        # Prioritize business measures (revenue, fee, salary, amount, profit, etc.) over row counts
        business_metrics = [
            c for c in num_cols
            if not any(k in str(c).lower() for k in ("row_count", "total_count", "_count"))
        ]
        matched_metrics = [
            c for c in num_cols
            if any(k in str(c).lower() for k in title.lower().split() if len(k) > 2)
        ]
        preferred_single_metric = matched_metrics[0] if matched_metrics else (business_metrics[0] if business_metrics else num_cols[0])

        if chosen_type in ("stacked_bar_100", "percent_stacked_bar") and len(pct_breakdowns) >= 2:
            breakdown_ys = pct_breakdowns
        elif len(count_breakdowns) >= 2 and chosen_type in ("stacked_bar", "stacked_horizontal_bar"):
            breakdown_ys = count_breakdowns
        elif len(pct_breakdowns) >= 2:
            breakdown_ys = pct_breakdowns
        else:
            breakdown_ys = [preferred_single_metric]

        ys = y_cols or breakdown_ys
        if not ys:
            ys = [preferred_single_metric]

        primary_y = ys[0]

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
                    barmode="stack",
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
                fig = px.bar(
                    df,
                    x=x,
                    y=primary_y,
                    color=color_col,
                    barmode="stack",
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
                    barmode="stack",
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
                    barmode="stack",
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
            color_col = dim_cols[1] if (len(dim_cols) >= 2 and len(ys) == 1) else None
            fig = px.line(
                df,
                x=x,
                y=ys if not color_col else ys[0],
                color=color_col,
                markers=True,
                color_discrete_sequence=OKABE_ITO_PALETTE,
            )

        elif chosen_type == "area":
            fig = px.area(
                df,
                x=x,
                y=ys,
                color_discrete_sequence=OKABE_ITO_PALETTE,
            )

        elif chosen_type == "waterfall":
            val_col = "variance_delta" if "variance_delta" in df.columns else primary_y
            fig = go.Figure(
                go.Waterfall(
                    name=str(val_col),
                    orientation="v",
                    measure=["relative"] * len(df),
                    x=df[x].astype(str).tolist(),
                    y=df[val_col].tolist(),
                    connector={"line": {"color": "#6B7280"}},
                    increasing={"marker": {"color": "#10B981"}},
                    decreasing={"marker": {"color": "#EF4444"}},
                )
            )

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
            total_val = df[primary_y].sum()
            formatted_total = f"{total_val:,.1f}" if isinstance(total_val, float) else f"{total_val:,}"
            fig.update_layout(
                annotations=[
                    dict(
                        text=f"<b>Total</b><br>{formatted_total}",
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
                ).fillna(0)
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
            height=380,
        )
        fig.update_xaxes(automargin=True)
        fig.update_yaxes(automargin=True)
        return fig
