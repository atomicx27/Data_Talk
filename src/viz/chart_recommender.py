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

        if dim_cols:
            first_dim = str(dim_cols[0]).lower()
            if any(k in first_dim for k in ("date", "month", "year", "quarter", "week", "day", "time")):
                return "area" if "area" in q_low else "line"

        # If 2 categorical dimensions and 1 numeric metric exist, stacked bar is ideal for breakdown queries
        if len(dim_cols) >= 2 and len(num_cols) == 1 and any(k in q_low for k in ("by", "breakdown", "per", "across")):
            return "stacked_bar"

        # If both a dollar/volume metric and a percentage/rate metric exist, recommend a Dual-Axis Combo chart!
        pct_cols = [c for c in num_cols if any(k in str(c).lower() for k in ("pct", "percent", "margin", "rate", "ratio"))]
        non_pct_cols = [c for c in num_cols if c not in pct_cols]
        if pct_cols and non_pct_cols and dim_cols and 2 <= len(df) <= 20:
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

        if not num_cols:
            return None

        x = x_col or (dim_cols[0] if dim_cols else df.columns[0])
        ys = y_cols or [c for c in num_cols if c != x][:2]
        if not ys:
            ys = [num_cols[0]]

        primary_y = ys[0]

        if chosen_type == "combo" and len(num_cols) >= 2:
            pct_cols = [c for c in num_cols if any(k in str(c).lower() for k in ("pct", "percent", "margin", "rate"))]
            bar_col = next((c for c in num_cols if c not in pct_cols), num_cols[0])
            line_col = pct_cols[0] if pct_cols else num_cols[1]
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
            fig = px.bar(
                df,
                x=x,
                y=ys,
                barmode="group",
                color_discrete_sequence=OKABE_ITO_PALETTE,
                text_auto=".2s" if len(ys) == 1 else False,
            )

        fig.update_layout(
            title=dict(text=title, font=dict(size=13, color="#94a3b8"), x=0.01),
            template="plotly_dark",
            paper_bgcolor="rgba(17,24,39,0.0)",
            plot_bgcolor="rgba(17,24,39,0.28)",
            font=dict(family="Inter, -apple-system, BlinkMacSystemFont, sans-serif", size=12),
            margin=dict(l=24, r=24, t=42 if title else 20, b=24),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hoverlabel=dict(bgcolor="#0f172a", font_size=12, font_family="Inter, sans-serif"),
            bargap=0.28,
            height=380,
        )
        return fig
