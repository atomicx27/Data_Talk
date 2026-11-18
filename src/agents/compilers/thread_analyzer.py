"""Hybrid Chart Discussion Thread engine combining in-memory statistics with live warehouse drill-downs."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import pandas as pd

from src.agents.compilers.text_matching import detect_relevant_tables, normalize_what_if_overrides
from src.agents.llm_router import LLMRouter
from src.agents.prompt_manager import PromptManager
from src.knowledge.okf_models import OKFWorkspace

if TYPE_CHECKING:
    from src.connectors.base import BaseConnector

_LOGGER = logging.getLogger(__name__)


def should_render_chart(
    question: str,
    df: pd.DataFrame | None,
    response_mode: str = "auto",
    is_thread: bool = False,
) -> bool:
    """Determine whether a response should render a visual chart or Text-Only."""
    mode = (response_mode or "auto").lower().strip()
    if mode == "text_only":
        return False
    if df is None or df.empty:
        return False
    if mode == "with_chart":
        return True

    q_low = (question or "").lower().strip()

    explicit_chart_kw = (
        "chart", "plot", "graph", "visual", "visualize", "waterfall",
        "bar chart", "line chart", "pie chart", "donut", "scatter",
        "treemap", "heatmap", "histogram", "funnel", "radar",
    )
    if any(k in q_low for k in explicit_chart_kw):
        return True

    drill_kw = (
        "break down by", "breakdown by", "broken down by", "split by",
        "drill down", "by month", "monthly trend", "by year", "yearly trend",
        "by quarter", "over time", "trend of", "distribution of",
    )
    if any(k in q_low for k in drill_kw) and len(df) > 1:
        return True

    text_only_kw = (
        "text only", "in text", "just tell me", "no chart", "without chart",
        "summarize", "summary", "explain", "key takeaway", "takeaways",
        "insight", "insights", "what does this mean", "what is the difference",
        "difference between", "gap between", "ratio between",
        "which one is highest", "which is highest", "which one is lowest",
        "which is lowest", "who is the top", "what is the top",
        "what percentage", "what percent", "share of", "how much is",
        "above average", "below average", "what is the average",
        "what is the total", "what is the median", "why did", "why is",
    )
    if any(k in q_low for k in text_only_kw):
        return False

    if is_thread:
        return False

    if len(df) <= 1:
        return False
    return True


def needs_warehouse_sql_for_thread(
    question: str,
    parent_df: pd.DataFrame,
    parent_question: str,
    workspace: OKFWorkspace,
) -> bool:
    """Check if a thread question requires querying the warehouse for unrepresented columns/dimensions."""
    if parent_df is None or parent_df.empty:
        return True

    q_low = (question or "").lower().strip()
    existing_cols_low = {str(c).lower() for c in parent_df.columns}

    existing_vals_low: set[str] = set()
    for col in parent_df.columns:
        if parent_df[col].dtype == object or str(parent_df[col].dtype).startswith("str"):
            for v in parent_df[col].dropna().unique()[:200]:
                existing_vals_low.add(str(v).lower().strip())

    breakdown_match = re.search(
        r"\b(?:break\s*(?:this|it|that)?\s*down\s*by|split\s*by|group\s*by|drill\s*(?:down\s*)?(?:into|by)|by)\s+([a-z0-9_ ]+)",
        q_low,
    )
    if breakdown_match:
        requested_dim = breakdown_match.group(1).strip().split()[0]
        if requested_dim not in ("the", "this", "that", "it", "segment", "category") and requested_dim not in existing_cols_low:
            return True

    temporal_kw = ("monthly", "by month", "by year", "yearly", "quarterly", "by quarter", "over time", "time series", "weekly")
    has_time_col = any(
        any(t in c for t in ("date", "month", "year", "quarter", "week", "term", "period"))
        for c in existing_cols_low
        if c not in ("period_prior", "period_current")
    )
    if any(k in q_low for k in temporal_kw) and not has_time_col:
        return True

    parent_q_low = (parent_question or "").lower()
    for table in workspace.tables:
        for col in table.columns:
            if col.is_blocked or col.is_primary_key:
                continue
            c_low = col.name.lower()
            c_words = c_low.replace("_", " ")
            if c_low in existing_cols_low or c_words in parent_q_low:
                continue
            if len(c_words) >= 4 and re.search(rf"\b{re.escape(c_words)}\b", q_low):
                if c_words not in existing_vals_low:
                    return True
            if col.business_alias:
                alias_low = col.business_alias.lower().strip()
                if len(alias_low) >= 4 and alias_low not in existing_cols_low and alias_low not in parent_q_low:
                    if re.search(rf"\b{re.escape(alias_low)}\b", q_low) and alias_low not in existing_vals_low:
                        return True

    return False


def analyze_chart_dataframe(
    question: str,
    parent_df: pd.DataFrame,
    parent_question: str = "",
    parent_summary: str = "",
    llm_router: LLMRouter | None = None,
) -> tuple[str, pd.DataFrame, list[str]]:
    """Perform exact in-memory statistical & comparative analysis on the chart's DataFrame."""
    df = parent_df.copy()
    num_cols = list(df.select_dtypes(include=["number"]).columns)
    dim_cols = [c for c in df.columns if c not in num_cols]

    if not num_cols:
        for c in df.columns:
            converted = pd.to_numeric(df[c], errors="coerce")
            if converted.notna().sum() > 0 and c not in dim_cols[:1]:
                df[c] = converted
                num_cols.append(c)
        dim_cols = [c for c in df.columns if c not in num_cols]

    dim_col = dim_cols[0] if dim_cols else df.columns[0]
    q_low = (question or "").lower().strip()

    from src.viz.chart_recommender import ChartRecommender

    val_col = ChartRecommender._select_preferred_metric(num_cols, question) if num_cols else df.columns[-1]

    known_metric_keywords = (
        "grade", "grades", "mark", "marks", "gpa", "cgpa", "score", "scores",
        "salary", "salaries", "compensation", "tuition", "fee", "fees",
        "cost", "costs", "expense", "expenses", "revenue", "sales",
        "profit", "profits", "margin", "margins", "refund", "refunds",
        "budget", "budgets", "expenditure", "expenditures", "discount", "discounts",
    )
    existing_cols_text = (
        " ".join(str(c).lower() for c in df.columns)
        + " "
        + (parent_question or "").lower()
        + " "
        + (parent_summary or "").lower()
    )
    missing_metric_concepts = [
        kw
        for kw in known_metric_keywords
        if re.search(rf"\b{re.escape(kw)}\b", q_low)
        and kw not in existing_cols_text
    ]
    val_label = val_col.replace("_", " ").title()
    dim_label = dim_col.replace("_", " ").title()

    scope_advisory = ""
    if missing_metric_concepts:
        missing_lbl = missing_metric_concepts[0].title()
        table_cols_str = ", ".join(f"`{c}`" for c in df.columns)
        scope_advisory = (
            f"> ℹ️ **Table Scope Notice:** The current chart table contains {table_cols_str}, "
            f"but does **not contain `{missing_lbl}`** data.\n"
            f"> Below is the analysis based strictly on the available **{val_label}** in this ready table.\n\n"
        )

    is_forecast_table = (
        ("projected_p50" in df.columns or "p50" in df.columns)
        and ("actual" in df.columns or "ols_trend_fit" in df.columns)
    )
    if is_forecast_table:
        p_col = "period" if "period" in df.columns else dim_col
        p50_col = "projected_p50" if "projected_p50" in df.columns else "p50"
        mom_col = "recent_momentum" if "recent_momentum" in df.columns else ("smart_momentum" if "smart_momentum" in df.columns else None)
        p10_col = "p10_downside" if "p10_downside" in df.columns else ("p10" if "p10" in df.columns else None)
        p90_col = "p90_upside" if "p90_upside" in df.columns else ("p90" if "p90" in df.columns else None)

        metric_display = "Target Metric"
        if ":" in (parent_question or ""):
            raw_m = parent_question.split(":", 1)[1].strip()
            metric_display = raw_m.split(".")[-1].replace("_", " ").title() + f" (`{raw_m}`)"

        act_series = pd.to_numeric(df["actual"], errors="coerce") if "actual" in df.columns else pd.Series(dtype=float)
        p50_series = pd.to_numeric(df[p50_col], errors="coerce")

        hist_df = df[act_series.notna()].copy()
        proj_df = df[p50_series.notna() & act_series.isna()].copy()
        if proj_df.empty:
            proj_df = df[p50_series.notna()].tail(4).copy()

        hist_vals = pd.to_numeric(hist_df["actual"], errors="coerce").dropna()
        n_hist = len(hist_vals)
        hist_total = float(hist_vals.sum()) if n_hist else 0.0
        hist_mean = float(hist_vals.mean()) if n_hist else 0.0
        first_hist_p = str(hist_df.iloc[0][p_col]) if not hist_df.empty else "Start"
        last_hist_p = str(hist_df.iloc[-1][p_col]) if not hist_df.empty else "Latest"
        last_hist_v = float(hist_vals.iloc[-1]) if n_hist else 0.0

        peak_idx = hist_vals.idxmax() if n_hist else None
        min_hist_idx = hist_vals.idxmin() if n_hist else None
        peak_p = str(df.loc[peak_idx, p_col]) if peak_idx is not None else last_hist_p
        peak_v = float(hist_vals.loc[peak_idx]) if peak_idx is not None else last_hist_v
        trough_p = str(df.loc[min_hist_idx, p_col]) if min_hist_idx is not None else first_hist_p
        trough_v = float(hist_vals.loc[min_hist_idx]) if min_hist_idx is not None else 0.0

        n_proj = len(proj_df)
        first_proj_p = str(proj_df.iloc[0][p_col]) if n_proj else "Next Period"
        last_proj_p = str(proj_df.iloc[-1][p_col]) if n_proj else first_proj_p
        first_p50 = float(pd.to_numeric(proj_df.iloc[0][p50_col], errors="coerce") or 0.0) if n_proj else 0.0
        last_p50 = float(pd.to_numeric(proj_df.iloc[-1][p50_col], errors="coerce") or 0.0) if n_proj else 0.0
        first_mom = float(pd.to_numeric(proj_df.iloc[0][mom_col], errors="coerce") or first_p50) if (n_proj and mom_col) else first_p50
        last_mom = float(pd.to_numeric(proj_df.iloc[-1][mom_col], errors="coerce") or last_p50) if (n_proj and mom_col) else last_p50
        first_p10 = float(pd.to_numeric(proj_df.iloc[0][p10_col], errors="coerce") or 0.0) if (n_proj and p10_col) else 0.0
        first_p90 = float(pd.to_numeric(proj_df.iloc[0][p90_col], errors="coerce") or 0.0) if (n_proj and p90_col) else 0.0
        last_p10 = float(pd.to_numeric(proj_df.iloc[-1][p10_col], errors="coerce") or 0.0) if (n_proj and p10_col) else 0.0
        last_p90 = float(pd.to_numeric(proj_df.iloc[-1][p90_col], errors="coerce") or 0.0) if (n_proj and p90_col) else 0.0

        ols_slope = ((last_p50 - first_p50) / max(1, n_proj - 1)) if n_proj > 1 else (first_p50 - hist_mean)
        mom_slope = ((last_mom - first_mom) / max(1, n_proj - 1)) if n_proj > 1 else ols_slope
        vs_mean_pct = ((first_p50 - hist_mean) / abs(hist_mean) * 100.0) if hist_mean != 0 else 0.0
        vs_last_pct = ((first_p50 - last_hist_v) / abs(last_hist_v) * 100.0) if last_hist_v != 0 else 0.0

        is_upward = ols_slope >= 0
        direction_word = "increasing (upward trajectory)" if is_upward else "decreasing (downward trajectory)"
        verdict_headline = (
            f"**Yes — {metric_display} follows a long-term {direction_word} of `{ols_slope:+,.2f}` per period (OLS) and `{mom_slope:+,.2f}` per period (Recent Momentum).**"
            if is_upward
            else f"**No — {metric_display} follows a {direction_word} of `{ols_slope:+,.2f}` per period (OLS).**"
        )

        if last_hist_v > first_p50 * 1.05 and is_upward:
            step_explanation = (
                f"- **Why `{first_proj_p}` (`{first_p50:,.2f}`) is lower than `{last_hist_p}` (`{last_hist_v:,.2f}`) despite an upward trend:** "
                f"Recent historical periods (`{peak_p}` at `{peak_v:,.2f}` and `{last_hist_p}` at `{last_hist_v:,.2f}`) were above-trend spikes compared to the `{n_hist}`-period historical average (`{hist_mean:,.2f}`). "
                f"In `{first_proj_p}`, the regression line normalizes back toward the long-term trendline (**`{vs_mean_pct:+.1f}%` above the historical mean**, though `{vs_last_pct:+.1f}%` below `{last_hist_p}`'s spike) and then **rises steadily every year from `{first_proj_p}` (`{first_p50:,.2f}`) to `{last_proj_p}` (`{last_p50:,.2f}`)**."
            )
        elif last_hist_v < first_p50 * 0.95 and not is_upward:
            step_explanation = (
                f"- **Transition from `{last_hist_p}` (`{last_hist_v:,.2f}`) to `{first_proj_p}` (`{first_p50:,.2f}`):** "
                f"`{last_hist_p}` was an unusually low dip below the `{hist_mean:,.2f}` historical average; the forecast normalizes to the trendline in `{first_proj_p}` before continuing its `{ols_slope:+,.2f}`/period trajectory."
            )
        else:
            step_explanation = (
                f"- **Transition from `{last_hist_p}` (`{last_hist_v:,.2f}`) to `{first_proj_p}` (`{first_p50:,.2f}`):** "
                f"Projects `{vs_mean_pct:+.1f}%` vs. the historical period mean (`{hist_mean:,.2f}`) and `{vs_last_pct:+.1f}%` vs. `{last_hist_p}`."
            )

        forecast_bullet_rows = []
        for _, pr in proj_df.iterrows():
            p_lbl = str(pr[p_col])
            p50_v = float(pd.to_numeric(pr[p50_col], errors="coerce") or 0.0)
            mom_v = float(pd.to_numeric(pr[mom_col], errors="coerce") or p50_v) if mom_col else p50_v
            p10_v = float(pd.to_numeric(pr[p10_col], errors="coerce") or 0.0) if p10_col else 0.0
            p90_v = float(pd.to_numeric(pr[p90_col], errors="coerce") or 0.0) if p90_col else 0.0
            forecast_bullet_rows.append(
                f"  - **`{p_lbl}`:** OLS P50 **`{p50_v:,.2f}`** · Recent Momentum **`{mom_v:,.2f}`** · 80% Band `[ {p10_v:,.2f} — {p90_v:,.2f} ]`"
            )

        answer_md = (
            f"### 📈 Future Trend & Forecast Interpretation: {metric_display}\n\n"
            f"{verdict_headline}\n\n"
            f"- **Long-Term Trend (`{first_hist_p}`–`{last_hist_p}`, `{n_hist}` periods):** Historical actuals total **`{hist_total:,.2f}`** with an average of **`{hist_mean:,.2f}`/period** (ranging from a low of `{trough_v:,.2f}` in **`{trough_p}`** to a peak of `{peak_v:,.2f}` in **`{peak_p}`**).\n"
            f"{step_explanation}\n"
            f"- **Dual Forecast Models (`{first_proj_p}` -> `{last_proj_p}`):**\n"
            f"  1. **OLS Linear Trend (P50 Green Dashed):** Grows by **`{ols_slope:+,.2f}`/period** from **`{first_p50:,.2f}`** (`{first_proj_p}`) to **`{last_p50:,.2f}`** (`{last_proj_p}`).\n"
            f"  2. **Recent-Weighted Momentum (Amber Dash-Dot):** Weights recent periods more heavily, growing by **`{mom_slope:+,.2f}`/period** from **`{first_mom:,.2f}`** (`{first_proj_p}`) to **`{last_mom:,.2f}`** (`{last_proj_p}`).\n"
            f"- **80% Prediction Interval (P10 Downside – P90 Upside):** In `{first_proj_p}`, there is an 80% statistical probability the outcome lands between **`{first_p10:,.2f}`** and **`{first_p90:,.2f}`** (widening to `[ {last_p10:,.2f} — {last_p90:,.2f} ]` by `{last_proj_p}`).\n\n"
            f"**Year-by-Year Projected Schedule:**\n" + "\n".join(forecast_bullet_rows)
        )

        if llm_router and llm_router.is_configured():
            try:
                sample_records = df.to_dict(orient="records")
                fallback_sys = (
                    "You are a Principal Quantitative BI Forecaster answering a user's question about a Future Trend & What-If Fan Chart. "
                    "Use ONLY the exact numbers from the Pre-computed Forecast Analysis below. "
                    "Directly answer the user's question in the very first sentence. Never treat null/None future actuals as zero."
                )
                fallback_usr = (
                    f"Parent Chart Context: {parent_question}\n"
                    f"User Thread Question: {question}\n"
                    f"Pre-computed Forecast Analysis:\n{answer_md}\n"
                    f"Chart Rows (JSON):\n{json.dumps(sample_records, default=str)}\n\n"
                    "Provide a clear, direct, executive answer to the User Thread Question:"
                )
                llm_text = llm_router.complete("chart_thread_enrichment", fallback_sys, fallback_usr)
                if llm_text and len(llm_text.strip()) > 30:
                    answer_md = llm_text.strip()
            except Exception as exc:
                _LOGGER.debug("Forecast thread LLM enrichment skipped: %s", exc)

        follow_ups = [
            f"Why does {first_proj_p} differ from {last_hist_p}?",
            "Compare OLS Linear Trend (P50) vs Recent Momentum",
            f"What is the P10 downside and P90 upside range for {first_proj_p}–{last_proj_p}?",
        ]
        return answer_md, df, follow_ups

    def _format_row_label(r: Any) -> str:
        main_val = str(r[dim_col]).strip()
        if len(dim_cols) > 1:
            extra_parts = [str(r[c]).strip() for c in dim_cols[1:] if c in r and pd.notna(r[c])]
            if extra_parts:
                return f"{main_val} ({', '.join(extra_parts)})"
        return main_val

    analytical_typo_map = {
        r"\bhights\b": "highest",
        r"\bhight\b": "highest",
        r"\bhighests\b": "highest",
        r"\bhigest\b": "highest",
        r"\bheighest\b": "highest",
        r"\bhieght\b": "highest",
        r"\blowst\b": "lowest",
        r"\blowes\b": "lowest",
        r"\bbottomm\b": "bottom",
        r"\bsecnd\b": "second",
        r"\bsecon\b": "second",
        r"\b2nd\b": "second",
        r"\b3rd\b": "third",
        r"\b4th\b": "fourth",
        r"\b5th\b": "fifth",
        r"\bratioo\b": "ratio",
        r"\bavrage\b": "average",
    }
    for pat, rep in analytical_typo_map.items():
        q_low = re.sub(pat, rep, q_low)

    if len(dim_cols) > 1:
        for extra_dim in dim_cols[1:]:
            unique_sub_vals = df[extra_dim].dropna().unique()
            for sub_val in unique_sub_vals:
                sub_str = str(sub_val).strip().lower()
                if len(sub_str) >= 3 and re.search(rf"\b{re.escape(sub_str)}\b", q_low):
                    sub_filtered = df[df[extra_dim].astype(str).str.lower() == sub_str].copy()
                    if not sub_filtered.empty:
                        df = sub_filtered
                        break

    series = pd.to_numeric(df[val_col], errors="coerce").fillna(0.0)
    total_val = float(series.sum())
    mean_val = float(series.mean()) if len(series) else 0.0
    median_val = float(series.median()) if len(series) else 0.0

    if llm_router and llm_router.is_configured():
        try:
            sample_records = df.head(60).to_dict(orient="records")
            sorted_desc_ai = df.assign(_v=series).sort_values("_v", ascending=False).reset_index(drop=True)
            top_items_ai = [
                f"#{i+1}: {_format_row_label(r)} = {float(r['_v']):,.2f}"
                for i, (_, r) in enumerate(sorted_desc_ai.head(10).iterrows())
            ]
            sys_prompt = (
                "You are Data_Talk's Senior Conversational BI Chart Assistant. "
                "You are answering a user's question directly about a specific visual chart that they are viewing.\n"
                "CORE PRINCIPLES:\n"
                "1. Directly, conversationally, and accurately answer the user's specific question in the very first sentence.\n"
                "2. Understand natural human phrasing and typos (e.g. 'hight' or 'hights' means highest, '2nd hights' means 2nd highest / runner-up, 'gap' means difference).\n"
                "3. Ground your answer strictly in the provided chart table data and pre-computed rankings. Never invent numbers.\n"
                "4. If there are multiple categorical dimensions (e.g. College Name and Gender), specify both so the user knows exactly which bar or segment you are referring to.\n"
                "5. Format key numbers and entities in **bold** or `code`."
            )
            usr_prompt = (
                f"Parent Chart Context: {parent_question}\n"
                f"Chart Primary Metric: {val_label} (Sum: {total_val:,.2f}, Mean: {mean_val:,.2f}, Median: {median_val:,.2f})\n"
                f"Top Ranked Segments:\n" + "\n".join(f"- {it}" for it in top_items_ai) + "\n\n"
                f"All Chart Table Records (JSON):\n{json.dumps(sample_records, default=str)}\n\n"
                f"User Thread Question: {question}\n\n"
                "Provide a direct, helpful, and executive response:"
            )
            ai_text = llm_router.complete("chart_thread_qa", sys_prompt, usr_prompt)
            if ai_text and len(ai_text.strip()) > 15:
                final_ans = (scope_advisory + ai_text.strip()) if scope_advisory and not ai_text.startswith("> ℹ️") else ai_text.strip()
                follow_ups = [
                    f"Which {dim_label} segments are above average?",
                    f"Compare top vs bottom {dim_label}",
                    f"What percentage share does the top {dim_label} hold?",
                ]
                return final_ans, df, follow_ups
        except Exception as exc:
            _LOGGER.warning("Primary LLM chart thread Q&A encountered error, falling back to deterministic analyzer: %s", exc)

    matched_indices: list[Any] = []
    for idx_val, raw_seg in df[dim_col].items():
        seg_str = str(raw_seg).strip()
        seg_low = seg_str.lower()
        if not seg_low or seg_low in ("all other segments", "(unassigned)"):
            continue
        if seg_low in q_low:
            matched_indices.append(idx_val)
        else:
            tokens = [t for t in re.split(r"[^a-z0-9]+", seg_low) if len(t) >= 4 and t not in ("college", "department", "school", "group", "tier", "type", "total")]
            if tokens and any(re.search(rf"\b{re.escape(tok)}\b", q_low) for tok in tokens):
                matched_indices.append(idx_val)

    working_df = df.copy()
    answer_md = ""

    ordinal_word_map = {
        "first": 1, "1st": 1,
        "second": 2, "2nd": 2,
        "third": 3, "3rd": 3,
        "fourth": 4, "4th": 4,
        "fifth": 5, "5th": 5,
        "sixth": 6, "6th": 6,
        "seventh": 7, "7th": 7,
        "eighth": 8, "8th": 8,
        "ninth": 9, "9th": 9,
        "tenth": 10, "10th": 10,
    }
    ord_m = re.search(
        r"\b(1st|first|2nd|second|3rd|third|4th|fourth|5th|fifth|6th|seventh|8th|ninth|10th|#\s*\d+|\brank\s*\d+)\b",
        q_low,
    )
    target_k: int | None = None
    if ord_m:
        raw_matched_ord = ord_m.group(1).replace(" ", "")
        if raw_matched_ord.startswith("#"):
            target_k = int(raw_matched_ord[1:])
        elif raw_matched_ord.startswith("rank"):
            target_k = int(raw_matched_ord.replace("rank", ""))
        elif raw_matched_ord in ordinal_word_map:
            target_k = ordinal_word_map[raw_matched_ord]
        elif raw_matched_ord[:-2].isdigit():
            target_k = int(raw_matched_ord[:-2])

    if len(matched_indices) >= 2:
        sub_df = df.loc[matched_indices].copy()
        working_df = sub_df
        r1 = sub_df.iloc[0]
        r2 = sub_df.iloc[1]
        s1, s2 = _format_row_label(r1), _format_row_label(r2)
        v1 = float(pd.to_numeric(r1[val_col], errors="coerce") or 0.0)
        v2 = float(pd.to_numeric(r2[val_col], errors="coerce") or 0.0)
        diff = v1 - v2
        pct_diff = ((v1 - v2) / abs(v2) * 100.0) if v2 != 0 else 0.0
        share1 = (v1 / total_val * 100.0) if total_val != 0 else 0.0
        share2 = (v2 / total_val * 100.0) if total_val != 0 else 0.0

        lines = [
            f"### 🔍 Direct Comparison: **{s1}** vs. **{s2}** ({val_label})",
            f"- **{s1}:** `{v1:,.2f}` ({share1:.1f}% of chart total)",
            f"- **{s2}:** `{v2:,.2f}` ({share2:.1f}% of chart total)",
            f"- **Net Gap / Difference:** `{diff:+,.2f}` (**{pct_diff:+.1f}%** {'higher' if diff >= 0 else 'lower'} in {s1})",
        ]
        if "period_prior" in sub_df.columns and "period_current" in sub_df.columns and "variance_delta" in sub_df.columns:
            lines.append(
                f"- **Period Shift:** **{s1}** moved from `{float(r1['period_prior']):,.2f}` -> `{float(r1['period_current']):,.2f}` (`{float(r1['variance_delta']):+,.2f}`), "
                f"while **{s2}** moved from `{float(r2['period_prior']):,.2f}` -> `{float(r2['period_current']):,.2f}` (`{float(r2['variance_delta']):+,.2f}`)."
            )
        answer_md = "\n".join(lines)

    elif len(matched_indices) == 1 and not (target_k and target_k >= 2):
        sorted_all = df.assign(_v=series).sort_values("_v", ascending=False).reset_index(drop=True)
        row = df.loc[matched_indices[0]]
        seg_name = _format_row_label(row)
        val = float(pd.to_numeric(row[val_col], errors="coerce") or 0.0)
        rank_matches = sorted_all.index[sorted_all[dim_col].astype(str) == str(row[dim_col])].tolist()
        rank = (rank_matches[0] + 1) if rank_matches else 1
        share = (val / total_val * 100.0) if total_val != 0 else 0.0
        vs_mean = val - mean_val
        vs_mean_pct = (vs_mean / abs(mean_val) * 100.0) if mean_val != 0 else 0.0

        metric_details = [
            f"**{c.replace('_', ' ').title()}:** `{float(row[c]):,.2f}`"
            if isinstance(row[c], (int, float))
            else f"**{c.replace('_', ' ').title()}:** `{row[c]}`"
            for c in df.columns
            if c not in dim_cols
        ]
        answer_md = (
            f"### 📌 Segment Spotlight: **{seg_name}**\n"
            f"- **Rank:** `#{rank}` out of `{len(df)}` segments in this chart\n"
            f"- **{val_label}:** `{val:,.2f}` (**{share:.1f}%** share of total `{total_val:,.2f}`)\n"
            f"- **vs. Chart Average (`{mean_val:,.2f}`):** `{vs_mean:+,.2f}` (**{vs_mean_pct:+.1f}%** {'above' if vs_mean >= 0 else 'below'} average)\n"
            + (f"- **All Metrics:** {' · '.join(metric_details)}" if len(metric_details) > 1 else "")
        )
        working_df = df.loc[matched_indices].copy()

    elif target_k is not None and target_k >= 2:
        is_lowest_dir = any(k in q_low for k in ("lowest", "bottom", "minimum", "smallest", "least", "worst", "last"))
        if is_lowest_dir:
            sorted_ranked = df.assign(_v=series).sort_values("_v", ascending=True).reset_index(drop=True)
            dir_label = "Lowest"
        else:
            sorted_ranked = df.assign(_v=series).sort_values("_v", ascending=False).reset_index(drop=True)
            dir_label = "Highest"
        top_row = sorted_ranked.iloc[0]
        top_label = _format_row_label(top_row)
        top_val = float(top_row["_v"])

        idx_k = min(target_k - 1, len(sorted_ranked) - 1)
        target_row = sorted_ranked.iloc[idx_k]
        target_label = _format_row_label(target_row)
        target_val = float(target_row["_v"])
        target_share = (target_val / total_val * 100.0) if total_val != 0 else 0.0
        gap_vs_top = target_val - top_val
        gap_pct = ((target_val - top_val) / abs(top_val) * 100.0) if top_val != 0 else 0.0

        ord_suffix = f"{target_k}th" if target_k in (4, 5, 6, 7, 8, 9, 10) else ("2nd" if target_k == 2 else ("3rd" if target_k == 3 else f"{target_k}th"))
        medal = "🥈" if target_k == 2 else ("🥉" if target_k == 3 else f"#{target_k}")

        items = [
            f"`#{i+1}` **{_format_row_label(r)}** — `{float(r['_v']):,.2f}`"
            + (" 👈 *(Selected)*" if i == idx_k else "")
            for i, (_, r) in enumerate(sorted_ranked.head(max(5, target_k + 1)).iterrows())
        ]

        answer_md = (
            f"### {medal} #{target_k} ({ord_suffix.title()} {dir_label}): **{target_label}**\n\n"
            f"The **{ord_suffix} {dir_label.lower()}** {dim_label} by **{val_label}** is **{target_label}** at **`{target_val:,.2f}`** "
            f"({target_share:.1f}% of chart total).\n\n"
            f"- **Difference vs #{1} ({top_label} at `{top_val:,.2f}`):** `{gap_vs_top:+,.2f}` (**{gap_pct:+.1f}%**)\n"
            f"- **vs Chart Average (`{mean_val:,.2f}`):** `{target_val - mean_val:+,.2f}`\n\n"
            f"**Ranking Overview:**\n" + "\n".join(f"- {it}" for it in items)
        )
        working_df = sorted_ranked.head(max(5, target_k + 1))

    elif "above average" in q_low or "greater than average" in q_low or "over average" in q_low:
        above_df = df[series > mean_val].copy()
        working_df = above_df if not above_df.empty else df
        items = [
            f"**{_format_row_label(r)}** (`{float(pd.to_numeric(r[val_col], errors='coerce') or 0.0):,.2f}`)"
            for _, r in above_df.iterrows()
        ]
        answer_md = (
            f"**{len(above_df)} out of {len(df)} segments** are above the chart average of **`{mean_val:,.2f}`** ({val_label}):\n\n"
            + ("\n".join(f"- {it}" for it in items[:10]) if items else "- None of the segments exceed the average.")
        )

    elif "below average" in q_low or "less than average" in q_low or "under average" in q_low:
        below_df = df[series < mean_val].copy()
        working_df = below_df if not below_df.empty else df
        items = [
            f"**{_format_row_label(r)}** (`{float(pd.to_numeric(r[val_col], errors='coerce') or 0.0):,.2f}`)"
            for _, r in below_df.iterrows()
        ]
        answer_md = (
            f"**{len(below_df)} out of {len(df)} segments** are below the chart average of **`{mean_val:,.2f}`** ({val_label}):\n\n"
            + ("\n".join(f"- {it}" for it in items[:10]) if items else "- None of the segments are below the average.")
        )

    elif any(k in q_low for k in ("negative", "declin", "drop", "decreas", "drag", "lost", "worse")):
        target_col = "variance_delta" if "variance_delta" in num_cols else val_col
        t_series = pd.to_numeric(df[target_col], errors="coerce").fillna(0.0)
        neg_df = df[t_series < 0].sort_values(by=target_col, ascending=True)
        if not neg_df.empty:
            working_df = neg_df
            items = [
                f"**{_format_row_label(r)}**: `{float(r[target_col]):+,.2f}`"
                for _, r in neg_df.iterrows()
            ]
            answer_md = (
                f"**{len(neg_df)} segment(s)** showed a negative change / decline (Total drag: **`{float(neg_df[target_col].sum()):+,.2f}`**):\n\n"
                + "\n".join(f"- 🔻 {it}" for it in items[:10])
            )
        else:
            bottom_df = df.assign(_v=series).sort_values("_v", ascending=True).head(3)
            working_df = bottom_df.drop(columns=["_v"])
            items = [f"**{_format_row_label(r)}**: `{float(r['_v']):,.2f}`" for _, r in bottom_df.iterrows()]
            answer_md = (
                f"No segments have negative values, but the **lowest-performing segments** for **{val_label}** are:\n\n"
                + "\n".join(f"- {it}" for it in items)
            )

    elif any(k in q_low for k in ("lowest", "bottom", "minimum", "smallest", "least", "worst", "lowst", "lowes")):
        sorted_asc = df.assign(_v=series).sort_values("_v", ascending=True).reset_index(drop=True)
        working_df = sorted_asc.drop(columns=["_v"]).head(5)
        bot = sorted_asc.iloc[0]
        bot_label = _format_row_label(bot)
        bot_val = float(bot["_v"])
        bot_share = (bot_val / total_val * 100.0) if total_val != 0 else 0.0
        gap_note = ""
        if len(sorted_asc) >= 2:
            sec = sorted_asc.iloc[1]
            sec_label = _format_row_label(sec)
            sec_val = float(sec["_v"])
            gap = bot_val - sec_val
            gap_note = f" (trailing `#{2}` lowest **{sec_label}** at `{sec_val:,.2f}` by `{gap:+,.2f}`)"
        items = [
            f"`#{i+1}` **{_format_row_label(r)}** — `{float(r['_v']):,.2f}`"
            for i, (_, r) in enumerate(sorted_asc.head(5).iterrows())
        ]
        answer_md = (
            f"### 🔻 Lowest {val_label}: **{bot_label}**\n\n"
            f"The lowest {dim_label} by **{val_label}** is **{bot_label}** at **`{bot_val:,.2f}`** "
            f"({bot_share:.1f}% of chart total){gap_note}.\n\n"
            f"**Bottom Segments:**\n" + "\n".join(f"- {it}" for it in items)
        )

    elif any(k in q_low for k in ("highest", "top", "maximum", "largest", "biggest", "most", "best", "leader", "hight", "hights")):
        sorted_desc = df.assign(_v=series).sort_values("_v", ascending=False).reset_index(drop=True)
        working_df = sorted_desc.drop(columns=["_v"]).head(5)
        top = sorted_desc.iloc[0]
        top_label = _format_row_label(top)
        top_val = float(top["_v"])
        top_share = (top_val / total_val * 100.0) if total_val != 0 else 0.0
        gap_note = ""
        if len(sorted_desc) >= 2:
            sec = sorted_desc.iloc[1]
            sec_label = _format_row_label(sec)
            sec_val = float(sec["_v"])
            gap = top_val - sec_val
            gap_note = f", leading `#{2}` **{sec_label}** (`{sec_val:,.2f}`) by **`{gap:+,.2f}`**"
        items = [
            f"`#{i+1}` **{_format_row_label(r)}** — `{float(r['_v']):,.2f}`"
            for i, (_, r) in enumerate(sorted_desc.head(5).iterrows())
        ]
        answer_md = (
            f"### 🥇 Highest {val_label}: **{top_label}**\n\n"
            f"The highest {dim_label} by **{val_label}** is **{top_label}** at **`{top_val:,.2f}`** "
            f"(**{top_share:.1f}%** of chart total){gap_note}.\n\n"
            f"**Top Segments:**\n" + "\n".join(f"- {it}" for it in items)
        )

    else:
        sorted_desc = df.assign(_v=series).sort_values("_v", ascending=False).reset_index(drop=True)
        top_row = sorted_desc.iloc[0]
        bot_row = sorted_desc.iloc[-1]
        top_label = _format_row_label(top_row)
        bot_label = _format_row_label(bot_row)
        top_val = float(top_row["_v"])
        bot_val = float(bot_row["_v"])
        top_share = (top_val / total_val * 100.0) if total_val != 0 else 0.0
        top3_share = (float(sorted_desc.head(3)["_v"].sum()) / total_val * 100.0) if total_val != 0 else 0.0
        spread = top_val - bot_val
        above_count = int((series > mean_val).sum())

        answer_md = (
            f"### 📊 Chart Analysis & Key Takeaways ({val_label} by {dim_label})\n"
            f"- **Aggregate Total:** `{total_val:,.2f}` across `{len(df)}` segments (Mean: `{mean_val:,.2f}` · Median: `{median_val:,.2f}`)\n"
            f"- **Top Performer:** **{top_label}** at **`{top_val:,.2f}`** ({top_share:.1f}% of total)\n"
            f"- **Lowest Segment:** **{bot_label}** at **`{bot_val:,.2f}`** (Spread / Range: `{spread:,.2f}`)\n"
            f"- **Concentration:** Top 3 segments account for **{top3_share:.1f}%** of total volume; **{above_count} of {len(df)}** segments perform above average."
        )

    if llm_router and llm_router.is_configured():
        try:
            sample_records = df.head(20).to_dict(orient="records")
            fallback_sys = (
                "You are a Principal BI Analyst answering a question about a specific chart. "
                "Use ONLY the exact numbers from the provided chart rows and pre-computed summary. "
                "Be concise, direct, and format key numbers in **bold** or `code`. Never invent or hallucinate metrics not present in the data."
            )
            fallback_usr = (
                f"Parent Chart Context: {parent_question}\n"
                f"User Thread Question: {question}\n"
                f"Pre-computed Deterministic Analysis:\n{answer_md}\n"
                f"Chart Rows (JSON):\n{json.dumps(sample_records, default=str)}\n\n"
                "Provide a direct, insightful answer to the User Thread Question:"
            )
            sys_prompt, user_prompt = PromptManager.render(
                "chart_thread_enrichment",
                fallback_system=fallback_sys,
                fallback_user=fallback_usr,
                parent_question=parent_question,
                question=question,
                answer_md=answer_md,
                sample_records_json=json.dumps(sample_records, default=str),
            )
            llm_text = llm_router.complete("chart_thread_enrichment", sys_prompt, user_prompt)
            if llm_text and len(llm_text.strip()) > 20:
                answer_md = llm_text.strip()
        except Exception as exc:
            _LOGGER.debug("Thread LLM enrichment skipped: %s", exc)

    follow_ups = [
        f"Which {dim_label} segments are above average?",
        f"Compare top vs bottom {dim_label}",
        f"What percentage share does the top {dim_label} hold?",
    ]
    final_md = (scope_advisory + answer_md) if scope_advisory and not answer_md.startswith("> ℹ️") else answer_md
    return final_md, working_df, follow_ups


def answer_chart_thread_question(
    question: str,
    parent_question: str,
    parent_sql: str,
    parent_summary: str,
    parent_rows: list[dict[str, Any]],
    parent_columns: list[str],
    thread_history: list[dict[str, Any]],
    response_mode: str,
    workspace: OKFWorkspace,
    okf_store: Any,
    rag_store: Any,
    connector: "BaseConnector",
    llm_router: LLMRouter | None = None,
    what_if_overrides: dict[str, float] | None = None,
    answer_question_fn: Callable[..., tuple[Any, pd.DataFrame]] | None = None,
) -> dict[str, Any]:
    """Hybrid Chart Thread Engine: answers from chart data directly or executes a warehouse SQL drill-down."""
    parent_df = pd.DataFrame(parent_rows) if parent_rows else pd.DataFrame()
    needs_sql = needs_warehouse_sql_for_thread(question, parent_df, parent_question, workspace)
    clean_overrides = normalize_what_if_overrides(what_if_overrides)

    source_mode = "chart_data"
    executed_query = ""
    result_df = parent_df
    answer_text = ""
    follow_ups: list[str] = []
    node_trace: list[str] = []
    node_telemetry: list[dict[str, Any]] = []
    error_diagnostic: dict[str, Any] | None = None

    if needs_sql:
        if answer_question_fn is None:
            from src.agents.query_agent import QueryAgent

            answer_question_fn = QueryAgent.answer_question

        # Forensic Fix #4: Infer involved_tables from parent & prior thread SQL so drill-downs preserve domain scope
        parent_tables = detect_relevant_tables(f"{parent_sql} {parent_question}", workspace, {})
        synthetic_history: list[dict[str, Any]] = []
        if parent_question or parent_sql:
            synthetic_history.append({
                "response": {
                    "user_question": parent_question,
                    "resolved_question": parent_question,
                    "success": True,
                    "executed_query": parent_sql or "SELECT 1",
                    "executive_summary": parent_summary,
                    "involved_tables": parent_tables,
                },
                "rows": parent_rows[:50],
                "columns": parent_columns,
            })
        for turn in (thread_history or []):
            if turn.get("executed_query"):
                turn_tables = turn.get("involved_tables") or detect_relevant_tables(
                    f"{turn.get('executed_query', '')} {turn.get('question', '')}",
                    workspace,
                    {},
                )
                synthetic_history.append({
                    "response": {
                        "user_question": turn.get("question", ""),
                        "resolved_question": turn.get("question", ""),
                        "success": True,
                        "executed_query": turn.get("executed_query", ""),
                        "involved_tables": turn_tables,
                    },
                    "rows": turn.get("rows", [])[:50],
                    "columns": turn.get("columns", []),
                })

        q_resp, sql_df = answer_question_fn(
            question=question.strip(),
            workspace=workspace,
            okf_store=okf_store,
            rag_store=rag_store,
            connector=connector,
            llm_router=llm_router,
            what_if_overrides=clean_overrides or None,
            conversation_history=synthetic_history,
        )
        node_trace = list(getattr(q_resp, "node_trace", []) or [])
        node_telemetry = [
            rec.model_dump() if hasattr(rec, "model_dump") else dict(rec)
            for rec in (getattr(q_resp, "node_telemetry", []) or [])
        ]
        raw_diag = getattr(q_resp, "error_diagnostic", None)
        if raw_diag is not None:
            error_diagnostic = raw_diag.model_dump() if hasattr(raw_diag, "model_dump") else dict(raw_diag)

        if q_resp.success and not sql_df.empty:
            source_mode = "sql_query"
            executed_query = q_resp.executed_query
            result_df = sql_df
            answer_text = q_resp.executive_summary
            follow_ups = q_resp.suggested_follow_ups[:3]
        elif not parent_df.empty:
            answer_text, result_df, follow_ups = analyze_chart_dataframe(
                question, parent_df, parent_question, parent_summary, llm_router
            )
        else:
            answer_text = q_resp.executive_summary or q_resp.error_message or "Could not resolve thread question."
    else:
        answer_text, result_df, follow_ups = analyze_chart_dataframe(
            question, parent_df, parent_question, parent_summary, llm_router
        )

    effective_is_thread = source_mode == "chart_data"
    include_chart = should_render_chart(
        question=question,
        df=result_df,
        response_mode=response_mode,
        is_thread=effective_is_thread,
    )

    chart_type = "none"
    plotly_spec = None
    if include_chart and result_df is not None and not result_df.empty:
        from src.api.utils import fig_to_json
        from src.viz.chart_recommender import ChartRecommender

        plot_df = result_df if len(result_df) >= 2 or parent_df.empty else parent_df
        chart_type = ChartRecommender.recommend_chart_type(plot_df, question)
        if chart_type in ("none", "table", "kpi"):
            chart_type = "bar"
        fig = ChartRecommender.build_figure(plot_df, chart_type=chart_type, title=question)
        plotly_spec = fig_to_json(fig) if fig else None

    rows_out = result_df.head(100).to_dict(orient="records") if (result_df is not None and not result_df.empty) else []
    cols_out = [str(c) for c in result_df.columns] if (result_df is not None and not result_df.empty) else []

    return {
        "question": question.strip(),
        "answer": answer_text,
        "response_mode": response_mode,
        "source_mode": source_mode,
        "include_chart": bool(include_chart and plotly_spec),
        "chart_type": chart_type,
        "plotly_spec": plotly_spec,
        "executed_query": executed_query,
        "rows": rows_out,
        "columns": cols_out,
        "suggested_follow_ups": follow_ups,
        "node_trace": node_trace,
        "node_telemetry": node_telemetry,
        "error_diagnostic": error_diagnostic,
    }
