"""Executive summary narratives, follow-up chips, hierarchy-aware drill-downs, and dataset overview."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from src.agents.llm_router import LLMRouter
from src.agents.prompt_manager import PromptManager
from src.knowledge.okf_models import OKFWorkspace, SemanticMeasure, TableRole
from src.knowledge.okf_store import JoinPlan

_LOGGER = logging.getLogger(__name__)


def build_executive_summary(
    question: str,
    df: pd.DataFrame,
    matched_measures: list[SemanticMeasure],
    applied_rules: list[str],
    llm_router: LLMRouter | None,
    workspace: OKFWorkspace | None = None,
) -> str:
    """Synthesize polarity- and non-additivity-aware narrative insight from a result DataFrame."""
    if df.empty:
        return "Query executed cleanly and returned 0 matching rows."

    num_cols = list(df.select_dtypes(include=["number"]).columns)
    non_num_cols = [c for c in df.columns if c not in num_cols]

    calendar_id_dims = [
        c
        for c in num_cols
        if str(c).lower().endswith(("_year", "_id", "_code", "_tier", "_quarter", "_month", "_day"))
        or str(c).lower()
        in (
            "year", "payment_year", "hire_year", "academic_year", "calendar_year",
            "birth_year", "fiscal_year", "id", "code", "quarter", "month", "day",
        )
    ]
    if calendar_id_dims:
        num_cols = [c for c in num_cols if c not in calendar_id_dims]
        non_num_cols = [c for c in non_num_cols if c not in calendar_id_dims] + calendar_id_dims

    if len(df) == 1 and num_cols:
        parts = [f"**{col.replace('_', ' ').title()}**: `{df[col].iloc[0]:,.2f}`" for col in num_cols[:4]]
        kpi_line = " | ".join(parts)
        insight = ""
        if "gross_margin_pct" in df.columns:
            gm = float(df["gross_margin_pct"].iloc[0])
            if gm >= 40.0:
                insight = f"\n- 💡 **Profitability Health**: Strong gross margin performance at **{gm:.1f}%**."
            elif gm < 20.0:
                insight = f"\n- ⚠️ **Margin Watch**: Gross margin is relatively low at **{gm:.1f}%**."
        return f"**KPI Summary**\n{kpi_line}{insight}"

    if non_num_cols and num_cols:
        dim_col = non_num_cols[0]
        second_dim_col = non_num_cols[1] if len(non_num_cols) >= 2 else None

        q_low = question.lower()
        q_tokens = [w.strip("?,.:;()'\"") for w in q_low.split() if len(w.strip("?,.:;()'\"")) > 2]
        business_metrics = [
            c
            for c in num_cols
            if not any(k in str(c).lower() for k in ("row_count", "total_count", "_count", "count"))
        ]
        count_cols = [c for c in num_cols if c not in business_metrics]
        only_part_to_whole_pct = (
            second_dim_col is not None
            and len(business_metrics) == 1
            and str(business_metrics[0]).lower().endswith(("_ratio_pct", "_share_pct"))
            and len(count_cols) >= 1
        )
        matched_business = [
            c
            for c in business_metrics
            if any(k in str(c).lower() for k in q_tokens)
        ]
        if only_part_to_whole_pct:
            metric_col = count_cols[0]
        elif matched_business:
            metric_col = matched_business[0]
        elif business_metrics and not any(k in q_low for k in ("count", "headcount", "how many", "number of")):
            metric_col = business_metrics[0]
        else:
            matched_any = [c for c in num_cols if any(k in str(c).lower() for k in q_tokens)]
            metric_col = matched_any[0] if matched_any else (business_metrics[0] if business_metrics else num_cols[0])

        sorted_df = df.sort_values(by=metric_col, ascending=False)
        top_row = sorted_df.iloc[0]
        bot_row = sorted_df.iloc[-1]
        top_val = float(top_row[metric_col])
        bot_val = float(bot_row[metric_col])
        mean_val = float(df[metric_col].mean())

        dim_title = dim_col.replace("_", " ").title()
        metric_title = metric_col.replace("_", " ").title()

        top_label = (
            f"**{top_row[dim_col]}** ({top_row[second_dim_col]})"
            if second_dim_col and pd.notna(top_row.get(second_dim_col))
            else f"**{top_row[dim_col]}**"
        )
        bot_label = (
            f"**{bot_row[dim_col]}** ({bot_row[second_dim_col]})"
            if second_dim_col and pd.notna(bot_row.get(second_dim_col))
            else f"**{bot_row[dim_col]}**"
        )

        mc_low = str(metric_col).lower()
        dc_low = str(dim_col).lower()
        is_non_additive = (
            any(mc_low.startswith(p) for p in ("avg_", "average_", "mean_", "median_", "pct_", "rate_"))
            or any(k in mc_low for k in ("_avg", "_mean", "_rate", "_pct", "_percent", "_ratio", "margin", "gpa", "cgpa", "score", "marks"))
            or mc_low in ("salary", "base_salary", "gpa", "cgpa", "rate", "percentage", "ratio", "score", "marks")
        )
        is_time_dim = (
            second_dim_col is None
            and any(k in dc_low for k in ("date", "month", "year", "quarter", "week", "day", "time", "semester"))
        )

        if "total_earnings" in df.columns and "total_expenditure" in df.columns:
            tot_earn = float(df["total_earnings"].sum())
            tot_exp = float(df["total_expenditure"].sum())
            net_diff = tot_earn - tot_exp
            top_earn_row = df.sort_values(by="total_earnings", ascending=False).iloc[0]
            top_exp_row = df.sort_values(by="total_expenditure", ascending=False).iloc[0]
            summary_lines = [
                f"Compared **Earnings vs. Expenditure** across **{len(df):,}** `{dim_title}` groups.",
                f"- 💰 **Highest Earnings**: **{top_earn_row[dim_col]}** at `{float(top_earn_row['total_earnings']):,.2f}` (Total Earnings: `{tot_earn:,.2f}`).",
                f"- 🏛️ **Highest Expenditure**: **{top_exp_row[dim_col]}** at `{float(top_exp_row['total_expenditure']):,.2f}` (Total Expenditure: `{tot_exp:,.2f}`).",
                f"- ⚖️ **Overall Net Position**: `{net_diff:+,.2f}` ({'Net Surplus' if net_diff >= 0 else 'Net Deficit'} across all groups).",
            ]
            return "\n".join(summary_lines)

        if is_time_dim and len(df) >= 2:
            time_df = df.sort_values(by=dim_col, ascending=True)
            start_row = time_df.iloc[0]
            end_row = time_df.iloc[-1]
            start_val = float(start_row[metric_col])
            end_val = float(end_row[metric_col])
            delta_val = end_val - start_val
            pct_change = ((end_val - start_val) / abs(start_val) * 100.0) if start_val else 0.0
            summary_lines = [
                f"Analyzed **{len(df):,}** time periods across `{dim_title}` for `{metric_title}`.",
                f"- 📈 **Trend Trajectory**: Moved from `{start_val:,.2f}` (**{start_row[dim_col]}**) to `{end_val:,.2f}` (**{end_row[dim_col]}**) (**{pct_change:+.1f}%** / `{delta_val:+,.2f}`).",
                f"- 🏔️ **Peak Period**: {top_label} at `{top_val:,.2f}` | 📉 **Lowest Period**: {bot_label} at `{bot_val:,.2f}`.",
            ]
            if is_non_additive:
                summary_lines.append(f"- 📊 **Period Benchmark**: Average across periods is `{mean_val:,.2f}`.")
            else:
                total_val = float(df[metric_col].sum())
                summary_lines.append(
                    f"- 📊 **Period Benchmark**: Average `{mean_val:,.2f}` per period (Cumulative Total: `{total_val:,.2f}`)."
                )
        elif is_non_additive:
            vs_mean = ((top_val - mean_val) / abs(mean_val) * 100.0) if mean_val else 0.0
            spread = top_val - bot_val
            summary_lines = [
                f"Analyzed **{len(df):,}** groups across `{dim_title}` for `{metric_title}`.",
                f"- ⬆️ **Highest Average / Rate**: {top_label} at `{top_val:,.2f}` (**{vs_mean:+.1f}%** vs. benchmark mean of `{mean_val:,.2f}`).",
            ]
            if len(df) > 1 and str(bot_row[dim_col]) != str(top_row[dim_col]):
                summary_lines.append(
                    f"- ⬇️ **Lowest Average / Rate**: {bot_label} at `{bot_val:,.2f}` (Spread: `{spread:,.2f}`)."
                )
        else:
            total_val = float(df[metric_col].sum())
            share = (top_val / total_val * 100.0) if total_val else 0.0
            is_financial_metric = any(
                k in mc_low
                for k in ("revenue", "rev", "sales", "profit", "fee", "tuition", "salary", "amount", "paid", "cost", "budget")
            )
            is_cost_metric = any(
                k in (mc_low + " " + q_low)
                for k in ("cost", "expense", "expenditure", "spend", "budget", "salary", "payroll", "outflow")
            )
            is_demographic_or_neutral = (not is_financial_metric) and any(
                k in (q_low + " " + dc_low + " " + mc_low)
                for k in (
                    "gender", "sex", "age", "race", "ethnicity", "religion",
                    "student", "enrollment", "faculty", "staff", "patient",
                    "ratio", "raio", "demographic", "distribution", "split", "mix", "breakdown", "count",
                )
            )
            if is_cost_metric:
                top_badge = "💰 **Highest Cost / Expenditure**"
                bot_badge = "📉 **Lowest Cost / Expenditure**"
            elif is_demographic_or_neutral:
                top_badge = "📊 **Highest Representation / Volume**"
                bot_badge = "📉 **Lowest Representation**"
            else:
                top_badge = "🏆 **Top Performer**"
                bot_badge = "📉 **Lowest Group**"

            summary_lines = [
                f"Analyzed **{len(df):,}** groups across `{dim_title}` for `{metric_title}`.",
                f"- {top_badge}: {top_label} with `{top_val:,.2f}` (**{share:.1f}%** of total `{total_val:,.2f}`).",
            ]
            if len(df) > 1 and str(bot_row[dim_col]) != str(top_row[dim_col]):
                summary_lines.append(
                    f"- {bot_badge}: {bot_label} at `{bot_val:,.2f}`."
                )

            if len(df) >= 3:
                top_2_sum = float(sorted_df.head(2)[metric_col].sum())
                top_2_share = (top_2_sum / (total_val or 1.0)) * 100.0
                if top_2_share >= 60.0:
                    summary_lines.append(
                        f"- ⚖️ **Concentration Alert**: Top 2 segments represent **{top_2_share:.1f}%** of aggregate volume."
                    )

        notice = ""
        if "gender" in q_low and not any("gender" in c.lower() for c in df.columns):
            notice = "ℹ️ **Domain Notice**: Gender is not recorded in this table. Available classifications and distributions are shown instead.\n\n"

        return notice + "\n".join(summary_lines)

    return f"Returned **{len(df):,}** rows across {len(df.columns)} columns."


def generate_drill_down_options(
    df: pd.DataFrame,
    workspace: OKFWorkspace,
    involved_tables: list[str] | None = None,
) -> list[dict[str, str]]:
    """Identify hierarchy-aware and schema-driven dimensions to drill into from the current result."""
    if df.empty:
        return []

    num_cols = list(df.select_dtypes(include=["number"]).columns)
    calendar_id_dims = [
        c
        for c in num_cols
        if str(c).lower().endswith(("_year", "_id", "_code", "_tier", "_quarter", "_month", "_day"))
        or str(c).lower()
        in (
            "year", "payment_year", "hire_year", "academic_year", "calendar_year",
            "birth_year", "fiscal_year", "id", "code", "quarter", "month", "day",
        )
    ]
    existing_dims = [str(c).lower() for c in df.columns if c not in num_cols or c in calendar_id_dims]
    current_dim = existing_dims[0] if existing_dims else ""
    inv_set = set(involved_tables or [])

    blocked_parents: set[str] = set()
    if "department" in current_dim:
        blocked_parents.add("college_name")
    elif "program" in current_dim:
        blocked_parents.update({"college_name", "department_name"})
    elif "course" in current_dim:
        blocked_parents.update({"college_name", "department_name", "program_name"})

    is_faculty_context = "faculty" in inv_set and "students" not in inv_set and "enrollments_marks" not in inv_set
    is_fee_context = "fee_invoices" in inv_set
    is_marks_context = "enrollments_marks" in inv_set or "courses" in inv_set

    candidates: list[tuple[str, str, str]] = [
        ("category", "Product Category", "products"),
        ("region", "Geographic Region", "customers"),
        ("customer_tier", "Customer Tier", "customers"),
        ("sales_channel", "Sales Channel", "orders"),
        ("order_month", "Monthly Trend", "orders"),
        ("department_name", "Department", "departments"),
        ("program_name", "Degree Program", "degree_programs"),
    ]
    if is_faculty_context:
        candidates.extend([
            ("academic_rank", "Academic Rank", "faculty"),
            ("employment_status", "Employment Status", "faculty"),
            ("college_name", "College", "colleges"),
        ])
    elif is_fee_context:
        candidates.extend([
            ("payment_status", "Payment Status", "fee_invoices"),
            ("payment_mode", "Payment Mode", "fee_invoices"),
            ("gender", "Gender", "students"),
            ("degree_level", "Degree Level", "degree_programs"),
            ("college_name", "College", "colleges"),
        ])
    elif is_marks_context:
        candidates.extend([
            ("course_title", "Course", "courses"),
            ("gender", "Gender", "students"),
            ("degree_level", "Degree Level", "degree_programs"),
            ("college_name", "College", "colleges"),
        ])
    else:
        candidates.extend([
            ("gender", "Gender", "students"),
            ("degree_level", "Degree Level", "degree_programs"),
            ("academic_rank", "Academic Rank", "faculty"),
            ("course_title", "Course", "courses"),
            ("payment_status", "Payment Status", "fee_invoices"),
            ("college_name", "College", "colleges"),
        ])

    options: list[dict[str, str]] = []
    seen_dims: set[str] = set()

    for dim_key, label, tbl_name in candidates:
        if dim_key in blocked_parents:
            continue
        if any(dim_key in ed or ed in dim_key for ed in existing_dims):
            continue
        if dim_key in seen_dims:
            continue
        if workspace.get_table(tbl_name):
            seen_dims.add(dim_key)
            options.append({"dimension": dim_key, "label": label, "table": tbl_name})

    if len(options) < 5:
        skip_dyn = {
            "first_name", "last_name", "email", "phone", "phone_number",
            "roll_number", "date_of_birth", "address", "remarks", "syllabus_summary",
        }
        for tbl in workspace.active_tables():
            for col in tbl.columns:
                c_low = col.name.lower()
                if col.is_blocked or col.is_pii or col.is_primary_key or col.is_foreign_key:
                    continue
                if c_low in skip_dyn or c_low in seen_dims or c_low in blocked_parents or c_low.endswith(("_id", "_code")):
                    continue
                if any(c_low in ed or ed in c_low for ed in existing_dims):
                    continue
                if any(ty in col.data_type.upper() for ty in ("CHAR", "TEXT", "VARCHAR", "STRING")) and 2 <= col.distinct_count <= 50:
                    seen_dims.add(c_low)
                    pretty_lbl = col.name.replace("_name", "").replace("_", " ").title()
                    options.append({"dimension": col.name, "label": pretty_lbl, "table": tbl.name})
                    if len(options) >= 5:
                        break
            if len(options) >= 5:
                break

    return options[:5]


def generate_suggested_follow_ups(
    df: pd.DataFrame,
    question: str,
    workspace: OKFWorkspace,
    join_plan: JoinPlan,
) -> list[str]:
    """Proactively suggests the 3 best follow-up questions for the current result."""
    if df.empty:
        return [
            "Show dataset overview and active tables",
            "What are the top KPIs in this dataset?",
            "Show sample records from active tables",
        ]

    num_cols = list(df.select_dtypes(include=["number"]).columns)
    non_num_cols = [c for c in df.columns if c not in num_cols]
    calendar_id_dims = [
        c
        for c in num_cols
        if str(c).lower().endswith(("_year", "_id", "_code", "_tier", "_quarter", "_month", "_day"))
        or str(c).lower()
        in (
            "year", "payment_year", "hire_year", "academic_year", "calendar_year",
            "birth_year", "fiscal_year", "id", "code", "quarter", "month", "day",
        )
    ]
    if calendar_id_dims:
        num_cols = [c for c in num_cols if c not in calendar_id_dims]
        non_num_cols = [c for c in non_num_cols if c not in calendar_id_dims] + calendar_id_dims

    if len(df) == 1:
        if workspace.get_table("products"):
            return [
                "Break down revenue by product category",
                "Show monthly revenue trend over time",
                "Compare performance across geographic regions",
            ]
        if workspace.get_table("colleges") or workspace.get_table("students"):
            return [
                "Break down student count by college",
                "Show gender ratio across departments",
                "Compare average marks by college",
            ]
        return [
            "Break this down by primary category",
            "Show trend over time",
            "View top 5 contributors",
        ]

    if non_num_cols and num_cols:
        dim_col = non_num_cols[0]
        business_metrics = [
            c
            for c in num_cols
            if not any(k in str(c).lower() for k in ("row_count", "total_count", "_count", "count"))
        ]
        sort_col = business_metrics[0] if business_metrics else num_cols[0]
        sorted_df = df.sort_values(by=sort_col, ascending=False)
        top_seg = str(sorted_df.iloc[0][dim_col])
        drill_opts = generate_drill_down_options(df, workspace, join_plan.required_tables)
        first_drill_lbl = drill_opts[0]["label"].lower() if drill_opts else "department"

        if any(k in dim_col.lower() for k in ("category", "product", "sku")):
            return [
                f"Drill down into {top_seg} by region",
                f"Show monthly trend for {top_seg}",
                "Why did revenue change? Period-over-period variance",
            ]
        if any(k in dim_col.lower() for k in ("region", "country", "geo")):
            breakdown_target = drill_opts[0]["label"] if drill_opts else ("product category" if workspace.get_table("products") else "category")
            return [
                f"Break down {top_seg} by {breakdown_target}",
                f"Compare customer tiers in {top_seg}",
                "Show monthly trend for top region",
            ]
        if any(k in dim_col.lower() for k in ("month", "date", "year")):
            breakdown_target = drill_opts[0]["label"] if drill_opts else (
                "College" if workspace.get_table("colleges") else (
                    "Department" if workspace.get_table("departments") else (
                        "product category" if workspace.get_table("products") else "category"
                    )
                )
            )
            return [
                "Why did it change? Period-over-period variance analysis",
                f"Break down this trend by {breakdown_target}",
                "Show top contributing drivers for peak period",
            ]
        return [
            f"Drill down into {top_seg} by {first_drill_lbl}",
            f"Break down by {first_drill_lbl}",
            f"Show trend over time for {top_seg}",
        ]

    return [
        "Break down by category",
        "Show monthly trend",
        "Filter to top 5",
    ]


def is_dataset_overview_question(question: str) -> bool:
    """Detect whether the user is asking for an exploratory narrative explanation of the dataset."""
    import re

    q = question.lower().strip()
    overview_patterns = [
        r"\btell\s+me\s+about\s+(this|the)\s+(data|data\s*set|dataset|database|schema)\b",
        r"\bwhat\s+(is|does)\s+(this|the)\s+(data|data\s*set|dataset|database|entire\s+dataset)\s+(about|show|contain|represent)\b",
        r"\bexplain\s+(to\s+me\s+in\s+words\s+)?what\s+(this|the)\s+(data|data\s*set|dataset|entire\s+dataset)\s+(is\s+about|shows|means)\b",
        r"\b(overview|summary|describe|description)\s+of\s+(this|the)\s+(data|data\s*set|dataset|database|schema)\b",
        r"\bwhat\s+is\s+this\s+(data|data\s*set|dataset|database)\s+about\b",
        r"\bexplain\s+(this|the)\s+(data|data\s*set|dataset|database)\b",
        r"\bwhat\s+does\s+this\s+(data|data\s*set|dataset|database)\s+(represent|show|mean)\b",
        r"\bwhat\s+tables?\s+(are|do\s+we\s+have|exist)\b",
        r"\bexplain\s+to\s+me\s+in\s+words\b",
    ]
    for pat in overview_patterns:
        if re.search(pat, q):
            return True
    phrases = (
        "tell me about this data",
        "tell me about this dataset",
        "about this data set",
        "about this dataset",
        "what this entire dataset is about",
        "what this dataset is about",
        "what this data is about",
        "overview of the data",
        "overview of this data",
        "explain to me in words",
    )
    return any(p in q for p in phrases)


def generate_dataset_overview_payload(
    question: str,
    workspace: OKFWorkspace,
    llm_router: LLMRouter | None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Generate a structured, in-words semantic explanation dict and overview DataFrame."""
    active_tables = workspace.active_tables()
    analytical_tables = [
        t
        for t in active_tables
        if sum(1 for c in t.columns if c.name.lower().startswith("unnamed_")) < max(1, len(t.columns) * 0.5)
    ]
    connected_names = {
        r.source_table for r in (workspace.relationships or []) if r.status != "rejected"
    } | {
        r.target_table for r in (workspace.relationships or []) if r.status != "rejected"
    }
    connected_analytical = [t for t in analytical_tables if t.name in connected_names]
    if len(connected_analytical) >= 4:
        display_tables = connected_analytical
    else:
        display_tables = analytical_tables if analytical_tables else active_tables

    total_rows = sum(t.row_count or 0 for t in display_tables)
    total_cols = sum(len(t.columns) for t in display_tables)

    all_table_names = [t.name.lower() for t in display_tables]
    suggested_clean: list[str] = []
    if any("college" in n for n in all_table_names):
        suggested_clean = [
            "Show average marks for each college",
            "Show count of students by degree program",
            "Show total fee due by college",
            "Show faculty salary trend year on year",
        ]
    elif any("order" in n for n in all_table_names):
        suggested_clean = [
            "Show net revenue, profit, and gross margin % by product category",
            "Monthly net revenue trend over time",
            "Compare net revenue and returns by customer tier",
        ]
    else:
        suggested_clean = ["Show summary statistics and top records"]

    if llm_router is not None and llm_router.is_configured():
        try:
            table_summaries = []
            for t in display_tables:
                col_names = [c.name for c in t.columns if not c.is_blocked]
                table_summaries.append(
                    f"- `{t.name}` ({t.table_role.value}, {t.row_count:,} rows): {', '.join(col_names[:10])}"
                )
            rels_summary = [
                f"- `{r.source_table}.{r.source_column}` -> `{r.target_table}.{r.target_column}`"
                for r in (workspace.relationships or [])
                if r.status != "rejected"
            ]

            fallback_sys = (
                "You are Data_Talk's Senior Business Intelligence Data Architect. "
                "The user is asking for an executive, in-words explanation of what this dataset is about and what it shows.\n"
                "Provide a clear, well-structured executive briefing in clean GitHub markdown:\n"
                "1. High-level Domain & Core Purpose (2-3 sentences explaining what real-world domain this represents)\n"
                "2. Key Entities & Master Tables (what core business entities are tracked)\n"
                "3. Core Operational Facts & Performance Metrics (what transactions, scores, or volume can be measured)\n"
                "4. Connected Relationship Paths (how key tables join together)\n"
                "5. 4 High-Value Starter Questions they can ask right now"
            )
            fallback_usr = (
                f"Workspace: {workspace.workspace_name}\n"
                f"User Question: {question}\n\n"
                f"Tables ({len(display_tables)} total, {total_rows:,} records):\n" + "\n".join(table_summaries) + "\n\n"
                f"Approved Relationships:\n" + ("\n".join(rels_summary) or "None")
            )
            sys_prompt, usr_prompt = PromptManager.render(
                "dataset_overview",
                fallback_system=fallback_sys,
                fallback_user=fallback_usr,
                workspace_name=workspace.workspace_name,
                question=question,
                total_tables=len(display_tables),
                total_rows=f"{total_rows:,}",
                table_summaries="\n".join(table_summaries),
                rels_summary="\n".join(rels_summary) or "None",
            )
            res = llm_router.complete_text("dataset_overview", sys_prompt, usr_prompt)
            if res and len(res.strip()) > 50:
                df_overview = pd.DataFrame([
                    {
                        "table_name": t.name,
                        "table_role": t.table_role.value,
                        "row_count": t.row_count,
                        "column_count": len(t.columns),
                        "key_columns": ", ".join(c.name for c in t.columns[:4] if not c.is_blocked),
                    }
                    for t in display_tables
                ])
                model_lbl = getattr(llm_router, "model", "LLM")
                return (
                    {
                        "user_question": question,
                        "success": True,
                        "executed_query": "/* Semantic Schema Catalog */ SHOW TABLES;",
                        "compiled_dax_equivalent": "// Overview of Semantic Model",
                        "involved_tables": [t.name for t in display_tables[:6]],
                        "executive_summary": res.strip(),
                        "recommended_chart_type": "table",
                        "compiler_mode": "llm",
                        "compiler_detail": f"LLM ({model_lbl})",
                        "suggested_follow_ups": suggested_clean[:4],
                    },
                    df_overview,
                )
        except Exception as exc:
            _LOGGER.warning("LLM dataset overview generation failed: %s", exc)

    facts = [t for t in display_tables if t.table_role == TableRole.FACT]
    dims = [t for t in display_tables if t.table_role != TableRole.FACT]

    domain_name = "Enterprise Operations & Analytics"
    domain_desc = "Tracks organizational master records, operational transactions, and business metrics."
    if any("student" in n or "college" in n or "enroll" in n for n in all_table_names):
        domain_name = "University Administration & Academic Performance"
        domain_desc = (
            "Tracks complete higher education lifecycle operations: student admissions, demographic records, "
            "college and department hierarchies, faculty assignments, course curriculum sections, "
            "student grade assessments, and tuition fee billings."
        )
    elif any("order" in n or "sales" in n or "product" in n for n in all_table_names):
        domain_name = "Retail, E-Commerce & Commercial Sales"
        domain_desc = "Tracks retail order processing, product catalogs, customer segments, sales revenue, and merchandise returns."
    elif any("patient" in n or "clinic" in n or "health" in n for n in all_table_names):
        domain_name = "Healthcare & Clinical Operations"
        domain_desc = "Tracks patient encounters, clinical appointments, diagnoses, and medical billing."

    dim_lines = [
        f"- **`{t.name}`** ({t.row_count:,} records): Master entity catalog with {len(t.columns)} attributes"
        for t in dims[:8]
    ]
    fact_lines = [
        f"- **`{t.name}`** ({t.row_count:,} records): Operational facts measuring {', '.join(c.name for c in t.columns if not c.is_blocked and not c.is_primary_key)[:65]}..."
        for t in facts[:6]
    ]

    rel_samples = [
        f"`{r.source_table}` -> `{r.target_table}`"
        for r in (workspace.relationships or [])[:6]
        if r.status != "rejected"
    ]
    rel_text = ", ".join(rel_samples) if rel_samples else "Self-contained analytical entities"

    overview_md = (
        f"### 🌐 Domain Overview: **{domain_name}**\n\n"
        f"{domain_desc}\n\n"
        f"The database consists of **{len(display_tables)} interconnected tables** comprising **{total_rows:,} total records** and **{total_cols} attributes**.\n\n"
        f"#### 🏛️ Master Entities & Dimensions:\n"
        + ("\n".join(dim_lines) if dim_lines else "- Master entity records across operational areas.")
        + "\n\n"
        f"#### 📊 Core Operational Facts & Metrics:\n"
        + ("\n".join(fact_lines) if fact_lines else "- Transactional activity and performance metrics.")
        + "\n\n"
        f"#### 🔗 Connected Relationship Paths:\n"
        f"Entities are linked across the join graph: {rel_text}.\n\n"
        f"#### 💡 Example Business Questions to Ask:\n"
        + "\n".join(f'- "{s}"' for s in suggested_clean[:4])
    )

    df_overview = pd.DataFrame([
        {
            "table_name": t.name,
            "table_role": t.table_role.value,
            "row_count": t.row_count,
            "column_count": len(t.columns),
            "key_columns": ", ".join(c.name for c in t.columns[:4] if not c.is_blocked),
        }
        for t in display_tables
    ])

    return (
        {
            "user_question": question,
            "success": True,
            "executed_query": "/* Semantic Schema Catalog */ SHOW TABLES;",
            "compiled_dax_equivalent": "// Overview of Semantic Model",
            "involved_tables": [t.name for t in display_tables[:6]],
            "executive_summary": overview_md,
            "recommended_chart_type": "table",
            "suggested_follow_ups": suggested_clean[:4],
        },
        df_overview,
    )
