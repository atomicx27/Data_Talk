"""Deterministic offline SQL & companion DAX compiler for Data_Talk."""

from __future__ import annotations

import re
from typing import Any

from src.agents.compilers.text_matching import (
    detect_categorical_value_matches,
    fuzzy_matches_word,
    normalize_what_if_overrides,
)
from src.agents.measure_compiler import MeasureCompiler
from src.knowledge.okf_models import OKFWorkspace, SemanticMeasure
from src.knowledge.okf_store import JoinPlan


def clean_agg_alias(col_name: str, agg_fn: str) -> str:
    """Format a clean SQL output alias respecting AVG/SUM prefixes without double-prefixing."""
    clean = re.sub(r"[^a-z0-9_]+", "_", col_name.lower()).strip("_")
    prefix = agg_fn.lower()
    if prefix in ("sum", "total"):
        if clean.startswith("total_") or clean.startswith("sum_"):
            return clean
        return f"total_{clean}"
    elif prefix in ("avg", "average", "mean"):
        if clean.startswith("avg_"):
            return clean
        if clean.startswith("total_"):
            return clean.replace("total_", "avg_", 1)
        if clean.startswith("sum_"):
            return clean.replace("sum_", "avg_", 1)
        return f"avg_{clean}"
    return f"{prefix}_{clean}"


def infer_agg_function(q_low: str, metric_col: str | None) -> str:
    """Choose AVG vs. SUM based on non-additive metric names and question intent."""
    mc = (metric_col or "").lower()
    if any(re.search(rf"\b{w}\b", q_low) for w in ("avg", "average", "mean")):
        return "AVG"
    if any(re.search(rf"\b{w}\b", q_low) for w in ("total", "sum", "payroll", "overall")) and not any(
        k in mc for k in ("gpa", "cgpa", "rate", "pct", "percent", "ratio", "score", "marks")
    ):
        return "SUM"
    if any(
        k in mc
        for k in (
            "salary", "compensation", "wage", "gpa", "cgpa", "rate",
            "pct", "percent", "ratio", "score", "marks", "age", "rating",
        )
    ):
        return "AVG"
    if any(w in q_low for w in ("salary", "salaries", "compensation", "rate", "gpa", "score")):
        return "AVG"
    return "SUM"


def infer_sort_direction(q_low: str) -> str:
    """Return ASC for lowest/bottom queries, otherwise DESC."""
    if any(re.search(rf"\b{w}\b", q_low) for w in ("bottom", "lowest", "worst", "least", "smallest", "minimum")):
        return "ASC"
    return "DESC"


def compile_deterministic_sql(
    question: str,
    workspace: OKFWorkspace,
    join_plan: JoinPlan,
    matched_measures: list[SemanticMeasure],
    what_if_overrides: dict[str, float] | None = None,
) -> str:
    """Built-in semantic SQL compiler for multi-table joins, fan-trap CTEs, time series, and rankings."""
    q_low = question.lower()
    overrides = normalize_what_if_overrides(what_if_overrides)

    filter_hints = [r.sql_filter_hint for r in workspace.business_rules if r.sql_filter_hint]
    orders_filter = next((f for f in filter_hints if "orders." in f), "")

    rev_mult = 1.0 + float(overrides.get("Revenue Multiplier %", 0.0))
    cost_mult = 1.0 + float(overrides.get("Cost Multiplier %", 0.0))

    group_col = None
    group_alias = None
    need_customers = False
    need_products = False

    by_match = re.search(r"\bby\s+([a-z0-9_]+)", q_low)
    by_dim = by_match.group(1).lower() if by_match else ""

    if ("region" in by_dim or "country" in by_dim) and workspace.get_table("customers"):
        group_col = "customers.region_code"
        group_alias = "region"
        need_customers = True
    elif ("tier" in by_dim or "segment" in by_dim) and workspace.get_table("customers"):
        group_col = "customers.cust_tier"
        group_alias = "customer_tier"
        need_customers = True
    elif ("category" in by_dim or "categories" in by_dim) and workspace.get_table("products"):
        group_col = "products.category"
        group_alias = "category"
        need_products = True
    elif "month" in by_dim or "trend" in by_dim or "timeline" in by_dim:
        group_col = "SUBSTR(CAST(orders.order_date AS VARCHAR), 1, 7)"
        group_alias = "order_month"
    elif any(w in q_low for w in ("month", "monthly", "trend", "over time", "timeline")):
        group_col = "SUBSTR(CAST(orders.order_date AS VARCHAR), 1, 7)"
        group_alias = "order_month"
    elif any(w in q_low for w in ("category", "categories")) and workspace.get_table("products"):
        group_col = "products.category"
        group_alias = "category"
        need_products = True
    elif any(w in q_low for w in ("product", "sku")) and workspace.get_table("products"):
        group_col = "products.sku_name"
        group_alias = "product"
        need_products = True
    elif any(w in q_low for w in ("region", "country", "geography")) and workspace.get_table("customers"):
        group_col = "customers.region_code"
        group_alias = "region"
        need_customers = True
    elif any(w in q_low for w in ("tier", "segment", "enterprise", "smb")) and workspace.get_table("customers"):
        group_col = "customers.cust_tier"
        group_alias = "customer_tier"
        need_customers = True
    elif "channel" in q_low and workspace.get_table("orders"):
        group_col = "orders.sales_channel"
        group_alias = "sales_channel"
    elif "status" in q_low or "ord_stat" in q_low:
        group_col = "orders.ord_stat"
        group_alias = "order_status"
    elif "reason" in q_low and workspace.get_table("returns"):
        return (
            "SELECT returns.ret_reason_cd AS return_reason, "
            "COUNT(*) AS return_count, "
            "ROUND(SUM(returns.refund_amt_usd), 2) AS total_refund_usd "
            "FROM returns GROUP BY returns.ret_reason_cd ORDER BY total_refund_usd DESC"
        )

    # Case A: Multi-Fact Fan/Chasm Trap Query (Orders + Returns together)
    if (
        workspace.get_table("orders")
        and workspace.get_table("returns")
        and (
            (join_plan.has_fan_trap and "orders" in join_plan.required_tables and "returns" in join_plan.required_tables)
            or (
                ("return" in q_low or "refund" in q_low)
                and any(w in q_low for w in ("revenue", "sales", "order", "rate", "net"))
            )
        )
    ):
        blocked_cols_low = {c.lower() for c in workspace.blocked_column_names()}
        order_id_blocked = "order_id" in blocked_cols_low or "orders.order_id" in blocked_cols_low
        return_id_blocked = "return_id" in blocked_cols_low or "returns.return_id" in blocked_cols_low
        order_cnt_cte = "COUNT(*)" if order_id_blocked else "COUNT(DISTINCT order_id)"
        return_cnt_cte = "COUNT(*)" if return_id_blocked else "COUNT(DISTINCT return_id)"
        dim_select = "products.category" if need_products or not need_customers else "customers.region_code"
        dim_alias = "category" if "products" in dim_select else "region"
        dim_table = "products" if "products" in dim_select else "customers"
        dim_key = "product_id" if dim_table == "products" else "customer_id"
        where_clause = f"WHERE {orders_filter}" if orders_filter else ""
        return (
            f"WITH sales_cte AS (\n"
            f"    SELECT {dim_key}, SUM(rev_net_usd * {rev_mult:.4f}) AS net_revenue_usd, {order_cnt_cte} AS total_orders\n"
            f"    FROM orders {where_clause}\n"
            f"    GROUP BY {dim_key}\n"
            f"),\n"
            f"returns_cte AS (\n"
            f"    SELECT {dim_key}, SUM(refund_amt_usd) AS refund_usd, {return_cnt_cte} AS return_count\n"
            f"    FROM returns\n"
            f"    GROUP BY {dim_key}\n"
            f")\n"
            f"SELECT d.{dim_select.split('.')[1]} AS {dim_alias},\n"
            f"       ROUND(SUM(COALESCE(s.net_revenue_usd, 0)), 2) AS net_revenue_usd,\n"
            f"       ROUND(SUM(COALESCE(r.refund_usd, 0)), 2) AS refund_usd,\n"
            f"       ROUND(SUM(COALESCE(s.net_revenue_usd, 0)) - SUM(COALESCE(r.refund_usd, 0)), 2) AS net_retained_revenue_usd,\n"
            f"       ROUND(100.0 * SUM(COALESCE(r.refund_usd, 0)) / NULLIF(SUM(COALESCE(s.net_revenue_usd, 0)), 0), 2) AS return_rate_pct\n"
            f"FROM {dim_table} d\n"
            f"LEFT JOIN sales_cte s ON d.{dim_key} = s.{dim_key}\n"
            f"LEFT JOIN returns_cte r ON d.{dim_key} = r.{dim_key}\n"
            f"GROUP BY d.{dim_select.split('.')[1]}\n"
            f"ORDER BY net_revenue_usd DESC"
        )

    # Case B: Standard Enterprise Demo Orders + Dimensions Query
    orders_demo = workspace.get_table("orders")
    if (
        orders_demo
        and orders_demo.get_column("rev_net_usd")
        and orders_demo.get_column("cogs_usd")
        and ("orders" in join_plan.required_tables or not join_plan.required_tables)
    ):
        extra_filters: list[str] = []
        if orders_filter and "ord_stat" not in (group_col or ""):
            extra_filters.append(orders_filter)

        if group_alias != "region" and workspace.get_table("customers"):
            if any(r in q_low for r in ("region na", "region us", "in us", "to us", "for us", "north america")):
                need_customers = True
                extra_filters.append("(customers.region_code = 'NA' OR customers.region_code = 'US')")
            elif any(r in q_low for r in ("region emea", "in emea", "to emea", "for emea", "europe")):
                need_customers = True
                extra_filters.append("customers.region_code = 'EMEA'")
            elif any(r in q_low for r in ("region apac", "in apac", "to apac", "for apac", "asia")):
                need_customers = True
                extra_filters.append("customers.region_code = 'APAC'")
            elif any(r in q_low for r in ("region latam", "in latam", "to latam", "for latam")):
                need_customers = True
                extra_filters.append("customers.region_code = 'LATAM'")

        if group_alias != "customer_tier" and workspace.get_table("customers"):
            if any(t in q_low for t in ("enterprise", "tier ent", "ent tier")):
                need_customers = True
                extra_filters.append("customers.cust_tier = 'ENT'")
            elif any(t in q_low for t in ("mid-market", "mid market", "tier mid")):
                need_customers = True
                extra_filters.append("customers.cust_tier = 'MID'")
            elif any(t in q_low for t in ("smb", "small business", "tier smb")):
                need_customers = True
                extra_filters.append("customers.cust_tier = 'SMB'")

        if group_alias != "category" and workspace.get_table("products"):
            for cat in ("cloud", "security", "analytics", "hardware", "technology", "edge", "furniture", "office supplies"):
                if cat in q_low:
                    need_products = True
                    extra_filters.append(f"LOWER(products.category) LIKE '%{cat}%'")
                    break

        joins: list[str] = []
        if need_customers:
            joins.append("LEFT JOIN customers ON orders.customer_id = customers.customer_id")
        if need_products:
            joins.append("LEFT JOIN products ON orders.product_id = products.product_id")
        join_str = ("\n" + "\n".join(joins)) if joins else ""
        where_str = ("\nWHERE " + " AND ".join(extra_filters)) if extra_filters else ""

        rev_expr = f"SUM(orders.rev_net_usd * {rev_mult:.4f})"
        profit_expr = f"SUM((orders.rev_net_usd * {rev_mult:.4f}) - (orders.cogs_usd * {cost_mult:.4f}))"
        margin_expr = f"ROUND(100.0 * ({profit_expr}) / NULLIF(({rev_expr}), 0), 2)"

        extra_measure_select = ""
        for m in matched_measures:
            if m.sql_expression and "returns." not in m.sql_expression:
                safe_alias = re.sub(r"[^a-z0-9_]+", "_", m.name.lower()).strip("_")
                extra_measure_select = f",\n       ROUND({m.sql_expression}, 2) AS {safe_alias}"
                break

        top_match = re.search(r"\b(?:top|bottom|limit)\s+(\d+)\b", q_low)
        limit_clause = f"\nLIMIT {int(top_match.group(1))}" if top_match else ""
        sort_dir = infer_sort_direction(q_low)

        blocked_cols_low = {c.lower() for c in workspace.blocked_column_names()}
        order_id_blocked = "order_id" in blocked_cols_low or "orders.order_id" in blocked_cols_low
        order_cnt_expr = "COUNT(*)" if order_id_blocked else "COUNT(DISTINCT orders.order_id)"

        if group_col and group_alias:
            if group_alias == "order_month":
                extra_filters_month = list(extra_filters) + ["orders.order_date IS NOT NULL"]
                where_str = "\nWHERE " + " AND ".join(extra_filters_month)
            order_by = f"{group_alias} ASC" if group_alias == "order_month" else f"net_revenue_usd {sort_dir}"
            return (
                f"SELECT {group_col} AS {group_alias},\n"
                f"       ROUND({rev_expr}, 2) AS net_revenue_usd,\n"
                f"       ROUND({profit_expr}, 2) AS profit_usd,\n"
                f"       {margin_expr} AS gross_margin_pct,\n"
                f"       {order_cnt_expr} AS order_count"
                f"{extra_measure_select}\n"
                f"FROM orders{join_str}{where_str}\n"
                f"GROUP BY {group_col}\n"
                f"ORDER BY {order_by}{limit_clause}"
            )

        return (
            f"SELECT ROUND({rev_expr}, 2) AS net_revenue_usd,\n"
            f"       ROUND({profit_expr}, 2) AS total_profit_usd,\n"
            f"       {margin_expr} AS gross_margin_pct,\n"
            f"       {order_cnt_expr} AS total_orders"
            f"{extra_measure_select}\n"
            f"FROM orders{join_str}{where_str}"
        )

    # Case B2: Multi-Fact Expenditure vs. Earnings Comparison across Colleges / Departments
    has_expenditure_intent = any(
        w in q_low
        for w in ("expenditure", "expenditures", "expense", "expenses", "cost", "costs", "budget", "spend", "spending", "outflow")
    )
    has_earning_intent = any(
        w in q_low
        for w in ("earning", "earnings", "revenue", "revenues", "income", "fee", "fees", "tuition", "collection", "collections", "receipts")
    )
    if (
        has_expenditure_intent
        and has_earning_intent
        and workspace.get_table("colleges")
        and workspace.get_table("departments")
        and workspace.get_table("fee_invoices")
        and workspace.get_table("students")
    ):
        join_plan.has_fan_trap = True
        if isinstance(join_plan, dict):
            join_plan["has_fan_trap"] = True
        has_faculty = workspace.get_table("faculty") is not None
        cat_matches_b2 = detect_categorical_value_matches(question, workspace)
        college_filter_val = next(
            (m["value"] for m in cat_matches_b2 if m["table"] == "colleges" and m["column"] == "college_name"),
            None,
        )
        dept_filter_val = next(
            (m["value"] for m in cat_matches_b2 if m["table"] == "departments" and m["column"] == "department_name"),
            None,
        )
        wants_dept_grain = (
            any(w in q_low for w in ("department", "departments", "dept"))
            or bool(college_filter_val)
            or bool(dept_filter_val)
        )
        if wants_dept_grain and workspace.get_table("degree_programs"):
            b2_where_parts: list[str] = []
            if college_filter_val:
                safe_c = college_filter_val.replace("'", "''")
                b2_where_parts.append(f'c."college_name" = \'{safe_c}\'')
            if dept_filter_val:
                safe_d = dept_filter_val.replace("'", "''")
                b2_where_parts.append(f'd."department_name" = \'{safe_d}\'')
            b2_where_sql = ("\nWHERE " + " AND ".join(b2_where_parts)) if b2_where_parts else ""
            fac_cte_dept = (
                ",\nfaculty_payroll_cte AS (\n"
                '    SELECT "department_id", SUM("salary") AS faculty_payroll\n'
                '    FROM "faculty"\n'
                '    GROUP BY "department_id"\n'
                ")"
                if has_faculty
                else ""
            )
            fac_join_dept = 'LEFT JOIN faculty_payroll_cte fp ON d."department_id" = fp."department_id"\n' if has_faculty else ""
            exp_expr = "COALESCE(db.department_budget, 0) + COALESCE(fp.faculty_payroll, 0)" if has_faculty else "COALESCE(db.department_budget, 0)"
            fac_col = ",\n       ROUND(COALESCE(fp.faculty_payroll, 0), 2) AS faculty_payroll" if has_faculty else ""
            return (
                "WITH dept_budget_cte AS (\n"
                '    SELECT "department_id", SUM("budget_allocated") AS department_budget\n'
                '    FROM "departments"\n'
                '    GROUP BY "department_id"\n'
                ")"
                f"{fac_cte_dept},\n"
                "fee_earnings_cte AS (\n"
                '    SELECT dp."department_id",\n'
                '           SUM(fi."amount_paid") AS total_earnings,\n'
                '           SUM(fi."net_amount_due") AS total_fee_billed\n'
                '    FROM "fee_invoices" fi\n'
                '    JOIN "students" s ON fi."student_id" = s."student_id"\n'
                '    JOIN "degree_programs" dp ON s."program_id" = dp."program_id"\n'
                '    GROUP BY dp."department_id"\n'
                ")\n"
                'SELECT d."department_name" AS department_name,\n'
                "       ROUND(COALESCE(er.total_earnings, 0), 2) AS total_earnings,\n"
                f"       ROUND({exp_expr}, 2) AS total_expenditure,\n"
                "       ROUND(COALESCE(db.department_budget, 0), 2) AS department_budget"
                f"{fac_col},\n"
                f"       ROUND(COALESCE(er.total_earnings, 0) - ({exp_expr}), 2) AS net_surplus_deficit\n"
                'FROM "departments" d\n'
                'LEFT JOIN "colleges" c ON d."college_id" = c."college_id"\n'
                'LEFT JOIN dept_budget_cte db ON d."department_id" = db."department_id"\n'
                f"{fac_join_dept}"
                f'LEFT JOIN fee_earnings_cte er ON d."department_id" = er."department_id"{b2_where_sql}\n'
                "ORDER BY total_earnings DESC"
            )

        fac_cte = (
            ",\nfaculty_payroll_cte AS (\n"
            '    SELECT d."college_id", SUM(f."salary") AS faculty_payroll\n'
            '    FROM "faculty" f\n'
            '    JOIN "departments" d ON f."department_id" = d."department_id"\n'
            '    GROUP BY d."college_id"\n'
            ")"
            if has_faculty
            else ""
        )
        fac_join = 'LEFT JOIN faculty_payroll_cte fp ON c."college_id" = fp."college_id"\n' if has_faculty else ""
        exp_expr = "COALESCE(db.department_budget, 0) + COALESCE(fp.faculty_payroll, 0)" if has_faculty else "COALESCE(db.department_budget, 0)"
        fac_col = ",\n       ROUND(COALESCE(fp.faculty_payroll, 0), 2) AS faculty_payroll" if has_faculty else ""
        return (
            "WITH dept_budget_cte AS (\n"
            '    SELECT "college_id", SUM("budget_allocated") AS department_budget\n'
            '    FROM "departments"\n'
            '    GROUP BY "college_id"\n'
            ")"
            f"{fac_cte},\n"
            "fee_earnings_cte AS (\n"
            '    SELECT s."college_id",\n'
            '           SUM(fi."amount_paid") AS total_earnings,\n'
            '           SUM(fi."net_amount_due") AS total_fee_billed\n'
            '    FROM "fee_invoices" fi\n'
            '    JOIN "students" s ON fi."student_id" = s."student_id"\n'
            '    GROUP BY s."college_id"\n'
            ")\n"
            'SELECT c."college_name" AS college_name,\n'
            "       ROUND(COALESCE(er.total_earnings, 0), 2) AS total_earnings,\n"
            f"       ROUND({exp_expr}, 2) AS total_expenditure,\n"
            "       ROUND(COALESCE(db.department_budget, 0), 2) AS department_budget"
            f"{fac_col},\n"
            f"       ROUND(COALESCE(er.total_earnings, 0) - ({exp_expr}), 2) AS net_surplus_deficit\n"
            'FROM "colleges" c\n'
            'LEFT JOIN dept_budget_cte db ON c."college_id" = db."college_id"\n'
            f"{fac_join}"
            'LEFT JOIN fee_earnings_cte er ON c."college_id" = er."college_id"\n'
            "ORDER BY total_earnings DESC"
        )

    # Case C: Multi-table / Cross-Database joined query
    if len(join_plan.required_tables) >= 2 and join_plan.join_edges:
        q_norm = re.sub(r"[_\s]+", " ", q_low)
        q_words_c = [w for w in re.findall(r"[a-z0-9]+", q_low) if len(w) >= 2]
        metric_tbl = None
        metric_col = None
        dim_tbl = None
        dim_col = None

        metric_candidates: list[tuple[int, str, str]] = []
        non_metric_cols = {
            "year", "hire_year", "birth_year", "established_year", "academic_year", "fiscal_year",
            "date", "month", "day", "rank", "zip", "zip_code", "postal_code", "phone", "phone_number",
            "id", "code", "term_id", "college_id", "department_id", "student_id", "faculty_id", "course_id",
            "current_semester", "current_semester_num", "semester", "semester_num",
            "section_number", "credit_hours", "course_level", "is_active",
        }
        generic_metric_words = {"total", "amount", "fee", "score", "marks", "usd", "net", "pct", "rate"}
        has_time_intent = any(w in q_low for w in ("year", "trend", "annual", "yearly", "over time", "yoy", "monthly"))
        for t_name in join_plan.required_tables:
            t_obj = workspace.get_table(t_name)
            if not t_obj:
                continue
            tbl_has_date = any(
                any(k in col.name.lower() for k in ("date", "year", "time"))
                and not any(b in col.name.lower() for b in ("birth", "dob", "established"))
                for col in t_obj.columns
                if not col.is_blocked
            )
            for c in t_obj.columns:
                if c.is_blocked or c.is_primary_key or c.is_foreign_key:
                    continue
                if not any(ty in c.data_type.upper() for ty in ("INT", "FLOAT", "DOUBLE", "NUMERIC", "DECIMAL", "BIGINT")):
                    continue
                c_low = c.name.lower()
                if c_low in non_metric_cols or c_low.endswith("_id") or c_low.endswith("_year"):
                    continue
                score = 0
                if any(w in q_low for w in ("salary", "salaries", "compensation", "pay", "wage", "wages", "payroll")) and any(k in c_low for k in ("salary", "pay", "comp", "wage")):
                    score = 100
                elif any(w in q_low for w in ("revenue", "sales", "turnover")) and any(k in c_low for k in ("revenue", "rev", "sales", "amount")):
                    score = 100
                elif any(w in q_low for w in ("profit", "margin", "earnings")) and any(k in c_low for k in ("profit", "margin")):
                    score = 100
                elif any(w in q_low for w in ("cost", "expense", "spend", "budget")) and any(k in c_low for k in ("cost", "expense", "spend", "budget", "salary")):
                    if "salary" in c_low:
                        score = 105 if (has_time_intent and tbl_has_date) else 92
                    else:
                        score = 95 if (has_time_intent and not tbl_has_date) else 100
                elif any(w in q_low for w in ("fee", "fees", "tuition", "invoice", "balance", "outstanding", "scholarship", "hostel", "mess", "library", "paid", "due")) and any(k in c_low for k in ("fee", "tuition", "amount", "balance", "paid", "scholarship", "discount")):
                    score = 100
                elif any(w in q_low for w in ("mark", "marks", "grade", "grades", "score", "scores", "gpa", "cgpa", "midterm", "internal", "assessment", "attendance")) and any(k in c_low for k in ("mark", "score", "grade", "gpa", "cgpa", "attendance")):
                    score = 100
                elif re.search(rf"\b{re.escape(c_low)}\b", q_low):
                    score = 95
                elif any(w in q_low for w in c_low.replace("_", " ").split() if len(w) > 3):
                    score = 75

                c_tokens = [w for w in c_low.split("_") if len(w) >= 3]
                distinctive_hits = sum(
                    1
                    for ct in c_tokens
                    if ct not in generic_metric_words
                    and (re.search(rf"\b{re.escape(ct)}\b", q_norm) or fuzzy_matches_word(q_words_c, ct, 0.84))
                )
                if distinctive_hits > 0:
                    score = max(score, 90) + (distinctive_hits * 25)

                if any(w in q_low for w in ("gpa", "cgpa", "grade point", "grade points")) and c_low in ("grade_points", "cgpa"):
                    score += 30
                elif "attendance" in q_low and "attendance" in c_low:
                    score += 35
                elif any(w in q_low for w in ("outstanding", "unpaid", "arrear", "arrears")) and "outstanding" in c_low:
                    score += 35
                elif any(w in q_low for w in ("paid", "collected", "collection")) and "paid" in c_low:
                    score += 30
                elif any(w in q_low for w in ("due", "billed")) and "due" in c_low:
                    score += 20

                if score > 0:
                    if "total" in c_low or "net_amount" in c_low:
                        score += 5
                    metric_candidates.append((score, t_name, c.name))

        if metric_candidates:
            metric_candidates.sort(key=lambda x: -x[0])
            _, metric_tbl, metric_col = metric_candidates[0]

        first_edge = join_plan.join_edges[0]
        base_tbl_name = (
            metric_tbl
            if metric_tbl and any(metric_tbl in (e.source_table, e.target_table) for e in join_plan.join_edges)
            else first_edge.source_table
        )
        join_clauses = []
        joined_tables = {base_tbl_name}

        edges_to_process = list(join_plan.join_edges)
        while edges_to_process:
            progress = False
            for edge in list(edges_to_process):
                if edge.source_table in joined_tables and edge.target_table not in joined_tables:
                    join_clauses.append(
                        f'{edge.join_type} JOIN "{edge.target_table}" ON "{edge.source_table}"."{edge.source_column}" = "{edge.target_table}"."{edge.target_column}"'
                    )
                    joined_tables.add(edge.target_table)
                    edges_to_process.remove(edge)
                    progress = True
                elif edge.source_table in joined_tables and edge.target_table in joined_tables:
                    edges_to_process.remove(edge)
                    progress = True
            if progress:
                continue
            for edge in list(edges_to_process):
                if edge.target_table in joined_tables and edge.source_table not in joined_tables:
                    join_clauses.append(
                        f'{edge.join_type} JOIN "{edge.source_table}" ON "{edge.target_table}"."{edge.target_column}" = "{edge.source_table}"."{edge.source_column}"'
                    )
                    joined_tables.add(edge.source_table)
                    edges_to_process.remove(edge)
                    progress = True
                    break
            if not progress:
                break

        join_str = ("\n" + "\n".join(join_clauses)) if join_clauses else ""

        where_filters: list[str] = []
        filtered_tables_cols: set[tuple[str, str]] = set()

        for cat_m in detect_categorical_value_matches(question, workspace):
            m_tbl, m_col, m_val = cat_m["table"], cat_m["column"], cat_m["value"]
            if m_tbl in joined_tables and (m_tbl, m_col) not in filtered_tables_cols:
                safe_val = str(m_val).replace("'", "''")
                where_filters.append(f'"{m_tbl}"."{m_col}" = \'{safe_val}\'')
                filtered_tables_cols.add((m_tbl, m_col))

        filter_seg_match = re.search(
            r"\b(?:for\s+['\"]([a-z0-9_ &]+?)['\"]|(?:into|filtered\s+to|filtered\s+by)\s+['\"]?([a-z0-9_ &]+?)['\"]?)(?:\s+broken\s+down|\s+break\s+down|\s+by\s+|\s*$)",
            q_low,
        )
        if filter_seg_match and not where_filters:
            raw_seg = (filter_seg_match.group(1) or filter_seg_match.group(2) or "").strip()
            stop_seg_words = {
                "each", "every", "all", "any", "the", "this", "that", "these", "those",
                "drill", "down", "into", "result", "results",
                "college", "colleges", "department", "departments", "student", "students",
                "faculty", "course", "courses", "program", "programs", "degree", "degrees",
                "order", "orders", "customer", "customers", "product", "products",
                "category", "categories", "region", "regions", "year", "years", "month", "months",
                "gender", "genders", "rank", "ranks", "status", "tier", "tiers", "channel", "channels",
            }
            seg_words = [w for w in raw_seg.split() if len(w) > 2 and w not in stop_seg_words]
            if seg_words:
                ordered_joined = sorted(
                    joined_tables,
                    key=lambda tn: 0 if any(w in tn.lower() for w in raw_seg.split()) else 1,
                )
                for t_name in ordered_joined:
                    t_obj = workspace.get_table(t_name)
                    if not t_obj:
                        continue
                    sorted_cols = sorted(
                        t_obj.columns,
                        key=lambda col: 0 if col.name.lower().endswith("name") or col.name.lower() in ("name", "title") else 1,
                    )
                    for c in sorted_cols:
                        if c.is_blocked:
                            continue
                        if any(ty in c.data_type.upper() for ty in ("CHAR", "TEXT", "VARCHAR", "STRING")):
                            c_low = c.name.lower()
                            if any(k in c_low for k in ("name", "title", "category", "region", "desc", "code")):
                                pattern = "%" + "%".join(seg_words[:3]) + "%"
                                where_filters.append(f'LOWER("{t_name}"."{c.name}") LIKE \'{pattern}\'')
                                filtered_tables_cols.add((t_name, c.name))
                                break
                    if where_filters:
                        break

        where_clause = ("\nWHERE " + " AND ".join(where_filters)) if where_filters else ""

        def _resolve_dim_phrase(
            dim_phrase: str,
            exclude_cols: set[tuple[str, str]],
        ) -> tuple[str | None, str | None]:
            if not dim_phrase:
                return None, None
            dp_clean = re.sub(r"\s+", "_", dim_phrase.strip().lower())
            dp_tokens = [
                w
                for w in dp_clean.split("_")
                if w and w not in ("for", "in", "where", "filtered", "to", "and", "the", "wise", "each", "every")
            ][:2]
            if not dp_tokens:
                return None, None
            dp_norm = "_".join(dp_tokens)
            dp_first = dp_tokens[0]

            def _matches_tbl(tn: str) -> bool:
                tl = tn.lower()
                ts = tl[:-1] if tl.endswith("s") else tl
                return (
                    dp_norm in tl
                    or dp_first in tl
                    or dp_first.rstrip("s") in tl
                    or fuzzy_matches_word(dp_tokens, tl)
                    or fuzzy_matches_word(dp_tokens, ts)
                )

            ordered_dim_tables = sorted(
                join_plan.required_tables,
                key=lambda tn: 0 if _matches_tbl(tn) else 1,
            )
            for t_name in ordered_dim_tables:
                t_obj = workspace.get_table(t_name)
                if not t_obj:
                    continue
                dim_candidates = sorted(
                    t_obj.columns,
                    key=lambda col: 0 if col.name.lower().endswith("name") or col.name.lower() in ("name", "title") else 1,
                )
                for c in dim_candidates:
                    if (t_name, c.name) in exclude_cols:
                        continue
                    c_low = c.name.lower()
                    if c.is_blocked or c.is_primary_key or c.is_foreign_key or c_low.endswith("_id") or c_low == "id":
                        continue
                    if c_low in ("first_name", "last_name", "phone", "phone_number", "email", "roll_number", "date_of_birth"):
                        continue
                    c_stem = c_low.replace("_name", "").replace("_code", "").replace("_title", "")
                    if (
                        dp_norm == c_low
                        or dp_norm == c_stem
                        or dp_norm in c_low
                        or dp_first == c_low
                        or dp_first == c_stem
                        or (len(dp_tokens) >= 2 and all(tok in (t_name.lower() + "_" + c_low) for tok in dp_tokens))
                        or fuzzy_matches_word([dp_first], c_stem)
                        or (_matches_tbl(t_name) and (c_low.endswith("name") or c_low.endswith("title") or c_low in ("name", "title")))
                    ):
                        return t_name, c.name
            return None, None

        wise_match = re.search(r"\b([a-z0-9_]+)\s+wise\b", q_low)
        break_match = re.search(
            r"\b(?:break\s+(?:that\s+|it\s+)?down\s+by|broken\s+down\s+by|drill\s+down\s+by|split\s+(?:that\s+|it\s+)?by)\s+([a-z0-9_]+(?:\s+(?:program|level|rank|status|mode|tier|category|channel|title|name|year|month))?)",
            q_low,
        )
        by_match = re.search(
            r"\b(?:by|per|across|for\s+each|for\s+every|each)\s+([a-z0-9_]+(?:\s+(?:program|level|rank|status|mode|tier|category|channel|title|name|year|month))?)",
            q_low,
        )

        break_phrase = break_match.group(1).strip().lower() if break_match else ""
        primary_phrase = (
            by_match.group(1).strip().lower()
            if by_match
            else (wise_match.group(1).strip().lower() if wise_match else "")
        )

        second_dim_tbl = None
        second_dim_col = None

        if primary_phrase and break_phrase and primary_phrase != break_phrase:
            prim_tbl, prim_col = _resolve_dim_phrase(primary_phrase, set())
            brk_tbl, brk_col = _resolve_dim_phrase(break_phrase, filtered_tables_cols)
            if prim_tbl and prim_col and (prim_tbl, prim_col) in filtered_tables_cols:
                dim_tbl, dim_col = brk_tbl, brk_col
            elif prim_tbl and prim_col and brk_tbl and brk_col and (prim_tbl, prim_col) != (brk_tbl, brk_col):
                dim_tbl, dim_col = prim_tbl, prim_col
                second_dim_tbl, second_dim_col = brk_tbl, brk_col
            else:
                dim_tbl, dim_col = (brk_tbl, brk_col) if brk_tbl else (prim_tbl, prim_col)
        else:
            target_dim_phrase = break_phrase or primary_phrase
            if target_dim_phrase:
                dim_tbl, dim_col = _resolve_dim_phrase(target_dim_phrase, filtered_tables_cols)
                if not dim_tbl:
                    dim_tbl, dim_col = _resolve_dim_phrase(target_dim_phrase, set())

        if not dim_tbl:
            for t_name in join_plan.required_tables:
                t_obj = workspace.get_table(t_name)
                if not t_obj:
                    continue
                for c in t_obj.columns:
                    c_low = c.name.lower()
                    if c.is_blocked or c.is_primary_key or c.is_foreign_key or c_low.endswith("_id") or c_low == "id":
                        continue
                    if (t_name, c.name) in filtered_tables_cols:
                        continue
                    if any(ty in c.data_type.upper() for ty in ("CHAR", "TEXT", "VARCHAR", "STRING")):
                        if c_low in q_low and len(c_low) >= 4:
                            dim_tbl = t_name
                            dim_col = c.name
                            break
                if dim_tbl:
                    break

        if not dim_tbl:
            def tbl_priority(t: str) -> int:
                tl = t.lower()
                if any(k in tl for k in ("college", "department", "program", "category", "region", "country", "territory")):
                    return 0
                return 1

            for t_name in sorted(join_plan.required_tables, key=tbl_priority):
                t_obj = workspace.get_table(t_name)
                if not t_obj:
                    continue
                t_norm = t_name.lower().replace("_", " ")
                singular = t_name.lower()[:-1] if t_name.lower().endswith("s") else t_name.lower()
                if t_name.lower() in q_low or t_norm in q_norm or singular in q_low or any(ft == t_name for ft, _ in filtered_tables_cols):
                    for c in t_obj.columns:
                        c_low = c.name.lower()
                        if c.is_blocked or c.is_primary_key or c.is_foreign_key or c_low.endswith("_id"):
                            continue
                        if c_low in ("first_name", "last_name", "phone", "email"):
                            continue
                        if c_low.endswith("name") or c_low in ("name", "title", "description", "label"):
                            dim_tbl = t_name
                            dim_col = c.name
                            break
                    if not dim_col:
                        for c in t_obj.columns:
                            c_low = c.name.lower()
                            if c.is_blocked or c.is_primary_key or c.is_foreign_key or c_low.endswith("_id") or c_low in ("first_name", "last_name", "phone", "email"):
                                continue
                            if any(k in c_low for k in ("name", "code", "title", "type", "category")):
                                dim_tbl = t_name
                                dim_col = c.name
                                break
                    if not dim_col:
                        for c in t_obj.columns:
                            c_low = c.name.lower()
                            if not c.is_blocked and not c.is_primary_key and not c.is_foreign_key and not c_low.endswith("_id") and any(ty in c.data_type.upper() for ty in ("CHAR", "TEXT", "VARCHAR", "STRING")):
                                if c_low in ("first_name", "last_name", "phone", "email"):
                                    continue
                                dim_tbl = t_name
                                dim_col = c.name
                                break
                if dim_tbl:
                    break

        if not second_dim_tbl and dim_tbl and dim_col and any(
            k in q_low
            for k in (
                "ratio", "ritio", "raio", "gender", "rank", "status", "breakdown",
                "by", "across", "distribution", "trend", "year", "annual", "yearly", "over time",
            )
        ):
            candidates: list[tuple[int, str, str]] = []
            for t_name in join_plan.required_tables:
                t_obj = workspace.get_table(t_name)
                if not t_obj:
                    continue
                for c in t_obj.columns:
                    c_low = c.name.lower()
                    if c.is_blocked or c.is_primary_key or c.is_foreign_key or c_low.endswith("_id") or c_low == "id":
                        continue
                    if t_name == dim_tbl and c.name == dim_col:
                        continue
                    if (t_name, c.name) in filtered_tables_cols:
                        continue
                    if metric_tbl and t_name == metric_tbl and c.name == metric_col:
                        continue
                    if c_low in ("first_name", "last_name", "phone", "phone_number", "email", "address", "remarks", "syllabus_summary"):
                        continue

                    score = 0
                    is_birth_col = any(b in c_low for b in ("birth", "dob")) and not any(b in q_low for b in ("birth", "dob", "age"))
                    if not is_birth_col and any(k in q_low for k in ("trend", "year", "annual", "yearly", "over time", "yoy")) and any(k in c_low for k in ("hire_year", "payment_year", "hire_date", "order_date", "date")):
                        score = 105 if (metric_tbl and t_name == metric_tbl) else 95
                    elif re.search(rf"\b{re.escape(c_low)}\b", q_low):
                        score = 100
                    elif not is_birth_col and any(k in q_low for k in ("trend", "year", "annual", "yearly", "over time", "yoy")) and "year" in c_low:
                        score = 90 if (metric_tbl and t_name == metric_tbl) else 75
                    elif any(k in q_low for k in ("gender", "sex", "male", "female", "demographic")) and c_low == "gender":
                        score = 100
                    elif any(k in q_low for k in ("rank", "title", "professor")) and "rank" in c_low:
                        score = 95
                    elif any(k in q_low for k in ("tier", "segment", "commercial")) and ("tier" in c_low or "segment" in c_low):
                        score = 95
                    elif any(k in q_low for k in ("category", "type", "class")) and "category" in c_low:
                        score = 90
                    elif "status" in q_low and "status" in c_low:
                        score = 90
                    elif not metric_col and any(k in q_low for k in ("ratio", "ritio", "raio", "demographic", "diversity")):
                        if c_low == "gender":
                            score = 50
                        elif "rank" in c_low:
                            score = 40

                    if score > 0:
                        candidates.append((score, t_name, c.name))

            if candidates:
                candidates.sort(key=lambda x: -x[0])
                _, second_dim_tbl, second_dim_col = candidates[0]

        is_ratio_query = bool(
            re.search(r"\b(?:ratio|raio|ritio|rio|pct|percent|share|proportion)\b", q_low)
        )
        sort_dir = infer_sort_direction(q_low)

        if dim_tbl and dim_col and second_dim_tbl and second_dim_col:
            d1_ref = f'"{dim_tbl}"."{dim_col}"'
            if any(k in second_dim_col.lower() for k in ("date", "time")) and any(w in q_low for w in ("year", "trend", "annual", "yearly", "over time")):
                d2_ref = f'STRFTIME(TRY_CAST("{second_dim_tbl}"."{second_dim_col}" AS DATE), \'%Y\')'
                d2_alias = re.sub(r"[^a-z0-9_]+", "_", second_dim_col.lower()).replace("date", "year").strip("_")
            else:
                d2_ref = f'"{second_dim_tbl}"."{second_dim_col}"'
                d2_alias = re.sub(r"[^a-z0-9_]+", "_", second_dim_col.lower()).strip("_")
            d1_alias = re.sub(r"[^a-z0-9_]+", "_", dim_col.lower()).strip("_")
            top_match = re.search(r"\b(?:top|bottom|limit)\s+(\d+)\b", q_low)
            limit_clause = f"\nLIMIT {int(top_match.group(1))}" if top_match else ""

            if metric_tbl and metric_col:
                m_clean = re.sub(r"[^a-z0-9_]+", "_", metric_col.lower()).strip("_")
                agg_fn = infer_agg_function(q_low, metric_col)
                metric_alias = clean_agg_alias(metric_col, agg_fn)
                ratio_alias = (m_clean.replace("total_", "") + "_ratio_pct").lstrip("_")
                ratio_select = ""
                if is_ratio_query:
                    ratio_select = (
                        f",\n       ROUND(100.0 * SUM(\"{metric_tbl}\".\"{metric_col}\") / NULLIF(SUM(SUM(\"{metric_tbl}\".\"{metric_col}\")) OVER (PARTITION BY {d1_ref}), 0), 2) AS {ratio_alias}"
                    )
                return (
                    f"SELECT {d1_ref} AS {d1_alias},\n"
                    f"       {d2_ref} AS {d2_alias},\n"
                    f"       ROUND({agg_fn}(\"{metric_tbl}\".\"{metric_col}\"), 2) AS {metric_alias},\n"
                    f"       COUNT(*) AS {d2_alias}_count{ratio_select}\n"
                    f'FROM "{base_tbl_name}"{join_str}{where_clause}\n'
                    f"GROUP BY {d1_ref}, {d2_ref}\n"
                    f"ORDER BY {d1_ref}, {d2_ref} ASC"
                    f"{limit_clause}"
                )

            return (
                f"SELECT {d1_ref} AS {d1_alias},\n"
                f"       {d2_ref} AS {d2_alias},\n"
                f"       COUNT(*) AS {d2_alias}_count,\n"
                f"       ROUND(100.0 * COUNT(*) / NULLIF(SUM(COUNT(*)) OVER (PARTITION BY {d1_ref}), 0), 2) AS {d2_alias}_ratio_pct\n"
                f'FROM "{base_tbl_name}"{join_str}{where_clause}\n'
                f"GROUP BY {d1_ref}, {d2_ref}\n"
                f"ORDER BY {d1_ref}, {d2_alias}_count {sort_dir}"
                f"{limit_clause}"
            )

        if dim_tbl and dim_col:
            dim_ref = f'"{dim_tbl}"."{dim_col}"'
            dim_alias = re.sub(r"[^a-z0-9_]+", "_", dim_col.lower()).strip("_")
            top_match = re.search(r"\b(?:top|bottom|limit)\s+(\d+)\b", q_low)
            limit_clause = f"\nLIMIT {int(top_match.group(1))}" if top_match else ""
            if metric_tbl and metric_col:
                agg_fn = infer_agg_function(q_low, metric_col)
                metric_ref = f'ROUND({agg_fn}("{metric_tbl}"."{metric_col}"), 2)'
                metric_alias = clean_agg_alias(metric_col, agg_fn)
                ratio_select = (
                    f",\n       ROUND(100.0 * COUNT(*) / NULLIF(SUM(COUNT(*)) OVER (), 0), 2) AS ratio_pct"
                    if is_ratio_query else ""
                )
                return (
                    f"SELECT {dim_ref} AS {dim_alias},\n"
                    f"       {metric_ref} AS {metric_alias},\n"
                    f"       COUNT(*) AS total_count{ratio_select}\n"
                    f'FROM "{base_tbl_name}"{join_str}{where_clause}\n'
                    f"GROUP BY {dim_ref}\n"
                    f"ORDER BY {metric_alias} {sort_dir}"
                    f"{limit_clause}"
                )
            else:
                ratio_select = (
                    f",\n       ROUND(100.0 * COUNT(*) / NULLIF(SUM(COUNT(*)) OVER (), 0), 2) AS ratio_pct"
                    if is_ratio_query else ""
                )
                return (
                    f"SELECT {dim_ref} AS {dim_alias},\n"
                    f"       COUNT(*) AS row_count{ratio_select}\n"
                    f'FROM "{base_tbl_name}"{join_str}{where_clause}\n'
                    f"GROUP BY {dim_ref}\n"
                    f"ORDER BY row_count {sort_dir}"
                    f"{limit_clause}"
                )

        select_cols: list[str] = []
        for t_name in [base_tbl_name] + [t for t in join_plan.required_tables if t != base_tbl_name]:
            t_obj = workspace.get_table(t_name)
            if t_obj:
                for c in t_obj.columns:
                    if not c.is_blocked and not c.is_pii:
                        select_cols.append(f'"{t_name}"."{c.name}"')
                        if len(select_cols) >= 8:
                            break
                if len(select_cols) >= 8:
                    break
        if select_cols:
            return f'SELECT {", ".join(select_cols)} FROM "{base_tbl_name}"{join_str}{where_clause} LIMIT 50'

    # Fallback for single-table or unjoined schema queries
    target_tbl = None
    if join_plan and join_plan.required_tables:
        best_t = None
        best_t_score = -1
        for rt in join_plan.required_tables:
            t_cand = workspace.get_table(rt)
            if not t_cand:
                continue
            t_sc = 0
            tl = t_cand.name.lower()
            if tl in q_low or tl.rstrip("s") in q_low:
                t_sc += 50
            for col in t_cand.columns:
                if not col.is_blocked and col.name.lower() in q_low:
                    t_sc += 30
            if t_sc > best_t_score:
                best_t_score = t_sc
                best_t = t_cand
        target_tbl = best_t or workspace.get_table(join_plan.required_tables[0])
    if not target_tbl:
        for t in workspace.active_tables():
            t_low = t.name.lower()
            if t_low in q_low or t_low.rstrip("s") in q_low:
                target_tbl = t
                break
    if not target_tbl:
        target_tbl = workspace.active_tables()[0]

    non_metric_single = {
        "year", "hire_year", "birth_year", "established_year", "academic_year", "fiscal_year", "payment_year",
        "date", "month", "day", "rank", "zip", "zip_code", "postal_code", "phone", "phone_number",
        "id", "code", "current_semester", "current_semester_num", "semester", "semester_num",
        "section_number", "credit_hours", "course_level", "is_active",
    }
    cols = [c.name for c in target_tbl.columns if not c.is_blocked]
    num_cols = [
        c.name
        for c in target_tbl.columns
        if not c.is_blocked
        and any(t in c.data_type.upper() for t in ("INT", "FLOAT", "DOUBLE", "DECIMAL", "NUMERIC", "BIGINT"))
        and not c.is_primary_key
        and not c.is_foreign_key
        and not c.name.lower().endswith(("_id", "_year"))
        and c.name.lower() not in non_metric_single
    ]
    matched_num_col = None
    for nc in num_cols:
        nc_low = nc.lower()
        if nc_low in q_low or any(w in q_low for w in nc_low.split("_") if len(w) > 3):
            matched_num_col = nc
            break
        if any(w in q_low for w in ("salary", "salaries", "compensation", "pay")) and "salary" in nc_low:
            matched_num_col = nc
            break
        if any(w in q_low for w in ("cost", "expense", "spend", "budget")) and any(k in nc_low for k in ("budget", "cost", "expense", "spend", "salary")):
            matched_num_col = nc
            break
    active_num_col = matched_num_col or (num_cols[0] if num_cols else None)

    single_where_parts: list[str] = []
    for cat_m in detect_categorical_value_matches(question, workspace):
        if cat_m["table"] == target_tbl.name:
            safe_v = str(cat_m["value"]).replace("'", "''")
            single_where_parts.append(f'"{target_tbl.name}"."{cat_m["column"]}" = \'{safe_v}\'')
    single_where_sql = f" WHERE {' AND '.join(single_where_parts)}" if single_where_parts else ""

    cat_cols = [
        c.name
        for c in target_tbl.columns
        if not c.is_blocked
        and c.name not in num_cols
        and not c.is_primary_key
        and not c.is_foreign_key
        and not c.name.lower().endswith("_id")
        and c.name.lower() not in ("first_name", "last_name", "email", "phone", "phone_number", "roll_number", "date_of_birth")
    ]
    if not cat_cols:
        cat_cols = [
            c.name
            for c in target_tbl.columns
            if not c.is_blocked and c.name not in num_cols and not c.is_primary_key
        ]
    cat_cols.sort(key=lambda col: 0 if col.lower().endswith("name") or col.lower() in ("name", "title") else 1)

    matched_cat_col = None
    by_single = re.search(
        r"\b(?:by|per|across|for\s+each|each|broken\s+down\s+by|break\s+(?:that\s+|it\s+)?down\s+by)\s+([a-z0-9_]+)",
        q_low,
    )
    target_single_dim = by_single.group(1).lower() if by_single else ""
    for cc in cat_cols:
        cc_low = cc.lower()
        cc_stem = cc_low.replace("_name", "").replace("_code", "")
        if (
            target_single_dim
            and (
                target_single_dim == cc_low
                or target_single_dim in cc_low
                or fuzzy_matches_word([target_single_dim], cc_stem)
            )
        ) or re.search(rf"\b{re.escape(cc_low)}\b", q_low):
            matched_cat_col = cc
            break
        if any(w in cc_low for w in q_low.split() if len(w) > 3 and w not in ("show", "total", "average", "count", "trend")):
            matched_cat_col = cc
            break
    if not matched_cat_col:
        matched_cat_col = next((c for c in cat_cols if c.lower().endswith("_name") or c.lower() == "name"), None)
    active_cat_col = matched_cat_col or (cat_cols[0] if cat_cols else None)
    sort_dir = infer_sort_direction(q_low)

    date_col = next(
        (
            c.name
            for c in target_tbl.columns
            if not c.is_blocked
            and any(k in c.name.lower() for k in ("date", "time", "year"))
            and not c.name.lower().endswith("_id")
            and (
                not any(b in c.name.lower() for b in ("birth", "dob"))
                or any(b in q_low for b in ("birth", "dob", "age"))
            )
        ),
        None,
    )
    if date_col and any(w in q_low for w in ("trend", "year", "monthly", "over time", "timeline")):
        agg_fn = infer_agg_function(q_low, active_num_col)
        dim_expr = (
            f'STRFTIME(TRY_CAST("{target_tbl.name}"."{date_col}" AS DATE), \'%Y\')'
            if "date" in date_col.lower()
            else f'"{target_tbl.name}"."{date_col}"'
        )
        dim_alias = date_col.lower().replace("date", "year") if "date" in date_col.lower() else date_col.lower()
        top_match = re.search(r"\b(?:top|bottom|limit)\s+(\d+)\b", q_low)
        limit_clause = f"\nLIMIT {int(top_match.group(1))}" if top_match else ""
        extra_date_cond = (" AND " + " AND ".join(single_where_parts)) if single_where_parts else ""
        if active_num_col:
            alias = clean_agg_alias(active_num_col, agg_fn)
            return (
                f'SELECT {dim_expr} AS {dim_alias},\n'
                f'       ROUND({agg_fn}("{target_tbl.name}"."{active_num_col}"), 2) AS {alias},\n'
                f'       COUNT(*) AS row_count\n'
                f'FROM "{target_tbl.name}"\n'
                f'WHERE "{target_tbl.name}"."{date_col}" IS NOT NULL{extra_date_cond}\n'
                f'GROUP BY {dim_alias}\n'
                f'ORDER BY {dim_alias} ASC'
                f'{limit_clause}'
            )

    if active_cat_col and active_num_col and not (any(w in q_low for w in ("count", "how many", "number of")) and not matched_num_col):
        agg_fn = infer_agg_function(q_low, active_num_col)
        alias = clean_agg_alias(active_num_col, agg_fn)
        top_match = re.search(r"\b(?:top|bottom|limit)\s+(\d+)\b", q_low)
        limit_clause = f" LIMIT {int(top_match.group(1))}" if top_match else ""
        return (
            f'SELECT "{active_cat_col}", ROUND({agg_fn}("{target_tbl.name}"."{active_num_col}"), 2) AS {alias}, COUNT(*) AS row_count '
            f'FROM "{target_tbl.name}"{single_where_sql} GROUP BY "{active_cat_col}" ORDER BY {alias} {sort_dir}{limit_clause}'
        )
    if active_cat_col:
        top_match = re.search(r"\b(?:top|bottom|limit)\s+(\d+)\b", q_low)
        limit_clause = f" LIMIT {int(top_match.group(1))}" if top_match else ""
        return (
            f'SELECT "{active_cat_col}", COUNT(*) AS row_count '
            f'FROM "{target_tbl.name}"{single_where_sql} GROUP BY "{active_cat_col}" ORDER BY row_count {sort_dir}{limit_clause}'
        )
    return f'SELECT {", ".join(f"{c}" for c in cols[:8])} FROM "{target_tbl.name}"{single_where_sql} LIMIT 50'


def build_dax_equivalent(
    question: str,
    workspace: OKFWorkspace,
    matched_measures: list[SemanticMeasure],
) -> str:
    """Build a companion Power BI DAX `DEFINE MEASURE ... EVALUATE SUMMARIZECOLUMNS` query."""
    q_low = question.lower()
    define_block = MeasureCompiler.format_dax_define_block(matched_measures or workspace.measures[:2])
    if workspace.get_table("orders"):
        group_dax = "'products'[category]"
        if "region" in q_low:
            group_dax = "'customers'[region_code]"
        elif "tier" in q_low:
            group_dax = "'customers'[cust_tier]"
        elif "channel" in q_low:
            group_dax = "'orders'[sales_channel]"
        elif "month" in q_low or "trend" in q_low:
            group_dax = "'orders'[order_date]"

        eval_block = (
            f"EVALUATE\n"
            f"SUMMARIZECOLUMNS(\n"
            f"  {group_dax},\n"
            f"  \"Net Revenue USD\", CALCULATE(SUM('orders'[rev_net_usd]), 'orders'[ord_stat] <> \"X\"),\n"
            f"  \"Profit USD\", CALCULATE(SUM('orders'[profit_usd]), 'orders'[ord_stat] <> \"X\"),\n"
            f"  \"Gross Margin %\", DIVIDE(SUM('orders'[profit_usd]), SUM('orders'[rev_net_usd]), 0)\n"
            f")"
        )
        return f"{define_block}\n{eval_block}".strip()

    active = workspace.active_tables()
    if not active:
        return "EVALUATE ROW(\"Count\", 0)"
    dim_tbl = active[0].name
    dim_col = active[0].columns[0].name if active[0].columns else "id"
    for t in active:
        for c in t.columns:
            if not c.is_blocked and not c.is_primary_key and not c.is_foreign_key:
                if c.name.lower() in q_low or t.name.lower().rstrip("s") in q_low:
                    if any(ty in c.data_type.upper() for ty in ("CHAR", "TEXT", "VARCHAR", "STRING")):
                        dim_tbl, dim_col = t.name, c.name
                        break
    measure_name = matched_measures[0].name if matched_measures else "Row Count"
    measure_dax = (
        matched_measures[0].dax_expression
        if (matched_measures and matched_measures[0].dax_expression)
        else f"COUNTROWS('{dim_tbl}')"
    )
    eval_block = (
        f"EVALUATE\n"
        f"SUMMARIZECOLUMNS(\n"
        f"  '{dim_tbl}'[{dim_col}],\n"
        f"  \"{measure_name}\", {measure_dax}\n"
        f")"
    )
    return f"{define_block}\n{eval_block}".strip()
