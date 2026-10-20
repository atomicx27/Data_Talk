"""Adaptive Smart Business Knowledge Interviewer Agent for Data_Talk.

Analyzes the OKFWorkspace + Statistical Column Profiles (without exposing PII) to:
1. Identify cryptic column names (e.g., `ord_stat`, `rev_gross_lcy`, `disc_amt`, `cust_tier`, `ret_reason_cd`).
2. Identify low-cardinality categorical status/code columns (`['C', 'P', 'X', 'H']`) and ask what each code means
   and which codes should be included/excluded in standard business metrics.
3. Identify multiple date columns or revenue/amount columns and ask which is the primary reporting grain.
4. Process user answers atomically into `OKFWorkspace.business_rules`, `ColumnProfile.value_meanings`, and `RAGStore`.
5. Compute a live **Knowledge Readiness Score (0–100%)**.
"""

from __future__ import annotations

import re
from typing import Any

from src.agents.llm_router import LLMRouter
from src.knowledge.okf_models import (
    BusinessRule,
    ColumnProfile,
    InterviewQuestion,
    JoinStatus,
    OKFWorkspace,
    TableSchema,
)
from src.knowledge.rag_store import RAGStore
from src.security.privacy_firewall import PrivacyFirewall

CRYPTIC_ABBREV_PATTERN = re.compile(
    r"(_cd|_stat|_amt|_lcy|_qty|_dt|_flg|_ind|_id|^ord_|^cust_|^rev_|^ret_|^disc_|^acct_)",
    re.IGNORECASE,
)


class InterviewerAgent:
    """Generates prioritized business knowledge questions and indexes user answers into OKF + RAG."""

    @classmethod
    def generate_prioritized_questions(
        cls,
        workspace: OKFWorkspace,
        llm_router: LLMRouter | None = None,
        max_questions: int = 8,
    ) -> list[InterviewQuestion]:
        """Generate 5-10 high-impact onboarding questions based on schema & statistical profile."""
        existing_by_id = {q.id: q for q in workspace.interview_questions}
        candidates: list[InterviewQuestion] = []

        # 1. Categorical Code / Status Columns (Highest Impact — prevents wrong filter assumptions!)
        for tbl in workspace.active_tables():
            for col in tbl.columns:
                if col.is_blocked or col.is_pii or col.is_primary_key or col.is_foreign_key:
                    continue
                if 1 < len(col.sample_values) <= 12 and (
                    any(len(str(v)) <= 4 for v in col.sample_values)
                    or any(
                        tok in col.name.lower()
                        for tok in ("stat", "status", "tier", "type", "code", "cd", "reason", "region", "channel")
                    )
                ):
                    qid = f"q_codes_{tbl.name}_{col.name}"
                    if qid in existing_by_id:
                        candidates.append(existing_by_id[qid])
                        continue
                    safe_samples = ", ".join(f"'{v}'" for v in col.sample_values[:8])
                    q_text = (
                        f"In table `{tbl.name}`, column `{col.name}` contains values [{safe_samples}]. "
                        f"What do these codes mean, and should any be excluded when calculating standard metrics?"
                    )
                    suggestions = cls._suggest_options_for_codes(col.name, col.sample_values)
                    candidates.append(
                        InterviewQuestion(
                            id=qid,
                            priority=1,
                            category="status_codes",
                            table_name=tbl.name,
                            column_name=col.name,
                            question_text=q_text,
                            context_hint=f"Distinct values: {col.distinct_count} | Null rate: {col.null_ratio*100:.1f}%",
                            suggested_options=suggestions,
                        )
                    )

        # 2. Ambiguous Revenue / Metric Columns (e.g., rev_gross_lcy vs rev_net_usd vs disc_amt)
        for tbl in workspace.active_tables():
            metric_cols = [
                c.name
                for c in tbl.columns
                if not c.is_blocked
                and any(
                    k in c.name.lower()
                    for k in ("rev", "sales", "amt", "price", "cost", "profit", "refund", "margin")
                )
            ]
            if len(metric_cols) >= 2:
                qid = f"q_metric_choice_{tbl.name}"
                if qid in existing_by_id:
                    candidates.append(existing_by_id[qid])
                else:
                    cols_str = ", ".join(f"`{c}`" for c in metric_cols)
                    candidates.append(
                        InterviewQuestion(
                            id=qid,
                            priority=1,
                            category="metric_definition",
                            table_name=tbl.name,
                            column_name=metric_cols[0],
                            question_text=(
                                f"Table `{tbl.name}` has multiple financial columns ({cols_str}). "
                                f"When a user asks for 'Revenue' or 'Sales', which column or formula should be the default?"
                            ),
                            context_hint=f"Table row count: {tbl.row_count:,}",
                            suggested_options=[
                                f"Use `{metric_cols[0]}` as Gross Revenue and `rev_net_usd` (or Gross minus Discount) as Default Recognized Revenue.",
                                f"Always use `{metric_cols[0]}` for all Revenue queries.",
                            ],
                        )
                    )

        # 3. Cryptic Column Abbreviations needing Business Aliases
        for tbl in workspace.active_tables():
            for col in tbl.columns:
                if col.is_blocked or col.is_pii or col.is_primary_key or col.is_foreign_key:
                    continue
                if CRYPTIC_ABBREV_PATTERN.search(col.name) and not col.description:
                    qid = f"q_codes_{tbl.name}_{col.name}"
                    if any(q.id == qid for q in candidates):
                        continue
                    alias_qid = f"q_alias_{tbl.name}_{col.name}"
                    if alias_qid in existing_by_id:
                        candidates.append(existing_by_id[alias_qid])
                        continue
                    candidates.append(
                        InterviewQuestion(
                            id=alias_qid,
                            priority=2,
                            category="cryptic_column",
                            table_name=tbl.name,
                            column_name=col.name,
                            question_text=(
                                f"What is the business meaning and unit of `{tbl.name}.{col.name}` "
                                f"(type: `{col.data_type}`, range: `{col.min_value}` to `{col.max_value}`)?"
                            ),
                            context_hint=f"Distinct values: {col.distinct_count:,}",
                            suggested_options=cls._suggest_alias_for_column(col.name),
                        )
                    )

        # 4. Date / Fiscal Calendar Grain Question
        date_cols = [
            f"{t.name}.{c.name}"
            for t in workspace.active_tables()
            for c in t.columns
            if not c.is_blocked and any(k in c.name.lower() or k in c.data_type.lower() for k in ("date", "time", "_dt"))
        ]
        if date_cols:
            qid = "q_fiscal_calendar"
            if qid in existing_by_id:
                candidates.append(existing_by_id[qid])
            else:
                candidates.append(
                    InterviewQuestion(
                        id=qid,
                        priority=2,
                        category="date_grain",
                        table_name=date_cols[0].split(".")[0],
                        column_name=date_cols[0].split(".")[1],
                        question_text=(
                            f"We detected date column(s) ({', '.join(f'`{d}`' for d in date_cols[:4])}). "
                            "Does your organization use a standard Calendar Year (Jan–Dec) or a custom Fiscal Year for YoY/QoQ reporting?"
                        ),
                        context_hint="Used for YoY, QoQ, and MoM time-intelligence calculations",
                        suggested_options=[
                            "Standard Calendar Year (Jan 1 – Dec 31), using `orders.order_date` as primary date grain.",
                            "Fiscal Year starting April 1 (Q1 = Apr–Jun), using `orders.order_date` as primary date grain.",
                        ],
                    )
                )

        # Optional LLM enrichment if configured and fewer than max_questions
        candidates.sort(key=lambda q: q.priority)
        selected = candidates[:max_questions]
        workspace.interview_questions = selected
        return selected

    @staticmethod
    def _suggest_options_for_codes(col_name: str, samples: list[str]) -> list[str]:
        c_low = col_name.lower()
        sample_set = {s.upper() for s in samples}
        if "stat" in c_low and {"C", "P", "X"}.issubset(sample_set):
            return [
                "C = Completed, P = Pending, X = Cancelled, H = On Hold. Exclude 'X' (Cancelled) from Recognized Revenue.",
                "C = Closed/Won, P = Processing, X = Cancelled, H = Hold. Include only 'C' in Revenue.",
            ]
        if "tier" in c_low and {"ENT", "MM", "SMB"}.issubset(sample_set):
            return [
                "ENT = Enterprise (>1000 employees), MM = Mid-Market, SMB = Small/Medium Business.",
            ]
        if "reason" in c_low and {"DEF", "LATE"}.issubset(sample_set):
            return [
                "DEF = Defective Product, LATE = Late Delivery, DUP = Duplicate Order, CHG = Customer Changed Mind.",
            ]
        return [f"Standard categorical values for `{col_name}` ({', '.join(samples[:5])})."]

    @staticmethod
    def _suggest_alias_for_column(col_name: str) -> list[str]:
        c_low = col_name.lower()
        if "rev_gross" in c_low:
            return ["Gross Revenue before discounts (in USD / Local Currency)."]
        if "disc_amt" in c_low:
            return ["Promotional Discount Amount in USD deducted from Gross Revenue."]
        if "refund_amt" in c_low:
            return ["Total Refunded Dollar Amount issued to customer upon return."]
        return [f"Business metric column `{col_name}`."]

    @classmethod
    def record_answer(
        cls,
        workspace: OKFWorkspace,
        rag_store: RAGStore,
        question_id: str,
        answer_text: str,
    ) -> BusinessRule | None:
        """Save user's interview answer into OKFWorkspace + RAGStore atomically."""
        cleaned = (answer_text or "").strip()
        if not cleaned:
            return None

        target_q: InterviewQuestion | None = None
        for q in workspace.interview_questions:
            if q.id == question_id:
                target_q = q
                break

        if target_q is None:
            return None

        target_q.answer = cleaned
        target_q.answered = True

        # Update column description & code mappings if applicable
        if target_q.table_name and target_q.column_name:
            tbl = workspace.get_table(target_q.table_name)
            if tbl:
                col = tbl.get_column(target_q.column_name)
                if col:
                    col.description = cleaned
                    # Extract 'CODE = Meaning' pairs if present
                    pairs = re.findall(r"\b([A-Z0-9_]{1,6})\s*=\s*([A-Za-z0-9 /\-]+)", cleaned)
                    for k, v in pairs:
                        col.value_meanings[k.strip()] = v.strip()

        # Infer SQL/DAX filter hints if answer mentions excluding a code (e.g. "Exclude 'X'")
        sql_hint = ""
        dax_hint = ""
        excl_match = re.search(r"exclude\s+['\"]?([A-Za-z0-9_]+)['\"]?", cleaned, re.IGNORECASE)
        if excl_match and target_q.table_name and target_q.column_name:
            ex_val = excl_match.group(1)
            sql_hint = f"{target_q.table_name}.{target_q.column_name} <> '{ex_val}'"
            dax_hint = f"'{target_q.table_name}'[{target_q.column_name}] <> \"{ex_val}\""

        rule_id = f"rule_{question_id}"
        rule = BusinessRule(
            id=rule_id,
            category=target_q.category,
            term=f"{target_q.table_name}.{target_q.column_name}".strip(".") or target_q.category,
            definition=cleaned,
            applies_to_tables=[target_q.table_name] if target_q.table_name else [],
            sql_filter_hint=sql_hint,
            dax_filter_hint=dax_hint,
            source="interview",
        )

        # Replace existing rule with same id or append
        workspace.business_rules = [r for r in workspace.business_rules if r.id != rule_id] + [rule]

        # Index in RAGStore for semantic retrieval during chat
        rag_store.upsert(
            collection="business_rules",
            doc_id=rule_id,
            content=f"Question: {target_q.question_text}\nBusiness Rule Answer: {cleaned}\nSQL Filter Hint: {sql_hint}",
            metadata={
                "table": target_q.table_name,
                "column": target_q.column_name,
                "category": target_q.category,
                "sql_filter_hint": sql_hint,
                "dax_filter_hint": dax_hint,
            },
        )
        return rule

    @staticmethod
    def sync_schema_to_rag(workspace: OKFWorkspace, rag_store: RAGStore) -> None:
        """Index all active table schemas and column profiles into the RAG `schema_glossary` collection."""
        for tbl in workspace.active_tables():
            col_summaries = []
            for c in tbl.columns:
                if c.is_blocked:
                    continue
                samples_str = (
                    PrivacyFirewall.wrap_untrusted_samples(c.sample_values)
                    if c.sample_values and not c.is_pii
                    else "[]"
                )
                meanings_str = (
                    f" | Codes: {c.value_meanings}" if c.value_meanings else ""
                )
                desc_str = f" ({c.description})" if c.description else ""
                col_summaries.append(
                    f"- {c.name} [{c.data_type}]{desc_str}: samples={samples_str}{meanings_str}"
                )
            doc_text = (
                f"Table `{tbl.name}` (Role: {tbl.table_role.value}, Rows: {tbl.row_count:,})\n"
                f"Description: {tbl.description or 'Core data table'}\n"
                f"Columns:\n" + "\n".join(col_summaries)
            )
            rag_store.upsert(
                collection="schema_glossary",
                doc_id=f"table_{tbl.name}",
                content=doc_text,
                metadata={"table_name": tbl.name, "role": tbl.table_role.value},
            )

    @staticmethod
    def compute_readiness_score(workspace: OKFWorkspace) -> dict[str, Any]:
        """Compute the 0–100% Knowledge Readiness Score across Schema, Joins, Interview, and Measures."""
        active_tbls = workspace.active_tables()
        if not active_tbls:
            return {"score": 0, "schema_pts": 0, "join_pts": 0, "interview_pts": 0, "measure_pts": 0}

        # 1. Schema Profiled (25 pts)
        schema_pts = 25 if any(t.row_count > 0 for t in active_tbls) else 15

        # 2. Joins Approved (25 pts)
        if len(active_tbls) == 1:
            join_pts = 25
        else:
            total_rels = [r for r in workspace.relationships if r.status != JoinStatus.REJECTED]
            approved_rels = workspace.approved_relationships()
            if not total_rels:
                join_pts = 10
            else:
                join_pts = int(25 * (len(approved_rels) / len(total_rels)))

        # 3. Business Interview Answered (30 pts)
        qs = workspace.interview_questions
        if not qs:
            interview_pts = 15
        else:
            answered = sum(1 for q in qs if q.answered)
            interview_pts = int(30 * min(1.0, answered / max(1, min(4, len(qs)))))

        # 4. Semantic Measures / Rules Defined (20 pts)
        n_assets = len(workspace.measures) + len(workspace.calculated_columns) + len(workspace.business_rules)
        measure_pts = min(20, n_assets * 5)

        total = min(100, schema_pts + join_pts + interview_pts + measure_pts)
        return {
            "score": total,
            "schema_pts": schema_pts,
            "join_pts": join_pts,
            "interview_pts": interview_pts,
            "measure_pts": measure_pts,
        }
