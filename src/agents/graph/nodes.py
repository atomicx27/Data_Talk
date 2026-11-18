"""11 pure-Python traced LangGraph nodes, conditional routers, and structured ErrorDiagnostic classification."""

from __future__ import annotations

import logging
import re
from typing import Any, Literal

import pandas as pd
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END
from langgraph.types import Command, interrupt

from src.agents.compilers.deterministic_compiler import build_dax_equivalent, compile_deterministic_sql
from src.agents.compilers.semantic_helpers import (
    build_adaptive_schema_context,
    check_just_in_time_clarification,
    expand_schema_from_error,
    find_matching_verified_query,
    generate_query,
    resolve_follow_up_context,
)
from src.agents.compilers.summary_and_followups import (
    build_executive_summary,
    generate_dataset_overview_payload,
    generate_drill_down_options,
    generate_suggested_follow_ups,
    is_dataset_overview_question,
)
from src.agents.compilers.text_matching import (
    detect_categorical_value_matches,
    detect_relevant_tables,
    ensure_categorical_samples,
    extract_what_if_from_prompt,
    match_measures,
    normalize_prompt_text,
    normalize_what_if_overrides,
)
from src.agents.graph.callbacks import emit_stream_event, trace_graph_node
from src.agents.graph.state import (
    BlockingAmbiguityPayload,
    CategoricalMatchRecord,
    ErrorDiagnostic,
    JoinPlanSnapshot,
    QueryGraphState,
    get_graph_context,
)
from src.agents.interviewer_agent import InterviewerAgent
from src.agents.prompt_manager import PromptManager
from src.knowledge.okf_models import BusinessRule, JoinStatus, OKFWorkspace, SemanticMeasure
from src.knowledge.okf_store import JoinPlan
from src.security.privacy_firewall import PrivacyFirewall, SecurityViolationError

_LOGGER = logging.getLogger(__name__)
MAX_RETRIES = 3


def classify_error_diagnostic(
    error_msg: str,
    failed_node: str,
    attempt: int = 1,
    max_retries: int = MAX_RETRIES,
    workspace: OKFWorkspace | None = None,
) -> ErrorDiagnostic:
    """Classify any SQL, AST security, LLM provider, domain, or runtime error into structured ErrorDiagnostic."""
    raw = str(error_msg or "")
    low = raw.lower()
    redacted = PrivacyFirewall.redact_text(raw)

    if any(
        tok in low
        for tok in (
            "security",
            "firewall",
            "read-only",
            "read only",
            "forbidden",
            "blocked table",
            "blocked column",
            "multi-statement",
            "ddl",
            "dml",
            "pii",
        )
    ):
        return ErrorDiagnostic(
            error_code="AST_SECURITY_VIOLATION",
            error_category="security",
            failed_node=failed_node,
            attempt=attempt,
            max_retries=max_retries,
            user_message="Query blocked by the Read-Only SQL AST Security Firewall.",
            technical_detail=redacted,
            remediation_hint="Retry with the Offline Heuristic Engine or ensure no blocked PII columns or write statements are requested.",
            recoverable_with_heuristic=True,
        )

    if "no matching tables" in low or "out_of_domain" in low:
        active_names = ", ".join(t.name for t in workspace.active_tables()[:8]) if workspace else ""
        hint = (
            f"Upload a dataset or query existing workspace tables ({active_names})."
            if active_names
            else "Upload a dataset or connect a database source from Connect Data."
        )
        return ErrorDiagnostic(
            error_code="OUT_OF_DOMAIN",
            error_category="domain",
            failed_node=failed_node,
            attempt=attempt,
            max_retries=max_retries,
            user_message="No matching tables, columns, or metrics were found in the active workspace.",
            technical_detail=redacted,
            remediation_hint=hint,
            recoverable_with_heuristic=False,
        )

    if any(tok in low for tok in ("401", "429", "api key", "api_key", "authentication", "rate limit", "timeout", "litellm")):
        return ErrorDiagnostic(
            error_code="LLM_PROVIDER_ERROR",
            error_category="llm_provider",
            failed_node=failed_node,
            attempt=attempt,
            max_retries=max_retries,
            user_message="The configured AI model provider failed or timed out while compiling the query.",
            technical_detail=redacted,
            remediation_hint="Retry using the Offline Heuristic Engine or inspect your API credentials in AI Model Settings.",
            recoverable_with_heuristic=True,
        )

    if any(
        tok in low
        for tok in (
            "catalog error",
            "binder error",
            "does not exist",
            "no such table",
            "no such column",
            "not found in from clause",
            "referenced table",
        )
    ):
        return ErrorDiagnostic(
            error_code="DB_CATALOG_ERROR",
            error_category="database",
            failed_node=failed_node,
            attempt=attempt,
            max_retries=max_retries,
            user_message="Database engine could not resolve a referenced table or column.",
            technical_detail=redacted,
            remediation_hint="Inspect OKF schema/joins or open the query in the SQL Scratchpad to adjust identifiers.",
            recoverable_with_heuristic=True,
        )

    if any(tok in low for tok in ("parser error", "syntax error", "sql", "dax", "parse")):
        return ErrorDiagnostic(
            error_code="SQL_SYNTAX_ERROR",
            error_category="sql_compilation",
            failed_node=failed_node,
            attempt=attempt,
            max_retries=max_retries,
            user_message="Compiled query could not be parsed by the database engine.",
            technical_detail=redacted,
            remediation_hint="Open the query in SQL Scratchpad to inspect syntax or retry with Offline Heuristic Engine.",
            recoverable_with_heuristic=True,
        )

    if "recursion limit" in low:
        return ErrorDiagnostic(
            error_code="GRAPH_RECURSION_LIMIT",
            error_category="internal",
            failed_node=failed_node,
            attempt=attempt,
            max_retries=max_retries,
            user_message="Orchestration exceeded maximum state transition steps.",
            technical_detail=redacted,
            remediation_hint="Retry with a simpler query or enable Offline Heuristic Engine.",
            recoverable_with_heuristic=True,
        )

    return ErrorDiagnostic(
        error_code="INTERNAL_NODE_ERROR",
        error_category="internal",
        failed_node=failed_node,
        attempt=attempt,
        max_retries=max_retries,
        user_message="Query execution failed after self-healing attempts.",
        technical_detail=redacted,
        remediation_hint="Inspect trace details or retry using Offline Heuristic Engine / SQL Scratchpad.",
        recoverable_with_heuristic=True,
    )


def _classify_clarification_severity(
    question: str,
    workspace: OKFWorkspace,
    join_plan: JoinPlan,
) -> tuple[str | None, str | None, BlockingAmbiguityPayload | None]:
    """Differentiate blocking HITL ambiguities from non-blocking advisory clarification hints."""
    # 1. Blocking Ambiguity: unapproved cross-database join needed for the query
    for edge in join_plan.join_edges:
        if edge.is_cross_source and edge.status == JoinStatus.PENDING:
            q_id = f"q_cross_link_{edge.source_table}"
            prompt_text = (
                f"Joining `{edge.source_table}.{edge.source_column}` to `{edge.target_table}.{edge.target_column}` "
                "crosses distinct data sources and is currently pending approval. Approve this cross-source join?"
            )
            payload = BlockingAmbiguityPayload(
                type="cross_source_join",
                question=question,
                clarification_prompt=prompt_text,
                suggested_options=[
                    f"Approve cross-source join ({edge.source_table} -> {edge.target_table})",
                    "Keep data sources separate",
                ],
                question_id=q_id,
            )
            return prompt_text, "blocking_modal", payload

    # 2. Blocking Ambiguity: explicit unanswered Priority-1 interview question flagged as blocking category
    req_set = set(join_plan.required_tables or [])
    for iq in getattr(workspace, "interview_questions", []) or []:
        if (
            not iq.answered
            and iq.priority == 1
            and iq.category in ("join_confirm", "metric_conflict")
            and (not iq.table_name or iq.table_name in req_set)
        ):
            payload = BlockingAmbiguityPayload(
                type=iq.category,
                question=question,
                clarification_prompt=iq.question_text,
                suggested_options=list(iq.suggested_options or []),
                question_id=iq.id,
            )
            return iq.question_text, "blocking_modal", payload

    # 3. Non-Blocking Inline Hint: advisory status code / filter best-practice suggestion
    advisory_hint = check_just_in_time_clarification(question, workspace)
    if advisory_hint:
        return advisory_hint, "non_blocking_inline", None

    return None, None, None


def repair_query_with_guard(
    question: str,
    failed_query: str,
    error_msg: str,
    workspace: OKFWorkspace,
    join_plan: JoinPlan,
    is_dax_target: bool,
    llm_router: Any | None,
    okf_store: Any | None = None,
    matched_measures: list[SemanticMeasure] | None = None,
    what_if_overrides: dict[str, float] | None = None,
    rag_bundle: dict[str, list[dict[str, Any]]] | None = None,
    current_tables: list[str] | None = None,
) -> tuple[str, str, str, list[str], JoinPlan, bool]:
    """Repair a failed query with Schema Expansion, Pre-Flight AST Firewall Guard, and Measure Preservation.

    Returns (repaired_query, compiler_mode, compiler_detail, expanded_tables, updated_join_plan, ast_fallback_used).
    """
    measures = list(matched_measures or [])
    overrides = normalize_what_if_overrides(what_if_overrides)
    expanded_tables = expand_schema_from_error(
        question=question,
        failed_query=failed_query,
        error_msg=error_msg,
        workspace=workspace,
        current_tables=list(current_tables or join_plan.required_tables or []),
    )
    updated_plan = join_plan
    if okf_store is not None and set(expanded_tables) != set(join_plan.required_tables or []):
        updated_plan = okf_store.get_join_plan(workspace, expanded_tables)

    ast_fallback_used = False

    if llm_router is not None and getattr(llm_router, "is_configured", lambda: False)():
        schema_str = build_adaptive_schema_context(
            workspace=workspace,
            join_plan=updated_plan,
            rag_bundle=rag_bundle,
            extra_tier1_tables=expanded_tables,
            compact_only=True,
        )
        blocked_tbls = workspace.blocked_table_names()
        if blocked_tbls:
            schema_str += f"\nBlocked Tables (FORBIDDEN): {', '.join(sorted(blocked_tbls))}"

        fallback_sys = (
            "You are Data_Talk's Self-Healing SQL/DAX Critic. Inspect the failed query and engine/AST firewall error, "
            'check the valid OKF table/column names, maintain strict read-only SELECT/WITH semantics, and return JSON: {"repaired_query": "..."}'
        )
        fallback_usr = (
            f"Question: {question}\nFailed Query:\n{failed_query}\n"
            f"Engine Error: {error_msg}\nValid Schema:\n{schema_str}"
        )
        sys_prompt, usr_prompt = PromptManager.render(
            "self_healing_critic",
            fallback_system=fallback_sys,
            fallback_user=fallback_usr,
            question=question,
            failed_query=failed_query,
            error_msg=error_msg,
            schema_str=schema_str,
        )
        try:
            res = llm_router.complete_json(
                purpose="self_healing_critic",
                system_prompt=sys_prompt,
                user_prompt=usr_prompt,
            )
            candidate = str(res.get("repaired_query") or "").strip()
            if candidate:
                if not is_dax_target:
                    PrivacyFirewall.validate_sql_read_only(
                        candidate,
                        dialect=workspace.dialect,
                        blocked_tables=workspace.blocked_table_names(),
                        blocked_columns=workspace.blocked_column_names(),
                    )
                model_lbl = getattr(llm_router, "model", "LLM")
                return (
                    candidate,
                    "llm",
                    f"LLM Self-Healed ({model_lbl})",
                    expanded_tables,
                    updated_plan,
                    False,
                )
        except SecurityViolationError as sec_exc:
            _LOGGER.warning("LLM self-healing candidate failed pre-flight AST guard: %s", sec_exc)
            ast_fallback_used = True
        except Exception as exc:
            _LOGGER.warning("LLM self-healing failed: %s", exc)

    # Deterministic column alias repair (guarded against unsafe writes/blocked tables too)
    repaired = failed_query
    col_repairs = {
        r"\brevenue\b": "rev_net_usd",
        r"\bsales\b": "rev_net_usd",
        r"\bprofit\b": "profit_usd",
        r"\bstatus\b": "ord_stat",
        r"\bdiscount\b": "disc_amt",
        r"\bregion\b": "region_code",
        r"\btier\b": "cust_tier",
    }
    for pattern, replacement in col_repairs.items():
        repaired = re.sub(pattern, replacement, repaired, flags=re.IGNORECASE)

    if repaired and repaired != failed_query:
        try:
            if not is_dax_target:
                PrivacyFirewall.validate_sql_read_only(
                    repaired,
                    dialect=workspace.dialect,
                    blocked_tables=workspace.blocked_table_names(),
                    blocked_columns=workspace.blocked_column_names(),
                )
            return (
                repaired,
                "heuristic",
                "Heuristic Column Alias Repair",
                expanded_tables,
                updated_plan,
                ast_fallback_used,
            )
        except SecurityViolationError:
            ast_fallback_used = True

    # Forensic Fix #2: Always preserve matched_measures and what_if_overrides during heuristic fallback!
    fallback_query = (
        build_dax_equivalent(question, workspace, measures)
        if is_dax_target
        else compile_deterministic_sql(question, workspace, updated_plan, measures, overrides or None)
    )
    return (
        fallback_query,
        "heuristic",
        "Deterministic Heuristic AST-Safe Fallback" if ast_fallback_used else "Heuristic Self-Healing Fallback",
        expanded_tables,
        updated_plan,
        ast_fallback_used,
    )


# ==============================================================================
# 11 Pure-Python Traced LangGraph Nodes & Conditional Routers
# ==============================================================================


@trace_graph_node("ingest_and_resolve_followup")
def ingest_and_resolve_followup(state: QueryGraphState, config: RunnableConfig) -> dict[str, Any]:
    """Node 1: Hydrate categorical samples once, normalize What-If levers, and resolve follow-ups."""
    emit_stream_event({"type": "stage", "stage": "intent", "label": "Resolving follow-up & intent..."})
    ctx = get_graph_context(config)
    effective_force_heuristic = bool(state.force_heuristic or ctx.force_heuristic)
    effective_llm = None if effective_force_heuristic else ctx.llm_router

    # Forensic Fix #5: single in-memory hydration on ctx.workspace without overwriting from disk
    ensure_categorical_samples(ctx.workspace, ctx.connector)
    q_clean = normalize_prompt_text(state.question.strip())

    resolved_q, is_follow_up, follow_up_ctx, inherited_tables = resolve_follow_up_context(
        q_clean, ctx.conversation_history, ctx.workspace, effective_llm
    )
    resolved_q = normalize_prompt_text(resolved_q)

    # Forensic Fix #3: strip zero-value What-If sliders before prompt shock extraction
    clean_overrides = normalize_what_if_overrides(state.what_if_overrides)
    if not clean_overrides:
        clean_overrides = extract_what_if_from_prompt(resolved_q)

    raw_cat_matches = detect_categorical_value_matches(resolved_q, ctx.workspace)
    cat_records = [CategoricalMatchRecord(**m) for m in raw_cat_matches]
    if raw_cat_matches and not follow_up_ctx:
        top_m = raw_cat_matches[0]
        if top_m.get("matched_phrase", "").lower() != top_m.get("value", "").lower():
            follow_up_ctx = (
                f"Matched '{top_m['matched_phrase']}' → {top_m['table']}.{top_m['column']} = '{top_m['value']}'"
            )

    return {
        "q_clean": q_clean,
        "resolved_question": resolved_q,
        "what_if_overrides": clean_overrides or None,
        "is_follow_up": is_follow_up,
        "follow_up_context": follow_up_ctx,
        "inherited_tables": inherited_tables,
        "cat_value_matches": cat_records,
        "is_overview": is_dataset_overview_question(resolved_q),
        "attempt": 1,
        "last_error": None,
        "execution_succeeded": False,
        "error_diagnostic": None,
        "interrupted": False,
        "interrupt_payload": None,
    }


def route_after_followup(
    state: QueryGraphState,
) -> Literal["dataset_overview", "check_verified_memory", "retrieve_okf_and_rag"]:
    """Conditional edge 1: overview -> Verified Query Memory -> full semantic retrieval."""
    if state.is_overview:
        return "dataset_overview"
    if not state.force_recompile:
        return "check_verified_memory"
    return "retrieve_okf_and_rag"


@trace_graph_node("dataset_overview")
def dataset_overview(state: QueryGraphState, config: RunnableConfig) -> dict[str, Any]:
    """Node 2: Summarize workspace catalog and return overview response."""
    emit_stream_event({"type": "stage", "stage": "overview", "label": "Summarizing workspace schema..."})
    ctx = get_graph_context(config)
    effective_llm = None if (state.force_heuristic or ctx.force_heuristic) else ctx.llm_router

    resp_meta, df_ov = generate_dataset_overview_payload(state.resolved_question, ctx.workspace, effective_llm)
    resp_meta["resolved_question"] = state.resolved_question
    resp_meta["is_follow_up"] = state.is_follow_up
    resp_meta["follow_up_context"] = state.follow_up_context
    if not resp_meta.get("suggested_follow_ups"):
        empty_plan = JoinPlan([], [], [], [], False, [], [], "")
        resp_meta["suggested_follow_ups"] = generate_suggested_follow_ups(
            df_ov, state.resolved_question, ctx.workspace, empty_plan
        )
    resp_meta["drill_down_dimensions"] = []
    ctx.dataframe_registry["result"] = df_ov

    return {
        "response_dict": resp_meta,
        "result_df_key": "result",
        "execution_succeeded": True,
        "row_count": len(df_ov),
    }


@trace_graph_node("check_verified_memory")
def check_verified_memory(
    state: QueryGraphState, config: RunnableConfig
) -> Command[Literal["retrieve_okf_and_rag", "__end__"]]:
    """Node 3: Execute Verified Query Memory recipe if matched; otherwise fall through to RAG+OKF."""
    emit_stream_event({"type": "stage", "stage": "memory", "label": "Checking Verified Query Memory..."})
    ctx = get_graph_context(config)
    effective_llm = None if (state.force_heuristic or ctx.force_heuristic) else ctx.llm_router
    matched_vq = find_matching_verified_query(state.resolved_question, ctx.workspace)

    if matched_vq is not None:
        try:
            df_vq = ctx.connector.execute_query(
                matched_vq.executed_query,
                max_rows=10000,
                blocked_tables=ctx.workspace.blocked_table_names(),
                blocked_columns=ctx.workspace.blocked_column_names(),
            )
            matched_vq.execution_count += 1
            # Persist counter on initial hit and every 25th hit to keep hot Verified Memory <20ms on Windows NTFS
            if matched_vq.execution_count == 1 or matched_vq.execution_count % 25 == 0:
                try:
                    ctx.okf_store.save(ctx.workspace)
                except Exception:
                    pass

            # Forensic Fix #4: infer involved_tables fast from canonical SQL identifiers (avoids ~6ms fuzzy scan)
            inferred_tables = [
                t.name
                for t in ctx.workspace.tables
                if not t.is_blocked
                and re.search(rf"(?<![A-Za-z0-9_]){re.escape(t.name)}(?![A-Za-z0-9_])", matched_vq.executed_query, re.I)
            ]
            if not inferred_tables:
                inferred_tables = detect_relevant_tables(
                    f"{matched_vq.executed_query} {state.resolved_question}", ctx.workspace, {}
                )
            summary_text = build_executive_summary(
                state.resolved_question, df_vq, [], [], effective_llm, ctx.workspace
            )
            empty_plan = JoinPlan(inferred_tables, [], [], [], False, [], [], "")
            follow_ups = generate_suggested_follow_ups(df_vq, state.resolved_question, ctx.workspace, empty_plan)
            drill_downs = generate_drill_down_options(df_vq, ctx.workspace, inferred_tables)
            ctx.dataframe_registry["result"] = df_vq

            heal_line = f"⚡ Executed canonical recipe directly from Verified Query Memory (ID: {matched_vq.id})."
            resp_dict = {
                "user_question": state.q_clean,
                "resolved_question": state.resolved_question,
                "success": True,
                "executed_query": matched_vq.executed_query,
                "compiled_dax_equivalent": matched_vq.dax_equivalent or "",
                "query_dialect": matched_vq.query_dialect or getattr(ctx.connector, "dialect", "duckdb"),
                "attempts_used": 1,
                "healing_trace": [heal_line],
                "involved_tables": inferred_tables,
                "join_clauses_used": [],
                "fan_trap_mitigated": False,
                "measures_applied": [],
                "business_rules_applied": [],
                "clarification_suggestion": None,
                "clarification_severity": None,
                "executive_summary": summary_text,
                "recommended_chart_type": matched_vq.chart_type or "bar",
                "compiler_mode": "verified_memory",
                "compiler_detail": f"Verified Query Memory (Matched: '{matched_vq.question_pattern}')",
                "is_follow_up": state.is_follow_up,
                "follow_up_context": state.follow_up_context,
                "suggested_follow_ups": follow_ups,
                "drill_down_dimensions": drill_downs,
                "verified_query_id": matched_vq.id,
                "from_verified_memory": True,
            }
            return Command(
                update={
                    "verified_query_id": matched_vq.id,
                    "from_verified_memory": True,
                    "target_tables": inferred_tables,
                    "current_query": matched_vq.executed_query,
                    "execution_succeeded": True,
                    "row_count": len(df_vq),
                    "result_df_key": "result",
                    "healing_trace": [heal_line],
                    "response_dict": resp_dict,
                },
                goto=END,
            )
        except Exception as vq_err:
            _LOGGER.warning("Verified query %s execution failed: %s; falling back to compiler", matched_vq.id, vq_err)

    return Command(update={"from_verified_memory": False}, goto="retrieve_okf_and_rag")


@trace_graph_node("retrieve_okf_and_rag")
def retrieve_okf_and_rag(state: QueryGraphState, config: RunnableConfig) -> dict[str, Any]:
    """Node 4: Run hybrid RAG retrieval, resolve OKF Steiner join graph, match measures, and classify ambiguity."""
    emit_stream_event({"type": "stage", "stage": "semantic", "label": "Resolving OKF Steiner join graph & RAG..."})
    ctx = get_graph_context(config)

    rag_bundle = ctx.rag_store.retrieve_context_bundle(state.resolved_question, top_k_per_collection=5)
    target_tables = detect_relevant_tables(state.resolved_question, ctx.workspace, rag_bundle)

    if state.is_follow_up:
        conflicting_pairs = [("faculty", "students"), ("orders", "returns")]
        for t_inh in state.inherited_tables:
            if t_inh not in target_tables and ctx.workspace.get_table(t_inh):
                is_conflicted = any(
                    (t_inh == p[0] and p[1] in target_tables) or (t_inh == p[1] and p[0] in target_tables)
                    for p in conflicting_pairs
                )
                if not is_conflicted:
                    target_tables.append(t_inh)

    raw_join_plan = ctx.okf_store.get_join_plan(ctx.workspace, target_tables)
    matched_measures = match_measures(state.resolved_question, ctx.workspace)
    for m in matched_measures:
        for mt in m.involved_tables:
            if mt not in target_tables and ctx.workspace.get_table(mt):
                target_tables.append(mt)

    out_of_domain = bool(not target_tables and not matched_measures and ctx.workspace.active_tables())
    if not out_of_domain and len(target_tables) > len(raw_join_plan.required_tables):
        raw_join_plan = ctx.okf_store.get_join_plan(ctx.workspace, target_tables)

    applied_rules = [
        f"{r.term}: {r.definition}" + (f" [Filter: {r.sql_filter_hint}]" if r.sql_filter_hint else "")
        for r in ctx.workspace.business_rules
    ]

    hint, severity, blocking_payload = _classify_clarification_severity(
        state.resolved_question, ctx.workspace, raw_join_plan
    )
    if severity == "non_blocking_inline" and hint:
        emit_stream_event(
            {
                "type": "clarification_hint",
                "stage": "clarification_hint",
                "severity": "non_blocking_inline",
                "message": hint,
            }
        )

    return {
        "rag_bundle": rag_bundle,
        "target_tables": target_tables,
        "join_plan": JoinPlanSnapshot.from_join_plan(raw_join_plan),
        "matched_measures": matched_measures,
        "applied_rules": applied_rules,
        "out_of_domain": out_of_domain,
        "clarification_suggestion": hint,
        "clarification_severity": severity,  # type: ignore[dict-item]
        "blocking_ambiguity_payload": blocking_payload,
    }


def route_after_retrieval(
    state: QueryGraphState, config: RunnableConfig
) -> Literal["build_out_of_domain", "clarification_interrupt", "compile_initial_query"]:
    """Conditional edge 2: route to Out-of-Domain, blocking HITL interrupt, or SQL/DAX compilation."""
    if state.out_of_domain:
        return "build_out_of_domain"
    ctx = get_graph_context(config)
    allow_int = bool(state.allow_interrupts or ctx.allow_interrupts)
    if (
        allow_int
        and state.clarification_severity == "blocking_modal"
        and state.blocking_ambiguity_payload is not None
    ):
        return "clarification_interrupt"
    return "compile_initial_query"


@trace_graph_node("build_out_of_domain")
def build_out_of_domain(state: QueryGraphState, config: RunnableConfig) -> dict[str, Any]:
    """Node 5: Build graceful Out-of-Domain ErrorDiagnostic and active table catalog advisory."""
    emit_stream_event({"type": "stage", "stage": "guardrail", "label": "No matching schema entities found"})
    ctx = get_graph_context(config)
    active_names = [t.name for t in ctx.workspace.active_tables()]
    err_msg = f"No matching tables found in workspace for query: '{state.q_clean}'"
    diag = classify_error_diagnostic(err_msg, "build_out_of_domain", 1, MAX_RETRIES, ctx.workspace)

    resp_dict = {
        "user_question": state.q_clean,
        "resolved_question": state.resolved_question,
        "success": False,
        "executed_query": "-- No matching tables found in active workspace",
        "query_dialect": ctx.workspace.dialect,
        "involved_tables": [],
        "join_clauses_used": [],
        "measures_applied": [],
        "executive_summary": (
            f"⚠️ **Requested Data Not Found in Workspace**\n\n"
            f"The active workspace (`{ctx.workspace.workspace_name}`) does not contain tables, columns, or measures relating to **\"{state.q_clean}\"**.\n\n"
            f"**Available tables in this workspace:**\n"
            + "\n".join(f"- `{t}`" for t in active_names[:10])
            + (f"\n- ...and {len(active_names) - 10} more tables" if len(active_names) > 10 else "")
            + "\n\n💡 *Tip: Upload a CSV/Excel file or connect another database from the 'Connect Data' tab to query this information.*"
        ),
        "recommended_chart_type": "table",
        "error_message": err_msg,
        "error_diagnostic": diag.model_dump(),
        "is_follow_up": state.is_follow_up,
        "follow_up_context": state.follow_up_context,
    }
    return {
        "response_dict": resp_dict,
        "error_diagnostic": diag,
        "execution_succeeded": False,
        "result_df_key": "",
    }


@trace_graph_node("clarification_interrupt")
def clarification_interrupt(state: QueryGraphState, config: RunnableConfig) -> dict[str, Any]:
    """Node 6: Pause execution via interrupt() for blocking ambiguities and persist resumed answer into OKF+RAG."""
    emit_stream_event({"type": "stage", "stage": "hitl", "label": "Waiting for critical disambiguation..."})
    ctx = get_graph_context(config)
    payload_obj = state.blocking_ambiguity_payload or BlockingAmbiguityPayload(
        type="clarification",
        question=state.resolved_question or state.question,
        clarification_prompt=state.clarification_suggestion or "Please clarify how to proceed.",
    )
    resume_val = interrupt(payload_obj.model_dump())

    answer_text = ""
    save_rule = True
    question_id = payload_obj.question_id
    if isinstance(resume_val, dict):
        answer_text = str(resume_val.get("answer") or "").strip()
        save_rule = bool(resume_val.get("save_rule", True))
        question_id = str(resume_val.get("question_id") or question_id)
    elif resume_val is not None:
        answer_text = str(resume_val).strip()

    updated_plan = state.join_plan.to_join_plan()
    heal_lines: list[str] = []

    if answer_text:
        if save_rule:
            recorded = None
            if question_id:
                recorded = InterviewerAgent.record_answer(
                    ctx.workspace, ctx.rag_store, question_id, answer_text
                )
            if recorded is None:
                rule_id = f"rule_hitl_{abs(hash(answer_text)) % 100000}"
                rule = BusinessRule(
                    id=rule_id,
                    category=payload_obj.type or "domain_logic",
                    term=payload_obj.type or "clarification",
                    definition=answer_text,
                    applies_to_tables=list(state.target_tables[:4]),
                    source="chat_correction",
                )
                ctx.workspace.business_rules = [
                    r for r in ctx.workspace.business_rules if r.id != rule_id
                ] + [rule]
                ctx.rag_store.upsert(
                    collection="business_rules",
                    doc_id=rule_id,
                    content=f"Clarification: {payload_obj.clarification_prompt}\nAnswer: {answer_text}",
                    metadata={"category": payload_obj.type, "source": "chat_hitl"},
                )

        if payload_obj.type == "cross_source_join":
            ans_low = answer_text.lower()
            approve = any(w in ans_low for w in ("yes", "approve", "connect", "link", "confirm"))
            reject = any(w in ans_low for w in ("no", "reject", "separate", "ignore"))
            if approve or reject:
                new_status = JoinStatus.APPROVED if approve else JoinStatus.REJECTED
                for rel in ctx.workspace.relationships:
                    if rel.is_cross_source and rel.status == JoinStatus.PENDING:
                        rel.status = new_status
                updated_plan = ctx.okf_store.get_join_plan(ctx.workspace, state.target_tables)

        try:
            ctx.okf_store.save(ctx.workspace)
        except Exception:
            pass
        heal_lines.append(f"Clarification applied: {answer_text}")

    applied_rules = [
        f"{r.term}: {r.definition}" + (f" [Filter: {r.sql_filter_hint}]" if r.sql_filter_hint else "")
        for r in ctx.workspace.business_rules
    ]
    return {
        "applied_rules": applied_rules,
        "join_plan": JoinPlanSnapshot.from_join_plan(updated_plan),
        "blocking_ambiguity_payload": None,
        "clarification_suggestion": f"Clarified: {answer_text}" if answer_text else None,
        "clarification_severity": None,
        "interrupted": False,
        "interrupt_payload": None,
        "healing_trace": heal_lines,
    }


@trace_graph_node("compile_initial_query")
def compile_initial_query(state: QueryGraphState, config: RunnableConfig) -> dict[str, Any]:
    """Node 7: Compile governed SQL + DAX with Adaptive Two-Tier Schema Pruning and Fan-Trap write-back."""
    emit_stream_event({"type": "stage", "stage": "compile", "label": "Compiling governed SQL & DAX..."})
    ctx = get_graph_context(config)
    effective_llm = None if (state.force_heuristic or ctx.force_heuristic) else ctx.llm_router
    is_dax_target = getattr(ctx.connector, "dialect", "duckdb") == "dax"

    raw_plan = state.join_plan.to_join_plan()
    sql_or_dax, dax_equiv, mode, detail = generate_query(
        question=state.resolved_question,
        workspace=ctx.workspace,
        join_plan=raw_plan,
        matched_measures=state.matched_measures,
        rag_bundle=state.rag_bundle,
        is_dax_target=is_dax_target,
        llm_router=effective_llm,
        what_if_overrides=state.what_if_overrides,
    )

    # Forensic Fix #1: preserve in-place Case B2 has_fan_trap=True mutation in strict JoinPlanSnapshot
    synced_snapshot = JoinPlanSnapshot.from_join_plan(raw_plan)
    return {
        "is_dax_target": is_dax_target,
        "attempt": 1,
        "current_query": sql_or_dax,
        "dax_equivalent": dax_equiv,
        "compiler_mode": mode,
        "compiler_detail": detail,
        "join_plan": synced_snapshot,
    }


@trace_graph_node("execute_sandbox")
def execute_sandbox(
    state: QueryGraphState, config: RunnableConfig
) -> Command[Literal["post_process_and_summarize", "critic_repair_query", "handle_exhausted_failure"]]:
    """Node 8: Execute compiled query in read-only sandbox; route to summary, repair, or error."""
    emit_stream_event(
        {"type": "stage", "stage": "execute", "label": f"Executing attempt {state.attempt} inside read-only AST sandbox..."}
    )
    ctx = get_graph_context(config)
    try:
        if not (state.current_query or "").strip():
            raise ValueError("Compiled query is empty.")
        df = ctx.connector.execute_query(
            state.current_query,
            max_rows=10000,
            blocked_tables=ctx.workspace.blocked_table_names(),
            blocked_columns=ctx.workspace.blocked_column_names(),
        )
        ctx.dataframe_registry["result"] = df
        heal_update = (
            [f"Attempt {state.attempt} succeeded ({len(df):,} rows returned)."]
            if state.attempt > 1
            else []
        )
        return Command(
            update={
                "execution_succeeded": True,
                "last_error": None,
                "row_count": len(df),
                "result_df_key": "result",
                "error_diagnostic": None,
                "healing_trace": heal_update,
            },
            goto="post_process_and_summarize",
        )
    except Exception as exc:
        err_str = str(exc)
        diag = classify_error_diagnostic(err_str, "execute_sandbox", state.attempt, MAX_RETRIES, ctx.workspace)
        next_node: Literal["critic_repair_query", "handle_exhausted_failure"] = (
            "critic_repair_query" if state.attempt < MAX_RETRIES else "handle_exhausted_failure"
        )
        return Command(
            update={
                "execution_succeeded": False,
                "last_error": err_str,
                "error_diagnostic": diag,
            },
            goto=next_node,
        )


@trace_graph_node("critic_repair_query")
def critic_repair_query(state: QueryGraphState, config: RunnableConfig) -> dict[str, Any]:
    """Node 9 (Loop Edge): Expand schema, run AST-guarded Self-Healing Critic, preserve measures."""
    next_attempt = state.attempt + 1
    emit_stream_event(
        {
            "type": "stage",
            "stage": "heal",
            "label": f"Self-healing attempt {next_attempt} against OKF & AST firewall...",
        }
    )
    ctx = get_graph_context(config)
    effective_llm = None if (state.force_heuristic or ctx.force_heuristic) else ctx.llm_router
    trace_line = (
        f"Attempt {state.attempt} failed with error: `{state.last_error or ''}`. "
        "Self-healing query against OKF schema..."
    )

    raw_plan = state.join_plan.to_join_plan()
    repaired_q, mode, detail, expanded_tables, updated_plan, ast_fallback = repair_query_with_guard(
        question=state.resolved_question,
        failed_query=state.current_query,
        error_msg=state.last_error or "",
        workspace=ctx.workspace,
        join_plan=raw_plan,
        is_dax_target=state.is_dax_target,
        llm_router=effective_llm,
        okf_store=ctx.okf_store,
        matched_measures=state.matched_measures,
        what_if_overrides=state.what_if_overrides,
        rag_bundle=state.rag_bundle,
        current_tables=state.target_tables,
    )

    heal_lines = [trace_line]
    if ast_fallback:
        heal_lines.append("Pre-flight AST Firewall Guard intercepted unsafe repair; fell back deterministically.")

    return {
        "attempt": next_attempt,
        "current_query": repaired_q,
        "compiler_mode": mode,
        "compiler_detail": detail,
        "target_tables": expanded_tables,
        "join_plan": JoinPlanSnapshot.from_join_plan(updated_plan),
        "healing_trace": heal_lines,
    }


@trace_graph_node("post_process_and_summarize")
def post_process_and_summarize(state: QueryGraphState, config: RunnableConfig) -> dict[str, Any]:
    """Node 10: Upsert clean query to RAG, synthesize summary, follow-ups, drill-downs, and final response."""
    emit_stream_event({"type": "stage", "stage": "summarize", "label": "Synthesizing executive summary..."})
    ctx = get_graph_context(config)
    effective_llm = None if (state.force_heuristic or ctx.force_heuristic) else ctx.llm_router
    df = ctx.dataframe_registry.get(state.result_df_key or "result", pd.DataFrame())
    raw_plan = state.join_plan.to_join_plan()
    dialect = getattr(ctx.connector, "dialect", ctx.workspace.dialect)

    is_clean_query = not re.search(r"'(?:Unknown|dummy|other|n/a)'\s+as\s+", state.current_query, re.IGNORECASE)
    if is_clean_query and not state.healing_trace:
        try:
            ctx.rag_store.upsert(
                collection="few_shot_queries",
                doc_id=f"chat_q_{abs(hash(state.resolved_question)) % 100000}",
                content=(
                    f"User Question: {state.resolved_question}\n"
                    f"Verified Query ({dialect}):\n{state.current_query}\n"
                    f"DAX Equivalent:\n{state.dax_equivalent}"
                ),
                metadata={"question": state.resolved_question, "dialect": dialect},
            )
        except Exception:
            pass

    summary_text = build_executive_summary(
        state.resolved_question,
        df,
        state.matched_measures,
        state.applied_rules,
        effective_llm,
        ctx.workspace,
    )
    follow_ups = generate_suggested_follow_ups(df, state.resolved_question, ctx.workspace, raw_plan)
    drill_downs = generate_drill_down_options(df, ctx.workspace, raw_plan.required_tables)

    applied_measures = [m.name for m in state.matched_measures]
    if state.what_if_overrides:
        for wk, wv in state.what_if_overrides.items():
            applied_measures.append(f"What-If ({wk}: {wv*100:+.1f}%)")

    resp_dict = {
        "user_question": state.q_clean,
        "resolved_question": state.resolved_question,
        "success": True,
        "executed_query": state.current_query,
        "compiled_dax_equivalent": state.dax_equivalent,
        "query_dialect": dialect,
        "attempts_used": state.attempt,
        "healing_trace": list(state.healing_trace),
        "involved_tables": raw_plan.required_tables,
        "join_clauses_used": raw_plan.sql_join_clauses,
        "fan_trap_mitigated": raw_plan.has_fan_trap,
        "measures_applied": applied_measures,
        "business_rules_applied": list(state.applied_rules[:5]),
        "clarification_suggestion": state.clarification_suggestion,
        "clarification_severity": state.clarification_severity,
        "executive_summary": summary_text,
        "recommended_chart_type": "bar",
        "compiler_mode": state.compiler_mode,
        "compiler_detail": state.compiler_detail,
        "is_follow_up": state.is_follow_up,
        "follow_up_context": state.follow_up_context,
        "suggested_follow_ups": follow_ups,
        "drill_down_dimensions": drill_downs,
    }
    return {
        "response_dict": resp_dict,
        "error_diagnostic": None,
    }


@trace_graph_node("handle_exhausted_failure")
def handle_exhausted_failure(state: QueryGraphState, config: RunnableConfig) -> dict[str, Any]:
    """Node 11: Emit structured redacted ErrorDiagnostic after all 3 attempts fail."""
    emit_stream_event({"type": "stage", "stage": "error", "label": "Diagnostic error report ready"})
    ctx = get_graph_context(config)
    raw_plan = state.join_plan.to_join_plan()
    exhausted_line = f"All {MAX_RETRIES} attempts exhausted. Final error: {state.last_error or ''}"
    final_trace = [*list(state.healing_trace), exhausted_line]
    diag = state.error_diagnostic or classify_error_diagnostic(
        state.last_error or "Unknown query failure",
        "execute_sandbox",
        MAX_RETRIES,
        MAX_RETRIES,
        ctx.workspace,
    )
    redacted_error = PrivacyFirewall.redact_text(state.last_error or "")

    resp_dict = {
        "user_question": state.q_clean,
        "resolved_question": state.resolved_question,
        "success": False,
        "executed_query": state.current_query,
        "compiled_dax_equivalent": state.dax_equivalent,
        "query_dialect": getattr(ctx.connector, "dialect", ctx.workspace.dialect),
        "attempts_used": MAX_RETRIES,
        "healing_trace": final_trace,
        "involved_tables": raw_plan.required_tables,
        "join_clauses_used": raw_plan.sql_join_clauses,
        "fan_trap_mitigated": raw_plan.has_fan_trap,
        "measures_applied": [m.name for m in state.matched_measures],
        "business_rules_applied": list(state.applied_rules[:5]),
        "clarification_suggestion": state.clarification_suggestion,
        "clarification_severity": state.clarification_severity,
        "error_message": redacted_error,
        "error_diagnostic": diag.model_dump(),
        "compiler_mode": state.compiler_mode,
        "compiler_detail": state.compiler_detail,
        "is_follow_up": state.is_follow_up,
        "follow_up_context": state.follow_up_context,
        "suggested_follow_ups": [
            "Show dataset overview and active tables",
            "What are the top KPIs?",
        ],
    }
    return {
        "healing_trace": [exhausted_line],
        "response_dict": resp_dict,
        "error_diagnostic": diag,
    }
