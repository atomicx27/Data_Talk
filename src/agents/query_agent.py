"""Agentic Query Compiler, 3x Self-Healing Execution Loop & Lineage Engine Facade.

Delegates end-to-end "Talk to Data" orchestration to the modular LangGraph StateGraph
(`src/agents/graph/query_graph.py`) and pure-Python compilers (`src/agents/compilers/`),
while preserving 100% backwards-compatible `QueryAgent` & `QueryResponse` contracts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd
from pydantic import BaseModel, Field

from src.agents.compilers.deterministic_compiler import (
    build_dax_equivalent,
    clean_agg_alias,
    compile_deterministic_sql,
    infer_agg_function,
    infer_sort_direction,
)
from src.agents.compilers.semantic_helpers import (
    check_just_in_time_clarification,
    find_matching_verified_query,
    generate_query,
    improve_user_prompt_payload,
    llm_compile_query,
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
    fuzzy_matches_word,
    match_measures,
    normalize_prompt_text,
)
from src.agents.compilers.thread_analyzer import (
    analyze_chart_dataframe,
    answer_chart_thread_question,
    needs_warehouse_sql_for_thread,
    should_render_chart,
)
from src.agents.graph.nodes import MAX_RETRIES, repair_query_with_guard
from src.agents.graph.query_graph import run_query_graph
from src.agents.graph.state import (
    BlockingAmbiguityPayload,
    ErrorDiagnostic,
    NodeTelemetryRecord,
)
from src.agents.llm_router import LLMRouter
from src.knowledge.okf_models import OKFWorkspace, SemanticMeasure, VerifiedQuery
from src.knowledge.okf_store import JoinPlan, OKFStore
from src.knowledge.rag_store import RAGStore

if TYPE_CHECKING:
    from src.connectors.base import BaseConnector


class QueryResponse(BaseModel):
    """Structured output from QueryAgent.answer_question()."""

    user_question: str
    resolved_question: str = ""
    success: bool
    executed_query: str = ""
    compiled_dax_equivalent: str = ""
    query_dialect: str = "duckdb"
    attempts_used: int = 1
    healing_trace: list[str] = Field(default_factory=list)
    involved_tables: list[str] = Field(default_factory=list)
    join_clauses_used: list[str] = Field(default_factory=list)
    fan_trap_mitigated: bool = False
    measures_applied: list[str] = Field(default_factory=list)
    business_rules_applied: list[str] = Field(default_factory=list)
    clarification_suggestion: str | None = None
    clarification_severity: str | None = None
    executive_summary: str = ""
    error_message: str = ""
    error_diagnostic: ErrorDiagnostic | None = None
    recommended_chart_type: str = "bar"
    compiler_mode: str = "heuristic"
    compiler_detail: str = "Deterministic Heuristic Engine"
    is_follow_up: bool = False
    follow_up_context: str | None = None
    suggested_follow_ups: list[str] = Field(default_factory=list)
    drill_down_dimensions: list[dict[str, str]] = Field(default_factory=list)
    verified_query_id: str | None = None
    from_verified_memory: bool = False
    node_trace: list[str] = Field(default_factory=list)
    node_telemetry: list[NodeTelemetryRecord] = Field(default_factory=list)
    total_latency_ms: float = 0.0
    interrupted: bool = False
    interrupt_payload: BlockingAmbiguityPayload | None = None
    thread_id: str | None = None


class QueryAgent:
    """Backwards-compatible public facade over the modular LangGraph QueryAgent engine."""

    MAX_RETRIES = MAX_RETRIES

    @classmethod
    def answer_question(
        cls,
        question: str,
        workspace: OKFWorkspace,
        okf_store: OKFStore,
        rag_store: RAGStore,
        connector: "BaseConnector",
        llm_router: LLMRouter | None = None,
        what_if_overrides: dict[str, float] | None = None,
        conversation_history: list[dict[str, Any]] | None = None,
        force_recompile: bool = False,
        allow_interrupts: bool = False,
        force_heuristic: bool = False,
        thread_id: str | None = None,
        checkpointer: Any | None = None,
    ) -> tuple[QueryResponse, pd.DataFrame]:
        """Execute a natural language analytical query via LangGraph StateGraph orchestration."""
        result = run_query_graph(
            question=question,
            workspace=workspace,
            okf_store=okf_store,
            rag_store=rag_store,
            connector=connector,
            llm_router=llm_router,
            what_if_overrides=what_if_overrides,
            conversation_history=conversation_history,
            force_recompile=force_recompile,
            allow_interrupts=allow_interrupts,
            force_heuristic=force_heuristic,
            thread_id=thread_id,
            checkpointer=checkpointer,
        )
        payload = dict(result.get("response_dict") or {})
        payload.setdefault("user_question", question)
        payload.setdefault("success", False)
        payload["interrupted"] = bool(result.get("interrupted", False))
        payload["interrupt_payload"] = result.get("interrupt_payload")
        payload["thread_id"] = result.get("thread_id")
        return QueryResponse.model_validate(payload), result.get("df", pd.DataFrame())

    # ------------------------------------------------------------------
    # Backwards-Compatible Helper Delegates
    # ------------------------------------------------------------------
    @classmethod
    def _extract_what_if_from_prompt(cls, question: str) -> dict[str, float]:
        return extract_what_if_from_prompt(question)

    @staticmethod
    def _normalize_prompt_text(text: str) -> str:
        return normalize_prompt_text(text)

    @classmethod
    def _ensure_categorical_samples(
        cls,
        workspace: OKFWorkspace,
        connector: "BaseConnector | None" = None,
    ) -> None:
        ensure_categorical_samples(workspace, connector)

    @classmethod
    def _detect_categorical_value_matches(
        cls,
        question: str,
        workspace: OKFWorkspace,
    ) -> list[dict[str, Any]]:
        return detect_categorical_value_matches(question, workspace)

    @staticmethod
    def _fuzzy_matches_word(q_words: list[str], candidate: str, threshold: float = 0.80) -> bool:
        return fuzzy_matches_word(q_words, candidate, threshold)

    @classmethod
    def _detect_relevant_tables(
        cls,
        question: str,
        workspace: OKFWorkspace,
        rag_bundle: dict[str, list[dict[str, Any]]],
    ) -> list[str]:
        return detect_relevant_tables(question, workspace, rag_bundle)

    @staticmethod
    def _match_measures(question: str, workspace: OKFWorkspace) -> list[SemanticMeasure]:
        return match_measures(question, workspace)

    @staticmethod
    def _check_just_in_time_clarification(question: str, workspace: OKFWorkspace) -> str | None:
        return check_just_in_time_clarification(question, workspace)

    @classmethod
    def _generate_query(
        cls,
        question: str,
        workspace: OKFWorkspace,
        join_plan: JoinPlan,
        matched_measures: list[SemanticMeasure],
        rag_bundle: dict[str, list[dict[str, Any]]],
        is_dax_target: bool,
        llm_router: LLMRouter | None,
        what_if_overrides: dict[str, float] | None = None,
    ) -> tuple[str, str, str, str]:
        return generate_query(
            question,
            workspace,
            join_plan,
            matched_measures,
            rag_bundle,
            is_dax_target,
            llm_router,
            what_if_overrides,
        )

    @classmethod
    def _llm_compile_query(
        cls,
        question: str,
        workspace: OKFWorkspace,
        join_plan: JoinPlan,
        matched_measures: list[SemanticMeasure],
        rag_bundle: dict[str, list[dict[str, Any]]],
        is_dax_target: bool,
        llm_router: LLMRouter,
        what_if_overrides: dict[str, float] | None = None,
    ) -> tuple[str, str]:
        return llm_compile_query(
            question,
            workspace,
            join_plan,
            matched_measures,
            rag_bundle,
            is_dax_target,
            llm_router,
            what_if_overrides,
        )

    @staticmethod
    def _clean_agg_alias(col_name: str, agg_fn: str) -> str:
        return clean_agg_alias(col_name, agg_fn)

    @staticmethod
    def _infer_agg_function(q_low: str, metric_col: str | None) -> str:
        return infer_agg_function(q_low, metric_col)

    @staticmethod
    def _infer_sort_direction(q_low: str) -> str:
        return infer_sort_direction(q_low)

    @classmethod
    def _deterministic_sql_compiler(
        cls,
        question: str,
        workspace: OKFWorkspace,
        join_plan: JoinPlan,
        matched_measures: list[SemanticMeasure],
        what_if_overrides: dict[str, float] | None = None,
    ) -> str:
        return compile_deterministic_sql(question, workspace, join_plan, matched_measures, what_if_overrides)

    @classmethod
    def _build_dax_equivalent(
        cls,
        question: str,
        workspace: OKFWorkspace,
        matched_measures: list[SemanticMeasure],
    ) -> str:
        return build_dax_equivalent(question, workspace, matched_measures)

    @classmethod
    def _repair_query(
        cls,
        question: str,
        failed_query: str,
        error_msg: str,
        workspace: OKFWorkspace,
        join_plan: JoinPlan,
        is_dax_target: bool,
        llm_router: LLMRouter | None,
        okf_store: OKFStore | None = None,
        matched_measures: list[SemanticMeasure] | None = None,
        what_if_overrides: dict[str, float] | None = None,
    ) -> tuple[str, str, str]:
        repaired, mode, detail, _, _, _ = repair_query_with_guard(
            question=question,
            failed_query=failed_query,
            error_msg=error_msg,
            workspace=workspace,
            join_plan=join_plan,
            is_dax_target=is_dax_target,
            llm_router=llm_router,
            okf_store=okf_store,
            matched_measures=matched_measures,
            what_if_overrides=what_if_overrides,
        )
        return repaired, mode, detail

    @classmethod
    def _build_executive_summary(
        cls,
        question: str,
        df: pd.DataFrame,
        matched_measures: list[SemanticMeasure],
        applied_rules: list[str],
        llm_router: LLMRouter | None,
        workspace: OKFWorkspace | None = None,
    ) -> str:
        return build_executive_summary(question, df, matched_measures, applied_rules, llm_router, workspace)

    @classmethod
    def _resolve_follow_up_context(
        cls,
        question: str,
        conversation_history: list[dict[str, Any]] | None,
        workspace: OKFWorkspace,
        llm_router: LLMRouter | None,
    ) -> tuple[str, bool, str | None, list[str]]:
        return resolve_follow_up_context(question, conversation_history, workspace, llm_router)

    @classmethod
    def improve_user_prompt(
        cls,
        question: str,
        workspace: OKFWorkspace,
        connector: "BaseConnector | None" = None,
        conversation_history: list[dict[str, Any]] | None = None,
        llm_router: LLMRouter | None = None,
    ) -> dict[str, Any]:
        return improve_user_prompt_payload(question, workspace, connector, conversation_history, llm_router)

    @classmethod
    def _generate_suggested_follow_ups(
        cls,
        df: pd.DataFrame,
        question: str,
        workspace: OKFWorkspace,
        join_plan: JoinPlan,
    ) -> list[str]:
        return generate_suggested_follow_ups(df, question, workspace, join_plan)

    @classmethod
    def _generate_drill_down_options(
        cls,
        df: pd.DataFrame,
        workspace: OKFWorkspace,
        involved_tables: list[str] | None = None,
    ) -> list[dict[str, str]]:
        return generate_drill_down_options(df, workspace, involved_tables)

    @classmethod
    def _find_matching_verified_query(cls, question: str, workspace: OKFWorkspace) -> VerifiedQuery | None:
        return find_matching_verified_query(question, workspace)

    @classmethod
    def _is_dataset_overview_question(cls, question: str) -> bool:
        return is_dataset_overview_question(question)

    @classmethod
    def _generate_dataset_overview(
        cls,
        question: str,
        workspace: OKFWorkspace,
        llm_router: LLMRouter | None,
    ) -> tuple[QueryResponse, pd.DataFrame]:
        payload, df_ov = generate_dataset_overview_payload(question, workspace, llm_router)
        return QueryResponse.model_validate(payload), df_ov

    @classmethod
    def should_render_chart(
        cls,
        question: str,
        df: pd.DataFrame | None,
        response_mode: str = "auto",
        is_thread: bool = False,
    ) -> bool:
        return should_render_chart(question, df, response_mode, is_thread)

    @classmethod
    def _needs_warehouse_sql_for_thread(
        cls,
        question: str,
        parent_df: pd.DataFrame,
        parent_question: str,
        workspace: OKFWorkspace,
    ) -> bool:
        return needs_warehouse_sql_for_thread(question, parent_df, parent_question, workspace)

    @classmethod
    def _analyze_chart_dataframe(
        cls,
        question: str,
        parent_df: pd.DataFrame,
        parent_question: str = "",
        parent_summary: str = "",
        llm_router: LLMRouter | None = None,
    ) -> tuple[str, pd.DataFrame, list[str]]:
        return analyze_chart_dataframe(question, parent_df, parent_question, parent_summary, llm_router)

    @classmethod
    def answer_chart_thread_question(
        cls,
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
    ) -> dict[str, Any]:
        return answer_chart_thread_question(
            question=question,
            parent_question=parent_question,
            parent_sql=parent_sql,
            parent_summary=parent_summary,
            parent_rows=parent_rows,
            parent_columns=parent_columns,
            thread_history=thread_history,
            response_mode=response_mode,
            workspace=workspace,
            okf_store=okf_store,
            rag_store=rag_store,
            connector=connector,
            llm_router=llm_router,
            what_if_overrides=what_if_overrides,
            answer_question_fn=cls.answer_question,
        )
