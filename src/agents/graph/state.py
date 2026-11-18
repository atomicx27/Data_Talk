"""Strict Pydantic v2 state schemas and runtime context for LangGraph orchestration."""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal

import pandas as pd
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.knowledge.okf_models import RelationshipEdge, SemanticMeasure


def _assert_json_safe(value: Any, path: str = "value") -> Any:
    """Reject non-serializable runtime objects (DataFrames, connections, graphs) inside Any fields."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, BaseModel):
        return value
    if isinstance(value, list):
        for idx, item in enumerate(value):
            _assert_json_safe(item, f"{path}[{idx}]")
        return value
    if isinstance(value, tuple):
        for idx, item in enumerate(value):
            _assert_json_safe(item, f"{path}[{idx}]")
        return value
    if isinstance(value, dict):
        for k, v in value.items():
            if not isinstance(k, (str, int, float, bool)):
                raise ValueError(f"Non-JSON key type {type(k).__name__} at {path}")
            _assert_json_safe(v, f"{path}.{k}")
        return value
    raise ValueError(
        f"Non-serializable object of type '{type(value).__name__}' is forbidden in checkpointed state at '{path}'"
    )


@dataclass
class QueryGraphContext:
    """Non-serializable runtime dependencies injected via RunnableConfig['configurable']['dt_context']."""

    workspace: Any                  # OKFWorkspace
    okf_store: Any                  # OKFStore (holds NetworkX join graph)
    rag_store: Any                  # RAGStore (SQLite BM25 + Vector RRF)
    connector: Any                  # BaseConnector (DuckDB / Federated / SQL / PowerBI)
    llm_router: Any | None = None   # LLMRouter
    conversation_history: list[dict[str, Any]] = field(default_factory=list)
    dataframe_registry: dict[str, pd.DataFrame] = field(default_factory=dict)
    allow_interrupts: bool = False
    force_heuristic: bool = False


def get_graph_context(config: RunnableConfig | dict[str, Any] | None) -> QueryGraphContext:
    """Extract QueryGraphContext from LangGraph RunnableConfig."""
    if not isinstance(config, dict):
        raise ValueError("RunnableConfig must be a dictionary containing configurable.dt_context")
    configurable = config.get("configurable") or {}
    ctx = configurable.get("dt_context")
    if not isinstance(ctx, QueryGraphContext):
        raise ValueError("Missing QueryGraphContext at config['configurable']['dt_context']")
    return ctx


class StrictStateModel(BaseModel):
    """Base Pydantic v2 model forbidding arbitrary non-serializable types in checkpointed state."""

    model_config = ConfigDict(arbitrary_types_allowed=False, extra="forbid")


class JoinPlanSnapshot(StrictStateModel):
    """Pydantic-validated JoinPlan preserving RelationshipEdge objects without dict-subclass loss."""

    required_tables: list[str] = Field(default_factory=list)
    bridge_tables: list[str] = Field(default_factory=list)
    join_edges: list[RelationshipEdge] = Field(default_factory=list)
    sql_join_clauses: list[str] = Field(default_factory=list)
    has_fan_trap: bool = False
    fact_tables_involved: list[str] = Field(default_factory=list)
    shared_dimensions: list[str] = Field(default_factory=list)
    fan_trap_guidance: str = ""

    @classmethod
    def from_join_plan(cls, plan: Any) -> "JoinPlanSnapshot":
        """Build a strict snapshot from a JoinPlan object or plain dict."""
        if plan is None:
            return cls()
        if isinstance(plan, cls):
            return plan
        # JoinPlan subclasses dict, so attribute access (.join_edges) keeps RelationshipEdge instances
        raw_edges = getattr(plan, "join_edges", None)
        if raw_edges is None and isinstance(plan, dict):
            raw_edges = plan.get("join_edges", [])
        edges: list[RelationshipEdge] = []
        for edge in raw_edges or []:
            if isinstance(edge, RelationshipEdge):
                edges.append(edge)
            elif isinstance(edge, dict):
                edges.append(RelationshipEdge.model_validate(edge))

        def _read(name: str, default: Any) -> Any:
            val = getattr(plan, name, None)
            if val is None and isinstance(plan, dict):
                val = plan.get(name, default)
            return default if val is None else val

        return cls(
            required_tables=list(_read("required_tables", [])),
            bridge_tables=list(_read("bridge_tables", [])),
            join_edges=edges,
            sql_join_clauses=list(_read("sql_join_clauses", [])),
            has_fan_trap=bool(_read("has_fan_trap", False)),
            fact_tables_involved=list(_read("fact_tables_involved", [])),
            shared_dimensions=list(_read("shared_dimensions", [])),
            fan_trap_guidance=str(_read("fan_trap_guidance", "")),
        )

    def to_join_plan(self) -> Any:
        """Reconstruct a full OKF JoinPlan with RelationshipEdge objects and dict keys synced."""
        from src.knowledge.okf_store import JoinPlan

        return JoinPlan(
            required_tables=list(self.required_tables),
            bridge_tables=list(self.bridge_tables),
            join_edges=list(self.join_edges),
            sql_join_clauses=list(self.sql_join_clauses),
            has_fan_trap=bool(self.has_fan_trap),
            fact_tables_involved=list(self.fact_tables_involved),
            shared_dimensions=list(self.shared_dimensions),
            fan_trap_guidance=str(self.fan_trap_guidance),
        )


class CategoricalMatchRecord(StrictStateModel):
    """Matched categorical filter phrase resolved against profiled table column samples."""

    table: str
    column: str
    value: str
    score: int = 0
    matched_phrase: str = ""


class NodeTelemetryRecord(StrictStateModel):
    """Per-node latency and status telemetry emitted cleanly without PII/row leakage."""

    node: str
    duration_ms: float = 0.0
    attempt: int = 1
    status: Literal["ok", "interrupted", "error"] = "ok"
    updated_keys: list[str] = Field(default_factory=list)


class BlockingAmbiguityPayload(StrictStateModel):
    """Structured HITL interrupt payload awaiting explicit user resolution."""

    severity: Literal["blocking_modal"] = "blocking_modal"
    type: str
    question: str
    clarification_prompt: str
    suggested_options: list[str] = Field(default_factory=list)
    question_id: str = ""


class ErrorDiagnostic(StrictStateModel):
    """Structured error payload surfaced to the user & frontend diagnostic recovery cards."""

    error_code: str
    error_category: Literal[
        "security",
        "sql_compilation",
        "database",
        "llm_provider",
        "domain",
        "internal",
    ]
    failed_node: str
    attempt: int = 1
    max_retries: int = 3
    user_message: str
    technical_detail: str = ""
    remediation_hint: str = ""
    recoverable_with_heuristic: bool = False


class QueryInputModel(StrictStateModel):
    """Input boundary schema for QueryAgentSubgraph."""

    question: str
    what_if_overrides: dict[str, float] | None = None
    force_recompile: bool = False
    allow_interrupts: bool = False
    force_heuristic: bool = False


class QueryOutputModel(StrictStateModel):
    """Output boundary schema for QueryAgentSubgraph."""

    response_dict: dict[str, Any] = Field(default_factory=dict)
    result_df_key: str = ""
    interrupted: bool = False
    interrupt_payload: BlockingAmbiguityPayload | None = None
    error_diagnostic: ErrorDiagnostic | None = None
    node_trace: list[str] = Field(default_factory=list)
    node_telemetry: list[NodeTelemetryRecord] = Field(default_factory=list)

    @field_validator("response_dict")
    @classmethod
    def _validate_response_dict_json_safe(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _assert_json_safe(value, "response_dict")


class QueryGraphState(QueryInputModel, QueryOutputModel):
    """Full checkpointed LangGraph state for the 11-node QueryAgent pipeline."""

    # Normalized & resolved intent
    q_clean: str = ""
    resolved_question: str = ""
    is_follow_up: bool = False
    follow_up_context: str | None = None
    inherited_tables: list[str] = Field(default_factory=list)
    cat_value_matches: list[CategoricalMatchRecord] = Field(default_factory=list)
    is_overview: bool = False

    # Verified Query Memory
    verified_query_id: str | None = None
    from_verified_memory: bool = False

    # OKF + RAG Context (strictly Pydantic-validated & JSON-serializable)
    rag_bundle: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)

    @field_validator("rag_bundle")
    @classmethod
    def _validate_rag_bundle_json_safe(cls, value: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
        return _assert_json_safe(value, "rag_bundle")
    target_tables: list[str] = Field(default_factory=list)
    join_plan: JoinPlanSnapshot = Field(default_factory=JoinPlanSnapshot)
    matched_measures: list[SemanticMeasure] = Field(default_factory=list)
    applied_rules: list[str] = Field(default_factory=list)
    clarification_suggestion: str | None = None
    clarification_severity: Literal["blocking_modal", "non_blocking_inline"] | None = None
    blocking_ambiguity_payload: BlockingAmbiguityPayload | None = None
    out_of_domain: bool = False

    # Compilation & 3x Self-Healing Execution Loop
    is_dax_target: bool = False
    attempt: int = 1
    current_query: str = ""
    dax_equivalent: str = ""
    compiler_mode: str = "heuristic"
    compiler_detail: str = "Deterministic Heuristic Engine"
    last_error: str | None = None
    execution_succeeded: bool = False
    row_count: int = 0
    healing_trace: Annotated[list[str], operator.add] = Field(default_factory=list)
    node_trace: Annotated[list[str], operator.add] = Field(default_factory=list)
    node_telemetry: Annotated[list[NodeTelemetryRecord], operator.add] = Field(default_factory=list)


class SupervisorState(StrictStateModel):
    """Top-level LangGraph Supervisor state unifying SQL/RAG, RootCause, and Simulation."""

    question: str
    intent: Literal["descriptive_sql", "diagnostic_variance", "what_if_forecast"] = "descriptive_sql"
    what_if_overrides: dict[str, float] | None = None
    force_recompile: bool = False
    allow_interrupts: bool = False
    force_heuristic: bool = False
    # Optional Root-Cause / Simulation parameters
    metric_col: str | None = None
    dimension_col: str | None = None
    target_metric_goal: float | None = None
    horizon_months: int = 6
    target_col: str | None = None
    date_col: str | None = None
    # Outputs
    response_dict: dict[str, Any] = Field(default_factory=dict)
    result_df_key: str = ""
    interrupted: bool = False
    interrupt_payload: BlockingAmbiguityPayload | None = None
    error_diagnostic: ErrorDiagnostic | None = None
    node_trace: Annotated[list[str], operator.add] = Field(default_factory=list)
    node_telemetry: Annotated[list[NodeTelemetryRecord], operator.add] = Field(default_factory=list)

    @field_validator("response_dict")
    @classmethod
    def _validate_supervisor_response_json_safe(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _assert_json_safe(value, "response_dict")
