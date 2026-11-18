"""Compiled LangGraph QueryAgent StateGraph with InMemorySaver, streaming, and HITL resume."""

from __future__ import annotations

import time
import uuid
import weakref
from collections.abc import Iterator
from typing import Any

import pandas as pd
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, RetryPolicy

from src.agents.graph.nodes import (
    MAX_RETRIES,
    build_out_of_domain,
    check_verified_memory,
    clarification_interrupt,
    classify_error_diagnostic,
    compile_initial_query,
    critic_repair_query,
    dataset_overview,
    execute_sandbox,
    handle_exhausted_failure,
    ingest_and_resolve_followup,
    post_process_and_summarize,
    retrieve_okf_and_rag,
    route_after_followup,
    route_after_retrieval,
)
from src.agents.graph.state import (
    BlockingAmbiguityPayload,
    ErrorDiagnostic,
    NodeTelemetryRecord,
    QueryGraphContext,
    QueryGraphState,
    QueryInputModel,
    QueryOutputModel,
)

ALLOWED_MSGPACK_MODULES: list[tuple[str, str]] = [
    ("src.agents.graph.state", "JoinPlanSnapshot"),
    ("src.agents.graph.state", "CategoricalMatchRecord"),
    ("src.agents.graph.state", "NodeTelemetryRecord"),
    ("src.agents.graph.state", "BlockingAmbiguityPayload"),
    ("src.agents.graph.state", "ErrorDiagnostic"),
    ("src.agents.graph.state", "QueryGraphState"),
    ("src.agents.graph.state", "QueryInputModel"),
    ("src.agents.graph.state", "QueryOutputModel"),
    ("src.agents.graph.state", "SupervisorState"),
    ("src.knowledge.okf_models", "RelationshipEdge"),
    ("src.knowledge.okf_models", "JoinCardinality"),
    ("src.knowledge.okf_models", "JoinStatus"),
    ("src.knowledge.okf_models", "SemanticMeasure"),
]

_GRAPH_CACHE: weakref.WeakKeyDictionary[Any, Any] = weakref.WeakKeyDictionary()
_DEFAULT_CHECKPOINTER: InMemorySaver | None = None
_UNSET: Any = object()


def build_default_checkpointer() -> InMemorySaver:
    """Create an InMemorySaver configured with explicit strict MsgPack type allowlisting."""
    serde = JsonPlusSerializer(allowed_msgpack_modules=ALLOWED_MSGPACK_MODULES)
    return InMemorySaver(serde=serde)


def get_default_checkpointer() -> InMemorySaver:
    """Return the process-wide fallback InMemorySaver instance."""
    global _DEFAULT_CHECKPOINTER
    if _DEFAULT_CHECKPOINTER is None:
        _DEFAULT_CHECKPOINTER = build_default_checkpointer()
    return _DEFAULT_CHECKPOINTER


def build_query_graph(checkpointer: Any = _UNSET) -> Any:
    """Build and compile the 11-node QueryAgent StateGraph with bounded retry and InMemorySaver."""
    cp = get_default_checkpointer() if checkpointer is _UNSET else checkpointer
    cached = None
    if cp is not None and cp is not False:
        try:
            cached = _GRAPH_CACHE.get(cp)
        except TypeError:
            cached = None
    if cached is not None:
        return cached

    builder = StateGraph(
        QueryGraphState,
        input_schema=QueryInputModel,
        output_schema=QueryOutputModel,
    )

    transient_retry = RetryPolicy(
        max_attempts=2,
        retry_on=(TimeoutError, ConnectionError, OSError),
    )

    builder.add_node("ingest_and_resolve_followup", ingest_and_resolve_followup)
    builder.add_node("dataset_overview", dataset_overview)
    builder.add_node(
        "check_verified_memory",
        check_verified_memory,
        destinations=("retrieve_okf_and_rag", END),
    )
    builder.add_node("retrieve_okf_and_rag", retrieve_okf_and_rag)
    builder.add_node("build_out_of_domain", build_out_of_domain)
    builder.add_node("clarification_interrupt", clarification_interrupt)
    builder.add_node(
        "compile_initial_query",
        compile_initial_query,
        retry_policy=transient_retry,
    )
    builder.add_node(
        "execute_sandbox",
        execute_sandbox,
        destinations=("post_process_and_summarize", "critic_repair_query", "handle_exhausted_failure"),
    )
    builder.add_node("critic_repair_query", critic_repair_query)
    builder.add_node("post_process_and_summarize", post_process_and_summarize)
    builder.add_node("handle_exhausted_failure", handle_exhausted_failure)

    builder.add_edge(START, "ingest_and_resolve_followup")
    builder.add_conditional_edges(
        "ingest_and_resolve_followup",
        route_after_followup,
        {
            "dataset_overview": "dataset_overview",
            "check_verified_memory": "check_verified_memory",
            "retrieve_okf_and_rag": "retrieve_okf_and_rag",
        },
    )
    builder.add_edge("dataset_overview", END)
    builder.add_conditional_edges(
        "retrieve_okf_and_rag",
        route_after_retrieval,
        {
            "build_out_of_domain": "build_out_of_domain",
            "clarification_interrupt": "clarification_interrupt",
            "compile_initial_query": "compile_initial_query",
        },
    )
    builder.add_edge("build_out_of_domain", END)
    builder.add_edge("clarification_interrupt", "compile_initial_query")
    builder.add_edge("compile_initial_query", "execute_sandbox")
    builder.add_edge("critic_repair_query", "execute_sandbox")
    builder.add_edge("post_process_and_summarize", END)
    builder.add_edge("handle_exhausted_failure", END)

    compiled = builder.compile(checkpointer=cp)
    try:
        _GRAPH_CACHE[cp] = compiled
    except TypeError:
        pass
    return compiled


def get_query_graph_topology() -> dict[str, Any]:
    """Return node list, directed edges, and Mermaid diagram for the compiled QueryAgent graph."""
    graph_repr = build_query_graph().get_graph()
    nodes = [
        {"id": nid, "name": getattr(ndata, "name", nid)}
        for nid, ndata in graph_repr.nodes.items()
    ]
    edges = [
        {
            "source": e.source,
            "target": e.target,
            "conditional": bool(getattr(e, "conditional", False)),
            "data": str(e.data) if getattr(e, "data", None) is not None else "",
        }
        for e in graph_repr.edges
    ]
    return {
        "graph_type": "query",
        "checkpointer": "InMemorySaver",
        "nodes": nodes,
        "edges": edges,
        "mermaid": graph_repr.draw_mermaid(),
    }


def _serialize_telemetry(records: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for rec in records or []:
        if isinstance(rec, NodeTelemetryRecord):
            out.append(rec.model_dump())
        elif isinstance(rec, dict):
            out.append(NodeTelemetryRecord.model_validate(rec).model_dump())
    return out


def _finalize_graph_result(
    raw_out: dict[str, Any],
    ctx: QueryGraphContext,
    elapsed_ms: float,
    thread_id: str,
    question: str = "",
) -> dict[str, Any]:
    """Normalize LangGraph output (completed or interrupted) into structured Data_Talk result payload."""
    interrupts = raw_out.get("__interrupt__")
    raw_trace = list(raw_out.get("node_trace") or [])
    telemetry_dicts = _serialize_telemetry(list(raw_out.get("node_telemetry") or []))
    step_sum_ms = round(sum(float(t.get("duration_ms", 0.0)) for t in telemetry_dicts), 3)
    total_latency_ms = round(max(float(elapsed_ms), step_sum_ms), 3)

    if interrupts:
        first_int = interrupts[0]
        raw_payload = first_int.value if hasattr(first_int, "value") else first_int
        payload_obj = BlockingAmbiguityPayload.model_validate(raw_payload)
        resp_dict = {
            "user_question": question,
            "resolved_question": str(raw_out.get("resolved_question") or question),
            "success": False,
            "executed_query": "",
            "compiled_dax_equivalent": "",
            "query_dialect": getattr(ctx.connector, "dialect", ctx.workspace.dialect),
            "attempts_used": 0,
            "healing_trace": [],
            "involved_tables": list(raw_out.get("target_tables") or []),
            "join_clauses_used": [],
            "fan_trap_mitigated": False,
            "measures_applied": [],
            "business_rules_applied": [],
            "clarification_suggestion": payload_obj.clarification_prompt,
            "clarification_severity": "blocking_modal",
            "executive_summary": f"⏸️ **Clarification Required**\n\n{payload_obj.clarification_prompt}",
            "compiler_mode": "hitl_interrupt",
            "compiler_detail": "Paused at LangGraph Interrupt",
            "node_trace": raw_trace,
            "node_telemetry": telemetry_dicts,
            "total_latency_ms": total_latency_ms,
        }
        return {
            "thread_id": thread_id,
            "response_dict": resp_dict,
            "df": pd.DataFrame(),
            "interrupted": True,
            "interrupt_payload": payload_obj.model_dump(),
            "error_diagnostic": None,
            "node_trace": raw_trace,
            "node_telemetry": telemetry_dicts,
            "total_latency_ms": total_latency_ms,
        }

    resp_dict = dict(raw_out.get("response_dict") or {})
    raw_diag = raw_out.get("error_diagnostic") or resp_dict.get("error_diagnostic")
    diag_dict: dict[str, Any] | None = None
    if isinstance(raw_diag, ErrorDiagnostic):
        diag_dict = raw_diag.model_dump()
    elif isinstance(raw_diag, dict):
        diag_dict = ErrorDiagnostic.model_validate(raw_diag).model_dump()

    resp_dict["node_trace"] = raw_trace
    resp_dict["node_telemetry"] = telemetry_dicts
    resp_dict["total_latency_ms"] = total_latency_ms
    if diag_dict is not None:
        resp_dict["error_diagnostic"] = diag_dict
    else:
        resp_dict.setdefault("error_diagnostic", None)

    df_key = raw_out.get("result_df_key") or "result"
    df = ctx.dataframe_registry.get(df_key, pd.DataFrame())
    if not resp_dict.get("success", True):
        df = pd.DataFrame()

    return {
        "thread_id": thread_id,
        "response_dict": resp_dict,
        "df": df,
        "interrupted": False,
        "interrupt_payload": None,
        "error_diagnostic": diag_dict,
        "node_trace": raw_trace,
        "node_telemetry": telemetry_dicts,
        "total_latency_ms": total_latency_ms,
    }


def run_query_graph(
    question: str,
    workspace: Any,
    okf_store: Any,
    rag_store: Any,
    connector: Any,
    llm_router: Any | None = None,
    what_if_overrides: dict[str, float] | None = None,
    conversation_history: list[dict[str, Any]] | None = None,
    force_recompile: bool = False,
    allow_interrupts: bool = False,
    force_heuristic: bool = False,
    thread_id: str | None = None,
    checkpointer: Any | None = None,
) -> dict[str, Any]:
    """Execute the 11-node QueryAgent StateGraph synchronously with strict Pydantic state."""
    tid = thread_id or str(uuid.uuid4())
    ctx = QueryGraphContext(
        workspace=workspace,
        okf_store=okf_store,
        rag_store=rag_store,
        connector=connector,
        llm_router=llm_router,
        conversation_history=list(conversation_history or []),
        allow_interrupts=allow_interrupts,
        force_heuristic=force_heuristic,
    )
    graph = build_query_graph(checkpointer)
    config: dict[str, Any] = {
        "configurable": {"thread_id": tid, "dt_context": ctx},
        "recursion_limit": 25,
    }
    t0 = time.perf_counter()
    try:
        raw_out = graph.invoke(
            {
                "question": question,
                "what_if_overrides": what_if_overrides,
                "force_recompile": force_recompile,
                "allow_interrupts": allow_interrupts,
                "force_heuristic": force_heuristic,
            },
            config=config,
        )
    except Exception as exc:
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        diag = classify_error_diagnostic(str(exc), "query_graph", 1, MAX_RETRIES, workspace)
        return {
            "thread_id": tid,
            "response_dict": {
                "user_question": question,
                "success": False,
                "error_message": diag.technical_detail,
                "error_diagnostic": diag.model_dump(),
                "total_latency_ms": elapsed_ms,
            },
            "df": pd.DataFrame(),
            "interrupted": False,
            "interrupt_payload": None,
            "error_diagnostic": diag.model_dump(),
            "node_trace": [],
            "node_telemetry": [],
            "total_latency_ms": elapsed_ms,
        }

    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)
    return _finalize_graph_result(raw_out, ctx, elapsed_ms, tid, question=question)


def resume_query_graph(
    thread_id: str,
    resume_payload: Any,
    workspace: Any,
    okf_store: Any,
    rag_store: Any,
    connector: Any,
    llm_router: Any | None = None,
    conversation_history: list[dict[str, Any]] | None = None,
    checkpointer: Any | None = None,
    force_heuristic: bool = False,
) -> dict[str, Any]:
    """Resume a paused QueryAgent checkpoint suspended at clarification_interrupt."""
    graph = build_query_graph(checkpointer)
    ctx = QueryGraphContext(
        workspace=workspace,
        okf_store=okf_store,
        rag_store=rag_store,
        connector=connector,
        llm_router=llm_router,
        conversation_history=list(conversation_history or []),
        allow_interrupts=True,
        force_heuristic=force_heuristic,
    )
    config: dict[str, Any] = {
        "configurable": {"thread_id": thread_id, "dt_context": ctx},
        "recursion_limit": 25,
    }
    snap = graph.get_state(config)
    if not snap or (not snap.values and not snap.next):
        raise KeyError(f"Unknown or expired LangGraph thread_id: '{thread_id}'")

    question_orig = str((snap.values or {}).get("question") or "")
    t0 = time.perf_counter()
    raw_out = graph.invoke(Command(resume=resume_payload), config=config)
    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)
    return _finalize_graph_result(raw_out, ctx, elapsed_ms, thread_id, question=question_orig)


def stream_query_graph(
    question: str,
    workspace: Any,
    okf_store: Any,
    rag_store: Any,
    connector: Any,
    llm_router: Any | None = None,
    what_if_overrides: dict[str, float] | None = None,
    conversation_history: list[dict[str, Any]] | None = None,
    force_recompile: bool = False,
    allow_interrupts: bool = False,
    force_heuristic: bool = False,
    thread_id: str | None = None,
    checkpointer: Any | None = None,
) -> Iterator[dict[str, Any]]:
    """Stream typed stage/telemetry/clarification/interrupt/error/complete events from QueryAgentGraph."""
    tid = thread_id or str(uuid.uuid4())
    ctx = QueryGraphContext(
        workspace=workspace,
        okf_store=okf_store,
        rag_store=rag_store,
        connector=connector,
        llm_router=llm_router,
        conversation_history=list(conversation_history or []),
        allow_interrupts=allow_interrupts,
        force_heuristic=force_heuristic,
    )
    graph = build_query_graph(checkpointer)
    config: dict[str, Any] = {
        "configurable": {"thread_id": tid, "dt_context": ctx},
        "recursion_limit": 25,
    }

    t0 = time.perf_counter()
    last_interrupts: Any = None
    try:
        for mode_chunk in graph.stream(
            {
                "question": question,
                "what_if_overrides": what_if_overrides,
                "force_recompile": force_recompile,
                "allow_interrupts": allow_interrupts,
                "force_heuristic": force_heuristic,
            },
            config=config,
            stream_mode=["updates", "custom"],
        ):
            if not isinstance(mode_chunk, tuple) or len(mode_chunk) != 2:
                continue
            mode, payload = mode_chunk
            if mode == "custom" and isinstance(payload, dict):
                evt_type = str(payload.get("type") or "stage")
                yield {"event": evt_type, "data": payload}
            elif mode == "updates" and isinstance(payload, dict):
                if "__interrupt__" in payload:
                    last_interrupts = payload["__interrupt__"]
    except Exception as exc:
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        diag = classify_error_diagnostic(str(exc), "stream_query_graph", 1, MAX_RETRIES, workspace)
        yield {"event": "error", "data": diag.model_dump()}
        yield {
            "event": "complete",
            "data": {
                "thread_id": tid,
                "response_dict": {
                    "user_question": question,
                    "success": False,
                    "error_message": diag.technical_detail,
                    "error_diagnostic": diag.model_dump(),
                    "total_latency_ms": elapsed_ms,
                },
                "df": pd.DataFrame(),
                "interrupted": False,
                "interrupt_payload": None,
                "error_diagnostic": diag.model_dump(),
                "node_trace": [],
                "node_telemetry": [],
                "total_latency_ms": elapsed_ms,
            },
        }
        return

    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)
    snap = graph.get_state(config)
    values = dict((snap.values if snap else {}) or {})
    if last_interrupts:
        values["__interrupt__"] = last_interrupts
        finalized = _finalize_graph_result(values, ctx, elapsed_ms, tid, question=question)
        yield {"event": "interrupt", "data": finalized}
        return

    finalized = _finalize_graph_result(values, ctx, elapsed_ms, tid, question=question)
    if finalized.get("error_diagnostic") is not None:
        yield {"event": "error", "data": finalized["error_diagnostic"]}
    yield {"event": "complete", "data": finalized}
