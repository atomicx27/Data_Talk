"""Top-level LangGraph Supervisor unifying QueryAgentSubgraph, RootCauseAgent, and SimulationEngine."""

from __future__ import annotations

import re
import time
import uuid
import weakref
from typing import Any, Literal

import pandas as pd
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from src.agents.compilers.text_matching import extract_what_if_from_prompt, normalize_what_if_overrides
from src.agents.graph.callbacks import emit_stream_event, trace_graph_node
from src.agents.graph.nodes import MAX_RETRIES, classify_error_diagnostic
from src.agents.graph.query_graph import (
    _finalize_graph_result,
    build_query_graph,
    get_default_checkpointer,
)
from src.agents.graph.state import QueryGraphContext, SupervisorState, get_graph_context
from src.agents.root_cause_agent import RootCauseAgent
from src.analytics.simulation_engine import SimulationEngine

_SUPERVISOR_CACHE: weakref.WeakKeyDictionary[Any, Any] = weakref.WeakKeyDictionary()


@trace_graph_node("classify_bi_intent")
def classify_bi_intent(state: SupervisorState, config: RunnableConfig) -> dict[str, Any]:
    """Classify user BI intent cleanly into SQL/RAG, Root-Cause Variance, or What-If Forecast."""
    emit_stream_event({"type": "stage", "stage": "supervisor", "label": "Classifying analytical intent..."})
    q_low = (state.question or "").lower().strip()

    if state.target_metric_goal is not None:
        return {"intent": "what_if_forecast"}

    if re.search(
        r"\b(why\s+did|why\s+has|root\s+cause|variance\s+analysis|period[- ]over[- ]period\s+variance|what\s+drove|driver\s+analysis|waterfall\s+bridge)\b",
        q_low,
    ):
        return {"intent": "diagnostic_variance"}

    if re.search(
        r"\b(goal[- ]?seek|monte\s+carlo|forecast\s+next|forecast\s+the|predict\s+future|future\s+trend|extrapolate)\b",
        q_low,
    ):
        return {"intent": "what_if_forecast"}

    return {"intent": "descriptive_sql"}


def route_supervisor_intent(
    state: SupervisorState,
) -> Literal["query_agent_subgraph", "run_root_cause", "run_simulation"]:
    """Conditional Supervisor router dispatching to the right analytical engine."""
    if state.intent == "diagnostic_variance":
        return "run_root_cause"
    if state.intent == "what_if_forecast":
        return "run_simulation"
    return "query_agent_subgraph"


@trace_graph_node("run_root_cause")
def run_root_cause(state: SupervisorState, config: RunnableConfig) -> dict[str, Any]:
    """Execute RootCauseAgent period-over-period variance decomposition."""
    emit_stream_event(
        {"type": "stage", "stage": "root_cause", "label": "Running period-over-period Root Cause decomposition..."}
    )
    ctx = get_graph_context(config)
    try:
        opts = RootCauseAgent.discover_options(ctx.workspace, ctx.connector)
        avail_metrics = [m["id"] if isinstance(m, dict) else str(m) for m in (opts.get("metrics") or [])]
        avail_dims = [d["id"] if isinstance(d, dict) else str(d) for d in (opts.get("dimensions") or [])]

        q_low = (state.question or "").lower()
        metric_col = state.metric_col or next(
            (m for m in avail_metrics if m.lower() in q_low),
            avail_metrics[0] if avail_metrics else "rev_net_usd",
        )
        dimension_col = state.dimension_col or next(
            (d for d in avail_dims if d.lower() in q_low),
            avail_dims[0] if avail_dims else "category",
        )

        rc_res = RootCauseAgent.analyze_variance(
            workspace=ctx.workspace,
            connector=ctx.connector,
            metric_col=metric_col,
            dimension=dimension_col,
        )
        df = rc_res.get("df")
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame()
        ctx.dataframe_registry["result"] = df

        receipt = rc_res.get("audit_receipt") or {}
        resp_dict = {
            "user_question": state.question,
            "resolved_question": f"Root Cause Variance Analysis: {metric_col} by {dimension_col}",
            "success": True,
            "executed_query": str(receipt.get("executed_sql") or rc_res.get("executed_sql") or ""),
            "compiled_dax_equivalent": str(receipt.get("formula_latex") or ""),
            "query_dialect": getattr(ctx.connector, "dialect", ctx.workspace.dialect),
            "attempts_used": 1,
            "healing_trace": [],
            "involved_tables": list(receipt.get("tables_used") or []),
            "join_clauses_used": [],
            "fan_trap_mitigated": False,
            "measures_applied": [metric_col],
            "business_rules_applied": [],
            "executive_summary": str(rc_res.get("summary") or "Completed variance decomposition."),
            "recommended_chart_type": str(rc_res.get("chart_type") or "waterfall"),
            "compiler_mode": "supervisor_root_cause",
            "compiler_detail": f"RootCauseAgent ({receipt.get('algorithm', 'Variance Decomposition')})",
            "suggested_follow_ups": list(rc_res.get("suggested_followups") or []),
            "drill_down_dimensions": [],
            "is_diagnostic": True,
            "audit_receipt": receipt,
            "kpi_ribbon": rc_res.get("kpi_ribbon"),
        }
        return {
            "response_dict": resp_dict,
            "result_df_key": "result",
            "error_diagnostic": None,
        }
    except Exception as exc:
        diag = classify_error_diagnostic(str(exc), "run_root_cause", 1, MAX_RETRIES, ctx.workspace)
        return {
            "response_dict": {
                "user_question": state.question,
                "success": False,
                "error_message": diag.technical_detail,
                "error_diagnostic": diag.model_dump(),
                "compiler_mode": "supervisor_root_cause",
                "compiler_detail": "RootCauseAgent Error",
            },
            "result_df_key": "",
            "error_diagnostic": diag,
        }


@trace_graph_node("run_simulation")
def run_simulation(state: SupervisorState, config: RunnableConfig) -> dict[str, Any]:
    """Execute OLS Linear Trend / What-If SimulationEngine (and Goal Seek if target_metric_goal given)."""
    emit_stream_event(
        {"type": "stage", "stage": "simulation", "label": "Running OLS Trend & What-If Simulation..."}
    )
    ctx = get_graph_context(config)
    try:
        overrides = normalize_what_if_overrides(state.what_if_overrides) or extract_what_if_from_prompt(
            state.question
        )
        rev_shock_pct = float(overrides.get("Revenue Multiplier %", 0.0)) * 100.0
        cost_shock_pct = float(overrides.get("Cost Multiplier %", 0.0)) * 100.0

        goal_seek_note = ""
        if state.target_metric_goal is not None:
            def _eval_fn(mult_dec: float) -> float:
                res_gs = SimulationEngine.simulate_workspace_kpis(
                    workspace=ctx.workspace,
                    connector=ctx.connector,
                    levers={"Revenue Multiplier %": mult_dec * 100.0},
                    target_metric=state.target_col,
                    forecast_periods=state.horizon_months,
                )
                return float((res_gs.get("kpis") or {}).get("sim_profit", 0.0))

            req_mult, achieved, solved = SimulationEngine.solve_goal_seek(
                eval_fn=_eval_fn,
                target_value=float(state.target_metric_goal),
            )
            if solved:
                rev_shock_pct = req_mult * 100.0
                overrides["Revenue Multiplier %"] = req_mult
                goal_seek_note = (
                    f"Goal-Seek solved required lever = `{rev_shock_pct:+.2f}%` "
                    f"to achieve target `{achieved:,.2f}`.\n\n"
                )

        sim_res = SimulationEngine.simulate_workspace_kpis(
            workspace=ctx.workspace,
            connector=ctx.connector,
            revenue_shock_pct=rev_shock_pct,
            cost_shock_pct=cost_shock_pct,
            levers={k: v * 100.0 for k, v in overrides.items()},
            target_metric=state.target_col,
            forecast_periods=state.horizon_months,
        )
        kpis = sim_res.get("kpis") or {}
        reg = sim_res.get("regression") or {}
        wf_steps = sim_res.get("waterfall_steps") or []

        rows: list[dict[str, Any]] = []
        for h in reg.get("historical_points", []):
            rows.append({
                "period": str(h["date"]),
                "phase": "Historical Actual",
                "actual": h["actual"],
                "ols_trend_fit": h["trend"],
                "projected_p50": None,
                "p10_downside": None,
                "p90_upside": None,
            })
        for f in reg.get("forecast_points", []):
            rows.append({
                "period": str(f["date"]),
                "phase": "Projected Forecast",
                "actual": None,
                "ols_trend_fit": f.get("organic_projected", f["projected"]),
                "projected_p50": f["projected"],
                "p10_downside": f["p10"],
                "p90_upside": f["p90"],
            })
        out_df = pd.DataFrame(rows) if rows else pd.DataFrame(wf_steps)
        ctx.dataframe_registry["result"] = out_df

        tbl_name = str(kpis.get("table_name") or "")
        exec_sql = str(reg.get("executed_sql") or (f'SELECT * FROM "{tbl_name}"' if tbl_name else "SELECT 1"))
        next_p50 = float(kpis.get("next_forecast_p50", kpis.get("sim_revenue", 0.0)))
        summary = (
            f"{goal_seek_note}"
            f"Simulated `{kpis.get('metric_name', 'Metric')}` on `{tbl_name or 'workspace'}`: "
            f"Projected P50 = **`{next_p50:,.2f}`** "
            f"(Baseline: `{float(kpis.get('base_revenue', 0.0)):,.2f}` · "
            f"Simulated: `{float(kpis.get('sim_revenue', 0.0)):,.2f}`)."
        )
        resp_dict = {
            "user_question": state.question,
            "resolved_question": state.question,
            "success": True,
            "executed_query": exec_sql,
            "compiled_dax_equivalent": f"OLS Slope = {reg.get('slope', 0.0):+,.2f} · R2 = {reg.get('r2', 0.0):.2f}" if reg else "",
            "query_dialect": getattr(ctx.connector, "dialect", ctx.workspace.dialect),
            "attempts_used": 1,
            "healing_trace": [],
            "involved_tables": [tbl_name] if tbl_name else [],
            "join_clauses_used": [],
            "fan_trap_mitigated": False,
            "measures_applied": [f"What-If ({k}: {v*100:+.1f}%)" for k, v in overrides.items()],
            "business_rules_applied": [],
            "executive_summary": summary,
            "recommended_chart_type": "fan_chart" if rows else "waterfall",
            "compiler_mode": "supervisor_simulation",
            "compiler_detail": "OLS Regression & What-If Simulation Engine",
            "suggested_follow_ups": [
                "Break down metric by primary dimension",
                "Why did metric change? Period-over-period variance",
            ],
            "drill_down_dimensions": [],
            "is_simulation": True,
            "simulation_kpis": kpis,
        }
        return {
            "response_dict": resp_dict,
            "result_df_key": "result",
            "error_diagnostic": None,
        }
    except Exception as exc:
        diag = classify_error_diagnostic(str(exc), "run_simulation", 1, MAX_RETRIES, ctx.workspace)
        return {
            "response_dict": {
                "user_question": state.question,
                "success": False,
                "error_message": diag.technical_detail,
                "error_diagnostic": diag.model_dump(),
                "compiler_mode": "supervisor_simulation",
                "compiler_detail": "SimulationEngine Error",
            },
            "result_df_key": "",
            "error_diagnostic": diag,
        }


def build_supervisor_graph(checkpointer: Any | None = None) -> Any:
    """Compile the top-level LangGraph Supervisor unifying SQL/RAG, RootCause, and Simulation."""
    cp = checkpointer if checkpointer is not None else get_default_checkpointer()
    try:
        cached = _SUPERVISOR_CACHE.get(cp)
    except TypeError:
        cached = None
    if cached is not None:
        return cached

    # Subgraph omits its own checkpointer so parent SupervisorGraph checkpointer manages full state
    query_subgraph = build_query_graph(checkpointer=None)

    builder = StateGraph(SupervisorState)
    builder.add_node("classify_bi_intent", classify_bi_intent)
    builder.add_node("query_agent_subgraph", query_subgraph)
    builder.add_node("run_root_cause", run_root_cause)
    builder.add_node("run_simulation", run_simulation)

    builder.add_edge(START, "classify_bi_intent")
    builder.add_conditional_edges(
        "classify_bi_intent",
        route_supervisor_intent,
        {
            "query_agent_subgraph": "query_agent_subgraph",
            "run_root_cause": "run_root_cause",
            "run_simulation": "run_simulation",
        },
    )
    builder.add_edge("query_agent_subgraph", END)
    builder.add_edge("run_root_cause", END)
    builder.add_edge("run_simulation", END)

    compiled = builder.compile(checkpointer=cp)
    try:
        _SUPERVISOR_CACHE[cp] = compiled
    except TypeError:
        pass
    return compiled


def get_supervisor_graph_topology() -> dict[str, Any]:
    """Return node list, directed edges, and Mermaid diagram for the compiled SupervisorGraph."""
    graph_repr = build_supervisor_graph().get_graph(xray=True)
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
        "graph_type": "supervisor",
        "checkpointer": "InMemorySaver",
        "nodes": nodes,
        "edges": edges,
        "mermaid": graph_repr.draw_mermaid(),
    }


def run_supervisor_graph(
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
    resume_payload: Any | None = None,
) -> dict[str, Any]:
    """Execute the multi-agent SupervisorGraph with unified telemetry and checkpointer isolation."""
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
    graph = build_supervisor_graph(checkpointer)
    config: dict[str, Any] = {
        "configurable": {"thread_id": tid, "dt_context": ctx},
        "recursion_limit": 30,
    }
    t0 = time.perf_counter()
    inp: Any = (
        Command(resume=resume_payload)
        if resume_payload is not None
        else {
            "question": question,
            "what_if_overrides": what_if_overrides,
            "force_recompile": force_recompile,
            "allow_interrupts": allow_interrupts,
            "force_heuristic": force_heuristic,
        }
    )
    raw_out = graph.invoke(inp, config=config)
    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)
    return _finalize_graph_result(raw_out, ctx, elapsed_ms, tid, question=question)
