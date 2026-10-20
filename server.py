"""Data_Talk — Production FastAPI Backend & Enterprise Web UI Server.

Exposes REST API endpoints for all 6 architectural pillars of Data_Talk:
1. Multi-Source Connectors (1-Click 4-Table Enterprise Demo, CSV/Excel/Parquet, SQL Databases, Power BI .pbix/Desktop/Cloud)
2. Statistical Profiler, PII Redaction Shield, & Column Privacy Blocklist
3. 3-Signal Join Inference Engine, Interactive Join Governance, & ER Graph Topology
4. Smart AI Business Interviewer & Knowledge Readiness Score (0-100%)
5. Unified Semantic Measure & Calculated Column Studio (DAX <-> SQL + Live Dry-Run Verification)
6. Conversational BI ('Talk to Data'), Self-Healing SQL/DAX Loop, Root-Cause Variance Analyzer, What-If Simulator, & Outbound Privacy Inspector

Run directly with:
    python server.py
Then open http://localhost:8000 in your browser.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.io as pio
import uvicorn
import yaml
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.agents.interviewer_agent import InterviewerAgent
from src.agents.llm_router import LLMRouter
from src.agents.measure_compiler import MeasureCompiler
from src.agents.query_agent import QueryAgent
from src.agents.root_cause_agent import RootCauseAgent
from src.connectors.base import BaseConnector
from src.connectors.duckdb_connector import DuckDBConnector
from src.connectors.powerbi_connector import PowerBIConnector
from src.connectors.profiler import StatisticalProfiler
from src.connectors.sql_connector import SQLConnector
from src.knowledge.join_inference import JoinInferenceEngine
from src.knowledge.okf_models import (
    BusinessRule,
    JoinCardinality,
    JoinStatus,
    OKFWorkspace,
    PinnedWidget,
    RelationshipEdge,
)
from src.knowledge.okf_store import OKFStore
from src.knowledge.rag_store import RAGStore
from src.security.privacy_firewall import CredentialVault, PrivacyFirewall, PrivacyMode
from src.viz.chart_recommender import ChartRecommender

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"
FRONTEND_DIR.mkdir(parents=True, exist_ok=True)


class AppRuntime:
    """Thread-safe singleton state manager for the Data_Talk FastAPI server."""

    def __init__(self, base_dir: Path | None = None) -> None:
        import os

        root = base_dir or BASE_DIR
        self.okf_store = OKFStore(base_dir=root / "workspaces")
        self.vault = CredentialVault(base_dir=root / ".secrets")
        for env_k in ("GROQ_API_KEY", "NVIDIA_NIM_API_KEY", "NVIDIA_API_KEY"):
            val = os.environ.get(env_k)
            if val:
                self.vault.set_secret(env_k, val)
        self.llm_router = LLMRouter(
            privacy_firewall=PrivacyFirewall(PrivacyMode.STRICT_SCHEMA_ONLY),
        )
        self.active_workspace_id = "enterprise_demo"
        self.connectors: dict[str, BaseConnector] = {}
        self.rag_stores: dict[str, RAGStore] = {}
        self.chat_history: dict[str, list[dict[str, Any]]] = {}
        self.what_if_overrides: dict[str, dict[str, float]] = {}

    def get_workspace(self, workspace_id: str | None = None) -> OKFWorkspace:
        ws_id = workspace_id or self.active_workspace_id
        ws = self.okf_store.load(ws_id)
        if ws is None:
            ws = OKFWorkspace(
                workspace_id=ws_id,
                workspace_name=ws_id.replace("_", " ").title(),
                source_type="duckdb",
                dialect="duckdb",
            )
            self.okf_store.save(ws)
        return ws

    def get_rag_store(self, workspace_id: str | None = None) -> RAGStore:
        ws_id = workspace_id or self.active_workspace_id
        if ws_id not in self.rag_stores:
            self.rag_stores[ws_id] = RAGStore(self.okf_store.workspace_dir(ws_id))
        return self.rag_stores[ws_id]

    def get_connector(self, workspace_id: str | None = None) -> BaseConnector:
        ws_id = workspace_id or self.active_workspace_id
        if ws_id not in self.connectors:
            ws = self.get_workspace(ws_id)
            if ws.source_type == "sql" and ws.credential_ref:
                secret_url = self.vault.get_secret(ws.credential_ref)
                if secret_url:
                    self.connectors[ws_id] = SQLConnector(secret_url, dialect_hint=ws.dialect)
                    return self.connectors[ws_id]
            self.connectors[ws_id] = DuckDBConnector(self.okf_store.workspace_dir(ws_id))
        return self.connectors[ws_id]

    def run_onboarding(self, workspace: OKFWorkspace, connector: BaseConnector) -> None:
        rag_store = self.get_rag_store(workspace.workspace_id)
        blocked_t = workspace.blocked_table_names()
        blocked_c = workspace.blocked_column_names()

        raw_tables, explicit_rels, imported_measures = connector.extract_schema(
            blocked_tables=blocked_t,
            blocked_columns=blocked_c,
        )
        profiled_tables = StatisticalProfiler.profile_tables(
            connector,
            raw_tables,
            blocked_tables=blocked_t,
            blocked_columns=blocked_c,
        )
        workspace.tables = profiled_tables
        workspace.relationships = JoinInferenceEngine.infer_relationships(
            profiled_tables,
            existing_relationships=explicit_rels,
            connector=connector,
        )
        if imported_measures:
            existing_ids = {m.id for m in workspace.measures}
            for im in imported_measures:
                if im.id not in existing_ids:
                    workspace.measures.append(im)

        self.okf_store.classify_table_roles(workspace)
        InterviewerAgent.sync_schema_to_rag(workspace, rag_store)
        InterviewerAgent.generate_prioritized_questions(workspace, llm_router=self.llm_router)
        self.okf_store.save(workspace)

    def serialize_state(self) -> dict[str, Any]:
        ws = self.get_workspace()
        rag = self.get_rag_store()
        readiness = InterviewerAgent.compute_readiness_score(ws)
        return {
            "active_workspace_id": self.active_workspace_id,
            "workspaces": sorted(list(set(self.okf_store.list_workspaces() + [self.active_workspace_id]))),
            "workspace": ws.model_dump(mode="json"),
            "readiness": readiness,
            "rag_counts": rag.count_by_collection(),
            "privacy_mode": self.llm_router.firewall.privacy_mode.value,
            "llm_model": self.llm_router.model,
            "has_api_key": self.llm_router.is_configured(),
            "supported_models": LLMRouter.SUPPORTED_MODELS,
            "supported_chart_types": ChartRecommender.SUPPORTED_CHART_TYPES,
            "starter_templates": MeasureCompiler.STARTER_MEASURE_TEMPLATES,
            "desktop_pbi_instances": PowerBIConnector.discover_local_desktop_instances(),
            "chat_history": self.chat_history.get(self.active_workspace_id, []),
            "what_if_overrides": self.what_if_overrides.get(
                self.active_workspace_id,
                {"Revenue Multiplier %": 0.0, "Cost Multiplier %": 0.0},
            ),
            "audit_log": self.llm_router.firewall.audit_log[-25:],
        }


runtime = AppRuntime()

app = FastAPI(
    title="Data_Talk Enterprise BI API",
    description="Two-Tier OKF + Hybrid RAG Conversational Business Intelligence Platform",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_no_cache_headers(request: Request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/static") or request.url.path == "/":
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


# ---------------------------------------------------------------------------
# Request Schemas
# ---------------------------------------------------------------------------
class SettingsUpdateRequest(BaseModel):
    workspace_id: str | None = None
    privacy_mode: str | None = None
    llm_model: str | None = None
    api_key: str | None = None
    api_base: str | None = None


class SQLConnectRequest(BaseModel):
    dialect: str = "postgres"
    credential_name: str = "prod_sql_db"
    connection_url: str


class PowerBIConnectRequest(BaseModel):
    mode: str = "powerbi_desktop"  # powerbi_desktop | powerbi_cloud
    localhost_port: int = 55000
    dataset_id: str = ""
    access_token: str = ""


class ColumnConfigUpdateRequest(BaseModel):
    table_name: str
    column_name: str
    is_blocked: bool | None = None
    description: str | None = None


class JoinUpdateItem(BaseModel):
    id: str
    source_table: str
    source_column: str
    join_type: str = "LEFT"
    target_table: str
    target_column: str
    cardinality: str = "N:1"
    status: str = "approved"


class JoinBatchUpdateRequest(BaseModel):
    relationships: list[JoinUpdateItem]


class CustomJoinRequest(BaseModel):
    source_table: str
    source_column: str
    target_table: str
    target_column: str
    join_type: str = "LEFT"
    cardinality: str = "N:1"


class InterviewAnswerRequest(BaseModel):
    question_id: str
    answer: str


class BusinessRuleCreateRequest(BaseModel):
    term: str
    definition: str
    sql_filter_hint: str = ""


class MeasureCreateRequest(BaseModel):
    name: str
    table_name: str
    user_formula: str
    input_type: str = "dax"
    description: str = ""


class CalculatedColumnCreateRequest(BaseModel):
    table_name: str
    column_name: str
    user_formula: str
    input_type: str = "dax"


class ChatQueryRequest(BaseModel):
    question: str
    revenue_shock_pct: float = 0.0
    cost_shock_pct: float = 0.0


class ChartBuildRequest(BaseModel):
    rows: list[dict[str, Any]]
    chart_type: str
    title: str = "Analytics Chart"


class TeachRuleRequest(BaseModel):
    question: str
    rule_text: str


class RootCauseRequest(BaseModel):
    metric_col: str = "rev_net_usd"
    dimension: str = "category"


class PinWidgetRequest(BaseModel):
    title: str
    user_question: str
    executed_query: str
    query_dialect: str = "duckdb"
    chart_type: str = "bar"


class WhatIfSimulateRequest(BaseModel):
    revenue_shock_pct: float = 0.0
    cost_shock_pct: float = 0.0


def _fig_to_json(fig: Any) -> dict[str, Any] | None:
    if fig is None:
        return None
    # Style Plotly figure for the executive dark theme
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.45)",
        font={"family": "Inter, system-ui, sans-serif", "color": "#E2E8F0", "size": 12},
        margin={"l": 48, "r": 24, "t": 48, "b": 42},
        title_font={"size": 14, "color": "#F8FAFC"},
    )
    return json.loads(pio.to_json(fig))


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------
@app.get("/api/state")
def get_state() -> dict[str, Any]:
    return runtime.serialize_state()


@app.post("/api/settings")
def update_settings(req: SettingsUpdateRequest) -> dict[str, Any]:
    if req.workspace_id and req.workspace_id.strip():
        clean_id = req.workspace_id.strip().lower().replace(" ", "_")
        runtime.active_workspace_id = clean_id
        runtime.get_workspace(clean_id)

    if req.privacy_mode:
        runtime.llm_router.firewall.privacy_mode = PrivacyMode(req.privacy_mode)
    if req.llm_model:
        runtime.llm_router.model = req.llm_model
    if req.api_key is not None and req.api_key.strip():
        runtime.llm_router.api_key = req.api_key.strip()
    if req.api_base is not None:
        runtime.llm_router.api_base = req.api_base.strip() or None

    return runtime.serialize_state()


@app.post("/api/connect/demo")
def connect_demo_dataset() -> dict[str, Any]:
    ws = runtime.get_workspace()
    ws_dir = runtime.okf_store.workspace_dir(ws.workspace_id)
    duck_conn = DuckDBConnector(ws_dir)
    duck_conn.load_builtin_demo_dataset()
    runtime.connectors[ws.workspace_id] = duck_conn

    ws.source_type = "duckdb"
    ws.dialect = "duckdb"
    runtime.run_onboarding(ws, duck_conn)

    # Pre-verify 2 starter DAX measures so the Measure Studio is immediately populated
    rag = runtime.get_rag_store(ws.workspace_id)
    for tmpl in MeasureCompiler.STARTER_MEASURE_TEMPLATES[:2]:
        MeasureCompiler.create_and_verify_measure(
            workspace=ws,
            rag_store=rag,
            connector=duck_conn,
            name=tmpl["name"],
            table_name=tmpl["table_name"],
            user_formula=tmpl["dax_expression"],
            input_type="dax",
            description=tmpl["description"],
            llm_router=runtime.llm_router,
        )
    runtime.okf_store.save(ws)
    return runtime.serialize_state()


@app.post("/api/connect/upload")
async def connect_uploaded_files(files: list[UploadFile] = File(...)) -> dict[str, Any]:
    ws = runtime.get_workspace()
    ws_dir = runtime.okf_store.workspace_dir(ws.workspace_id)
    upload_dir = ws_dir / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    duck_conn = DuckDBConnector(ws_dir)
    for uf in files:
        fname = uf.filename or "uploaded.csv"
        dest = upload_dir / fname
        content = await uf.read()
        dest.write_bytes(content)
        suffix = Path(fname).suffix.lower()
        if suffix in (".pbix", ".pbit"):
            pbi_conn = PowerBIConnector(mode="powerbi_pbix", workspace_dir=ws_dir, pbix_path=dest)
            runtime.connectors[ws.workspace_id] = pbi_conn
            ws.source_type = "powerbi_pbix"
            ws.dialect = "duckdb"
            runtime.run_onboarding(ws, pbi_conn)
            return runtime.serialize_state()
        else:
            duck_conn.ingest_file(dest, original_filename=fname)

    runtime.connectors[ws.workspace_id] = duck_conn
    ws.source_type = "duckdb"
    ws.dialect = "duckdb"
    runtime.run_onboarding(ws, duck_conn)
    return runtime.serialize_state()


@app.post("/api/connect/sql")
def connect_sql_database(req: SQLConnectRequest) -> dict[str, Any]:
    if not req.connection_url.strip():
        raise HTTPException(status_code=400, detail="Connection URL is required.")
    try:
        ws = runtime.get_workspace()
        runtime.vault.set_secret(req.credential_name, req.connection_url.strip())
        sql_conn = SQLConnector(req.connection_url.strip(), dialect_hint=req.dialect)
        runtime.connectors[ws.workspace_id] = sql_conn
        ws.source_type = "sql"
        ws.dialect = sql_conn.dialect
        ws.credential_ref = req.credential_name
        runtime.run_onboarding(ws, sql_conn)
        return runtime.serialize_state()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"SQL Connection Failed: {exc}") from exc


@app.post("/api/connect/powerbi")
def connect_powerbi(req: PowerBIConnectRequest) -> dict[str, Any]:
    ws = runtime.get_workspace()
    try:
        if req.mode == "powerbi_desktop":
            pbi_conn = PowerBIConnector(mode="powerbi_desktop", localhost_port=req.localhost_port)
            ws.source_type = "powerbi_desktop"
            ws.dialect = "dax"
        else:
            runtime.vault.set_secret("pbi_cloud_token", req.access_token.strip())
            pbi_conn = PowerBIConnector(
                mode="powerbi_cloud",
                dataset_id=req.dataset_id.strip(),
                access_token=req.access_token.strip(),
            )
            ws.source_type = "powerbi_cloud"
            ws.dialect = "dax"
            ws.credential_ref = "pbi_cloud_token"

        runtime.connectors[ws.workspace_id] = pbi_conn
        runtime.run_onboarding(ws, pbi_conn)
        return runtime.serialize_state()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/schema/column-config")
def update_column_config(req: ColumnConfigUpdateRequest) -> dict[str, Any]:
    ws = runtime.get_workspace()
    tbl = ws.get_table(req.table_name)
    if not tbl:
        # Also search blocked tables
        for t in ws.tables:
            if t.name.lower() == req.table_name.lower():
                tbl = t
                break
    if not tbl:
        raise HTTPException(status_code=404, detail=f"Table '{req.table_name}' not found.")

    col = tbl.get_column(req.column_name)
    if not col:
        raise HTTPException(status_code=404, detail=f"Column '{req.column_name}' not found.")

    if req.is_blocked is not None:
        col.is_blocked = req.is_blocked
    if req.description is not None:
        col.description = req.description.strip()

    runtime.okf_store.save(ws)
    InterviewerAgent.sync_schema_to_rag(ws, runtime.get_rag_store())
    return runtime.serialize_state()


@app.post("/api/joins/approve-all")
def approve_all_joins() -> dict[str, Any]:
    ws = runtime.get_workspace()
    for rel in ws.relationships:
        rel.status = JoinStatus.APPROVED
    runtime.okf_store.save(ws)
    return runtime.serialize_state()


@app.post("/api/joins/update")
def update_joins(req: JoinBatchUpdateRequest) -> dict[str, Any]:
    ws = runtime.get_workspace()
    by_id = {r.id: r for r in ws.relationships}
    for item in req.relationships:
        if item.id in by_id:
            r = by_id[item.id]
            r.source_table = item.source_table
            r.source_column = item.source_column
            r.join_type = item.join_type
            r.target_table = item.target_table
            r.target_column = item.target_column
            r.cardinality = JoinCardinality(item.cardinality)
            r.status = JoinStatus(item.status)
    runtime.okf_store.save(ws)
    return runtime.serialize_state()


@app.post("/api/joins/add")
def add_custom_join(req: CustomJoinRequest) -> dict[str, Any]:
    ws = runtime.get_workspace()
    new_id = f"user_rel_{req.source_table}_{req.source_column}_to_{req.target_table}_{req.target_column}"
    ws.relationships.append(
        RelationshipEdge(
            id=new_id,
            source_table=req.source_table,
            source_column=req.source_column,
            target_table=req.target_table,
            target_column=req.target_column,
            join_type=req.join_type,
            cardinality=JoinCardinality(req.cardinality),
            confidence=1.0,
            discovery_method="user_defined",
            status=JoinStatus.APPROVED,
            overlap_ratio=1.0,
        )
    )
    runtime.okf_store.save(ws)
    return runtime.serialize_state()


@app.post("/api/interview/answer")
def answer_interview_question(req: InterviewAnswerRequest) -> dict[str, Any]:
    ws = runtime.get_workspace()
    rag = runtime.get_rag_store()
    InterviewerAgent.record_answer(ws, rag, req.question_id, req.answer)
    runtime.okf_store.save(ws)
    return runtime.serialize_state()


@app.post("/api/interview/add-rule")
def add_business_rule(req: BusinessRuleCreateRequest) -> dict[str, Any]:
    ws = runtime.get_workspace()
    rag = runtime.get_rag_store()
    rid = f"manual_rule_{len(ws.business_rules) + 1}"
    rule = BusinessRule(
        id=rid,
        term=req.term.strip(),
        definition=req.definition.strip(),
        sql_filter_hint=req.sql_filter_hint.strip(),
        source="manual",
    )
    ws.business_rules.append(rule)
    rag.upsert(
        "business_rules",
        rid,
        f"Term: {rule.term}\nDefinition: {rule.definition}\nSQL Filter: {rule.sql_filter_hint}",
        metadata={"term": rule.term, "sql_filter_hint": rule.sql_filter_hint},
    )
    runtime.okf_store.save(ws)
    return runtime.serialize_state()


@app.post("/api/measures/create")
def create_measure(req: MeasureCreateRequest) -> dict[str, Any]:
    ws = runtime.get_workspace()
    rag = runtime.get_rag_store()
    conn = runtime.get_connector()
    meas = MeasureCompiler.create_and_verify_measure(
        workspace=ws,
        rag_store=rag,
        connector=conn,
        name=req.name.strip(),
        table_name=req.table_name.strip(),
        user_formula=req.user_formula.strip(),
        input_type=req.input_type,
        description=req.description.strip(),
        llm_router=runtime.llm_router,
    )
    runtime.okf_store.save(ws)
    state = runtime.serialize_state()
    state["latest_measure"] = meas.model_dump(mode="json")
    return state


@app.post("/api/measures/calculated-column")
def create_calculated_column(req: CalculatedColumnCreateRequest) -> dict[str, Any]:
    ws = runtime.get_workspace()
    rag = runtime.get_rag_store()
    conn = runtime.get_connector()
    cc = MeasureCompiler.create_and_verify_calculated_column(
        workspace=ws,
        rag_store=rag,
        connector=conn,
        table_name=req.table_name.strip(),
        column_name=req.column_name.strip(),
        user_formula=req.user_formula.strip(),
        input_type=req.input_type,
        llm_router=runtime.llm_router,
    )
    runtime.okf_store.save(ws)
    state = runtime.serialize_state()
    state["latest_calculated_column"] = cc.model_dump(mode="json")
    return state


@app.post("/api/chat/query")
def run_chat_query(req: ChatQueryRequest) -> dict[str, Any]:
    ws = runtime.get_workspace()
    rag = runtime.get_rag_store()
    conn = runtime.get_connector()

    overrides = {
        "Revenue Multiplier %": req.revenue_shock_pct / 100.0,
        "Cost Multiplier %": req.cost_shock_pct / 100.0,
    }
    runtime.what_if_overrides[ws.workspace_id] = overrides

    resp, df = QueryAgent.answer_question(
        question=req.question.strip(),
        workspace=ws,
        okf_store=runtime.okf_store,
        rag_store=rag,
        connector=conn,
        llm_router=runtime.llm_router,
        what_if_overrides=overrides,
    )
    rec_chart = ChartRecommender.recommend_chart_type(df, req.question)
    resp.recommended_chart_type = rec_chart

    fig = ChartRecommender.build_figure(df, chart_type=rec_chart, title=req.question)
    rows = df.head(250).to_dict(orient="records") if not df.empty else []
    columns = [str(c) for c in df.columns]

    chat_entry = {
        "id": f"msg_{len(runtime.chat_history.get(ws.workspace_id, [])) + 1}",
        "response": resp.model_dump(mode="json"),
        "rows": rows,
        "columns": columns,
        "row_count": len(df),
        "chart_type": rec_chart,
        "plotly_spec": _fig_to_json(fig),
    }
    runtime.chat_history.setdefault(ws.workspace_id, []).append(chat_entry)
    return {
        "entry": chat_entry,
        "state": runtime.serialize_state(),
    }


@app.post("/api/chat/chart")
def build_chart_for_rows(req: ChartBuildRequest) -> dict[str, Any]:
    df = pd.DataFrame(req.rows)
    fig = ChartRecommender.build_figure(df, chart_type=req.chart_type, title=req.title)
    return {
        "chart_type": req.chart_type,
        "plotly_spec": _fig_to_json(fig),
    }


@app.post("/api/chat/teach")
def teach_rule_from_chat(req: TeachRuleRequest) -> dict[str, Any]:
    ws = runtime.get_workspace()
    rag = runtime.get_rag_store()
    rid = f"chat_rule_{len(ws.business_rules) + 1}"
    sql_hint = "orders.ord_stat <> 'X'" if "ord_stat" in req.rule_text or "'X'" in req.rule_text else ""
    rule = BusinessRule(
        id=rid,
        term=f"Correction on: {req.question[:40]}",
        definition=req.rule_text.strip(),
        sql_filter_hint=sql_hint,
        source="chat_correction",
    )
    ws.business_rules.append(rule)
    rag.upsert("business_rules", rid, req.rule_text.strip(), metadata={"sql_filter_hint": sql_hint})
    runtime.okf_store.save(ws)
    return runtime.serialize_state()


@app.post("/api/root-cause")
def run_root_cause_analysis(req: RootCauseRequest) -> dict[str, Any]:
    ws = runtime.get_workspace()
    conn = runtime.get_connector()
    res = RootCauseAgent.analyze_variance(
        workspace=ws,
        connector=conn,
        metric_col=req.metric_col,
        dimension=req.dimension,
    )
    rc_df: pd.DataFrame = res["df"]
    fig_wf = ChartRecommender.build_figure(
        rc_df,
        chart_type="waterfall",
        title=f"Period-over-Period Variance of {req.metric_col} by {req.dimension}",
        x_col="segment",
    )
    return {
        "summary": res["summary"],
        "rows": rc_df.to_dict(orient="records") if not rc_df.empty else [],
        "columns": [str(c) for c in rc_df.columns],
        "plotly_spec": _fig_to_json(fig_wf),
    }


@app.post("/api/dashboard/pin")
def pin_widget(req: PinWidgetRequest) -> dict[str, Any]:
    ws = runtime.get_workspace()
    widget = PinnedWidget(
        id=f"pin_{len(ws.pinned_widgets) + 1}",
        title=req.title[:70],
        user_question=req.user_question,
        executed_query=req.executed_query,
        query_dialect=req.query_dialect,
        chart_type=req.chart_type,
    )
    ws.pinned_widgets.append(widget)
    runtime.okf_store.save(ws)
    return runtime.serialize_state()


@app.delete("/api/dashboard/pin/{widget_id}")
def delete_pinned_widget(widget_id: str) -> dict[str, Any]:
    ws = runtime.get_workspace()
    ws.pinned_widgets = [w for w in ws.pinned_widgets if w.id != widget_id]
    runtime.okf_store.save(ws)
    return runtime.serialize_state()


@app.post("/api/dashboard/simulate")
def simulate_dashboard(req: WhatIfSimulateRequest) -> dict[str, Any]:
    ws = runtime.get_workspace()
    conn = runtime.get_connector()
    rm = 1.0 + (req.revenue_shock_pct / 100.0)
    cm = 1.0 + (req.cost_shock_pct / 100.0)
    runtime.what_if_overrides[ws.workspace_id] = {
        "Revenue Multiplier %": req.revenue_shock_pct / 100.0,
        "Cost Multiplier %": req.cost_shock_pct / 100.0,
    }

    kpis: dict[str, Any] = {
        "sim_revenue": 0.0,
        "sim_profit": 0.0,
        "sim_margin_pct": 0.0,
        "base_revenue": 0.0,
        "base_profit": 0.0,
    }
    if ws.get_table("orders"):
        try:
            sim_df = conn.execute_query(
                f"SELECT "
                f"ROUND(SUM(rev_net_usd), 2) AS base_rev, "
                f"ROUND(SUM(rev_net_usd - cogs_usd), 2) AS base_prof, "
                f"ROUND(SUM(rev_net_usd * {rm}), 2) AS sim_rev, "
                f"ROUND(SUM((rev_net_usd * {rm}) - (cogs_usd * {cm})), 2) AS sim_prof "
                f"FROM orders WHERE ord_stat <> 'X'",
                max_rows=1,
            )
            if not sim_df.empty:
                r0 = sim_df.iloc[0]
                sim_rev = float(r0["sim_rev"] or 0.0)
                sim_prof = float(r0["sim_prof"] or 0.0)
                kpis = {
                    "base_revenue": float(r0["base_rev"] or 0.0),
                    "base_profit": float(r0["base_prof"] or 0.0),
                    "sim_revenue": sim_rev,
                    "sim_profit": sim_prof,
                    "sim_margin_pct": round((sim_prof / sim_rev) * 100.0, 2) if sim_rev else 0.0,
                }
        except Exception:
            pass

    rendered_widgets: list[dict[str, Any]] = []
    for pw in ws.pinned_widgets:
        try:
            w_df = conn.execute_query(pw.executed_query, max_rows=300)
            fig = ChartRecommender.build_figure(w_df, chart_type=pw.chart_type, title=pw.title)
            rendered_widgets.append(
                {
                    "widget": pw.model_dump(mode="json"),
                    "rows": w_df.head(50).to_dict(orient="records"),
                    "plotly_spec": _fig_to_json(fig),
                    "error": None,
                }
            )
        except Exception as exc:
            rendered_widgets.append(
                {
                    "widget": pw.model_dump(mode="json"),
                    "rows": [],
                    "plotly_spec": None,
                    "error": str(exc),
                }
            )

    return {
        "kpis": kpis,
        "widgets": rendered_widgets,
    }


@app.get("/api/export/okf", response_class=PlainTextResponse)
def export_okf_yaml() -> PlainTextResponse:
    ws = runtime.get_workspace()
    yaml_str = yaml.safe_dump(ws.model_dump(mode="json"), sort_keys=False, allow_unicode=True)
    return PlainTextResponse(
        content=yaml_str,
        media_type="text/yaml",
        headers={"Content-Disposition": f'attachment; filename="{ws.workspace_id}_okf.yaml"'},
    )


@app.get("/api/export/dax", response_class=PlainTextResponse)
def export_dax_measures() -> PlainTextResponse:
    ws = runtime.get_workspace()
    dax_str = MeasureCompiler.format_dax_define_block(ws.measures) or "-- No measures defined yet"
    return PlainTextResponse(
        content=dax_str,
        media_type="text/plain",
        headers={"Content-Disposition": f'attachment; filename="{ws.workspace_id}_measures.dax"'},
    )


class ScratchpadQueryRequest(BaseModel):
    query: str


@app.post("/api/sql/execute")
def execute_scratchpad_query(req: ScratchpadQueryRequest) -> dict[str, Any]:
    """Execute a live read-only SQL or DAX query in the Studio Scratchpad (guarded by PrivacyFirewall AST)."""
    ws = runtime.get_workspace()
    conn = runtime.get_connector()
    try:
        df = conn.execute_query(
            req.query.strip(),
            max_rows=200,
            blocked_tables=ws.blocked_table_names(),
            blocked_columns=ws.blocked_column_names(),
        )
        return {
            "success": True,
            "row_count": len(df),
            "columns": [str(c) for c in df.columns],
            "rows": df.head(100).to_dict(orient="records"),
            "error": None,
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/chat/clear")
def clear_chat_history() -> dict[str, Any]:
    ws_id = runtime.active_workspace_id
    runtime.chat_history[ws_id] = []
    return runtime.serialize_state()


@app.get("/api/health")
def health_check() -> dict[str, Any]:
    import os
    import duckdb

    ws = runtime.get_workspace()
    return {
        "status": "healthy",
        "product": "Data_Talk Enterprise Semantic BI",
        "version": "2.0.0",
        "duckdb_version": duckdb.__version__,
        "active_workspace": ws.workspace_id,
        "active_llm_model": runtime.llm_router.model,
        "providers_ready": {
            "groq": bool(os.environ.get("GROQ_API_KEY")),
            "nvidia_nim": bool(os.environ.get("NVIDIA_NIM_API_KEY") or os.environ.get("NVIDIA_API_KEY")),
        },
        "privacy_mode": runtime.llm_router.firewall.privacy_mode.value,
    }


@app.get("/api/export/report", response_class=PlainTextResponse)
def export_executive_html_report() -> PlainTextResponse:
    """Generate a self-contained, boardroom-ready Executive HTML/PDF Report of the workspace."""
    import html as html_lib

    ws = runtime.get_workspace()
    readiness = InterviewerAgent.compute_readiness_score(ws)
    history = runtime.chat_history.get(ws.workspace_id, [])

    tables_rows = "".join(
        f"<tr><td><strong>{html_lib.escape(t.name)}</strong></td>"
        f"<td>{html_lib.escape(t.table_role.value)}</td>"
        f"<td>{t.row_count:,}</td>"
        f"<td>{len(t.columns)}</td>"
        f"<td>{sum(1 for c in t.columns if c.is_pii)} PII shielded</td></tr>"
        for t in ws.active_tables()
    )

    measures_rows = "".join(
        f"<tr><td><strong>{html_lib.escape(m.table_name)}[{html_lib.escape(m.name)}]</strong></td>"
        f"<td><code>{html_lib.escape(m.dax_expression)}</code></td>"
        f"<td><code>{html_lib.escape(m.sql_expression)}</code></td>"
        f"<td>{f'{m.sample_value:,.2f}' if isinstance(m.sample_value, (int, float)) else 'Verified'}</td></tr>"
        for m in ws.measures
    )

    insights_blocks = "".join(
        f"<div class='card'>"
        f"<h4>Q: {html_lib.escape(item.get('response', {}).get('user_question', ''))}</h4>"
        f"<p>{html_lib.escape(item.get('response', {}).get('executive_summary', ''))}</p>"
        f"<pre>{html_lib.escape(item.get('response', {}).get('executed_query', ''))}</pre>"
        f"</div>"
        for item in history[-10:]
    ) or "<p>No conversational queries recorded in this session yet.</p>"

    report_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>Data_Talk Executive BI Report — {html_lib.escape(ws.workspace_name)}</title>
  <style>
    body {{ font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; color: #0f172a; max-width: 1040px; margin: 36px auto; padding: 0 24px; line-height: 1.55; }}
    .header {{ display: flex; justify-content: space-between; align-items: flex-end; border-bottom: 2px solid #0f172a; padding-bottom: 16px; margin-bottom: 28px; }}
    .badge {{ background: #eff6ff; color: #1d4ed8; border: 1px solid #bfdbfe; padding: 4px 10px; border-radius: 6px; font-weight: 600; font-size: 12px; }}
    table {{ width: 100%; border-collapse: collapse; margin: 14px 0 28px; font-size: 13px; }}
    th, td {{ border: 1px solid #cbd5e1; padding: 9px 12px; text-align: left; }}
    th {{ background: #f8fafc; font-weight: 600; }}
    code, pre {{ font-family: 'JetBrains Mono', Consolas, monospace; font-size: 12px; background: #f1f5f9; padding: 2px 6px; border-radius: 4px; }}
    pre {{ padding: 12px; overflow-x: auto; }}
    .card {{ border: 1px solid #cbd5e1; border-radius: 8px; padding: 16px; margin-bottom: 14px; background: #ffffff; }}
    @media print {{ .no-print {{ display: none; }} }}
  </style>
</head>
<body>
  <div class="header">
    <div>
      <h1 style="margin:0;">Data_Talk — Executive Semantic BI Brief</h1>
      <p style="margin:4px 0 0; color:#475569;">Workspace: <strong>{html_lib.escape(ws.workspace_name)}</strong> • Engine: {html_lib.escape(ws.source_type.upper())} ({html_lib.escape(ws.dialect.upper())})</p>
    </div>
    <div>
      <span class="badge">AI Readiness Score: {readiness['score']}%</span>
      <button class="no-print" onclick="window.print()" style="margin-left:10px; padding:6px 12px; cursor:pointer;">Print / Save as PDF</button>
    </div>
  </div>

  <h2>1. Profiled Data Assets &amp; PII Shield Summary</h2>
  <table>
    <thead><tr><th>Table</th><th>Semantic Role</th><th>Row Count</th><th>Columns</th><th>Privacy Status</th></tr></thead>
    <tbody>{tables_rows}</tbody>
  </table>

  <h2>2. Governed Semantic Measures (Power BI DAX &harr; SQL)</h2>
  <table>
    <thead><tr><th>Measure</th><th>Power BI DAX</th><th>Compiled SQL</th><th>Live Sample</th></tr></thead>
    <tbody>{measures_rows}</tbody>
  </table>

  <h2>3. Conversational BI Insights &amp; Verified Query Lineage</h2>
  {insights_blocks}
</body>
</html>"""

    return PlainTextResponse(
        content=report_html,
        media_type="text/html",
        headers={"Content-Disposition": f'attachment; filename="{ws.workspace_id}_executive_report.html"'},
    )


# Serve static assets & index.html
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/")
def serve_index() -> FileResponse:
    return FileResponse(str(FRONTEND_DIR / "index.html"))


if __name__ == "__main__":
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False)

