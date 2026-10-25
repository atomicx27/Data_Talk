"""Runtime state for the Data_Talk API.

State is split into two scopes:

* **Shared** (:class:`AppRuntime`): persisted workspaces (OKF), RAG stores, live data connectors,
  the credential vault, and per-workspace locks. These represent on-disk data and are shared by all users.
* **Per-session** (:class:`SessionState`): which workspace a browser is looking at, its chat history,
  What-If overrides, and its own LLM settings / privacy mode / outbound audit log.
"""

from __future__ import annotations

import os
import threading
import time
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.agents.interviewer_agent import InterviewerAgent
from src.agents.knowledge_enricher import BusinessKnowledgeEnricher
from src.agents.llm_router import LLMRouter
from src.agents.measure_compiler import MeasureCompiler
from src.connectors.base import BaseConnector
from src.connectors.duckdb_connector import DuckDBConnector
from src.connectors.powerbi_connector import PowerBIConnector
from src.connectors.profiler import StatisticalProfiler
from src.connectors.sql_connector import SQLConnector
from src.knowledge.join_inference import JoinInferenceEngine
from src.knowledge.okf_models import OKFWorkspace
from src.knowledge.okf_store import OKFStore
from src.knowledge.rag_store import RAGStore
from src.security.privacy_firewall import CredentialVault, PrivacyFirewall, PrivacyMode
from src.viz.chart_recommender import ChartRecommender

DEFAULT_WORKSPACE_ID = "enterprise_demo"
DEFAULT_WHAT_IF = {"Revenue Multiplier %": 0.0, "Cost Multiplier %": 0.0}


def _new_llm_router() -> LLMRouter:
    return LLMRouter(privacy_firewall=PrivacyFirewall(PrivacyMode.STRICT_SCHEMA_ONLY))


@dataclass
class SessionState:
    """Everything that is private to one browser session (or to bearer-token API clients)."""

    session_id: str
    active_workspace_id: str = DEFAULT_WORKSPACE_ID
    chat_history: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    what_if_overrides: dict[str, dict[str, float]] = field(default_factory=dict)
    llm_router: LLMRouter = field(default_factory=_new_llm_router)
    last_seen: float = field(default_factory=time.monotonic)


class AppRuntime:
    """Shared, thread-safe state manager for the Data_Talk FastAPI server."""

    MAX_SESSIONS = 200

    def __init__(self, base_dir: Path | None = None, default_workspace_id: str = DEFAULT_WORKSPACE_ID) -> None:
        root = base_dir or Path(__file__).resolve().parent.parent.parent
        self.base_dir = root
        self.okf_store = OKFStore(base_dir=root / "workspaces")
        self.vault = CredentialVault(base_dir=root / ".secrets")
        for env_k in ("GROQ_API_KEY", "NVIDIA_NIM_API_KEY", "NVIDIA_API_KEY"):
            val = os.environ.get(env_k)
            if val and self.vault.get_secret(env_k) != val:
                try:
                    self.vault.set_secret(env_k, val)
                except Exception:  # unreadable vault must not prevent the server from starting
                    pass
        self.default_workspace_id = default_workspace_id
        self.connectors: dict[str, BaseConnector] = {}
        self.rag_stores: dict[str, RAGStore] = {}
        self._sessions: OrderedDict[str, SessionState] = OrderedDict()
        self._state_lock = threading.RLock()
        self._ws_locks: dict[str, threading.RLock] = {}

    # ------------------------------------------------------------------ sessions
    def get_session(self, session_id: str) -> SessionState:
        """Return (creating if needed) the session, evicting the least-recently-used beyond the cap."""
        with self._state_lock:
            sess = self._sessions.get(session_id)
            if sess is None:
                sess = SessionState(session_id=session_id, active_workspace_id=self.default_workspace_id)
                self._sessions[session_id] = sess
                while len(self._sessions) > self.MAX_SESSIONS:
                    self._sessions.popitem(last=False)
            else:
                self._sessions.move_to_end(session_id)
            sess.last_seen = time.monotonic()
            return sess

    @property
    def session_count(self) -> int:
        return len(self._sessions)

    # ------------------------------------------------------------------ locking
    @contextmanager
    def workspace_lock(self, workspace_id: str) -> Iterator[None]:
        """Serialise load -> mutate -> save cycles on a single workspace."""
        with self._state_lock:
            lock = self._ws_locks.setdefault(workspace_id, threading.RLock())
        with lock:
            yield

    # ------------------------------------------------------------------ shared resources
    def get_workspace(self, workspace_id: str) -> OKFWorkspace:
        ws = self.okf_store.load(workspace_id)
        if ws is None:
            ws = OKFWorkspace(
                workspace_id=workspace_id,
                workspace_name=workspace_id.replace("_", " ").title(),
                source_type="duckdb",
                dialect="duckdb",
            )
            self.okf_store.save(ws)
        return ws

    def get_rag_store(self, workspace_id: str) -> RAGStore:
        with self._state_lock:
            if workspace_id not in self.rag_stores:
                self.rag_stores[workspace_id] = RAGStore(self.okf_store.workspace_dir(workspace_id))
            return self.rag_stores[workspace_id]

    def set_connector(self, workspace_id: str, connector: BaseConnector) -> None:
        with self._state_lock:
            self.connectors[workspace_id] = connector

    def get_connector(self, workspace_id: str) -> BaseConnector:
        with self._state_lock:
            if workspace_id not in self.connectors:
                ws = self.get_workspace(workspace_id)
                if ws.source_type == "federated" or ws.data_sources:
                    from src.connectors.federated_connector import FederatedConnector

                    fed_conn = FederatedConnector(self.okf_store.workspace_dir(workspace_id))
                    for s in ws.data_sources:
                        if s.source_type == "sql" and s.credential_ref:
                            secret_url = self.vault.get_secret(s.credential_ref)
                            if secret_url:
                                fed_conn.register_sql_connector(s.id, secret_url, dialect_hint=s.dialect)
                    self.connectors[workspace_id] = fed_conn
                    return self.connectors[workspace_id]
                if ws.source_type == "sql" and ws.credential_ref:
                    secret_url = self.vault.get_secret(ws.credential_ref)
                    if secret_url:
                        self.connectors[workspace_id] = SQLConnector(secret_url, dialect_hint=ws.dialect)
                        return self.connectors[workspace_id]
                self.connectors[workspace_id] = DuckDBConnector(self.okf_store.workspace_dir(workspace_id))
            return self.connectors[workspace_id]

    def list_workspaces(self, extra: str | None = None) -> list[str]:
        names = set(self.okf_store.list_workspaces())
        if extra:
            names.add(extra)
        return sorted(names)

    def delete_workspace(self, workspace_id: str, session: SessionState | None = None) -> bool:
        """Safely release connectors and locks, permanently delete workspace files from disk, and reassign sessions."""
        from src.security.validators import normalize_workspace_id
        ws_id = normalize_workspace_id(workspace_id)
        with self._state_lock:
            # 1. Close and remove any active connector for this workspace
            conn = self.connectors.pop(ws_id, None)
            if conn is not None and hasattr(conn, "close"):
                try:
                    conn.close()
                except Exception:
                    pass

            # 2. Drop RAG store cache
            self.rag_stores.pop(ws_id, None)

            # 3. Force garbage collection to release file handles on Windows
            import gc
            gc.collect()

            # 4. Remove workspace directory and all files from disk
            deleted = self.okf_store.delete_workspace(ws_id)

            # 5. Remove any workspace locks
            self._ws_locks.pop(ws_id, None)

            # 6. Reassign any session currently pointing to the deleted workspace
            remaining = self.okf_store.list_workspaces()
            fallback_ws = remaining[0] if remaining else self.default_workspace_id

            for sess in self._sessions.values():
                if sess.active_workspace_id == ws_id:
                    sess.active_workspace_id = fallback_ws
                    sess.chat_history.pop(ws_id, None)
                    sess.what_if_overrides.pop(ws_id, None)

            if session and session.active_workspace_id == ws_id:
                session.active_workspace_id = fallback_ws
                session.chat_history.pop(ws_id, None)
                session.what_if_overrides.pop(ws_id, None)

            # Ensure the fallback workspace exists in OKF
            self.get_workspace(fallback_ws)

            return deleted

    # ------------------------------------------------------------------ onboarding
    def run_onboarding(self, workspace: OKFWorkspace, connector: BaseConnector, llm_router: LLMRouter) -> None:
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
        BusinessKnowledgeEnricher.auto_enrich_workspace(workspace, llm_router=llm_router, rag_store=rag_store)
        InterviewerAgent.sync_schema_to_rag(workspace, rag_store)
        InterviewerAgent.generate_prioritized_questions(workspace, llm_router=llm_router)
        self.okf_store.save(workspace)

    # ------------------------------------------------------------------ serialisation
    def serialize_state(self, session: SessionState) -> dict[str, Any]:
        ws_id = session.active_workspace_id
        ws = self.get_workspace(ws_id)
        rag = self.get_rag_store(ws_id)
        readiness = InterviewerAgent.compute_readiness_score(ws)
        domain_info = BusinessKnowledgeEnricher.detect_domain(ws)
        router = session.llm_router
        return {
            "active_workspace_id": ws_id,
            "workspaces": self.list_workspaces(extra=ws_id),
            "workspace": ws.model_dump(mode="json"),
            "domain_info": domain_info,
            "readiness": readiness,
            "rag_counts": rag.count_by_collection(),
            "privacy_mode": router.firewall.privacy_mode.value,
            "llm_model": router.model,
            "has_api_key": router.is_configured(),
            "supported_models": LLMRouter.SUPPORTED_MODELS,
            "supported_chart_types": ChartRecommender.SUPPORTED_CHART_TYPES,
            "starter_templates": MeasureCompiler.STARTER_MEASURE_TEMPLATES,
            "desktop_pbi_instances": PowerBIConnector.discover_local_desktop_instances(),
            "chat_history": session.chat_history.get(ws_id, []),
            "what_if_overrides": session.what_if_overrides.get(ws_id, dict(DEFAULT_WHAT_IF)),
            "audit_log": router.firewall.audit_log[-25:],
        }
