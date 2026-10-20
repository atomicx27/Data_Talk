"""AI business interviewer and rule ingestion routes."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends

from src.agents.interviewer_agent import InterviewerAgent
from src.api.deps import get_runtime, get_session
from src.api.runtime import AppRuntime, SessionState
from src.api.schemas import BusinessRuleCreateRequest, InterviewAnswerRequest
from src.knowledge.okf_models import BusinessRule

router = APIRouter(prefix="/api/interview", tags=["Interview"])


@router.post("/answer")
def answer_interview_question(
    req: InterviewAnswerRequest,
    runtime: Annotated[AppRuntime, Depends(get_runtime)],
    session: Annotated[SessionState, Depends(get_session)],
) -> dict[str, Any]:
    ws_id = session.active_workspace_id
    with runtime.workspace_lock(ws_id):
        ws = runtime.get_workspace(ws_id)
        rag = runtime.get_rag_store(ws.workspace_id)
        InterviewerAgent.record_answer(ws, rag, req.question_id, req.answer)
        runtime.okf_store.save(ws)
    return runtime.serialize_state(session)


@router.post("/add-rule")
def add_business_rule(
    req: BusinessRuleCreateRequest,
    runtime: Annotated[AppRuntime, Depends(get_runtime)],
    session: Annotated[SessionState, Depends(get_session)],
) -> dict[str, Any]:
    ws_id = session.active_workspace_id
    with runtime.workspace_lock(ws_id):
        ws = runtime.get_workspace(ws_id)
        rag = runtime.get_rag_store(ws.workspace_id)
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
    return runtime.serialize_state(session)
