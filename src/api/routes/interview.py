"""AI business interviewer and rule ingestion routes."""

from __future__ import annotations

import re
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException

from src.agents.interviewer_agent import InterviewerAgent
from src.agents.knowledge_enricher import BusinessKnowledgeEnricher
from src.api.deps import get_runtime, get_session
from src.api.runtime import AppRuntime, SessionState
from src.api.schemas import (
    BusinessRuleCreateRequest,
    BusinessRuleUpdateRequest,
    DomainOverrideRequest,
    InterviewAnswerRequest,
)
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


@router.post("/auto-enrich")
def auto_enrich_knowledge(
    runtime: Annotated[AppRuntime, Depends(get_runtime)],
    session: Annotated[SessionState, Depends(get_session)],
) -> dict[str, Any]:
    ws_id = session.active_workspace_id
    with runtime.workspace_lock(ws_id):
        ws = runtime.get_workspace(ws_id)
        rag = runtime.get_rag_store(ws.workspace_id)
        report = BusinessKnowledgeEnricher.auto_enrich_workspace(
            ws, llm_router=session.llm_router, rag_store=rag
        )
        InterviewerAgent.sync_schema_to_rag(ws, rag)
        runtime.okf_store.save(ws)
    return {
        "report": report,
        "state": runtime.serialize_state(session),
    }


@router.get("/domain")
def get_domain_info(
    runtime: Annotated[AppRuntime, Depends(get_runtime)],
    session: Annotated[SessionState, Depends(get_session)],
) -> dict[str, Any]:
    ws = runtime.get_workspace(session.active_workspace_id)
    return BusinessKnowledgeEnricher.detect_domain(ws)


@router.post("/domain-override")
def override_domain_info(
    req: DomainOverrideRequest,
    runtime: Annotated[AppRuntime, Depends(get_runtime)],
    session: Annotated[SessionState, Depends(get_session)],
) -> dict[str, Any]:
    ws_id = session.active_workspace_id
    with runtime.workspace_lock(ws_id):
        ws = runtime.get_workspace(ws_id)
        ws.domain_title = req.domain_title.strip()
        if req.domain_tag:
            ws.domain_tag = req.domain_tag.strip().lower()
        else:
            ws.domain_tag = re.sub(r"[^a-z0-9]+", "_", req.domain_title.lower()).strip("_")

        domain_info = {
            "domain_key": ws.domain_tag,
            "domain_title": ws.domain_title,
            "industry": req.industry or ws.domain_title,
            "summary": f"Business domain: {ws.domain_title}. Custom domain defined by business owner.",
        }
        rag = runtime.get_rag_store(ws.workspace_id)
        BusinessKnowledgeEnricher.sync_enriched_knowledge_to_rag(ws, rag, domain_info)
        InterviewerAgent.sync_schema_to_rag(ws, rag)
        runtime.okf_store.save(ws)
    return runtime.serialize_state(session)


@router.put("/rule/{rule_id}")
def update_business_rule(
    rule_id: str,
    req: BusinessRuleUpdateRequest,
    runtime: Annotated[AppRuntime, Depends(get_runtime)],
    session: Annotated[SessionState, Depends(get_session)],
) -> dict[str, Any]:
    ws_id = session.active_workspace_id
    with runtime.workspace_lock(ws_id):
        ws = runtime.get_workspace(ws_id)
        target_rule = next((r for r in ws.business_rules if r.id == rule_id), None)
        if not target_rule:
            raise HTTPException(status_code=404, detail=f"Business rule '{rule_id}' not found.")

        target_rule.term = req.term.strip()
        target_rule.definition = req.definition.strip()
        target_rule.sql_filter_hint = req.sql_filter_hint.strip()

        rag = runtime.get_rag_store(ws.workspace_id)
        rag.upsert(
            "business_rules",
            target_rule.id,
            f"Term: {target_rule.term}\nDefinition: {target_rule.definition}\nSQL Filter: {target_rule.sql_filter_hint}",
            metadata={"term": target_rule.term, "sql_filter_hint": target_rule.sql_filter_hint},
        )
        runtime.okf_store.save(ws)
    return runtime.serialize_state(session)


@router.delete("/rule/{rule_id}")
def delete_business_rule(
    rule_id: str,
    runtime: Annotated[AppRuntime, Depends(get_runtime)],
    session: Annotated[SessionState, Depends(get_session)],
) -> dict[str, Any]:
    ws_id = session.active_workspace_id
    with runtime.workspace_lock(ws_id):
        ws = runtime.get_workspace(ws_id)
        target_rule = next((r for r in ws.business_rules if r.id == rule_id), None)
        if not target_rule:
            raise HTTPException(status_code=404, detail=f"Business rule '{rule_id}' not found.")

        ws.business_rules = [r for r in ws.business_rules if r.id != rule_id]
        rag = runtime.get_rag_store(ws.workspace_id)
        rag.delete("business_rules", rule_id)
        runtime.okf_store.save(ws)
    return runtime.serialize_state(session)
