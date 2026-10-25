"""Pydantic request models for the Data_Talk REST API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class SettingsUpdateRequest(BaseModel):
    workspace_id: str | None = None
    privacy_mode: str | None = None
    llm_model: str | None = None
    api_key: str | None = None
    api_base: str | None = None


class WorkspaceDeleteRequest(BaseModel):
    workspace_id: str


class SQLConnectRequest(BaseModel):
    dialect: str = "postgres"
    credential_name: str = Field("prod_sql_db", pattern=r"^[A-Za-z0-9_.-]{1,64}$")
    connection_url: str
    source_id: str | None = None
    source_name: str | None = None
    attach_mode: bool = True


class RemoveSourceRequest(BaseModel):
    source_id: str


class PowerBIConnectRequest(BaseModel):
    mode: str = "powerbi_desktop"  # powerbi_desktop | powerbi_cloud
    localhost_port: int = Field(55000, ge=1, le=65535)
    dataset_id: str = ""
    access_token: str = ""


class ColumnConfigUpdateRequest(BaseModel):
    table_name: str
    column_name: str
    business_alias: str | None = None
    is_blocked: bool | None = None
    description: str | None = None
    value_meanings: dict[str, str] | None = None


class TableConfigUpdateRequest(BaseModel):
    table_name: str
    business_name: str | None = None
    description: str | None = None
    table_role: str | None = None


class DomainOverrideRequest(BaseModel):
    domain_title: str
    domain_tag: str | None = None
    industry: str | None = None


class BusinessRuleUpdateRequest(BaseModel):
    term: str
    definition: str
    sql_filter_hint: str = ""


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
    question: str = Field(..., max_length=4000)
    revenue_shock_pct: float = 0.0
    cost_shock_pct: float = 0.0


class ChartBuildRequest(BaseModel):
    rows: list[dict[str, Any]] = Field(..., max_length=5000)
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
    revenue_shock_pct: float = Field(0.0, ge=-100.0, le=1000.0)
    cost_shock_pct: float = Field(0.0, ge=-100.0, le=1000.0)


class ScratchpadQueryRequest(BaseModel):
    query: str = Field(..., max_length=20000)
