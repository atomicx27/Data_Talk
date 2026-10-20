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
Then open http://localhost:8504 in your browser.
"""

from __future__ import annotations

import os
from pathlib import Path

import uvicorn

from src.api import create_app
from src.api.runtime import AppRuntime, SessionState
from src.api.schemas import *  # noqa: F403 (backwards-compat export)
from src.security.auth import AuthConfig, Authenticator

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"

# Global runtime & auth instances
runtime = AppRuntime(base_dir=BASE_DIR)
auth_config = AuthConfig.from_env()
authenticator = Authenticator(auth_config)

# FastAPI application instance
app = create_app(runtime=runtime, auth_config=auth_config)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8504"))
    host = os.environ.get("HOST", "127.0.0.1")
    uvicorn.run("server:app", host=host, port=port, reload=False)
