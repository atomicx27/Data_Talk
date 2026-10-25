"""Shared pytest fixtures for Data_Talk test suites."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Generator

# Force deterministic offline mode during pytest suites to prevent network latency and LLM rate-limit flakes
os.environ["DATA_TALK_OFFLINE_MODE"] = "1"

import pytest
from fastapi.testclient import TestClient

from src.api import create_app
from src.api.runtime import AppRuntime
from src.security.auth import AuthConfig, Authenticator


@pytest.fixture
def temp_workspace() -> Generator[Path, None, None]:
    """Provide an isolated temporary directory for test workspaces and storage."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def test_runtime(temp_workspace: Path) -> AppRuntime:
    """Create an AppRuntime instance backed by a temporary workspace."""
    return AppRuntime(base_dir=temp_workspace)


@pytest.fixture
def test_client(test_runtime: AppRuntime) -> TestClient:
    """Create an unauthenticated FastAPI TestClient in open mode."""
    import server

    server.runtime = test_runtime
    server.authenticator = Authenticator(AuthConfig(api_token=None))
    app = create_app(runtime=test_runtime, auth_config=server.authenticator.config)
    return TestClient(app)


@pytest.fixture
def authed_client(test_runtime: AppRuntime) -> tuple[TestClient, str]:
    """Create a FastAPI TestClient configured with a required API token."""
    import server

    token = "secret-test-token-12345"
    server.runtime = test_runtime
    server.authenticator = Authenticator(AuthConfig(api_token=token))
    app = create_app(runtime=test_runtime, auth_config=server.authenticator.config)
    client = TestClient(app)
    return client, token
