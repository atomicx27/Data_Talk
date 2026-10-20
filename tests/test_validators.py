"""Unit tests for input validation, path containment, and filename safety."""

from pathlib import Path
import pytest

from src.security.validators import (
    ValidationError,
    ensure_within,
    normalize_workspace_id,
    safe_upload_name,
    validate_http_url,
)


def test_normalize_workspace_id_valid():
    assert normalize_workspace_id("my_workspace") == "my_workspace"
    assert normalize_workspace_id("Sales Demo 2026") == "sales_demo_2026"
    assert normalize_workspace_id("finance-dept") == "finance-dept"
    assert normalize_workspace_id("  team_a  ") == "team_a"


def test_normalize_workspace_id_invalid_traversal():
    with pytest.raises(ValidationError):
        normalize_workspace_id("../../etc/passwd")

    with pytest.raises(ValidationError):
        normalize_workspace_id("..\\..\\windows\\system32")

    with pytest.raises(ValidationError):
        normalize_workspace_id("/root")

    with pytest.raises(ValidationError):
        normalize_workspace_id("C:\\Windows")

    with pytest.raises(ValidationError):
        normalize_workspace_id("")

    with pytest.raises(ValidationError):
        normalize_workspace_id("a" * 100)


def test_safe_upload_name():
    assert safe_upload_name("data.csv") == "data.csv"
    assert safe_upload_name("../../sales.csv") == "sales.csv"  # stripped directory component
    # Note: server.py has .py extension which must be rejected
    with pytest.raises(ValidationError):
        safe_upload_name("../../server.py")

    assert safe_upload_name("nested/path/sales_2026.parquet") == "sales_2026.parquet"
    assert safe_upload_name("C:\\Users\\admin\\Desktop\\orders.xlsx") == "orders.xlsx"

    with pytest.raises(ValidationError):
        safe_upload_name("malware.exe")

    with pytest.raises(ValidationError):
        safe_upload_name("script.sh")


def test_ensure_within(tmp_path: Path):
    base = tmp_path / "workspaces"
    base.mkdir()
    child = base / "demo"
    child.mkdir()

    assert ensure_within(base, child) == child.resolve()

    outside = tmp_path / "other"
    outside.mkdir()
    with pytest.raises(ValidationError):
        ensure_within(base, outside)


def test_validate_http_url():
    assert validate_http_url("http://localhost:11434/v1") == "http://localhost:11434/v1"
    assert validate_http_url("https://api.groq.com/openai/v1") == "https://api.groq.com/openai/v1"

    with pytest.raises(ValidationError):
        validate_http_url("file:///etc/passwd")

    with pytest.raises(ValidationError):
        validate_http_url("javascript:alert(1)")

    with pytest.raises(ValidationError):
        validate_http_url("not-a-url")
