"""Unit and integration tests for PromptManager (dynamic prompt externalization & hot-reloading)."""

import os
import time
from pathlib import Path

import pytest

from src.agents.prompt_manager import PromptManager


@pytest.fixture(autouse=True)
def clean_prompt_cache():
    """Clear cached prompts before and after each test."""
    PromptManager.clear_cache()
    yield
    PromptManager.clear_cache()


def test_load_all_built_in_templates():
    """Verify that all 8 root prompt templates exist, parse cleanly into system and user sections."""
    expected_templates = [
        "sql_compiler",
        "self_healing_critic",
        "follow_up_resolver",
        "prompt_improver",
        "dataset_overview",
        "chart_thread_enrichment",
        "measure_compiler",
        "dax_transpiler",
    ]

    for name in expected_templates:
        sys_tmpl, usr_tmpl = PromptManager.load_template(name)
        assert sys_tmpl, f"Template '{name}' missing system prompt section"
        assert usr_tmpl, f"Template '{name}' missing user prompt section"
        assert len(sys_tmpl.strip()) > 20, f"Template '{name}' system section is too short"
        assert len(usr_tmpl.strip()) > 10, f"Template '{name}' user section is too short"


def test_safe_formatting_with_raw_json():
    """Verify that PromptManager._safe_format substitutes kwargs without choking on raw JSON braces."""
    template = (
        "Return strictly JSON: {\"sql_query\": \"SELECT * FROM {table_name}\", \"status\": \"{status}\"}\n"
        "Unrelated curly braces: {foo} and {bar}."
    )
    rendered = PromptManager._safe_format(
        template,
        table_name="students",
        status="ok",
        foo="hello",
        bar="world",
    )
    assert '{"sql_query": "SELECT * FROM students", "status": "ok"}' in rendered
    assert "Unrelated curly braces: hello and world." in rendered


def test_render_with_actual_template():
    """Test rendering the sql_compiler template with real parameters."""
    sys_rendered, usr_rendered = PromptManager.render(
        "sql_compiler",
        fallback_system="fallback sys",
        fallback_user="fallback usr",
        dialect="duckdb",
        fan_trap_guidance="Prevent fan traps using CTEs.",
        schema_lines="- students (id, name)",
        joins_text="No joins needed.",
        measures_text="None",
        rules_text="None",
        what_if_text="",
        question="Show total students",
    )

    assert "duckdb" in sys_rendered
    assert "Prevent fan traps using CTEs." in sys_rendered
    assert "students (id, name)" in usr_rendered
    assert "Show total students" in usr_rendered
    # Check that raw JSON format example is preserved
    assert '{"sql_query":' in sys_rendered


def test_fallback_when_template_missing(tmp_path):
    """Verify safe fallback to in-code defaults when a template file does not exist."""
    missing_name = "non_existent_special_template_12345"
    sys, usr = PromptManager.render(
        missing_name,
        fallback_system="System Default {dialect}",
        fallback_user="User Default {question}",
        dialect="postgres",
        question="What is total revenue?",
    )

    assert sys == "System Default postgres"
    assert usr == "User Default What is total revenue?"


def test_hot_reloading(tmp_path, monkeypatch):
    """Verify that editing a template file on disk updates the prompt without server restart."""
    # Point prompts directory to tmp_path
    monkeypatch.setenv("DATA_TALK_PROMPTS_DIR", str(tmp_path))

    template_file = tmp_path / "custom_task.md"
    template_file.write_text(
        "# System Prompt\nInitial system prompt for {task}\n\n# User Prompt\nInitial user question: {query}",
        encoding="utf-8",
    )

    # Initial render
    sys1, usr1 = PromptManager.render(
        "custom_task",
        fallback_system="",
        fallback_user="",
        task="analysis",
        query="show count",
    )
    assert sys1 == "Initial system prompt for analysis"
    assert usr1 == "Initial user question: show count"

    # Sleep briefly to ensure mtime timestamp differences on Windows file systems
    time.sleep(0.05)

    # Simulate user modifying the file on disk
    template_file.write_text(
        "# System Prompt\nMODIFIED system prompt: Be extra formal for {task}\n\n# User Prompt\nMODIFIED user: {query}",
        encoding="utf-8",
    )
    # Ensure mtime is strictly newer
    new_mtime = time.time() + 1.0
    os.utime(template_file, (new_mtime, new_mtime))

    # Second render should hot-reload the updated template immediately!
    sys2, usr2 = PromptManager.render(
        "custom_task",
        fallback_system="",
        fallback_user="",
        task="executive briefing",
        query="show breakdown",
    )
    assert sys2 == "MODIFIED system prompt: Be extra formal for executive briefing"
    assert usr2 == "MODIFIED user: show breakdown"
