"""Unit tests for LLMRouter and PromptManager."""
from __future__ import annotations

from unittest.mock import patch

from src.agents.llm_router import LLMRouter
from src.agents.prompt_manager import PromptManager


def test_llm_router_provider_credentials_and_json_array(monkeypatch) -> None:
    monkeypatch.delenv("DATA_TALK_OFFLINE_MODE", raising=False)
    for var in (
        "GROQ_API_KEY",
        "NVIDIA_NIM_API_KEY",
        "NVIDIA_API_KEY",
        "OPENAI_API_KEY",
        "GEMINI_API_KEY",
        "ANTHROPIC_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)

    router = LLMRouter(model="openai/gpt-4o")
    assert router.is_configured() is False

    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_key_123")
    groq_router = LLMRouter(model="groq/llama-3.3-70b-versatile")
    assert groq_router.is_configured() is True

    # Explicit Groq key resolves while OpenAI remains unconfigured
    _, openai_key, _ = router._resolve_provider_credentials("openai/gpt-4o")
    _, groq_key, _ = router._resolve_provider_credentials("groq/llama-3.3-70b-versatile")
    assert openai_key == "gsk_test_key_123"
    assert groq_key == "gsk_test_key_123"

    # complete_json handles JSON arrays cleanly
    with patch.object(router, "complete", return_value='```json\n[{"a": 1}, {"a": 2}]\n```'):
        parsed = router.complete_json("task", "sys", "usr")
        assert parsed == [{"a": 1}, {"a": 2}]


def test_prompt_manager_single_pass_safe_format() -> None:
    # A value containing {braces} must NOT be re-interpolated on a second pass
    rendered = PromptManager._safe_format(
        "Question: {question} | Schema: {schema}",
        question="Check {schema} literal",
        schema="orders(id)",
    )
    assert rendered == "Question: Check {schema} literal | Schema: orders(id)"
