"""Multi-Provider LiteLLM Router with Groq + NVIDIA NIM Failover & Privacy Firewall.

Supports:
- Groq Cloud (`groq/openai/gpt-oss-120b`, `groq/qwen/qwen3.8-27b`, `groq/openai/gpt-oss-20b`)
- NVIDIA NIM (`nvidia_nim/nvidia/nemotron-3.5-lightning-30b-a3b`, `nvidia_nim/nvidia/nemotron-3-super-120b-a12b`, `nvidia_nim/nv-mistralai/mistral-nemo-12b-instruct`)
- OpenAI (`gpt-4o`, `gpt-4o-mini`)
- Google Gemini (`gemini/gemini-2.5-pro`, `gemini/gemini-2.5-flash`)
- Anthropic (`claude-3-7-sonnet-latest`)
- Local Air-Gapped Ollama (`ollama/qwen2.5-coder:14b`, `ollama/llama3.2`)

Every prompt is scrubbed and recorded via `PrivacyFirewall.inspect_and_scrub_outbound_prompt()`
before calling `litellm.completion()`, giving users a transparent Outbound Payload Inspector.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import litellm
from dotenv import load_dotenv

from src.security.privacy_firewall import PrivacyFirewall, PrivacyMode

# Load .env from project root automatically
_ROOT_ENV = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(dotenv_path=_ROOT_ENV, override=False)

# Suppress verbose litellm debug logs
litellm.suppress_debug_info = True


class LLMRouter:
    """Unified LiteLLM completion and structured-JSON router with Groq <-> NVIDIA NIM failover."""

    SUPPORTED_MODELS = [
        "groq/openai/gpt-oss-120b",
        "groq/qwen/qwen3.8-27b",
        "groq/openai/gpt-oss-20b",
        "nvidia_nim/nvidia/nemotron-3.5-lightning-30b-a3b",
        "nvidia_nim/nvidia/nemotron-3-super-120b-a12b",
        "nvidia_nim/nv-mistralai/mistral-nemo-12b-instruct",
        "gpt-4o",
        "gpt-4o-mini",
        "gemini/gemini-2.5-flash",
        "gemini/gemini-2.5-pro",
        "claude-3-7-sonnet-latest",
        "ollama/qwen2.5-coder:14b",
        "ollama/llama3.2",
    ]

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        privacy_firewall: PrivacyFirewall | None = None,
    ) -> None:
        default_model = os.environ.get("DATA_TALK_DEFAULT_MODEL", "groq/openai/gpt-oss-120b")
        self.model = model or default_model
        self.api_key = api_key
        self.api_base = api_base
        self.firewall = privacy_firewall or PrivacyFirewall(PrivacyMode.STRICT_SCHEMA_ONLY)

    @staticmethod
    def _resolve_provider_credentials(
        target_model: str,
        explicit_key: str | None = None,
        explicit_base: str | None = None,
    ) -> tuple[str, str | None, str | None]:
        """Resolve LiteLLM model string, API key, and API base URL for Groq, NVIDIA NIM, or other providers."""
        if explicit_key and len(explicit_key.strip()) > 5:
            if target_model.startswith("nvidia_nim/"):
                raw_model_id = target_model.removeprefix("nvidia_nim/")
                return f"openai/{raw_model_id}", explicit_key.strip(), explicit_base or "https://integrate.api.nvidia.com/v1"
            return target_model, explicit_key.strip(), explicit_base

        if target_model.startswith("groq/"):
            key = os.environ.get("GROQ_API_KEY")
            return target_model, key, explicit_base

        if target_model.startswith("nvidia_nim/"):
            key = os.environ.get("NVIDIA_NIM_API_KEY") or os.environ.get("NVIDIA_API_KEY")
            raw_model_id = target_model.removeprefix("nvidia_nim/")
            base_url = explicit_base or "https://integrate.api.nvidia.com/v1"
            return f"openai/{raw_model_id}", key, base_url

        for env_var in ("OPENAI_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY", "GROQ_API_KEY", "NVIDIA_NIM_API_KEY"):
            val = os.environ.get(env_var)
            if val:
                return target_model, val, explicit_base

        return target_model, None, explicit_base

    def is_configured(self) -> bool:
        """Return True if an API key is configured for the active/fallback provider or Ollama is selected."""
        if os.environ.get("DATA_TALK_OFFLINE_MODE", "").lower() in ("1", "true", "yes"):
            return False
        if self.model.startswith("ollama/"):
            return True
        if self.api_key and len(self.api_key.strip()) > 5:
            return True
        for env_var in (
            "GROQ_API_KEY",
            "NVIDIA_NIM_API_KEY",
            "NVIDIA_API_KEY",
            "OPENAI_API_KEY",
            "GEMINI_API_KEY",
            "ANTHROPIC_API_KEY",
        ):
            if os.environ.get(env_var):
                return True
        return False

    def complete(
        self,
        purpose: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
    ) -> str:
        """Send a privacy-scrubbed chat completion request with automatic Groq <-> NVIDIA NIM failover."""
        raw_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        safe_messages = self.firewall.inspect_and_scrub_outbound_prompt(
            purpose=purpose,
            model=self.model,
            messages=raw_messages,
        )

        # Build candidate failover list: primary model first, then automatic Groq/NVIDIA backup
        candidate_models = [self.model]
        if self.model.startswith("groq/") and (os.environ.get("NVIDIA_NIM_API_KEY") or os.environ.get("NVIDIA_API_KEY")):
            candidate_models.append("nvidia_nim/nvidia/nemotron-3.5-lightning-30b-a3b")
        elif self.model.startswith("nvidia_nim/") and os.environ.get("GROQ_API_KEY"):
            candidate_models.append("groq/openai/gpt-oss-120b")

        last_exc: Exception | None = None
        for cand_model in candidate_models:
            litellm_model, resolved_key, resolved_base = self._resolve_provider_credentials(
                cand_model,
                explicit_key=self.api_key if cand_model == self.model else None,
                explicit_base=self.api_base if cand_model == self.model else None,
            )
            kwargs: dict[str, Any] = {
                "model": litellm_model,
                "messages": safe_messages,
                "temperature": temperature,
                "timeout": 20,
            }
            if resolved_key:
                kwargs["api_key"] = resolved_key
            if resolved_base:
                kwargs["api_base"] = resolved_base

            try:
                response = litellm.completion(**kwargs)
                msg = response.choices[0].message
                content = getattr(msg, "content", None) or getattr(msg, "reasoning_content", None) or ""
                return str(content).strip()
            except Exception as exc:
                last_exc = exc
                continue

        if last_exc is not None:
            raise last_exc
        raise RuntimeError("No LLM provider succeeded.")

    def complete_json(
        self,
        purpose: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        """Request a JSON object from LiteLLM and parse it reliably."""
        raw_text = self.complete(
            purpose=purpose,
            system_prompt=system_prompt + "\nReturn ONLY valid JSON with no markdown prose.",
            user_prompt=user_prompt,
            temperature=temperature,
        )
        cleaned = re.sub(r"^```(?:json)?\s*", "", raw_text.strip(), flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned.strip())
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            cleaned = cleaned[start : end + 1]
        return json.loads(cleaned)
