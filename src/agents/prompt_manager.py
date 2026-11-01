"""Dynamic Prompt Template Manager with Hot-Reloading and Safe Code Fallbacks.

Loads prompt templates from the `prompts/` directory at runtime.
Automatically detects when a template file is edited on disk and hot-reloads it
without requiring a server restart, with safe fallbacks to built-in code defaults.
"""

from __future__ import annotations

import logging
import os
import re
import threading
from pathlib import Path
from typing import Any

_LOGGER = logging.getLogger("datatalk.prompts")

# Resolve project root (two levels up from src/agents)
BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROMPTS_DIR = BASE_DIR / "prompts"


class PromptManager:
    """Manages loading, caching, hot-reloading, and rendering of external prompt templates."""

    _cache: dict[str, dict[str, Any]] = {}
    _lock = threading.Lock()

    @classmethod
    def get_prompts_dir(cls) -> Path:
        """Return the active prompts directory (overrideable via DATA_TALK_PROMPTS_DIR env var)."""
        env_dir = os.environ.get("DATA_TALK_PROMPTS_DIR")
        if env_dir:
            return Path(env_dir)
        return PROMPTS_DIR

    @classmethod
    def _parse_template_file(cls, content: str) -> tuple[str, str]:
        """Split a template markdown file into (system_prompt, user_prompt) sections.

        Expects sections marked by:
        # System Prompt (or ## System Prompt / # System)
        # User Prompt (or ## User Prompt / # User)
        """
        lines = content.splitlines()
        system_lines: list[str] = []
        user_lines: list[str] = []
        current_section: str | None = None

        for line in lines:
            stripped = line.strip().lower()
            if re.match(r"^#{1,3}\s+(system\s+prompt|system)\b", stripped):
                current_section = "system"
                continue
            elif re.match(r"^#{1,3}\s+(user\s+prompt|user)\b", stripped):
                current_section = "user"
                continue

            if current_section == "system":
                system_lines.append(line)
            elif current_section == "user":
                user_lines.append(line)
            else:
                # Content before any header is treated as system prompt by default
                system_lines.append(line)

        system_str = "\n".join(system_lines).strip()
        user_str = "\n".join(user_lines).strip()
        return system_str, user_str

    @classmethod
    def load_template(
        cls,
        name: str,
        fallback_system: str = "",
        fallback_user: str = "",
    ) -> tuple[str, str]:
        """Load and return the raw (system_template, user_template) tuple.

        Checks file mtime on disk and hot-reloads automatically if modified.
        Falls back to provided defaults if the file is missing or unreadable.
        """
        prompts_dir = cls.get_prompts_dir()
        file_path = prompts_dir / f"{name}.md"
        if not file_path.exists():
            file_path = prompts_dir / f"{name}.txt"

        if not file_path.exists():
            return fallback_system, fallback_user

        try:
            mtime = file_path.stat().st_mtime
            with cls._lock:
                cached = cls._cache.get(name)
                if cached and cached.get("mtime") == mtime and cached.get("path") == str(file_path):
                    return cached["system"], cached["user"]

                # File modified or not cached yet: hot-reload from disk
                content = file_path.read_text(encoding="utf-8")
                sys_tmpl, usr_tmpl = cls._parse_template_file(content)

                # If either section is empty, allow fallback to supply the missing section
                final_sys = sys_tmpl if sys_tmpl else fallback_system
                final_usr = usr_tmpl if usr_tmpl else fallback_user

                cls._cache[name] = {
                    "mtime": mtime,
                    "path": str(file_path),
                    "system": final_sys,
                    "user": final_usr,
                }
                _LOGGER.info("Hot-reloaded prompt template '%s' from %s", name, file_path)
                return final_sys, final_usr
        except Exception as exc:
            _LOGGER.warning("Error loading prompt template '%s' from %s: %s (using code fallback)", name, file_path, exc)
            return fallback_system, fallback_user

    @classmethod
    def _safe_format(cls, template: str, **kwargs: Any) -> str:
        """Safely substitute {variable} placeholders in a template.

        Uses regex replacement so raw JSON braces (like {"sql_query": "..."})
        do not trigger Python KeyError or ValueError.
        """
        if not template:
            return ""

        # Replace known {var_name} patterns while ignoring raw json like {"sql_query": "..."}
        result = template
        for k, v in kwargs.items():
            pattern = re.compile(rf"\{{{re.escape(k)}\}}")
            result = pattern.sub(lambda _: str(v), result)
        return result

    @classmethod
    def render(
        cls,
        name: str,
        fallback_system: str,
        fallback_user: str,
        **kwargs: Any,
    ) -> tuple[str, str]:
        """Load template by name, apply kwargs substitution, and return (rendered_system, rendered_user).

        If formatting fails or placeholders are missing, returns the safely formatted code fallbacks.
        """
        raw_sys, raw_usr = cls.load_template(name, fallback_system, fallback_user)
        try:
            rendered_sys = cls._safe_format(raw_sys, **kwargs)
            rendered_usr = cls._safe_format(raw_usr, **kwargs)
            return rendered_sys, rendered_usr
        except Exception as exc:
            _LOGGER.warning("Formatting error in prompt '%s': %s (using code fallback)", name, exc)
            return cls._safe_format(fallback_system, **kwargs), cls._safe_format(fallback_user, **kwargs)

    @classmethod
    def clear_cache(cls) -> None:
        """Clear cached templates in memory."""
        with cls._lock:
            cls._cache.clear()
