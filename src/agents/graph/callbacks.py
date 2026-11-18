"""Zero-dependency per-node lifecycle telemetry & safe stream emission hooks."""

from __future__ import annotations

import functools
import inspect
import time
from collections.abc import Callable
from typing import Any, TypeVar

from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer
from langgraph.errors import GraphBubbleUp
from langgraph.types import Command

from src.agents.graph.state import NodeTelemetryRecord

F = TypeVar("F", bound=Callable[..., Any])


def emit_stream_event(event: dict[str, Any]) -> None:
    """Safely emit a custom LangGraph stream event; no-ops outside a runnable stream context."""
    try:
        writer = get_stream_writer()
    except RuntimeError:
        return
    except Exception:
        return
    if callable(writer):
        try:
            writer(event)
        except Exception:
            pass


def _extract_attempt(state: Any, update: dict[str, Any]) -> int:
    """Read current retry attempt cleanly from Pydantic state or update dict."""
    raw = None
    if isinstance(state, dict):
        raw = state.get("attempt")
    else:
        raw = getattr(state, "attempt", None)
    if raw is None:
        raw = update.get("attempt", 1)
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return 1


def _resolve_status(update: dict[str, Any]) -> str:
    """Determine node completion status without inspecting sensitive values."""
    if update.get("interrupted"):
        return "interrupted"
    if update.get("execution_succeeded") is False or update.get("error_diagnostic") is not None:
        return "error"
    return "ok"


def _sanitize_updated_keys(update: dict[str, Any]) -> list[str]:
    """Return sorted updated state field names only — never raw values, SQL rows, or PII."""
    ignored = {"node_trace", "node_telemetry"}
    return sorted(str(k) for k in update.keys() if k not in ignored)


def trace_graph_node(node_name: str) -> Callable[[F], F]:
    """Wrap a LangGraph node with timing telemetry while preserving Command & GraphInterrupt."""

    def decorator(func: F) -> F:
        params = list(inspect.signature(func).parameters.values())
        accepts_config = len(params) >= 2

        @functools.wraps(func)
        def wrapper(state: Any, config: RunnableConfig | None = None, *args: Any, **kwargs: Any) -> Any:
            t0 = time.perf_counter()
            try:
                if accepts_config:
                    result = func(state, config, *args, **kwargs)
                else:
                    result = func(state, *args, **kwargs)
            except GraphBubbleUp:
                # Explicitly preserve GraphInterrupt / ParentCommand bubble-up control flow
                raise

            duration_ms = round((time.perf_counter() - t0) * 1000.0, 3)

            if isinstance(result, Command):
                update = dict(result.update or {})
                attempt = _extract_attempt(state, update)
                status = _resolve_status(update)
                updated_keys = _sanitize_updated_keys(update)
                record = NodeTelemetryRecord(
                    node=node_name,
                    duration_ms=duration_ms,
                    attempt=attempt,
                    status=status,  # type: ignore[arg-type]
                    updated_keys=updated_keys,
                )
                update["node_trace"] = [*list(update.get("node_trace") or []), node_name]
                update["node_telemetry"] = [*list(update.get("node_telemetry") or []), record]
                emit_stream_event(
                    {
                        "type": "node_update",
                        "node": node_name,
                        "duration_ms": duration_ms,
                        "attempt": attempt,
                        "status": status,
                        "updated_keys": updated_keys,
                    }
                )
                return Command(
                    update=update,
                    goto=result.goto,
                    graph=result.graph,
                    resume=result.resume,
                )

            if isinstance(result, dict):
                update = dict(result)
                attempt = _extract_attempt(state, update)
                status = _resolve_status(update)
                updated_keys = _sanitize_updated_keys(update)
                record = NodeTelemetryRecord(
                    node=node_name,
                    duration_ms=duration_ms,
                    attempt=attempt,
                    status=status,  # type: ignore[arg-type]
                    updated_keys=updated_keys,
                )
                update["node_trace"] = [*list(update.get("node_trace") or []), node_name]
                update["node_telemetry"] = [*list(update.get("node_telemetry") or []), record]
                emit_stream_event(
                    {
                        "type": "node_update",
                        "node": node_name,
                        "duration_ms": duration_ms,
                        "attempt": attempt,
                        "status": status,
                        "updated_keys": updated_keys,
                    }
                )
                return update

            return result

        return wrapper  # type: ignore[return-value]

    return decorator
