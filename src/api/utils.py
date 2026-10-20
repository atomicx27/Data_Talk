"""Shared helpers for API route formatting and figures."""

from __future__ import annotations

import json
from typing import Any

import plotly.io as pio


def fig_to_json(fig: Any) -> dict[str, Any] | None:
    """Style Plotly figure for the executive dark theme and serialize to JSON dict."""
    if fig is None:
        return None
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.45)",
        font={"family": "Inter, system-ui, sans-serif", "color": "#E2E8F0", "size": 12},
        margin={"l": 48, "r": 24, "t": 48, "b": 42},
        title_font={"size": 14, "color": "#F8FAFC"},
    )
    return json.loads(pio.to_json(fig))
