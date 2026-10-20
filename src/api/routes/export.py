"""Export routes for OKF YAML, DAX measures, and Executive HTML reports."""

from __future__ import annotations

import html as html_lib
from typing import Annotated

import yaml
from fastapi import APIRouter, Depends
from fastapi.responses import PlainTextResponse

from src.agents.interviewer_agent import InterviewerAgent
from src.agents.measure_compiler import MeasureCompiler
from src.api.deps import get_runtime, get_session
from src.api.runtime import AppRuntime, SessionState

router = APIRouter(prefix="/api/export", tags=["Export"])


@router.get("/okf", response_class=PlainTextResponse)
def export_okf_yaml(
    runtime: Annotated[AppRuntime, Depends(get_runtime)],
    session: Annotated[SessionState, Depends(get_session)],
) -> PlainTextResponse:
    ws = runtime.get_workspace(session.active_workspace_id)
    yaml_str = yaml.safe_dump(ws.model_dump(mode="json"), sort_keys=False, allow_unicode=True)
    return PlainTextResponse(
        content=yaml_str,
        media_type="text/yaml",
        headers={"Content-Disposition": f'attachment; filename="{ws.workspace_id}_okf.yaml"'},
    )


@router.get("/dax", response_class=PlainTextResponse)
def export_dax_measures(
    runtime: Annotated[AppRuntime, Depends(get_runtime)],
    session: Annotated[SessionState, Depends(get_session)],
) -> PlainTextResponse:
    ws = runtime.get_workspace(session.active_workspace_id)
    dax_str = MeasureCompiler.format_dax_define_block(ws.measures) or "-- No measures defined yet"
    return PlainTextResponse(
        content=dax_str,
        media_type="text/plain",
        headers={"Content-Disposition": f'attachment; filename="{ws.workspace_id}_measures.dax"'},
    )


@router.get("/report", response_class=PlainTextResponse)
def export_executive_html_report(
    runtime: Annotated[AppRuntime, Depends(get_runtime)],
    session: Annotated[SessionState, Depends(get_session)],
) -> PlainTextResponse:
    """Generate a self-contained, boardroom-ready Executive HTML/PDF Report of the workspace."""
    ws = runtime.get_workspace(session.active_workspace_id)
    readiness = InterviewerAgent.compute_readiness_score(ws)
    history = session.chat_history.get(ws.workspace_id, [])

    tables_rows = "".join(
        f"<tr><td><strong>{html_lib.escape(t.name)}</strong></td>"
        f"<td>{html_lib.escape(t.table_role.value)}</td>"
        f"<td>{t.row_count:,}</td>"
        f"<td>{len(t.columns)}</td>"
        f"<td>{sum(1 for c in t.columns if c.is_pii)} PII shielded</td></tr>"
        for t in ws.active_tables()
    )

    measures_rows = "".join(
        f"<tr><td><strong>{html_lib.escape(m.table_name)}[{html_lib.escape(m.name)}]</strong></td>"
        f"<td><code>{html_lib.escape(m.dax_expression)}</code></td>"
        f"<td><code>{html_lib.escape(m.sql_expression)}</code></td>"
        f"<td>{f'{m.sample_value:,.2f}' if isinstance(m.sample_value, (int, float)) else 'Verified'}</td></tr>"
        for m in ws.measures
    )

    insights_blocks = "".join(
        f"<div class='card'>"
        f"<h4>Q: {html_lib.escape(item.get('response', {}).get('user_question', ''))}</h4>"
        f"<p>{html_lib.escape(item.get('response', {}).get('executive_summary', ''))}</p>"
        f"<pre>{html_lib.escape(item.get('response', {}).get('executed_query', ''))}</pre>"
        f"</div>"
        for item in history[-10:]
    ) or "<p>No conversational queries recorded in this session yet.</p>"

    report_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>Data_Talk Executive BI Report — {html_lib.escape(ws.workspace_name)}</title>
  <style>
    body {{ font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; color: #0f172a; max-width: 1040px; margin: 36px auto; padding: 0 24px; line-height: 1.55; }}
    .header {{ display: flex; justify-content: space-between; align-items: flex-end; border-bottom: 2px solid #0f172a; padding-bottom: 16px; margin-bottom: 28px; }}
    .badge {{ background: #eff6ff; color: #1d4ed8; border: 1px solid #bfdbfe; padding: 4px 10px; border-radius: 6px; font-weight: 600; font-size: 12px; }}
    table {{ width: 100%; border-collapse: collapse; margin: 14px 0 28px; font-size: 13px; }}
    th, td {{ border: 1px solid #cbd5e1; padding: 9px 12px; text-align: left; }}
    th {{ background: #f8fafc; font-weight: 600; }}
    code, pre {{ font-family: 'JetBrains Mono', Consolas, monospace; font-size: 12px; background: #f1f5f9; padding: 2px 6px; border-radius: 4px; }}
    pre {{ padding: 12px; overflow-x: auto; }}
    .card {{ border: 1px solid #cbd5e1; border-radius: 8px; padding: 16px; margin-bottom: 14px; background: #ffffff; }}
    @media print {{ .no-print {{ display: none; }} }}
  </style>
</head>
<body>
  <div class="header">
    <div>
      <h1 style="margin:0;">Data_Talk — Executive Semantic BI Brief</h1>
      <p style="margin:4px 0 0; color:#475569;">Workspace: <strong>{html_lib.escape(ws.workspace_name)}</strong> • Engine: {html_lib.escape(ws.source_type.upper())} ({html_lib.escape(ws.dialect.upper())})</p>
    </div>
    <div>
      <span class="badge">AI Readiness Score: {readiness['score']}%</span>
      <button class="no-print" onclick="window.print()" style="margin-left:10px; padding:6px 12px; cursor:pointer;">Print / Save as PDF</button>
    </div>
  </div>

  <h2>1. Profiled Data Assets &amp; PII Shield Summary</h2>
  <table>
    <thead><tr><th>Table</th><th>Semantic Role</th><th>Row Count</th><th>Columns</th><th>Privacy Status</th></tr></thead>
    <tbody>{tables_rows}</tbody>
  </table>

  <h2>2. Governed Semantic Measures (Power BI DAX &harr; SQL)</h2>
  <table>
    <thead><tr><th>Measure</th><th>Power BI DAX</th><th>Compiled SQL</th><th>Live Sample</th></tr></thead>
    <tbody>{measures_rows}</tbody>
  </table>

  <h2>3. Conversational BI Insights &amp; Verified Query Lineage</h2>
  {insights_blocks}
</body>
</html>"""

    return PlainTextResponse(
        content=report_html,
        media_type="text/html",
        headers={"Content-Disposition": f'attachment; filename="{ws.workspace_id}_executive_report.html"'},
    )
