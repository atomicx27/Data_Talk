/**
 * langgraph_3d_trace.js — LangGraph SSE Streaming, Dual HITL Interrupt/Resume,
 * 3D Lineage Stage Pipeline, and Live Graph Topology Inspector.
 */

function esc(str) {
  return String(str ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

const NODE_DISPLAY_NAMES = {
  normalize_intent: 'Normalize Intent',
  plan_schema_and_joins: 'Steiner Join Plan',
  check_semantic_ambiguity: 'Ambiguity Check',
  clarification_interrupt: 'HITL Clarify',
  build_out_of_domain: 'Out-of-Domain Guard',
  compile_initial_query: 'SQL & DAX Compile',
  execute_sandbox: 'Firewall & DuckDB',
  critic_repair_query: 'AST Self-Healing',
  post_process_and_summarize: 'Executive Synthesis',
  handle_exhausted_failure: 'Diagnostic Fallback',
  classify_bi_intent: 'Supervisor Route',
  run_query_agent: 'QueryAgent Subgraph',
  run_root_cause: 'Root-Cause Subgraph',
  run_simulation: 'Simulation Subgraph',
};

function labelForNode(nodeId) {
  return NODE_DISPLAY_NAMES[nodeId] || String(nodeId || 'node').replace(/_/g, ' ');
}

/**
 * Stream POST /api/chat/query/stream using Fetch ReadableStream SSE parser.
 */
export async function streamQuestion(body, callbacks = {}, signal = undefined) {
  const resp = await fetch('/api/chat/query/stream', {
    method: 'POST',
    credentials: 'same-origin',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal,
  });

  if (resp.status === 401) {
    throw new Error('Unauthorized: session expired.');
  }
  if (!resp.ok || !resp.body) {
    let errText = `HTTP ${resp.status}`;
    try {
      const parsed = await resp.json();
      errText = parsed.detail || JSON.stringify(parsed);
    } catch (_) {
      errText = (await resp.text().catch(() => '')) || errText;
    }
    throw new Error(errText);
  }

  const reader = resp.body.getReader();
  const decoder = new TextDecoder('utf-8');
  let buffer = '';
  let completedPayload = null;
  let interruptedPayload = null;

  const processBlock = (rawBlock) => {
    const lines = rawBlock.split(/\r?\n/);
    let eventName = 'message';
    const dataLines = [];
    for (const line of lines) {
      if (!line) continue;
      if (line.startsWith('event:')) {
        eventName = line.slice(6).trim();
      } else if (line.startsWith('data:')) {
        dataLines.push(line.slice(5).trimStart());
      }
    }
    if (!dataLines.length) return;
    let data = dataLines.join('\n');
    try {
      data = JSON.parse(data);
    } catch (_) {
      // Keep raw text fallback
    }

    if (eventName === 'stage' && typeof callbacks.onStage === 'function') {
      callbacks.onStage(data);
    } else if (eventName === 'node_update' && typeof callbacks.onNodeUpdate === 'function') {
      callbacks.onNodeUpdate(data);
    } else if (eventName === 'clarification_hint' && typeof callbacks.onClarificationHint === 'function') {
      callbacks.onClarificationHint(data);
    } else if (eventName === 'interrupt') {
      interruptedPayload = data;
      if (typeof callbacks.onInterrupt === 'function') callbacks.onInterrupt(data);
    } else if (eventName === 'error' && typeof callbacks.onError === 'function') {
      callbacks.onError(data);
    } else if (eventName === 'complete') {
      completedPayload = data;
      if (typeof callbacks.onComplete === 'function') callbacks.onComplete(data);
    }
  };

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let sepIdx;
    while ((sepIdx = buffer.indexOf('\n\n')) !== -1) {
      const block = buffer.slice(0, sepIdx);
      buffer = buffer.slice(sepIdx + 2);
      if (block.trim()) processBlock(block);
    }
  }
  if (buffer.trim()) {
    processBlock(buffer);
  }

  return { completed: completedPayload, interrupted: interruptedPayload };
}

/**
 * Render live SSE node pills inside the streaming typing card.
 */
export function renderLiveStreamStatus(el, state) {
  if (!el) return;
  const { stageLabel = 'Orchestrating LangGraph state engine…', nodes = [], hint = null } = state || {};
  const pills = nodes
    .map((n) => {
      const cls = n.status === 'error' ? 'tone-error' : n.status === 'interrupted' ? 'tone-warn' : 'tone-ok';
      const ms = n.duration_ms !== undefined ? ` ${Number(n.duration_ms).toFixed(1)}ms` : '';
      return `<span class="dt-sse-pill ${cls}">${esc(labelForNode(n.node))}${ms}</span>`;
    })
    .join('');

  const hintHtml = hint
    ? `<div class="dt-inline-hint-banner"><strong>Clarification hint:</strong> ${esc(
        hint.clarification_prompt || hint.message || JSON.stringify(hint)
      )}</div>`
    : '';

  el.innerHTML = `
    <div class="dt-sse-live-wrap">
      <div class="dt-sse-stage-row">
        <span class="dt-pulse-dot"></span>
        <span class="dt-sse-stage-label">${esc(stageLabel)}</span>
      </div>
      ${pills ? `<div class="dt-sse-pills-row">${pills}</div>` : ''}
      ${hintHtml}
    </div>
  `;
}

export function renderNonBlockingHintBanner(resp) {
  if (!resp || resp.clarification_severity !== 'non_blocking_inline' || !resp.clarification_suggestion) {
    return '';
  }
  return `
    <div class="dt-inline-hint-banner">
      <svg class="icon" style="width:14px;height:14px;flex-shrink:0"><use href="#i-alert-circle"/></svg>
      <div><strong>Ambiguity Notice (Non-Blocking):</strong> ${esc(resp.clarification_suggestion)}</div>
    </div>
  `;
}

export function renderTelemetrySummaryBadge(resp) {
  if (!resp) return '';
  const totalMs = Number(resp.total_latency_ms || 0);
  const steps = Array.isArray(resp.node_trace) ? resp.node_trace.length : 0;
  if (!totalMs && !steps) return '';
  return `<span class="dt-telemetry-chip" title="LangGraph execution telemetry: ${steps} nodes">${steps} nodes · ${totalMs.toFixed(
    1
  )} ms</span>`;
}

/**
 * Dual HITL blocking_modal dialog population & submission.
 */
export function openInterruptModal(interruptInfo, onResumeSubmit) {
  const dialog = document.getElementById('clarificationInterruptDialog');
  if (!dialog || !interruptInfo) return;

  const payload = interruptInfo.interrupt_payload || interruptInfo;
  const threadId = interruptInfo.thread_id || '';
  const question = payload.question || interruptInfo.question || '';
  const promptText = payload.clarification_prompt || 'Please clarify your intent so compilation can continue safely.';
  const options = Array.isArray(payload.suggested_options) ? payload.suggested_options : [];
  const questionId = payload.question_id || '';

  const qEl = document.getElementById('hitlQuestionContext');
  const pEl = document.getElementById('hitlClarificationPrompt');
  const optsEl = document.getElementById('hitlOptionsContainer');
  const ansInput = document.getElementById('hitlAnswerInput');
  const saveChk = document.getElementById('hitlSaveRuleCheckbox');
  const tidInput = document.getElementById('hitlThreadIdInput');
  const qidInput = document.getElementById('hitlQuestionIdInput');
  const form = document.getElementById('clarificationInterruptForm');
  const submitBtn = document.getElementById('hitlResumeSubmitBtn');

  if (qEl) qEl.textContent = question ? `Original question: "${question}"` : '';
  if (pEl) pEl.textContent = promptText;
  if (tidInput) tidInput.value = threadId;
  if (qidInput) qidInput.value = questionId;
  if (ansInput) ansInput.value = options[0] || '';
  if (saveChk) saveChk.checked = true;

  if (optsEl) {
    if (options.length) {
      optsEl.innerHTML = options
        .map(
          (opt, idx) =>
            `<button type="button" class="btn btn-secondary btn-xs dt-hitl-option-chip ${
              idx === 0 ? 'active' : ''
            }" data-opt="${esc(opt)}">${esc(opt)}</button>`
        )
        .join('');
      optsEl.querySelectorAll('.dt-hitl-option-chip').forEach((btn) => {
        btn.addEventListener('click', () => {
          optsEl.querySelectorAll('.dt-hitl-option-chip').forEach((b) => b.classList.remove('active'));
          btn.classList.add('active');
          if (ansInput) ansInput.value = btn.getAttribute('data-opt') || '';
        });
      });
    } else {
      optsEl.innerHTML = '<span class="muted small">Provide your preferred table, metric, or business definition below.</span>';
    }
  }

  if (form) {
    // Rebind cleanly by replacing onsubmit handler
    form.onsubmit = async (e) => {
      e.preventDefault();
      const answer = (ansInput?.value || '').trim();
      if (!answer) return;
      if (submitBtn) {
        submitBtn.disabled = true;
        submitBtn.textContent = 'Resuming LangGraph…';
      }
      try {
        await onResumeSubmit({
          thread_id: tidInput?.value || threadId,
          answer,
          save_rule: Boolean(saveChk?.checked),
          question_id: qidInput?.value || questionId || undefined,
        });
        dialog.close();
      } finally {
        if (submitBtn) {
          submitBtn.disabled = false;
          submitBtn.textContent = 'Resume Execution';
        }
      }
    };
  }

  if (!dialog.open) {
    dialog.showModal();
  }
}

/**
 * Render isometric 3D stage plates inside #lineage3dPipeline using node_trace & node_telemetry.
 */
export function renderLineage3DPipeline(container, resp) {
  if (!container || !resp) return;
  const telemetry = Array.isArray(resp.node_telemetry) && resp.node_telemetry.length ? resp.node_telemetry : null;
  const trace = Array.isArray(resp.node_trace) ? resp.node_trace : [];

  const nodes = telemetry
    ? telemetry
    : trace.map((nid, idx) => ({
        node: nid,
        duration_ms: 0,
        attempt: idx + 1,
        status: 'ok',
        updated_keys: [],
      }));

  if (!nodes.length) {
    container.innerHTML = `
      <div class="dt-lineage3d-empty">
        Executed via <code class="mono">${esc(resp.compiler_detail || resp.compiler_mode || 'Deterministic Compiler')}</code>
      </div>
    `;
    return;
  }

  const totalMs = Number(resp.total_latency_ms || 0);
  const platesHtml = nodes
    .map((n, idx) => {
      const status = n.status || 'ok';
      const tone = status === 'error' ? 'stage-err' : status === 'interrupted' ? 'stage-warn' : 'stage-ok';
      const ms = Number(n.duration_ms || 0);
      const arrow = idx < nodes.length - 1 ? '<span class="dt-stage-connector" aria-hidden="true">→</span>' : '';
      return `
        <div class="dt-stage-plate ${tone}" style="--stage-idx:${idx};" title="${esc(
        `Updated keys: ${(n.updated_keys || []).join(', ') || 'none'}`
      )}">
          <div class="dt-stage-step">#${idx + 1} · att ${esc(n.attempt || 1)}</div>
          <div class="dt-stage-name">${esc(labelForNode(n.node))}</div>
          <div class="dt-stage-ms">${ms > 0 ? `${ms.toFixed(1)} ms` : 'fast path'}</div>
        </div>
        ${arrow}
      `;
    })
    .join('');

  container.innerHTML = `
    <div class="dt-lineage3d-header">
      <span class="dt-lineage3d-title">3D LangGraph Execution Pipeline</span>
      <span class="dt-telemetry-chip">Total: ${totalMs.toFixed(1)} ms · ${nodes.length} steps</span>
    </div>
    <div class="dt-lineage3d-track">${platesHtml}</div>
  `;
}

const _TOPOLOGY_CACHE = {};

export async function loadAndRenderTopology(panelEl, graphType = 'query') {
  if (!panelEl) return;
  panelEl.innerHTML = '<div class="muted small">Fetching live LangGraph topology…</div>';
  try {
    if (!_TOPOLOGY_CACHE[graphType]) {
      const resp = await fetch(`/api/chat/graph-topology?graph_type=${encodeURIComponent(graphType)}`, {
        credentials: 'same-origin',
      });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      _TOPOLOGY_CACHE[graphType] = await resp.json();
    }
    const topo = _TOPOLOGY_CACHE[graphType];
    const nodeBadges = (topo.nodes || [])
      .map((n) => `<span class="meta-chip mono">${esc(n.id || n.name)}</span>`)
      .join(' ');
    const edgeRows = (topo.edges || [])
      .map(
        (e) =>
          `<div class="dt-topo-edge"><code class="mono">${esc(e.source)}</code> → <code class="mono">${esc(
            e.target
          )}</code> ${e.conditional ? '<span class="dt-cond-tag">conditional</span>' : ''}</div>`
      )
      .join('');

    panelEl.innerHTML = `
      <div class="dt-topo-summary">
        <strong>Graph:</strong> <code class="mono">${esc(topo.graph_type)}</code> ·
        <strong>Checkpointer:</strong> <code class="mono">${esc(topo.checkpointer)}</code> ·
        <span>${(topo.nodes || []).length} nodes / ${(topo.edges || []).length} edges</span>
      </div>
      <div class="dt-topo-nodes">${nodeBadges}</div>
      <div class="dt-topo-edges-grid">${edgeRows}</div>
      <details class="dt-diag-details" style="margin-top:8px;">
        <summary>Mermaid Graph Definition</summary>
        <pre class="code-box mono">${esc(topo.mermaid || '')}</pre>
      </details>
    `;
  } catch (err) {
    panelEl.innerHTML = `<div class="muted small">Could not inspect topology: ${esc(err?.message || err)}</div>`;
  }
}
