/**
 * error_boundary.js — End-to-End Structured Error Control & Diagnostic UI System.
 * Normalizes backend ErrorDiagnostic payloads, HTTP errors, SSE failures, JS exceptions,
 * and WebGL context loss into actionable cards with node badges & 1-click recovery actions.
 */

function esc(str) {
  return String(str ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

const STRATEGY_BUTTONS = {
  retry_recompile: [
    { action: 'retry-recompile', label: 'AI Recompile (Bypass Cache)', tone: 'primary' },
    { action: 'force-heuristic', label: 'Run Safe Heuristic Fallback', tone: 'secondary' },
    { action: 'open-scratchpad', label: 'Inspect in SQL Scratchpad', tone: 'ghost' },
  ],
  force_heuristic: [
    { action: 'force-heuristic', label: 'Switch to Deterministic Heuristic', tone: 'primary' },
    { action: 'retry-recompile', label: 'Retry AI Compile', tone: 'secondary' },
  ],
  open_joins: [
    { action: 'open-joins', label: 'Inspect ER Joins', tone: 'primary' },
    { action: 'retry-recompile', label: 'Re-run Query', tone: 'secondary' },
  ],
  open_llm_settings: [
    { action: 'open-llm', label: 'Configure LLM / Privacy', tone: 'primary' },
    { action: 'force-heuristic', label: 'Run Local Heuristic (No LLM)', tone: 'secondary' },
  ],
  open_scratchpad: [
    { action: 'open-scratchpad', label: 'Open SQL Scratchpad', tone: 'primary' },
    { action: 'retry-recompile', label: 'AI Recompile', tone: 'secondary' },
  ],
  reconnect_stream: [
    { action: 'retry-sync', label: 'Retry Standard Sync Query', tone: 'primary' },
    { action: 'force-heuristic', label: 'Run Heuristic Fallback', tone: 'secondary' },
  ],
};

function inferCategoryFromText(text) {
  const lower = String(text || '').toLowerCase();
  if (lower.includes('401') || lower.includes('403') || lower.includes('forbidden') || lower.includes('unauthorized')) {
    return {
      code: 'ERR_AUTH_PERMISSION',
      category: 'security',
      failed_node: 'api_gateway',
      remediation_hint: 'Session authorization or CORS origin check failed. Reload or verify API token settings.',
      retry_strategy: 'open_llm_settings',
    };
  }
  if (lower.includes('timeout') || lower.includes('network') || lower.includes('failed to fetch') || lower.includes('sse')) {
    return {
      code: 'ERR_NETWORK_STREAM',
      category: 'network',
      failed_node: 'sse_transport',
      remediation_hint: 'Network or SSE stream interrupted before completion. Retry via synchronous query or safe heuristic.',
      retry_strategy: 'reconnect_stream',
    };
  }
  if (lower.includes('join') || lower.includes('table') || lower.includes('binder') || lower.includes('column')) {
    return {
      code: 'ERR_SQL_SCHEMA_BINDING',
      category: 'execution',
      failed_node: 'execute_and_heal',
      remediation_hint: 'DuckDB could not bind referenced table or column. Inspect ER relationships or recompile with full schema.',
      retry_strategy: 'retry_recompile',
    };
  }
  if (lower.includes('firewall') || lower.includes('read-only') || lower.includes('pii') || lower.includes('privacy')) {
    return {
      code: 'ERR_PRIVACY_FIREWALL',
      category: 'security',
      failed_node: 'sql_firewall',
      remediation_hint: 'Privacy & Read-Only Firewall blocked unsafe statement or restricted identifier.',
      retry_strategy: 'open_scratchpad',
    };
  }
  if (lower.includes('llm') || lower.includes('api key') || lower.includes('quota') || lower.includes('rate limit')) {
    return {
      code: 'ERR_LLM_PROVIDER',
      category: 'llm',
      failed_node: 'generate_sql',
      remediation_hint: 'LLM provider returned an error or is unavailable. Configure credentials or run local heuristic mode.',
      retry_strategy: 'open_llm_settings',
    };
  }
  if (lower.includes('webgl') || lower.includes('3d') || lower.includes('force-graph')) {
    return {
      code: 'ERR_RENDER_3D_CONTEXT',
      category: 'render',
      failed_node: 'frontend_3d_stage',
      remediation_hint: 'WebGL / 3D engine could not initialize (offline or GPU limit). 2.5D CSS-3D / 2D fallback remains active.',
      retry_strategy: 'open_joins',
    };
  }
  return {
    code: 'ERR_RUNTIME_EXECUTION',
    category: 'runtime',
    failed_node: 'query_graph',
    remediation_hint: 'Execution encountered an unexpected error. Inspect details below or retry with AI Recompile.',
    retry_strategy: 'retry_recompile',
  };
}

export function normalizeDiagnostic(errOrPayload, fallbackContext = {}) {
  const rawObj =
    errOrPayload && typeof errOrPayload === 'object' && !(errOrPayload instanceof Error)
      ? errOrPayload
      : null;

  const tech =
    rawObj?.technical_detail ||
    rawObj?.detail ||
    rawObj?.error_message ||
    rawObj?.user_message ||
    (errOrPayload instanceof Error ? `${errOrPayload.name}: ${errOrPayload.message}` : String(errOrPayload || 'Unknown error'));

  const inferred = inferCategoryFromText(tech);
  const category = String(rawObj?.error_category || rawObj?.category || fallbackContext.category || inferred.category);
  const code = String(rawObj?.error_code || rawObj?.code || fallbackContext.code || inferred.code);

  // Choose default retry strategy cleanly from backend fields + category
  let retryStrategy = rawObj?.retry_strategy || fallbackContext.retry_strategy;
  if (!retryStrategy) {
    if (rawObj?.recoverable_with_heuristic) retryStrategy = 'force_heuristic';
    else if (category === 'security') retryStrategy = 'open_scratchpad';
    else if (category === 'llm_provider') retryStrategy = 'open_llm_settings';
    else retryStrategy = inferred.retry_strategy;
  }

  return {
    code,
    category,
    failed_node: String(rawObj?.failed_node || fallbackContext.failed_node || inferred.failed_node),
    user_message: String(rawObj?.user_message || fallbackContext.user_message || ''),
    technical_detail: String(tech),
    remediation_hint: String(rawObj?.remediation_hint || fallbackContext.remediation_hint || inferred.remediation_hint),
    recoverable: rawObj?.recoverable !== undefined ? Boolean(rawObj.recoverable) : true,
    retry_strategy: String(retryStrategy),
    attempt: Number(rawObj?.attempt || fallbackContext.attempt || 1),
    max_retries: Number(rawObj?.max_retries || fallbackContext.max_retries || 3),
    question: String(rawObj?.question || fallbackContext.question || ''),
    timestamp: new Date().toISOString(),
  };
}

export function renderDiagnosticCard(errOrDiag, options = {}) {
  const diag = normalizeDiagnostic(errOrDiag, options.fallbackContext || {});
  const idx = options.idx !== undefined ? String(options.idx) : '';
  const questionAttr = esc(diag.question || options.question || '');
  const btns = STRATEGY_BUTTONS[diag.retry_strategy] || STRATEGY_BUTTONS.retry_recompile;
  const serialized = esc(JSON.stringify(diag, null, 2));

  const actionHtml = btns
    .map(
      (b) =>
        `<button type="button" class="btn btn-${esc(b.tone)} btn-xs dt-diag-action-btn" data-diag-action="${esc(
          b.action
        )}" data-idx="${esc(idx)}" data-question="${questionAttr}">${esc(b.label)}</button>`
    )
    .join('');

  return `
    <div class="dt-diag-card" role="alert" data-diag-code="${esc(diag.code)}">
      <div class="dt-diag-header">
        <div class="dt-diag-badges">
          <span class="dt-diag-code-badge"><svg class="icon" style="width:13px;height:13px"><use href="#i-alert-circle"/></svg> ${esc(
            diag.code
          )}</span>
          <span class="dt-diag-node-badge">Node: <code class="mono">${esc(diag.failed_node)}</code></span>
          <span class="dt-diag-attempt-badge">Attempt ${esc(diag.attempt)} · ${esc(diag.category)}</span>
        </div>
        ${
          options.dismissible
            ? `<button type="button" class="btn btn-ghost btn-icon-only btn-xs dt-diag-dismiss-btn" aria-label="Dismiss diagnostic"><svg class="icon" style="width:13px;height:13px"><use href="#i-x"/></svg></button>`
            : ''
        }
      </div>
      <div class="dt-diag-remediation">${esc(diag.remediation_hint)}</div>
      <div class="dt-diag-actions">
        ${actionHtml}
        <button type="button" class="btn btn-ghost btn-xs dt-diag-copy-btn" data-diag-json="${serialized}">Copy Diagnostic JSON</button>
      </div>
      <details class="dt-diag-details">
        <summary>Technical trace &amp; root exception</summary>
        <pre class="code-box mono dt-diag-pre">${esc(diag.technical_detail)}</pre>
      </details>
    </div>
  `;
}

export function bindDiagnosticActions(container, handlers = {}) {
  if (!container) return;

  container.querySelectorAll('.dt-diag-action-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      const action = btn.getAttribute('data-diag-action');
      const idx = btn.getAttribute('data-idx');
      const question = btn.getAttribute('data-question') || '';
      if (typeof handlers.onAction === 'function') {
        handlers.onAction(action, { idx: idx === '' ? null : Number(idx), question, button: btn });
      }
    });
  });

  container.querySelectorAll('.dt-diag-copy-btn').forEach((btn) => {
    btn.addEventListener('click', async () => {
      const payload = btn.getAttribute('data-diag-json') || '{}';
      try {
        await navigator.clipboard.writeText(payload);
        const old = btn.textContent;
        btn.textContent = 'Copied!';
        setTimeout(() => {
          btn.textContent = old;
        }, 1400);
      } catch (_) {
        // Silent fallback if clipboard is restricted
      }
    });
  });

  container.querySelectorAll('.dt-diag-dismiss-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      const card = btn.closest('.dt-diag-card');
      if (card) card.remove();
    });
  });
}

export function showGlobalDiagnostic(errOrDiag, options = {}) {
  const mount = document.getElementById('globalDiagnosticMount');
  if (!mount) return;
  mount.innerHTML = renderDiagnosticCard(errOrDiag, { ...options, dismissible: true });
  mount.classList.remove('hidden');
  bindDiagnosticActions(mount, {
    onAction: (action, ctx) => {
      if (typeof options.onAction === 'function') {
        options.onAction(action, ctx);
      }
    },
  });
}

let _boundaryInstalled = false;

export function installGlobalErrorBoundary(options = {}) {
  if (_boundaryInstalled || typeof window === 'undefined') return;
  _boundaryInstalled = true;

  const report = (errObj, fallbackContext) => {
    const diag = normalizeDiagnostic(errObj, fallbackContext);
    showGlobalDiagnostic(diag, {
      fallbackContext,
      onAction: options.onAction,
    });
    if (typeof options.onDiagnostic === 'function') {
      options.onDiagnostic(diag);
    }
  };

  window.addEventListener('error', (event) => {
    // Ignore benign cross-origin script noise without details
    if (!event.message && !event.error) return;
    report(event.error || event.message, {
      code: 'ERR_FRONTEND_UNCAUGHT',
      category: 'frontend',
      failed_node: event.filename ? `${event.filename.split('/').pop()}:${event.lineno}` : 'window.onerror',
      remediation_hint: 'An unexpected UI error occurred. View details below or reload the active workspace.',
      retry_strategy: 'retry_sync',
    });
  });

  window.addEventListener('unhandledrejection', (event) => {
    const reason = event.reason;
    // Do not spam banner for intentional AbortError cancellations
    if (reason && reason.name === 'AbortError') return;
    report(reason || 'Unhandled promise rejection', {
      code: 'ERR_ASYNC_PROMISE',
      category: 'frontend',
      failed_node: 'async_task',
      remediation_hint: 'An asynchronous UI operation failed unexpectedly. Inspect the trace or retry.',
      retry_strategy: 'retry_sync',
    });
  });

  document.addEventListener(
    'webglcontextlost',
    (e) => {
      e.preventDefault();
      report('WebGL 3D context lost (GPU resource reclaimed).', {
        code: 'ERR_WEBGL_CONTEXT_LOST',
        category: 'render',
        failed_node: 'webgl_canvas',
        remediation_hint: 'Browser GPU memory limit reached. Switching back to CSS 2.5D Isometric or 2D mode recovers cleanly.',
        retry_strategy: 'open_joins',
      });
    },
    true
  );
}
