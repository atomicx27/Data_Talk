/**
 * Data_Talk Enterprise BI — Production Frontend Controller
 * Connects to the FastAPI backend (/api/...) and powers all 6 interactive BI studio views.
 */

let APP_STATE = null;
let ACTIVE_VIEW = 'chat';
let FOCAL_TABLE_FILTER = null;
let CONNECTED_SOURCE_FILTER = null;
let INTERVIEW_SOURCE_FILTER = null;
let CURRENT_LINEAGE_ENTRY = null;
let CMD_ACTIVE_INDEX = 0;
let CMD_FILTERED_ITEMS = [];

const VIEW_TITLES = {
  chat: 'Ask Data',
  dashboard: 'Pinboard',
  connect: 'Data sources',
  joins: 'Relationships',
  interview: 'Business rules',
  measures: 'Measures',
};

const STARTER_PROMPTS = [
  {
    prompt: 'Show net revenue, profit, and gross margin % by product category',
    title: 'Net revenue & margin % by category',
    sub: 'Dual-axis combo chart across orders & products',
    icon: 'i-bar',
  },
  {
    prompt: 'Show monthly net revenue trend over time',
    title: 'Monthly net revenue trend',
    sub: 'Time-series trajectory excluding cancelled orders',
    icon: 'i-trend-up',
  },
  {
    prompt: 'Compare net revenue, refunds, and return rate % by product category',
    title: 'Revenue vs. refunds by category',
    sub: 'Multi-fact CTE query (orders + returns)',
    icon: 'i-shield',
  },
  {
    prompt: 'Show revenue and order count by customer tier',
    title: 'Revenue by customer tier',
    sub: 'Segment breakdown across enterprise & SMB tiers',
    icon: 'i-users',
  },
  {
    prompt: 'Show breakdown of returns by return reason',
    title: 'Returns by reason code',
    sub: 'Distribution of refund volume by root cause',
    icon: 'i-undo',
  },
];

function getWorkspaceStarterPrompts() {
  const ws = APP_STATE?.workspace || {};
  const tables = (ws.tables || []).filter((t) => !t.is_blocked);
  const tableNames = tables.map((t) => (t.name || '').toLowerCase());

  // Adapt for education/students dataset
  if (tableNames.some((n) => n.includes('student') || n.includes('academic'))) {
    return [
      {
        prompt: 'Show count of students by Gender',
        title: 'Students by Gender',
        sub: 'Gender distribution breakdown with bar chart',
        icon: 'i-bar',
      },
      {
        prompt: 'Show average Total Marks and Percentage by Grade',
        title: 'Average scores by Grade',
        sub: 'Academic metrics grouped by letter grade',
        icon: 'i-trend-up',
      },
      {
        prompt: 'Show student count and total annual fee by Stream',
        title: 'Enrollments & fees by Stream',
        sub: 'Aggregate enrollments and fee billing by stream',
        icon: 'i-pie',
      },
      {
        prompt: 'Show student count by City and Fee Status',
        title: 'Fee status distribution',
        sub: 'Outstanding vs paid breakdown by city',
        icon: 'i-schema',
      },
    ];
  }

  // Default retail/ecommerce warehouse
  if (tableNames.includes('orders') || tableNames.includes('order_items')) {
    return STARTER_PROMPTS;
  }

  // Dynamic fallback based on available table metadata
  if (tables.length) {
    const prompts = [];
    const mainTable = tables.find((t) => t.table_role === 'FACT') || tables[0];
    const catCol = (mainTable.columns || []).find((c) => /char|text|varchar/i.test(c.data_type || ''));
    const numCol = (mainTable.columns || []).find((c) => /int|float|double|num/i.test(c.data_type || ''));

    if (catCol) {
      prompts.push({
        prompt: `Show count of rows by ${catCol.name} in ${mainTable.name}`,
        title: `Distribution by ${catCol.name}`,
        sub: `Row count grouped by ${catCol.name}`,
        icon: 'i-bar',
      });
    }
    if (numCol && catCol) {
      prompts.push({
        prompt: `Show sum of ${numCol.name} by ${catCol.name} in ${mainTable.name}`,
        title: `${numCol.name} by ${catCol.name}`,
        sub: `Metric totals grouped by ${catCol.name}`,
        icon: 'i-trend-up',
      });
    }
    prompts.push({
      prompt: `Show summary statistics for ${mainTable.name}`,
      title: `Summary of ${mainTable.name}`,
      sub: `High-level table overview`,
      icon: 'i-schema',
    });
    prompts.push({
      prompt: `Show top 10 rows from ${mainTable.name}`,
      title: `Preview ${mainTable.name}`,
      sub: `Inspect first 10 records`,
      icon: 'i-table',
    });
    return prompts;
  }

  return STARTER_PROMPTS;
}

// ============================================================================
// 1. TOP-LAYER DIALOG ANIMATION & LIGHT-DISMISS FALLBACKS
//    (modern-web-guidance: animate-to-from-top-layer & light-dismiss-a-dialog)
// ============================================================================
let supportsDisplayTransition;
function canTransitionDisplay() {
  if (supportsDisplayTransition !== undefined) return supportsDisplayTransition;
  if (!window.CSS?.supports?.('transition-behavior', 'allow-discrete') || !document.body) {
    return false;
  }
  const probe = document.createElement('div');
  probe.style.cssText = 'transition: display 1s allow-discrete !important; display: block;';
  document.body.appendChild(probe);
  getComputedStyle(probe).display;
  probe.style.display = 'none';
  supportsDisplayTransition = getComputedStyle(probe).display === 'block';
  probe.remove();
  return supportsDisplayTransition;
}

function supportsTopLayerExit() {
  return window.CSS?.supports?.('overlay', 'auto') && canTransitionDisplay();
}

async function closeTopLayer(dialog) {
  if (!dialog || !dialog.open) return;
  if (!supportsTopLayerExit()) {
    dialog.setAttribute('data-closing', '');
    const animations = dialog.getAnimations({ subtree: true });
    if (animations.length > 0) {
      await Promise.race([
        Promise.allSettled(animations.map((a) => a.finished)),
        new Promise((r) => setTimeout(r, 400)),
      ]);
    }
    if (!dialog.hasAttribute('data-closing')) return;
    dialog.removeAttribute('data-closing');
  }
  dialog.close();
}

function initModalDialogs() {
  const dialogs = document.querySelectorAll('dialog.modal-dialog');
  dialogs.forEach((dialog) => {
    dialog.addEventListener('cancel', (event) => {
      event.preventDefault();
      closeTopLayer(dialog);
    });

    dialog.querySelectorAll('form[method="dialog"]').forEach((form) => {
      form.addEventListener('submit', (event) => {
        event.preventDefault();
        closeTopLayer(dialog);
      });
    });

    // Fallback for browsers without closedby support
    if (!('closedBy' in HTMLDialogElement.prototype)) {
      dialog.addEventListener('click', (event) => {
        if (event.target !== dialog) return;
        const rect = dialog.getBoundingClientRect();
        const isDialogContent =
          rect.top <= event.clientY &&
          event.clientY <= rect.top + rect.height &&
          rect.left <= event.clientX &&
          event.clientX <= rect.left + rect.width;
        if (isDialogContent) return;
        closeTopLayer(dialog);
      });
    }
  });
}

function showToast(message) {
  const container = document.getElementById('toastContainer');
  if (!container) return;
  const el = document.createElement('div');
  el.className = 'toast';
  el.innerHTML = `<svg class="icon" style="color:var(--accent-primary)"><use href="#i-check"/></svg><span>${escapeHtml(message)}</span>`;
  container.appendChild(el);
  setTimeout(() => {
    el.remove();
  }, 3400);
}

async function apiRequest(url, options = {}) {
  const headers = options.headers || {};
  if (options.body && !(options.body instanceof FormData)) {
    headers['Content-Type'] = 'application/json';
  }
  const opts = {
    credentials: 'same-origin',
    ...options,
    headers,
  };
  const resp = await fetch(url, opts);
  if (resp.status === 401) {
    showToast('Session expired or unauthorized. Please reload or sign in.', 'error');
    setTimeout(() => {
      window.location.reload();
    }, 1500);
    throw new Error('Unauthorized');
  }
  if (!resp.ok) {
    let errText = `HTTP ${resp.status}`;
    try {
      const errJson = await resp.json();
      errText = errJson.detail || JSON.stringify(errJson);
    } catch (_) {
      errText = await resp.text();
    }
    throw new Error(errText);
  }
  return resp.json();
}

// ============================================================================
// 2. THEME MANAGEMENT & PLOTLY THEME ADAPTATION
// ============================================================================
function getActiveTheme() {
  return document.documentElement.getAttribute('data-theme') === 'light' ? 'light' : 'dark';
}

function applyPlotlyThemeLayout(baseLayout = {}) {
  if (!baseLayout) return {};
  const isLight = getActiveTheme() === 'light';
  if (!isLight) {
    // In dark theme, return the original server layout directly with zero modifications
    return baseLayout;
  }
  // In light theme, only adjust paper and plot background and font color without touching any axes
  return {
    ...baseLayout,
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(241, 245, 249, 0.7)',
    font: {
      ...(baseLayout.font || {}),
      color: '#0f172a',
    },
  };
}

function toggleTheme() {
  const next = getActiveTheme() === 'dark' ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', next);
  try {
    localStorage.setItem('dt-theme', next);
  } catch (_) {}
  renderChatView();
  renderERDiagram();
  if (ACTIVE_VIEW === 'dashboard') {
    refreshDashboardSimulation();
  }
}

// ============================================================================
// 3. NAVIGATION & GLOBAL TELEMETRY HEADER
// ============================================================================
function switchView(viewName) {
  ACTIVE_VIEW = viewName;
  const targetPanel = document.getElementById(`view-${viewName}`);
  if (!targetPanel) return;

  // Close mobile sidebar if open
  document.getElementById('appSidebar')?.classList.remove('open');
  document.getElementById('sidebarScrim')?.classList.remove('open');

  const applyDomUpdate = () => {
    document.querySelectorAll('.nav-item').forEach((btn) => {
      btn.classList.toggle('active', btn.dataset.view === viewName);
    });
    document.querySelectorAll('.view-panel').forEach((panel) => {
      const isTarget = panel.id === `view-${viewName}`;
      panel.classList.toggle('active', isTarget);
      panel.classList.toggle('inactive', !isTarget);
      panel.setAttribute('aria-hidden', isTarget ? 'false' : 'true');
    });
    const titleEl = document.getElementById('activeViewTitle');
    if (titleEl && VIEW_TITLES[viewName]) {
      titleEl.textContent = VIEW_TITLES[viewName];
    }
    if (viewName === 'joins') {
      renderERDiagram();
    } else if (viewName === 'dashboard') {
      refreshDashboardSimulation();
    }
  };

  if (!document.startViewTransition) {
    applyDomUpdate();
    targetPanel.focus({ preventScroll: true });
    return;
  }

  const transition = document.startViewTransition(() => {
    applyDomUpdate();
  });
  transition.finished.finally(() => {
    targetPanel.focus({ preventScroll: true });
  });
}

function renderGlobalTelemetry() {
  if (!APP_STATE) return;
  const ws = APP_STATE.workspace || {};
  const readiness = APP_STATE.readiness || {
    score: 0,
    schema_pts: 0,
    join_pts: 0,
    interview_pts: 0,
    measure_pts: 0,
  };
  const rag = APP_STATE.rag_counts || { schema_glossary: 0, business_rules: 0, few_shot_queries: 0 };

  // Workspace selector & breadcrumb
  const wsSelect = document.getElementById('workspaceSelect');
  if (wsSelect) {
    wsSelect.innerHTML = (APP_STATE.workspaces || [])
      .map(
        (id) =>
          `<option value="${escapeHtml(id)}" ${id === APP_STATE.active_workspace_id ? 'selected' : ''}>${escapeHtml(id)}</option>`
      )
      .join('');
  }
  const crumbWs = document.getElementById('breadcrumbWorkspace');
  if (crumbWs) {
    crumbWs.textContent = APP_STATE.active_workspace_id || 'workspace';
  }

  // Topbar meta badges
  const tables = (ws.tables || []).filter((t) => !t.is_blocked);
  const dialectBadge = document.getElementById('activeDialectBadge');
  if (dialectBadge) {
    dialectBadge.textContent = `${tables.length} tables · ${(ws.dialect || 'duckdb').toUpperCase()}`;
  }
  const ragBadge = document.getElementById('ragStatsBadge');
  if (ragBadge) {
    const ruleCount = (rag.business_rules || 0) + (rag.few_shot_queries || 0);
    ragBadge.textContent = `${ruleCount} rules active`;
  }

  // Segmented AI Readiness Card
  const score = readiness.score || 0;
  const pctEl = document.getElementById('readinessTotalPct');
  if (pctEl) pctEl.textContent = `${score}%`;

  const setSeg = (id, pts, max) => {
    const el = document.getElementById(id);
    if (el) el.style.width = `${Math.min(100, Math.round(((pts || 0) / max) * 100))}%`;
  };
  setSeg('segSchema', readiness.schema_pts, 25);
  setSeg('segJoins', readiness.join_pts, 25);
  setSeg('segInterview', readiness.interview_pts, 30);
  setSeg('segMeasures', readiness.measure_pts, 20);

  document.getElementById('pillarSchema').textContent = `Schema ${readiness.schema_pts}/25`;
  document.getElementById('pillarJoins').textContent = `Joins ${readiness.join_pts}/25`;
  document.getElementById('pillarInterview').textContent = `Rules ${readiness.interview_pts}/30`;
  document.getElementById('pillarMeasures').textContent = `DAX ${readiness.measure_pts}/20`;

  // Sidebar navigation badges & setup stepper completion states
  const rels = ws.relationships || [];
  const approvedRels = rels.filter((r) => r.status === 'approved');
  const qs = ws.interview_questions || [];
  const answeredQs = qs.filter((q) => q.answered);
  const measures = ws.measures || [];
  const pins = ws.pinned_widgets || [];

  document.getElementById('badgeTables').textContent = `${tables.length}`;
  document.getElementById('badgeJoins').textContent = `${approvedRels.length}/${rels.length}`;
  document.getElementById('badgeInterview').textContent = `${answeredQs.length}/${qs.length}`;
  document.getElementById('badgeMeasures').textContent = `${measures.length}`;
  document.getElementById('badgePins').textContent = `${pins.length}`;

  const stepCompletion = {
    connect: tables.length > 0,
    joins: rels.length > 0 && approvedRels.length === rels.length,
    interview: qs.length > 0 && answeredQs.length > 0,
    measures: measures.length > 0,
  };
  document.querySelectorAll('.step-item[data-step]').forEach((btn) => {
    const stepKey = btn.dataset.step;
    btn.classList.toggle('completed', Boolean(stepCompletion[stepKey]));
  });

  // Privacy & LLM status
  const privSel = document.getElementById('privacyModeSelect');
  if (privSel && APP_STATE.privacy_mode) {
    privSel.value = APP_STATE.privacy_mode;
  }
  const rawModel = APP_STATE.llm_model || 'gpt-oss-120b';
  const shortModel = rawModel.split('/').pop();
  const modelLbl = document.getElementById('llmModelLabel');
  if (modelLbl) modelLbl.textContent = shortModel;
  const dot = document.getElementById('llmStatusDot');
  if (dot) {
    dot.classList.toggle('online', Boolean(APP_STATE.has_api_key));
  }

  // LLM Modal options
  const modelSel = document.getElementById('llmModelSelect');
  if (modelSel && APP_STATE.supported_models) {
    modelSel.innerHTML = APP_STATE.supported_models
      .map(
        (m) =>
          `<option value="${escapeHtml(m)}" ${m === APP_STATE.llm_model ? 'selected' : ''}>${escapeHtml(m)}</option>`
      )
      .join('');
  }
}

// ============================================================================
// 4. VIEW: DATA SOURCES & SCHEMA EXPLORER
// ============================================================================
function renderConnectView() {
  const container = document.getElementById('schemaTablesContainer');
  if (!container || !APP_STATE) return;
  const ws = APP_STATE.workspace || {};
  const tables = ws.tables || [];
  const dataSources = ws.data_sources || [];

  // Render Connected Databases & Data Sources cards
  const sourcesContainer = document.getElementById('connectedSourcesContainer');
  if (sourcesContainer) {
    if (!dataSources.length && !tables.length) {
      sourcesContainer.innerHTML = `
        <div style="grid-column: 1 / -1; padding: 20px; text-align: center; color: var(--text-muted); font-size: 13px;">
          No active database sources yet. Ingest files above or connect an SQL database in multi-source mode.
        </div>
      `;
    } else {
      const activeSources = dataSources.length
        ? dataSources
        : [
            {
              id: 'default',
              name: 'Default Source',
              source_type: ws.source_type || 'duckdb',
              dialect: ws.dialect || 'duckdb',
              table_names: tables.map((t) => t.name),
            },
          ];

      sourcesContainer.innerHTML = activeSources
        .map((s) => {
          const tCount = (s.table_names || []).length;
          const badgeColor = s.source_type === 'sql' ? 'var(--accent-primary)' : 'var(--accent-emerald)';
          return `
            <div class="source-card">
              <div class="source-card-header">
                <div class="source-card-title">
                  <svg class="icon" style="color:${badgeColor};"><use href="#i-database"/></svg>
                  <span>${escapeHtml(s.name || s.id)}</span>
                </div>
                <span class="source-badge">${escapeHtml((s.dialect || s.source_type || 'sql').toUpperCase())}</span>
              </div>
              <div class="source-card-meta">
                <span class="mono">ID: ${escapeHtml(s.id)}</span>
                <span>·</span>
                <span>${tCount} table${tCount === 1 ? '' : 's'}</span>
                ${s.credential_ref ? `<span>·</span><span class="muted" title="Vault alias">🔑 ${escapeHtml(s.credential_ref)}</span>` : ''}
              </div>
              <div style="display:flex; justify-content:space-between; align-items:center; margin-top:6px;">
                <span class="meta-chip">${escapeHtml((s.source_type || 'SOURCE').toUpperCase())}</span>
                ${
                  activeSources.length > 1 || s.id !== 'default'
                    ? `
                  <button type="button" class="btn btn-secondary btn-sm" onclick="removeSource('${escapeHtml(s.id)}')" title="Disconnect this database" style="color:var(--accent-rose); padding: 2px 8px; font-size:11px;">
                    <svg class="icon" style="width:11px;height:11px;display:inline;vertical-align:-1px;margin-right:2px;"><use href="#i-trash"/></svg>Remove
                  </button>
                `
                    : ''
                }
              </div>
            </div>
          `;
        })
        .join('');
    }
  }

  // Render source filter chips
  const filterChips = document.getElementById('sourceTableFilterChips');
  if (filterChips) {
    if (dataSources.length > 1) {
      const allChip = `<button type="button" class="filter-chip ${!CONNECTED_SOURCE_FILTER ? 'active' : ''}" data-source="">All sources (${tables.length})</button>`;
      const srcChips = dataSources
        .map((s) => {
          const count = tables.filter((t) => t.source_id === s.id).length;
          return `<button type="button" class="filter-chip ${CONNECTED_SOURCE_FILTER === s.id ? 'active' : ''}" data-source="${escapeHtml(s.id)}">${escapeHtml(s.name || s.id)} (${count})</button>`;
        })
        .join('');
      filterChips.innerHTML = allChip + srcChips;
      filterChips.querySelectorAll('.filter-chip').forEach((btn) => {
        btn.addEventListener('click', () => {
          CONNECTED_SOURCE_FILTER = btn.dataset.source || null;
          renderConnectView();
        });
      });
    } else {
      filterChips.innerHTML = '';
    }
  }

  if (!tables.length) {
    container.innerHTML = `
      <div class="panel-card" style="text-align:center; padding: 36px;">
        <h3 class="card-title">No tables connected yet</h3>
        <p class="card-subtitle" style="margin-top:6px;">
          Click <strong>Load demo data</strong> above or upload CSV, Excel, Parquet, or connect SQL databases.
        </p>
      </div>
    `;
    return;
  }

  const visibleTables = CONNECTED_SOURCE_FILTER
    ? tables.filter((t) => (t.source_id || 'default') === CONNECTED_SOURCE_FILTER)
    : tables;

  const totalRows = visibleTables.reduce((acc, t) => acc + Number(t.row_count || 0), 0);
  const totalCols = visibleTables.reduce((acc, t) => acc + (t.columns || []).length, 0);
  const totalPii = visibleTables.reduce(
    (acc, t) => acc + (t.columns || []).filter((c) => c.is_pii).length,
    0
  );

  const summaryHtml = `
    <div class="schema-summary-strip">
      <div class="schema-stat-tile">
        <div class="schema-stat-label">Profiled tables ${CONNECTED_SOURCE_FILTER ? '(filtered)' : ''}</div>
        <div class="schema-stat-val">${visibleTables.length}</div>
      </div>
      <div class="schema-stat-tile">
        <div class="schema-stat-label">Total rows profiled</div>
        <div class="schema-stat-val">${totalRows.toLocaleString()}</div>
      </div>
      <div class="schema-stat-tile">
        <div class="schema-stat-label">Total columns</div>
        <div class="schema-stat-val">${totalCols}</div>
      </div>
      <div class="schema-stat-tile">
        <div class="schema-stat-label">PII columns shielded</div>
        <div class="schema-stat-val" style="color:var(--accent-emerald)">${totalPii}</div>
      </div>
    </div>
  `;

  const accordionHtml = visibleTables
    .map((tbl, idx) => {
      const piiCount = (tbl.columns || []).filter((c) => c.is_pii).length;
      const alertsHtml = (tbl.quality_alerts || [])
        .map(
          (a) =>
            `<div class="anomaly-banner"><svg class="icon"><use href="#i-alert"/></svg><span><strong>Anomaly watchdog:</strong> ${escapeHtml(a)}</span></div>`
        )
        .join('');

      const rowsHtml = (tbl.columns || [])
        .map((c) => {
          const keyBadge = c.is_primary_key
            ? `<span class="meta-chip" title="Primary Key"><svg class="icon" style="width:11px;height:11px;display:inline;vertical-align:-1px;margin-right:3px;"><use href="#i-key"/></svg>PK</span>`
            : c.is_foreign_key
            ? `<span class="meta-chip" title="Foreign Key"><svg class="icon" style="width:11px;height:11px;display:inline;vertical-align:-1px;margin-right:3px;"><use href="#i-link"/></svg>FK</span>`
            : '<span class="muted">—</span>';
          const piiBadge = c.is_pii
            ? `<span class="pii-badge pii-redacted"><svg class="icon" style="width:11px;height:11px"><use href="#i-shield"/></svg>Redacted</span>`
            : `<span class="pii-badge pii-safe"><svg class="icon" style="width:11px;height:11px"><use href="#i-check"/></svg>Safe</span>`;
          const samples = (c.sample_values || []).slice(0, 5).map(escapeHtml).join(', ') || '—';
          const nullPct = ((c.null_ratio || 0) * 100).toFixed(1);

          return `
            <tr>
              <td>
                <input
                  type="checkbox"
                  class="col-block-toggle"
                  data-table="${escapeHtml(tbl.name)}"
                  data-column="${escapeHtml(c.name)}"
                  aria-label="Block ${escapeHtml(c.name)} from AI"
                  ${c.is_blocked ? 'checked' : ''}
                />
              </td>
              <td class="mono" style="color:var(--text-primary); font-weight:600;">${escapeHtml(c.name)}</td>
              <td class="mono muted">${escapeHtml(c.data_type)}</td>
              <td>${keyBadge}</td>
              <td>${piiBadge}</td>
              <td class="mono">${Number(c.distinct_count || 0).toLocaleString()}</td>
              <td class="mono">
                <span class="null-bar-wrap">
                  <span>${nullPct}%</span>
                  <span class="null-bar"><span class="null-bar-fill" style="width:${Math.min(100, Number(nullPct))}%"></span></span>
                </span>
              </td>
              <td class="mono">${escapeHtml(c.min_value || '—')}</td>
              <td class="mono">${escapeHtml(c.max_value || '—')}</td>
              <td class="mono" style="max-width:240px; overflow:hidden; text-overflow:ellipsis;">${samples}</td>
            </tr>
          `;
        })
        .join('');

      return `
        <details class="schema-table-card" name="schema-table-accordion" ${idx === 0 ? 'open' : ''}>
          <summary class="schema-table-header">
            <div style="display:flex; align-items:center; gap:10px; flex-wrap:wrap;">
              <span class="role-badge role-${escapeHtml(tbl.table_role)}">${escapeHtml(tbl.table_role)}</span>
              <strong class="mono" style="font-size:13.5px; color:var(--text-primary);">${escapeHtml(tbl.name)}</strong>
              ${
                tbl.source_name || tbl.source_id
                  ? `<span class="source-badge" title="Source: ${escapeHtml(tbl.source_name || tbl.source_id)}"><svg class="icon" style="width:10px;height:10px;vertical-align:-1px;margin-right:2px;"><use href="#i-database"/></svg>${escapeHtml(tbl.source_name || tbl.source_id)}</span>`
                  : ''
              }
              <span class="muted" style="font-size:12px;">${Number(tbl.row_count || 0).toLocaleString()} rows · ${(tbl.columns || []).length} cols</span>
            </div>
            <div style="display:flex; align-items:center; gap:10px;">
              ${
                piiCount
                  ? `<span class="pii-badge pii-redacted"><svg class="icon" style="width:11px;height:11px"><use href="#i-shield"/></svg>${piiCount} PII shielded</span>`
                  : `<span class="pii-badge pii-safe"><svg class="icon" style="width:11px;height:11px"><use href="#i-check"/></svg>Verified clean</span>`
              }
              <svg class="icon muted"><use href="#i-chevron-down"/></svg>
            </div>
          </summary>
          ${alertsHtml}
          <div class="table-responsive">
            <table class="data-table">
              <thead>
                <tr>
                  <th>Block AI</th>
                  <th>Column</th>
                  <th>Type</th>
                  <th>Key</th>
                  <th>Privacy</th>
                  <th>Distinct</th>
                  <th>Null %</th>
                  <th>Min</th>
                  <th>Max</th>
                  <th>Sample values</th>
                </tr>
              </thead>
              <tbody>${rowsHtml}</tbody>
            </table>
          </div>
        </details>
      `;
    })
    .join('');

  container.innerHTML = summaryHtml + accordionHtml;

  container.querySelectorAll('.col-block-toggle').forEach((chk) => {
    chk.addEventListener('change', async (e) => {
      const table_name = e.target.dataset.table;
      const column_name = e.target.dataset.column;
      const is_blocked = e.target.checked;
      try {
        APP_STATE = await apiRequest('/api/schema/column-config', {
          method: 'POST',
          body: JSON.stringify({ table_name, column_name, is_blocked }),
        });
        renderAllViews();
        showToast(`Updated privacy blocklist for ${table_name}.${column_name}`);
      } catch (err) {
        showToast(`Error: ${err.message}`);
      }
    });
  });
}

// ============================================================================
// 5. VIEW: RELATIONSHIPS & STAR-SCHEMA ER DIAGRAM
// ============================================================================
function renderJoinsView() {
  if (!APP_STATE) return;
  const ws = APP_STATE.workspace || {};
  const tables = (ws.tables || []).filter((t) => !t.is_blocked);
  const rels = ws.relationships || [];

  const chipsContainer = document.getElementById('erFocalChips');
  if (chipsContainer) {
    const allChip = `<button type="button" class="filter-chip ${!FOCAL_TABLE_FILTER ? 'active' : ''}" data-focal="">All (${tables.length})</button>`;
    const tblChips = tables
      .map(
        (t) =>
          `<button type="button" class="filter-chip ${FOCAL_TABLE_FILTER === t.name ? 'active' : ''}" data-focal="${escapeHtml(t.name)}">${escapeHtml(t.name)}</button>`
      )
      .join('');
    chipsContainer.innerHTML = allChip + tblChips;
    chipsContainer.querySelectorAll('.filter-chip').forEach((btn) => {
      btn.addEventListener('click', () => {
        FOCAL_TABLE_FILTER = btn.dataset.focal || null;
        renderJoinsView();
      });
    });
  }

  renderERDiagram();

  const tbody = document.getElementById('joinTableBody');
  if (tbody) {
    if (!rels.length) {
      tbody.innerHTML = `<tr><td colspan="9" style="text-align:center; padding:24px;">No relationships discovered yet. Connect a dataset first.</td></tr>`;
    } else {
      tbody.innerHTML = rels
        .map(
          (r, idx) => `
          <tr data-rel-id="${escapeHtml(r.id)}">
            <td class="mono" style="color:var(--text-primary); font-weight:600;">
              ${escapeHtml(r.source_table)}
              ${r.source_source_id && r.source_source_id !== 'default' ? `<span class="source-badge" style="font-size:9.5px;padding:1px 5px;margin-left:4px;">${escapeHtml(r.source_source_id)}</span>` : ''}
            </td>
            <td class="mono">${escapeHtml(r.source_column)}</td>
            <td>
              <select class="input-select input-sm join-type-sel" data-idx="${idx}">
                <option value="LEFT" ${r.join_type === 'LEFT' ? 'selected' : ''}>LEFT JOIN</option>
                <option value="INNER" ${r.join_type === 'INNER' ? 'selected' : ''}>INNER JOIN</option>
                <option value="FULL" ${r.join_type === 'FULL' ? 'selected' : ''}>FULL JOIN</option>
              </select>
            </td>
            <td class="mono" style="color:var(--text-primary); font-weight:600;">
              ${escapeHtml(r.target_table)}
              ${r.target_source_id && r.target_source_id !== 'default' ? `<span class="source-badge" style="font-size:9.5px;padding:1px 5px;margin-left:4px;">${escapeHtml(r.target_source_id)}</span>` : ''}
            </td>
            <td class="mono">${escapeHtml(r.target_column)}</td>
            <td>
              <select class="input-select input-sm join-card-sel" data-idx="${idx}">
                <option value="N:1" ${r.cardinality === 'N:1' ? 'selected' : ''}>N:1</option>
                <option value="1:N" ${r.cardinality === '1:N' ? 'selected' : ''}>1:N</option>
                <option value="1:1" ${r.cardinality === '1:1' ? 'selected' : ''}>1:1</option>
              </select>
            </td>
            <td>
              ${
                r.is_cross_source
                  ? `<span class="meta-chip meta-chip-cross-db" title="Cross-database link between ${escapeHtml(r.source_source_id || 'source 1')} and ${escapeHtml(r.target_source_id || 'source 2')}"><svg class="icon" style="width:10px;height:10px;display:inline;vertical-align:-1px;margin-right:2px;"><use href="#i-link"/></svg>🌐 Cross-DB</span> `
                  : ''
              }
              <span class="meta-chip">${escapeHtml(r.discovery_method)}</span>
            </td>
            <td class="mono">${Math.round((r.confidence || 0) * 100)}% · ${((r.overlap_ratio || 0) * 100).toFixed(0)}% overlap</td>
            <td>
              <select class="input-select input-sm join-status-sel" data-idx="${idx}">
                <option value="approved" ${r.status === 'approved' ? 'selected' : ''}>Approved</option>
                <option value="pending" ${r.status === 'pending' ? 'selected' : ''}>Pending</option>
                <option value="rejected" ${r.status === 'rejected' ? 'selected' : ''}>Rejected</option>
              </select>
            </td>
          </tr>
        `
        )
        .join('');
    }
  }

  const sTblSel = document.getElementById('cjSourceTable');
  const tTblSel = document.getElementById('cjTargetTable');
  if (sTblSel && tTblSel && tables.length) {
    const opts = tables
      .map((t) => {
        const srcLabel = t.source_name || (t.source_id && t.source_id !== 'default' ? t.source_id : '');
        const display = srcLabel ? `${t.name} (${srcLabel})` : t.name;
        return `<option value="${escapeHtml(t.name)}">${escapeHtml(display)}</option>`;
      })
      .join('');
    sTblSel.innerHTML = opts;
    tTblSel.innerHTML = opts;
    if (tables.length > 1) tTblSel.selectedIndex = 1;
    populateCustomJoinColumns();
  }
}

function populateCustomJoinColumns() {
  if (!APP_STATE) return;
  const tables = APP_STATE.workspace.tables || [];
  const sTblName = document.getElementById('cjSourceTable')?.value;
  const tTblName = document.getElementById('cjTargetTable')?.value;
  const sColSel = document.getElementById('cjSourceCol');
  const tColSel = document.getElementById('cjTargetCol');

  const sTbl = tables.find((t) => t.name === sTblName);
  const tTbl = tables.find((t) => t.name === tTblName);
  if (sColSel && sTbl) {
    sColSel.innerHTML = (sTbl.columns || [])
      .map((c) => `<option value="${escapeHtml(c.name)}">${escapeHtml(c.name)}</option>`)
      .join('');
  }
  if (tColSel && tTbl) {
    tColSel.innerHTML = (tTbl.columns || [])
      .map((c) => `<option value="${escapeHtml(c.name)}">${escapeHtml(c.name)}</option>`)
      .join('');
  }
}

function renderERDiagram() {
  const viewport = document.getElementById('erDiagramViewport');
  if (!viewport || !APP_STATE) return;
  const ws = APP_STATE.workspace || {};
  const allTables = (ws.tables || []).filter((t) => !t.is_blocked);
  const rels = (ws.relationships || []).filter((r) => r.status !== 'rejected');

  if (!allTables.length) {
    viewport.innerHTML = `<div class="muted">Connect a dataset in Step 1 to view the entity-relationship graph.</div>`;
    return;
  }

  const isLight = getActiveTheme() === 'light';
  const nodeBg = isLight ? '#ffffff' : '#111622';
  const nodeStroke = isLight ? '#cbd5e1' : 'rgba(255,255,255,0.15)';
  const colText = isLight ? '#1e293b' : '#e2e8f0';
  const typeText = isLight ? '#64748b' : '#64748b';
  const pillBg = isLight ? '#f1f5f9' : '#0b0f17';

  let visibleTables = allTables;
  if (FOCAL_TABLE_FILTER) {
    const neighborSet = new Set([FOCAL_TABLE_FILTER]);
    rels.forEach((r) => {
      if (r.source_table === FOCAL_TABLE_FILTER) neighborSet.add(r.target_table);
      if (r.target_table === FOCAL_TABLE_FILTER) neighborSet.add(r.source_table);
    });
    visibleTables = allTables.filter((t) => neighborSet.has(t.name));
  }

  const facts = visibleTables.filter((t) => t.table_role === 'FACT');
  const dims = visibleTables.filter((t) => t.table_role !== 'FACT');

  const leftList = facts.length ? facts : visibleTables.slice(0, Math.ceil(visibleTables.length / 2));
  const rightList = facts.length ? dims : visibleTables.slice(Math.ceil(visibleTables.length / 2));

  const cardW = 248;
  const cardH = 152;
  const gapY = 32;
  const maxRows = Math.max(leftList.length, rightList.length, 1);
  const svgW = 880;
  const svgH = Math.max(340, maxRows * (cardH + gapY) + 36);

  const positions = {};
  leftList.forEach((t, idx) => {
    positions[t.name] = { x: 56, y: 24 + idx * (cardH + gapY), table: t };
  });
  rightList.forEach((t, idx) => {
    positions[t.name] = { x: svgW - cardW - 56, y: 24 + idx * (cardH + gapY), table: t };
  });

  const edgesSvg = rels
    .filter((r) => positions[r.source_table] && positions[r.target_table])
    .map((r) => {
      const p1 = positions[r.source_table];
      const p2 = positions[r.target_table];
      const sameCol = p1.x === p2.x;
      const x1 = sameCol ? p1.x + cardW : p1.x < p2.x ? p1.x + cardW : p1.x;
      const y1 = p1.y + cardH / 2;
      const x2 = sameCol ? p2.x + cardW : p1.x < p2.x ? p2.x : p2.x + cardW;
      const y2 = p2.y + cardH / 2;
      const cx1 = sameCol ? x1 + 90 : (x1 + x2) / 2;
      const cx2 = sameCol ? x2 + 90 : (x1 + x2) / 2;

      const color = r.is_cross_source ? '#8b5cf6' : r.status === 'approved' ? '#10b981' : '#f59e0b';
      const dash = r.status === 'approved' && !r.is_cross_source ? '' : 'stroke-dasharray="6,4"';
      const midX = sameCol ? x1 + 74 : (x1 + x2) / 2;
      const midY = (y1 + y2) / 2 - 7;
      const labelText = r.is_cross_source
        ? `🌐 ${r.source_column} → ${r.target_column} (${r.cardinality})`
        : `${r.source_column} → ${r.target_column} (${r.cardinality})`;

      return `
        <path d="M ${x1} ${y1} C ${cx1} ${y1}, ${cx2} ${y2}, ${x2} ${y2}" fill="none" stroke="${color}" stroke-width="2" ${dash} />
        <circle cx="${x1}" cy="${y1}" r="4" fill="${color}" />
        <circle cx="${x2}" cy="${y2}" r="4" fill="${color}" />
        <rect x="${midX - 92}" y="${midY - 11}" width="184" height="16" rx="8" fill="${pillBg}" stroke="${color}" stroke-opacity="0.4" />
        <text x="${midX}" y="${midY}" text-anchor="middle" fill="${color}" font-family="JetBrains Mono" font-size="9.5" font-weight="600">
          ${escapeHtml(labelText)}
        </text>
      `;
    })
    .join('');

  const nodesSvg = Object.values(positions)
    .map(({ x, y, table }) => {
      const isFact = table.table_role === 'FACT';
      const headerColor = isFact ? '#2563eb' : '#059669';
      const topCols = (table.columns || []).slice(0, 5);
      const colLines = topCols
        .map((c, idx) => {
          const prefix = c.is_primary_key ? 'PK ' : c.is_foreign_key ? 'FK ' : '· ';
          const prefixColor = c.is_primary_key ? '#f59e0b' : c.is_foreign_key ? '#3b82f6' : typeText;
          const cy = y + 48 + idx * 19;
          return `
            <text x="${x + 12}" y="${cy}" font-family="JetBrains Mono" font-size="10.5">
              <tspan fill="${prefixColor}" font-weight="600">${prefix}</tspan><tspan fill="${colText}">${escapeHtml(c.name)}</tspan>
            </text>
            <text x="${x + cardW - 12}" y="${cy}" text-anchor="end" fill="${typeText}" font-family="JetBrains Mono" font-size="9.5">${escapeHtml(c.data_type)}</text>
          `;
        })
        .join('');

      const srcTag = table.source_name || (table.source_id && table.source_id !== 'default' ? table.source_id : '');
      const tableLabel = srcTag ? `${table.name} [${srcTag}]` : table.name;

      return `
        <g class="er-node" data-table="${escapeHtml(table.name)}" style="cursor:pointer;">
          <rect x="${x}" y="${y}" width="${cardW}" height="${cardH}" rx="10" fill="${nodeBg}" stroke="${nodeStroke}" stroke-width="1.2" />
          <rect x="${x}" y="${y}" width="${cardW}" height="28" rx="10" fill="${headerColor}" />
          <text x="${x + 12}" y="${y + 18}" fill="#ffffff" font-family="Inter" font-size="11.5" font-weight="700">${escapeHtml(tableLabel)} · ${escapeHtml(table.table_role)}</text>
          <text x="${x + cardW - 10}" y="${y + 18}" text-anchor="end" fill="rgba(255,255,255,0.85)" font-family="JetBrains Mono" font-size="9.5">${Number(table.row_count || 0).toLocaleString()}r</text>
          ${colLines}
        </g>
      `;
    })
    .join('');

  viewport.innerHTML = `
    <svg width="${svgW}" height="${svgH}" viewBox="0 0 ${svgW} ${svgH}" style="max-width:100%; height:auto;">
      ${edgesSvg}
      ${nodesSvg}
    </svg>
  `;

  viewport.querySelectorAll('.er-node').forEach((node) => {
    node.addEventListener('click', () => {
      const tbl = node.dataset.table;
      FOCAL_TABLE_FILTER = FOCAL_TABLE_FILTER === tbl ? null : tbl;
      renderJoinsView();
    });
  });
}

// ============================================================================
// 6. VIEW: BUSINESS RULES & GUIDED AI INTERVIEW
// ============================================================================
function renderInterviewView() {
  if (!APP_STATE) return;
  const ws = APP_STATE.workspace || {};
  const qs = ws.interview_questions || [];
  const answered = qs.filter((q) => q.answered).length;
  const pct = qs.length ? Math.round((answered / qs.length) * 100) : 0;

  const banner = document.getElementById('interviewProgressBanner');
  if (banner) {
    banner.innerHTML = `
      <div style="display:flex; justify-content:space-between; align-items:center; font-size:12.5px; font-weight:600;">
        <span>Domain rules captured: <strong>${answered} of ${qs.length}</strong> ambiguities resolved</span>
        <span class="mono" style="color:var(--accent-emerald);">${pct}% complete</span>
      </div>
      <div class="progress-bar-track">
        <div class="progress-bar-fill" style="width:${pct}%;"></div>
      </div>
    `;
  }

  // Multi-source question filter chips
  const filterChipsEl = document.getElementById('interviewFilterChips');
  if (filterChipsEl) {
    const distinctSources = [...new Set(qs.map((q) => q.source_id).filter(Boolean))];
    const crossCount = qs.filter(
      (q) => q.id.startsWith('q_cross_link_') || q.category === 'CROSS_DATABASE_LINK'
    ).length;

    if (distinctSources.length > 1 || crossCount > 0) {
      let chipsHtml = `<button type="button" class="filter-chip ${!INTERVIEW_SOURCE_FILTER ? 'active' : ''}" data-isource="">All (${qs.length})</button>`;
      distinctSources.forEach((sId) => {
        const count = qs.filter((q) => (q.source_id || 'default') === sId).length;
        const sObj = (ws.data_sources || []).find((s) => s.id === sId);
        const label = sObj ? sObj.name || sObj.id : sId;
        chipsHtml += `<button type="button" class="filter-chip ${INTERVIEW_SOURCE_FILTER === sId ? 'active' : ''}" data-isource="${escapeHtml(sId)}">${escapeHtml(label)} (${count})</button>`;
      });
      if (crossCount > 0) {
        chipsHtml += `<button type="button" class="filter-chip ${INTERVIEW_SOURCE_FILTER === '__cross__' ? 'active' : ''}" data-isource="__cross__">🌐 Cross-DB Links (${crossCount})</button>`;
      }
      filterChipsEl.innerHTML = chipsHtml;
      filterChipsEl.querySelectorAll('.filter-chip').forEach((btn) => {
        btn.addEventListener('click', () => {
          INTERVIEW_SOURCE_FILTER = btn.dataset.isource || null;
          renderInterviewView();
        });
      });
    } else {
      filterChipsEl.innerHTML = `<span class="muted" style="font-size:12px;">Showing all rules</span>`;
    }
  }

  const container = document.getElementById('interviewQuestionsContainer');
  if (!container) return;

  if (!qs.length) {
    container.innerHTML = `<div class="panel-card">Connect a dataset in Step 1 to generate schema-tailored business rule questions.</div>`;
    return;
  }

  const filteredQs = qs.filter((q) => {
    if (!INTERVIEW_SOURCE_FILTER) return true;
    if (INTERVIEW_SOURCE_FILTER === '__cross__') {
      return q.id.startsWith('q_cross_link_') || q.category === 'CROSS_DATABASE_LINK';
    }
    return (q.source_id || 'default') === INTERVIEW_SOURCE_FILTER;
  });

  if (!filteredQs.length) {
    container.innerHTML = `<div class="panel-card" style="text-align:center; padding:24px; color:var(--text-muted);">No rules or questions found matching the selected database filter.</div>`;
    return;
  }

  container.innerHTML = filteredQs
    .map((q) => {
      const isCross = q.id.startsWith('q_cross_link_') || q.category === 'CROSS_DATABASE_LINK';
      const sourceTag = isCross
        ? `<span class="meta-chip meta-chip-cross-db">🌐 Cross-DB Link</span>`
        : q.source_id && q.source_id !== 'default'
        ? `<span class="source-badge"><svg class="icon" style="width:10px;height:10px"><use href="#i-database"/></svg>${escapeHtml(q.source_id)}</span>`
        : '';

      const statusBadge = q.answered
        ? `<span class="pii-badge pii-safe"><svg class="icon" style="width:11px;height:11px"><use href="#i-check"/></svg>Indexed in RAG</span>`
        : `<span class="meta-chip">Priority P${q.priority} · ${escapeHtml(q.category)}</span>`;

      const sugButtons = (q.suggested_options || [])
        .map((opt) => {
          const isSelected = q.answered && q.answer === opt;
          return `
            <button type="button" class="sug-chip ${isSelected ? 'selected' : ''}" data-qid="${escapeHtml(q.id)}" data-opt="${escapeHtml(opt)}">
              ${escapeHtml(opt)}
            </button>
          `;
        })
        .join('');

      return `
        <div class="interview-card ${q.answered ? 'answered' : ''}">
          <div style="display:flex; justify-content:space-between; align-items:center; gap:8px; flex-wrap:wrap;">
            <div style="display:flex; align-items:center; gap:6px;">
              <span class="mono" style="font-size:11px; color:var(--accent-primary); font-weight:600;">${escapeHtml(q.table_name || 'Global')}${q.column_name ? '.' + escapeHtml(q.column_name) : ''}</span>
              ${sourceTag}
            </div>
            ${statusBadge}
          </div>
          <div style="font-weight:600; font-size:13.5px; color:var(--text-primary);">${escapeHtml(q.question_text)}</div>
          <div style="font-size:11.5px; color:var(--text-muted);">${escapeHtml(q.context_hint || '')}</div>
          ${sugButtons ? `<div class="suggestion-chips">${sugButtons}</div>` : ''}
          <div class="input-with-btn" style="margin-top:auto; padding-top:4px;">
            <input
              type="text"
              class="input-text interview-ans-input"
              id="ans_${escapeHtml(q.id)}"
              value="${escapeHtml(q.answer || '')}"
              placeholder="Click a recommendation above or type a custom rule…"
            />
            <button type="button" class="btn btn-primary btn-save-ans" data-qid="${escapeHtml(q.id)}">Save</button>
          </div>
        </div>
      `;
    })
    .join('');

  container.querySelectorAll('.sug-chip').forEach((chip) => {
    chip.addEventListener('click', async () => {
      await submitInterviewAnswer(chip.dataset.qid, chip.dataset.opt);
    });
  });

  container.querySelectorAll('.btn-save-ans').forEach((btn) => {
    btn.addEventListener('click', async () => {
      const qid = btn.dataset.qid;
      const inputEl = document.getElementById(`ans_${qid}`);
      if (inputEl && inputEl.value.trim()) {
        await submitInterviewAnswer(qid, inputEl.value.trim());
      }
    });
  });
}

async function submitInterviewAnswer(question_id, answer) {
  try {
    APP_STATE = await apiRequest('/api/interview/answer', {
      method: 'POST',
      body: JSON.stringify({ question_id, answer }),
    });
    renderAllViews();
    showToast('Saved business rule to OKF & Hybrid RAG');
  } catch (err) {
    showToast(`Error: ${err.message}`);
  }
}

// ============================================================================
// 7. VIEW: MEASURES & DAX <-> SQL STUDIO
// ============================================================================
function renderMeasuresView() {
  if (!APP_STATE) return;
  const ws = APP_STATE.workspace || {};
  const tables = (ws.tables || []).filter((t) => !t.is_blocked);
  const templates = APP_STATE.starter_templates || [];
  const measures = ws.measures || [];
  const calcCols = ws.calculated_columns || [];

  const tableOpts = tables
    .map((t) => `<option value="${escapeHtml(t.name)}">${escapeHtml(t.name)}</option>`)
    .join('');
  const measTblSel = document.getElementById('measTableSelect');
  const ccTblSel = document.getElementById('ccTableSelect');
  if (measTblSel) measTblSel.innerHTML = tableOpts;
  if (ccTblSel) ccTblSel.innerHTML = tableOpts;

  const tmplGrid = document.getElementById('starterTemplatesGrid');
  if (tmplGrid) {
    tmplGrid.innerHTML = templates
      .map(
        (t, idx) => `
        <div class="template-card">
          <div>
            <div style="font-weight:600; font-size:13px; color:var(--text-primary);">${escapeHtml(t.name)}</div>
            <div style="font-size:11.5px; color:var(--text-muted); margin:3px 0 8px;">${escapeHtml(t.description)}</div>
            <pre class="code-block" style="font-size:10.5px; padding:7px;">${escapeHtml(t.dax_expression)}</pre>
          </div>
          <button type="button" class="btn btn-secondary btn-xs btn-block btn-use-tmpl" data-idx="${idx}">
            <svg class="icon" style="width:12px;height:12px"><use href="#i-plus"/></svg> Add '${escapeHtml(t.name)}'
          </button>
        </div>
      `
      )
      .join('');

    tmplGrid.querySelectorAll('.btn-use-tmpl').forEach((btn) => {
      btn.addEventListener('click', async () => {
        const tmpl = templates[Number(btn.dataset.idx)];
        if (!tmpl) return;
        try {
          APP_STATE = await apiRequest('/api/measures/create', {
            method: 'POST',
            body: JSON.stringify({
              name: tmpl.name,
              table_name: tmpl.table_name,
              user_formula: tmpl.dax_expression,
              input_type: 'dax',
              description: tmpl.description,
            }),
          });
          renderAllViews();
          showToast(`Verified & added '${tmpl.name}'`);
        } catch (err) {
          showToast(`Error: ${err.message}`);
        }
      });
    });
  }

  const ccList = document.getElementById('calcColListContainer');
  if (ccList) {
    if (!calcCols.length) {
      ccList.innerHTML = '';
    } else {
      ccList.innerHTML = `
        <div style="margin-top:14px; border-top:1px solid var(--border-subtle); padding-top:10px;">
          <div class="section-label" style="padding-left:0">Calculated columns (${calcCols.length})</div>
          ${calcCols
            .map(
              (cc) => `
              <div style="padding:8px 10px; background:var(--bg-surface); border:1px solid var(--border-subtle); border-radius:6px; margin-top:6px; font-size:12px;">
                <strong class="mono">${escapeHtml(cc.table_name)}[${escapeHtml(cc.column_name)}]</strong>
                <span class="pii-badge pii-safe" style="margin-left:6px;">${cc.verified ? 'Verified' : 'Stored'}</span>
                <div class="mono muted" style="font-size:11px; margin-top:4px;">SQL: ${escapeHtml(cc.sql_expression)}</div>
              </div>
            `
            )
            .join('')}
        </div>
      `;
    }
  }

  const catalogGrid = document.getElementById('measuresCatalogGrid');
  if (catalogGrid) {
    if (!measures.length) {
      catalogGrid.innerHTML = `<div class="panel-card">No measures defined yet. Pick a starter template below to compile and verify one.</div>`;
    } else {
      catalogGrid.innerHTML = measures
        .map((m) => {
          const valBadge =
            typeof m.sample_value === 'number'
              ? `<span class="meta-chip meta-chip-accent">Live: ${Number(m.sample_value).toLocaleString(undefined, { maximumFractionDigits: 4 })}</span>`
              : '';
          return `
            <div class="measure-card">
              <div style="display:flex; justify-content:space-between; align-items:center; gap:8px; flex-wrap:wrap;">
                <strong class="mono" style="font-size:13.5px; color:var(--text-primary);">${escapeHtml(m.table_name)}[${escapeHtml(m.name)}]</strong>
                <div style="display:flex; gap:6px; align-items:center;">
                  ${valBadge}
                  <span class="pii-badge pii-safe"><svg class="icon" style="width:11px;height:11px"><use href="#i-check"/></svg>${m.verified ? 'Verified' : 'Stored'}</span>
                </div>
              </div>
              <div style="font-size:12px; color:var(--text-secondary); margin-top:4px;">${escapeHtml(m.description || '')}</div>
              <div class="code-comparison-grid">
                <div>
                  <div class="code-header">Power BI DAX</div>
                  <pre class="code-block">MEASURE '${escapeHtml(m.table_name)}'[${escapeHtml(m.name)}] = ${escapeHtml(m.dax_expression)}</pre>
                </div>
                <div>
                  <div class="code-header">Compiled SQL</div>
                  <pre class="code-block">${escapeHtml(m.sql_expression)}</pre>
                </div>
              </div>
            </div>
          `;
        })
        .join('');
    }
  }
}

// ============================================================================
// 8. VIEW: ASK DATA (CONVERSATIONAL BI & ROOT-CAUSE STUDIO)
// ============================================================================
function renderChatView() {
  if (!APP_STATE) return;
  const stream = document.getElementById('chatMessagesStream');
  const sugBar = document.getElementById('composerSuggestions');
  if (!stream) return;

  const prompts = getWorkspaceStarterPrompts();

  // Render quick prompt chips above sticky composer
  if (sugBar) {
    sugBar.innerHTML = prompts.map(
      (item) => `
        <button type="button" class="prompt-chip" data-prompt="${escapeHtml(item.prompt)}">
          <svg class="icon" style="width:13px;height:13px;color:var(--accent-primary)"><use href="#${item.icon}"/></svg>
          <span>${escapeHtml(item.title)}</span>
        </button>
      `
    ).join('');

    sugBar.querySelectorAll('.prompt-chip').forEach((chip) => {
      chip.addEventListener('click', () => submitChatQuestion(chip.dataset.prompt));
    });
  }

  const history = APP_STATE.chat_history || [];
  if (!history.length) {
    const cardsHtml = prompts.map(
      (item) => `
        <button type="button" class="starter-card" data-prompt="${escapeHtml(item.prompt)}">
          <span class="starter-card-icon"><svg class="icon"><use href="#${item.icon}"/></svg></span>
          <span>
            <div class="starter-card-title">${escapeHtml(item.title)}</div>
            <div class="starter-card-sub">${escapeHtml(item.sub)}</div>
          </span>
        </button>
      `
    ).join('');

    stream.innerHTML = `
      <div class="chat-empty-hero">
        <span class="chat-empty-badge"><svg class="icon" style="width:13px;height:13px"><use href="#i-sparkles"/></svg> Governed Semantic BI</span>
        <h2 class="chat-empty-title">Ask anything about your connected data</h2>
        <p class="chat-empty-sub">
          Questions compile into verified multi-table SQL &amp; Power BI DAX using your approved join paths and business rules.
        </p>
        <div class="starter-cards-grid">${cardsHtml}</div>
      </div>
    `;

    stream.querySelectorAll('.starter-card').forEach((card) => {
      card.addEventListener('click', () => submitChatQuestion(card.dataset.prompt));
    });
    return;
  }

  stream.innerHTML = history
    .map((entry, idx) => {
      const resp = entry.response || {};
      const isSingleRow = (entry.row_count || (entry.rows || []).length) === 1;
      const chartTypes = isSingleRow
        ? ['kpi', 'table']
        : APP_STATE.supported_chart_types || [
            'bar',
            'horizontal_bar',
            'combo',
            'line',
            'area',
            'scatter',
            'pie',
            'waterfall',
            'table',
          ];
      const chartOpts = chartTypes
        .map(
          (ct) =>
            `<option value="${escapeHtml(ct)}" ${ct === entry.chart_type ? 'selected' : ''}>${ct === 'kpi' ? 'KPI Summary Card' : 'Chart: ' + escapeHtml(ct.replace('_', ' ').toUpperCase())}</option>`
        )
        .join('');

      const fanTrapBadge = resp.fan_trap_mitigated
        ? `<span class="pii-badge pii-safe" title="Pre-aggregated CTEs prevented multi-fact fan trap"><svg class="icon" style="width:11px;height:11px"><use href="#i-shield"/></svg>Multi-fact CTE</span>`
        : '';

      const tablePreviewRows = (entry.rows || []).slice(0, 10);
      const cols = entry.columns || [];
      const totalRowsCount = entry.row_count || (entry.rows || []).length;
      const miniTableHtml =
        cols.length && tablePreviewRows.length
          ? `
          <details class="clean-disclosure" style="margin-top:12px; padding:10px 14px;">
            <summary style="font-size:12px; color:var(--text-secondary);">
              <svg class="icon" style="width:14px;height:14px"><use href="#i-table"/></svg>
              <span>Result table (${totalRowsCount} ${totalRowsCount === 1 ? 'row' : 'rows'})</span>
            </summary>
            <div class="table-responsive" style="max-height:240px;">
              <table class="data-table">
                <thead>
                  <tr>${cols.map((c) => `<th>${escapeHtml(c)}</th>`).join('')}</tr>
                </thead>
                <tbody>
                  ${tablePreviewRows
                    .map(
                      (r) =>
                        `<tr>${cols
                          .map((c) => `<td class="mono">${escapeHtml(formatCellVal(r[c]))}</td>`)
                          .join('')}</tr>`
                    )
                    .join('')}
                </tbody>
              </table>
            </div>
          </details>
        `
          : '';

      const tablesUsed = (resp.involved_tables || []).length ? resp.involved_tables : ['orders'];

      return `
        <article class="chat-card" data-entry-idx="${idx}">
          <div class="chat-question-header">
            <div class="chat-q-left">
              <span class="chat-q-avatar"><svg class="icon" style="width:14px;height:14px"><use href="#i-chat"/></svg></span>
              <h3 class="chat-question-title">${escapeHtml(resp.user_question || '')}</h3>
            </div>
            <div class="chat-q-actions">
              ${fanTrapBadge}
              <button type="button" class="btn btn-secondary btn-xs btn-copy-sql" data-idx="${idx}" title="Copy compiled SQL">
                <svg class="icon" style="width:13px;height:13px"><use href="#i-copy"/></svg> SQL
              </button>
              <button type="button" class="btn btn-secondary btn-xs btn-export-csv" data-idx="${idx}" title="Download CSV">
                <svg class="icon" style="width:13px;height:13px"><use href="#i-download"/></svg> CSV
              </button>
              <button type="button" class="btn btn-secondary btn-xs btn-explain-lineage" data-idx="${idx}">
                <svg class="icon" style="width:13px;height:13px"><use href="#i-code"/></svg> Explain
              </button>
              <button type="button" class="btn btn-secondary btn-xs btn-pin-chart" data-idx="${idx}">
                <svg class="icon" style="width:13px;height:13px"><use href="#i-pin"/></svg> Pin
              </button>
            </div>
          </div>

          <div class="chat-summary-text">${formatMarkdownBold(resp.executive_summary || '')}</div>

          <div class="chat-toolbar">
            <div class="chat-toolbar-left">
              <select class="input-select input-sm chart-type-switcher" data-idx="${idx}" aria-label="Chart type" style="width:190px;">
                ${chartOpts}
              </select>
            </div>
            <div style="display:flex; gap:6px; align-items:center; flex-wrap:wrap;">
              ${tablesUsed.map((t) => `<span class="meta-chip">${escapeHtml(t)}</span>`).join('')}
            </div>
          </div>

          <div id="plotly_chat_${idx}" class="chart-box"></div>
          ${miniTableHtml}
        </article>
      `;
    })
    .join('');

  // Render Plotly charts with theme-aware layout
  history.forEach((entry, idx) => {
    const plotEl = document.getElementById(`plotly_chat_${idx}`);
    if (plotEl && entry.plotly_spec && window.Plotly) {
      try {
        const themedLayout = applyPlotlyThemeLayout(entry.plotly_spec.layout);
        window.Plotly.react(plotEl, entry.plotly_spec.data, themedLayout, {
          responsive: true,
          displayModeBar: false,
        });
      } catch (plotErr) {
        console.warn(`Plotly render failed for chat #${idx}:`, plotErr);
        plotEl.innerHTML = `<div style="padding:12px; color:var(--text-muted); font-size:12px;">Visual chart temporarily unavailable.</div>`;
      }
    } else if (plotEl && entry.chart_type === 'table') {
      plotEl.style.minHeight = '44px';
      plotEl.innerHTML = `<div style="padding:10px 12px; color:var(--text-muted); font-size:12px;">Tabular view active — inspect the result table below.</div>`;
    } else if (plotEl && (entry.chart_type === 'kpi' || (!entry.plotly_spec && (entry.row_count || (entry.rows || []).length) === 1))) {
      plotEl.style.display = 'block';
      plotEl.style.minHeight = '52px';
      const kpis = (entry.columns || []).map((c) => `
        <div class="schema-stat-tile" style="flex:1; min-width:140px; padding:10px 14px;">
          <div class="schema-stat-label">${escapeHtml(c.replace(/_/g, ' '))}</div>
          <div class="schema-stat-val" style="font-size:18px;">${escapeHtml(formatCellVal(entry.rows[0]?.[c]))}</div>
        </div>
      `).join('');
      plotEl.innerHTML = `<div style="display:flex; gap:10px; flex-wrap:wrap; margin:4px 0 10px 0;">${kpis}</div>`;
    } else if (plotEl && !entry.plotly_spec) {
      plotEl.style.display = 'none';
    }
  });

  stream.querySelectorAll('.chart-type-switcher').forEach((sel) => {
    sel.addEventListener('change', async (e) => {
      const idx = Number(e.target.dataset.idx);
      const entry = history[idx];
      const newType = e.target.value;
      try {
        const res = await apiRequest('/api/chat/chart', {
          method: 'POST',
          body: JSON.stringify({
            rows: entry.rows,
            chart_type: newType,
            title: entry.response.user_question,
          }),
        });
        entry.chart_type = newType;
        entry.plotly_spec = res.plotly_spec;
        renderChatView();
      } catch (err) {
        showToast(`Chart error: ${err.message}`);
      }
    });
  });

  stream.querySelectorAll('.btn-copy-sql').forEach((btn) => {
    btn.addEventListener('click', async () => {
      const idx = Number(btn.dataset.idx);
      const sql = history[idx]?.response?.executed_query || '';
      if (navigator.clipboard && sql) {
        await navigator.clipboard.writeText(sql);
        const originalHtml = btn.innerHTML;
        btn.innerHTML = `<svg class="icon" style="width:13px;height:13px;color:var(--accent-emerald)"><use href="#i-check"/></svg> Copied`;
        setTimeout(() => { btn.innerHTML = originalHtml; }, 1800);
        showToast('Compiled SQL copied to clipboard');
      }
    });
  });

  stream.querySelectorAll('.btn-export-csv').forEach((btn) => {
    btn.addEventListener('click', () => {
      const idx = Number(btn.dataset.idx);
      exportEntryToCsv(history[idx]);
    });
  });

  stream.querySelectorAll('.btn-explain-lineage').forEach((btn) => {
    btn.addEventListener('click', () => {
      const idx = Number(btn.dataset.idx);
      openLineageModal(history[idx]);
    });
  });

  stream.querySelectorAll('.btn-pin-chart').forEach((btn) => {
    btn.addEventListener('click', async () => {
      const idx = Number(btn.dataset.idx);
      const entry = history[idx];
      const resp = entry.response;
      try {
        APP_STATE = await apiRequest('/api/dashboard/pin', {
          method: 'POST',
          body: JSON.stringify({
            title: resp.user_question,
            user_question: resp.user_question,
            executed_query: resp.executed_query,
            query_dialect: resp.query_dialect || 'duckdb',
            chart_type: entry.chart_type || 'bar',
          }),
        });
        const originalHtml = btn.innerHTML;
        btn.innerHTML = `<svg class="icon" style="width:13px;height:13px;color:var(--accent-emerald)"><use href="#i-check"/></svg> Pinned`;
        setTimeout(() => { btn.innerHTML = originalHtml; }, 1800);
        renderGlobalTelemetry();
        showToast('Pinned chart to Pinboard');
      } catch (err) {
        showToast(`Error: ${err.message}`);
      }
    });
  });
}

function formatCellVal(v) {
  if (v === null || v === undefined) return '—';
  if (typeof v === 'number') {
    return Number.isInteger(v)
      ? v.toLocaleString()
      : v.toLocaleString(undefined, { maximumFractionDigits: 3 });
  }
  return String(v);
}

function exportEntryToCsv(entry) {
  if (!entry || !entry.rows || !entry.rows.length) {
    showToast('No rows available to export.');
    return;
  }
  const cols = entry.columns && entry.columns.length ? entry.columns : Object.keys(entry.rows[0]);
  const escapeCsvCell = (val) => {
    const str = String(val ?? '');
    if (str.includes(',') || str.includes('"') || str.includes('\n')) {
      return `"${str.replace(/"/g, '""')}"`;
    }
    return str;
  };
  const headerLine = cols.map(escapeCsvCell).join(',');
  const dataLines = entry.rows.map((row) => cols.map((c) => escapeCsvCell(row[c])).join(','));
  const csvContent = [headerLine, ...dataLines].join('\n');
  const blob = new Blob([csvContent], { type: 'text/csv;charset-utf-8;' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `datatalk_export_${Date.now()}.csv`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
  showToast(`Exported ${entry.rows.length} rows to CSV`);
}

function openLineageModal(entry) {
  if (!entry) return;
  CURRENT_LINEAGE_ENTRY = entry;
  const resp = entry.response || {};
  document.getElementById('lineageDialogQuestion').textContent = resp.user_question || '';
  document.getElementById('lineageSqlCode').textContent = resp.executed_query || '';
  document.getElementById('lineageDaxCode').textContent = resp.compiled_dax_equivalent || '';

  const badges = [];
  if (resp.fan_trap_mitigated) {
    badges.push(
      `<span class="pii-badge pii-safe"><svg class="icon" style="width:11px;height:11px"><use href="#i-shield"/></svg>Multi-fact chasm/fan trap mitigated via pre-aggregated CTEs</span>`
    );
  }
  badges.push(`<span class="meta-chip">Attempts: ${resp.attempts_used || 1}</span>`);
  document.getElementById('lineageBadgesRow').innerHTML = badges.join(' ');

  document.getElementById('lineageMetaDetails').innerHTML = `
    <div><strong>Tables used:</strong> <code class="mono">${escapeHtml((resp.involved_tables || []).join(', '))}</code></div>
    <div style="margin-top:4px;"><strong>Approved join paths:</strong> <code class="mono">${escapeHtml((resp.join_clauses_used || []).join(' ; ') || 'Single-table query')}</code></div>
  `;

  const dialog = document.getElementById('lineageDialog');
  if (dialog) dialog.showModal();
}

// ============================================================================
// 9. VIEW: PINBOARD & WHAT-IF SIMULATOR
// ============================================================================
async function refreshDashboardSimulation() {
  const revSlider = document.getElementById('revShockSlider');
  const costSlider = document.getElementById('costShockSlider');
  const revenue_shock_pct = Number(revSlider?.value || 0);
  const cost_shock_pct = Number(costSlider?.value || 0);

  document.getElementById('revShockVal').textContent = `${revenue_shock_pct >= 0 ? '+' : ''}${revenue_shock_pct}%`;
  document.getElementById('costShockVal').textContent = `${cost_shock_pct >= 0 ? '+' : ''}${cost_shock_pct}%`;

  try {
    const sim = await apiRequest('/api/dashboard/simulate', {
      method: 'POST',
      body: JSON.stringify({ revenue_shock_pct, cost_shock_pct }),
    });

    const kpis = sim.kpis || {};
    document.getElementById('simRevValue').textContent = `$${Number(kpis.sim_revenue || 0).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
    document.getElementById('simProfitValue').textContent = `$${Number(kpis.sim_profit || 0).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
    document.getElementById('simMarginValue').textContent = `${Number(kpis.sim_margin_pct || 0).toFixed(1)}%`;

    const revDeltaEl = document.getElementById('simRevDelta');
    if (revDeltaEl) {
      revDeltaEl.textContent =
        revenue_shock_pct === 0
          ? 'Baseline (0% shock)'
          : `Revenue shock: ${revenue_shock_pct > 0 ? '+' : ''}${revenue_shock_pct}%`;
      revDeltaEl.className = `kpi-delta ${revenue_shock_pct < 0 ? 'negative' : revenue_shock_pct === 0 ? 'muted' : ''}`;
    }

    const profitDeltaEl = document.getElementById('simProfitDelta');
    if (profitDeltaEl) {
      profitDeltaEl.textContent =
        cost_shock_pct === 0
          ? 'Baseline (0% shock)'
          : `COGS shock: ${cost_shock_pct > 0 ? '+' : ''}${cost_shock_pct}%`;
      profitDeltaEl.className = `kpi-delta ${cost_shock_pct > 0 ? 'negative' : cost_shock_pct === 0 ? 'muted' : ''}`;
    }

    const grid = document.getElementById('pinnedWidgetsGrid');
    if (!grid) return;
    const widgets = sim.widgets || [];
    if (!widgets.length) {
      grid.innerHTML = `
        <div class="panel-card" style="grid-column: 1 / -1; text-align:center; padding:36px;">
          <h3 class="card-title">No pinned charts yet</h3>
          <p class="card-subtitle" style="margin-top:6px;">
            Ask any question in <strong>Ask Data</strong> and click <strong>Pin</strong> on the answer card.
          </p>
        </div>
      `;
      return;
    }

    grid.innerHTML = widgets
      .map(
        (item, idx) => `
        <div class="panel-card">
          <div style="display:flex; justify-content:space-between; align-items:center; gap:10px; margin-bottom:10px;">
            <strong style="font-size:13.5px; color:var(--text-primary);">${escapeHtml(item.widget.title)}</strong>
            <button type="button" class="btn btn-ghost btn-xs btn-unpin" data-id="${escapeHtml(item.widget.id)}">
              <svg class="icon" style="width:13px;height:13px"><use href="#i-x"/></svg> Remove
            </button>
          </div>
          <div id="dash_plot_${idx}" class="chart-box"></div>
        </div>
      `
      )
      .join('');

    widgets.forEach((item, idx) => {
      const el = document.getElementById(`dash_plot_${idx}`);
      if (el && item.plotly_spec && window.Plotly) {
        try {
          const themedLayout = applyPlotlyThemeLayout(item.plotly_spec.layout);
          window.Plotly.react(el, item.plotly_spec.data, themedLayout, {
            responsive: true,
            displayModeBar: false,
          });
        } catch (dErr) {
          console.warn(`Plotly render failed for dashboard widget #${idx}:`, dErr);
        }
      }
    });

    grid.querySelectorAll('.btn-unpin').forEach((btn) => {
      btn.addEventListener('click', async () => {
        try {
          APP_STATE = await apiRequest(`/api/dashboard/pin/${btn.dataset.id}`, { method: 'DELETE' });
          renderGlobalTelemetry();
          await refreshDashboardSimulation();
        } catch (err) {
          showToast(`Error: ${err.message}`);
        }
      });
    });
  } catch (err) {
    showToast(`Pinboard error: ${err.message}`);
  }
}

// ============================================================================
// 10. HELPERS & INITIALIZATION
// ============================================================================
function escapeHtml(str) {
  return String(str ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function formatMarkdownBold(text) {
  return escapeHtml(text)
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/`([^`]+)`/g, '<code class="mono">$1</code>');
}

function renderAllViews() {
  try { renderGlobalTelemetry(); } catch (e) { console.error('Error rendering telemetry:', e); }
  try { renderConnectView(); } catch (e) { console.error('Error rendering connect view:', e); }
  try { renderJoinsView(); } catch (e) { console.error('Error rendering joins view:', e); }
  try { renderInterviewView(); } catch (e) { console.error('Error rendering interview view:', e); }
  try { renderMeasuresView(); } catch (e) { console.error('Error rendering measures view:', e); }
  try { renderChatView(); } catch (e) { console.error('Error rendering chat view:', e); }
}

async function loadDemoDataset() {
  try {
    showToast('Loading 4-table demo warehouse…');
    APP_STATE = await apiRequest('/api/connect/demo', { method: 'POST' });
    renderAllViews();
    showToast('Demo warehouse profiled and ready');
  } catch (err) {
    showToast(`Error loading demo: ${err.message}`);
  }
}

window.removeSource = async function (sourceId) {
  if (
    !confirm(
      `Are you sure you want to disconnect database "${sourceId}"?\n\nThis will remove its mirrored tables, relationships, and business rule questions from the active workspace.`
    )
  ) {
    return;
  }
  try {
    showToast(`Disconnecting source "${sourceId}"…`);
    APP_STATE = await apiRequest('/api/connect/remove-source', {
      method: 'POST',
      body: JSON.stringify({ source_id: sourceId }),
    });
    if (CONNECTED_SOURCE_FILTER === sourceId) {
      CONNECTED_SOURCE_FILTER = null;
    }
    if (INTERVIEW_SOURCE_FILTER === sourceId) {
      INTERVIEW_SOURCE_FILTER = null;
    }
    renderAllViews();
    showToast(`Disconnected database "${sourceId}"`);
  } catch (err) {
    showToast(`Error removing database: ${err.message}`);
  }
};

async function submitChatQuestion(questionText) {
  const trimmed = (questionText || '').trim();
  if (!trimmed) return;

  const stream = document.getElementById('chatMessagesStream');
  const sendBtn = document.getElementById('chatSendBtn');
  if (sendBtn) sendBtn.disabled = true;

  // Insert an immediate loading skeleton card for responsive UX
  if (stream) {
    const skel = document.createElement('div');
    skel.id = 'activeChatSkeleton';
    skel.className = 'chat-card chat-skeleton-card';
    skel.innerHTML = `
      <div class="chat-question-header">
        <div class="chat-q-left">
          <span class="chat-q-avatar"><svg class="icon" style="width:14px;height:14px"><use href="#i-sparkles"/></svg></span>
          <h3 class="chat-question-title">${escapeHtml(trimmed)}</h3>
        </div>
        <span class="meta-chip meta-chip-accent">Compiling SQL &amp; DAX…</span>
      </div>
      <div class="skeleton-line" style="width:78%"></div>
      <div class="skeleton-line" style="width:52%"></div>
      <div class="skeleton-chart"></div>
    `;
    stream.appendChild(skel);
    skel.scrollIntoView({ behavior: 'smooth', block: 'end' });
  }

  const revSlider = document.getElementById('revShockSlider');
  const costSlider = document.getElementById('costShockSlider');
  try {
    const res = await apiRequest('/api/chat/query', {
      method: 'POST',
      body: JSON.stringify({
        question: trimmed,
        revenue_shock_pct: Number(revSlider?.value || 0),
        cost_shock_pct: Number(costSlider?.value || 0),
      }),
    });
    APP_STATE = res.state;
    document.getElementById('activeChatSkeleton')?.remove();
    renderAllViews();
  } catch (err) {
    document.getElementById('activeChatSkeleton')?.remove();
    showToast(`Query error: ${err.message}`);
  } finally {
    if (sendBtn) sendBtn.disabled = false;
  }
}

document.addEventListener('DOMContentLoaded', async () => {
  initModalDialogs();

  // Theme toggle
  document.getElementById('themeToggleBtn')?.addEventListener('click', toggleTheme);

  // Mobile menu drawer
  const sidebar = document.getElementById('appSidebar');
  const scrim = document.getElementById('sidebarScrim');
  document.getElementById('mobileMenuBtn')?.addEventListener('click', () => {
    sidebar?.classList.toggle('open');
    scrim?.classList.toggle('open');
  });
  scrim?.addEventListener('click', () => {
    sidebar?.classList.remove('open');
    scrim?.classList.remove('open');
  });

  // Sidebar navigation
  document.querySelectorAll('.nav-item').forEach((btn) => {
    btn.addEventListener('click', () => switchView(btn.dataset.view));
  });
  document.querySelectorAll('[data-goto]').forEach((btn) => {
    btn.addEventListener('click', () => switchView(btn.dataset.goto));
  });

  // Tabbed Data Source Connector Switcher (Files / SQL / Power BI)
  document.querySelectorAll('.source-tab').forEach((tabBtn) => {
    tabBtn.addEventListener('click', () => {
      const targetTab = tabBtn.dataset.sourceTab;
      document.querySelectorAll('.source-tab').forEach((b) => {
        b.classList.toggle('active', b === tabBtn);
      });
      ['files', 'sql', 'pbi'].forEach((key) => {
        const panel = document.getElementById(`sourcePanel-${key}`);
        if (panel) {
          panel.classList.toggle('hidden', key !== targetTab);
        }
      });
    });
  });

  // Collapsible Root-Cause Analyzer Drawer Toggle
  const rcDrawer = document.getElementById('rootCauseDrawer');
  document.getElementById('toggleRootCauseBtn')?.addEventListener('click', () => {
    rcDrawer?.classList.toggle('hidden');
  });
  document.getElementById('closeRootCauseBtn')?.addEventListener('click', () => {
    rcDrawer?.classList.add('hidden');
  });

  // Demo Dataset Buttons
  document.getElementById('quickLoadDemoBtn')?.addEventListener('click', loadDemoDataset);
  document.getElementById('heroLoadDemoBtn')?.addEventListener('click', loadDemoDataset);

  // File Input & Drag-and-Drop Feedback
  const fileIn = document.getElementById('fileUploadInput');
  const dropzone = document.getElementById('dropzoneLabel');
  fileIn?.addEventListener('change', () => {
    const names = Array.from(fileIn.files || []).map((f) => f.name);
    document.getElementById('selectedFilesLabel').textContent = names.length
      ? `Selected: ${names.join(', ')}`
      : 'CSV, Excel, Parquet or .pbix — loaded into a local DuckDB warehouse';
  });
  if (dropzone && fileIn) {
    ['dragenter', 'dragover'].forEach((ev) => {
      dropzone.addEventListener(ev, (e) => {
        e.preventDefault();
        dropzone.classList.add('dragover');
      });
    });
    ['dragleave', 'drop'].forEach((ev) => {
      dropzone.addEventListener(ev, (e) => {
        e.preventDefault();
        dropzone.classList.remove('dragover');
      });
    });
    dropzone.addEventListener('drop', (e) => {
      if (e.dataTransfer?.files?.length) {
        fileIn.files = e.dataTransfer.files;
        fileIn.dispatchEvent(new Event('change'));
      }
    });
  }

  // File Upload Submit
  document.getElementById('fileUploadForm')?.addEventListener('submit', async (e) => {
    e.preventDefault();
    if (!fileIn?.files?.length) {
      showToast('Select one or more CSV, Excel, Parquet, or .pbix files first.');
      return;
    }
    const fd = new FormData();
    Array.from(fileIn.files).forEach((f) => fd.append('files', f));
    try {
      showToast('Ingesting & profiling uploaded files…');
      APP_STATE = await apiRequest('/api/connect/upload', { method: 'POST', body: fd });
      renderAllViews();
      showToast('Files ingested and profiled');
    } catch (err) {
      showToast(`Upload error: ${err.message}`);
    }
  });

  // SQL Connect Submit
  document.getElementById('sqlConnectForm')?.addEventListener('submit', async (e) => {
    e.preventDefault();
    try {
      const source_id = document.getElementById('sqlSourceIdInput')?.value?.trim() || undefined;
      const source_name = document.getElementById('sqlSourceNameInput')?.value?.trim() || undefined;
      const attach_mode = document.getElementById('sqlAttachModeInput')?.checked ?? true;

      APP_STATE = await apiRequest('/api/connect/sql', {
        method: 'POST',
        body: JSON.stringify({
          dialect: document.getElementById('sqlDialectInput').value,
          credential_name: document.getElementById('sqlCredNameInput').value,
          connection_url: document.getElementById('sqlUrlInput').value,
          source_id: source_id || null,
          source_name: source_name || null,
          attach_mode: attach_mode,
        }),
      });
      renderAllViews();
      showToast(attach_mode ? 'Connected and attached SQL database' : 'Connected & profiled SQL database');
    } catch (err) {
      showToast(`SQL Error: ${err.message}`);
    }
  });

  // Power BI Mode Toggle & Submit
  const pbiMode = document.getElementById('pbiModeInput');
  pbiMode?.addEventListener('change', () => {
    const isDesktop = pbiMode.value === 'powerbi_desktop';
    document.getElementById('pbiDesktopFields').classList.toggle('hidden', !isDesktop);
    document.getElementById('pbiCloudFields').classList.toggle('hidden', isDesktop);
  });

  document.getElementById('pbiConnectForm')?.addEventListener('submit', async (e) => {
    e.preventDefault();
    try {
      APP_STATE = await apiRequest('/api/connect/powerbi', {
        method: 'POST',
        body: JSON.stringify({
          mode: pbiMode.value,
          localhost_port: Number(document.getElementById('pbiPortInput').value || 55000),
          dataset_id: document.getElementById('pbiDatasetInput').value,
          access_token: document.getElementById('pbiTokenInput').value,
        }),
      });
      renderAllViews();
      showToast('Connected to Power BI semantic model');
    } catch (err) {
      showToast(`Power BI Error: ${err.message}`);
    }
  });

  // Join Actions
  document.getElementById('approveAllJoinsBtn')?.addEventListener('click', async () => {
    APP_STATE = await apiRequest('/api/joins/approve-all', { method: 'POST' });
    renderAllViews();
    showToast('All candidate relationships approved');
  });

  document.getElementById('saveJoinEditsBtn')?.addEventListener('click', async () => {
    const rels = APP_STATE.workspace.relationships || [];
    const updated = rels.map((r, idx) => ({
      id: r.id,
      source_table: r.source_table,
      source_column: r.source_column,
      join_type: document.querySelector(`.join-type-sel[data-idx="${idx}"]`)?.value || r.join_type,
      target_table: r.target_table,
      target_column: r.target_column,
      cardinality: document.querySelector(`.join-card-sel[data-idx="${idx}"]`)?.value || r.cardinality,
      status: document.querySelector(`.join-status-sel[data-idx="${idx}"]`)?.value || r.status,
    }));
    APP_STATE = await apiRequest('/api/joins/update', {
      method: 'POST',
      body: JSON.stringify({ relationships: updated }),
    });
    renderAllViews();
    showToast('Saved relationship governance changes');
  });

  document.getElementById('cjSourceTable')?.addEventListener('change', populateCustomJoinColumns);
  document.getElementById('cjTargetTable')?.addEventListener('change', populateCustomJoinColumns);

  document.getElementById('customJoinForm')?.addEventListener('submit', async (e) => {
    e.preventDefault();
    APP_STATE = await apiRequest('/api/joins/add', {
      method: 'POST',
      body: JSON.stringify({
        source_table: document.getElementById('cjSourceTable').value,
        source_column: document.getElementById('cjSourceCol').value,
        target_table: document.getElementById('cjTargetTable').value,
        target_column: document.getElementById('cjTargetCol').value,
        cardinality: document.getElementById('cjCardinality').value,
      }),
    });
    renderAllViews();
    showToast('Custom relationship added & approved');
  });

  // Custom Business Rule Form
  document.getElementById('customRuleForm')?.addEventListener('submit', async (e) => {
    e.preventDefault();
    APP_STATE = await apiRequest('/api/interview/add-rule', {
      method: 'POST',
      body: JSON.stringify({
        term: document.getElementById('ruleTermInput').value,
        definition: document.getElementById('ruleDefInput').value,
        sql_filter_hint: document.getElementById('ruleSqlHintInput').value,
      }),
    });
    e.target.reset();
    renderAllViews();
    showToast('Added glossary rule to OKF & RAG');
  });

  // Measure & Calculated Column Creation
  document.getElementById('measureCreateForm')?.addEventListener('submit', async (e) => {
    e.preventDefault();
    try {
      APP_STATE = await apiRequest('/api/measures/create', {
        method: 'POST',
        body: JSON.stringify({
          name: document.getElementById('measNameInput').value,
          table_name: document.getElementById('measTableSelect').value,
          input_type: document.getElementById('measLangSelect').value,
          user_formula: document.getElementById('measFormulaInput').value,
          description: document.getElementById('measDescInput').value,
        }),
      });
      e.target.reset();
      renderAllViews();
      showToast('Measure compiled, verified & indexed');
    } catch (err) {
      showToast(`Measure error: ${err.message}`);
    }
  });

  document.getElementById('calcColCreateForm')?.addEventListener('submit', async (e) => {
    e.preventDefault();
    try {
      APP_STATE = await apiRequest('/api/measures/calculated-column', {
        method: 'POST',
        body: JSON.stringify({
          column_name: document.getElementById('ccNameInput').value,
          table_name: document.getElementById('ccTableSelect').value,
          input_type: document.getElementById('ccLangSelect').value,
          user_formula: document.getElementById('ccFormulaInput').value,
        }),
      });
      e.target.reset();
      renderAllViews();
      showToast('Calculated column compiled & verified');
    } catch (err) {
      showToast(`Calculated column error: ${err.message}`);
    }
  });

  // Auto-expanding Chat Textarea & Enter-to-Send
  const chatInput = document.getElementById('chatQuestionInput');
  chatInput?.addEventListener('input', () => {
    chatInput.style.height = 'auto';
    chatInput.style.height = `${Math.min(140, chatInput.scrollHeight)}px`;
  });
  chatInput?.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      document.getElementById('chatInputForm')?.requestSubmit();
    }
  });

  document.getElementById('chatInputForm')?.addEventListener('submit', async (e) => {
    e.preventDefault();
    if (!chatInput) return;
    const q = chatInput.value;
    chatInput.value = '';
    chatInput.style.height = 'auto';
    await submitChatQuestion(q);
  });

  // Teach AI Form inside Lineage Modal
  document.getElementById('teachAiForm')?.addEventListener('submit', async (e) => {
    e.preventDefault();
    const ruleText = document.getElementById('teachAiRuleInput').value;
    if (!ruleText.trim()) return;
    APP_STATE = await apiRequest('/api/chat/teach', {
      method: 'POST',
      body: JSON.stringify({
        question: CURRENT_LINEAGE_ENTRY?.response?.user_question || 'Chat Query',
        rule_text: ruleText,
      }),
    });
    document.getElementById('teachAiRuleInput').value = '';
    closeTopLayer(document.getElementById('lineageDialog'));
    renderAllViews();
    showToast('Saved business rule to OKF & RAG');
  });

  // Root-Cause Variance Form
  document.getElementById('rootCauseForm')?.addEventListener('submit', async (e) => {
    e.preventDefault();
    const metric_col = document.getElementById('rcMetricSelect').value;
    const dimension = document.getElementById('rcDimensionSelect').value;
    try {
      const res = await apiRequest('/api/root-cause', {
        method: 'POST',
        body: JSON.stringify({ metric_col, dimension }),
      });
      const box = document.getElementById('rootCauseResultBox');
      box.innerHTML = `
        <div class="chat-summary-text" style="margin-top:12px;">${formatMarkdownBold(res.summary || '')}</div>
        <div id="rcWaterfallPlot" class="chart-box" style="min-height:280px; margin-top:10px;"></div>
      `;
      const plotEl = document.getElementById('rcWaterfallPlot');
      if (plotEl && res.plotly_spec && window.Plotly) {
        try {
          const themedLayout = applyPlotlyThemeLayout(res.plotly_spec.layout);
          window.Plotly.react(plotEl, res.plotly_spec.data, themedLayout, {
            responsive: true,
            displayModeBar: false,
          });
        } catch (rcErr) {
          console.warn('Plotly waterfall error:', rcErr);
        }
      }
    } catch (err) {
      showToast(`Variance error: ${err.message}`);
    }
  });

  // What-If Sliders
  document.getElementById('revShockSlider')?.addEventListener('input', refreshDashboardSimulation);
  document.getElementById('costShockSlider')?.addEventListener('input', refreshDashboardSimulation);
  document.getElementById('refreshDashboardBtn')?.addEventListener('click', refreshDashboardSimulation);

  // Workspace Switcher & New Workspace
  document.getElementById('workspaceSelect')?.addEventListener('change', async (e) => {
    APP_STATE = await apiRequest('/api/settings', {
      method: 'POST',
      body: JSON.stringify({ workspace_id: e.target.value }),
    });
    renderAllViews();
  });

  document.getElementById('newWorkspaceBtn')?.addEventListener('click', async () => {
    const name = prompt('Enter new workspace ID (e.g., finance_q4):');
    if (name && name.trim()) {
      APP_STATE = await apiRequest('/api/settings', {
        method: 'POST',
        body: JSON.stringify({ workspace_id: name.trim() }),
      });
      renderAllViews();
      showToast(`Switched to workspace: ${APP_STATE.active_workspace_id}`);
    }
  });

  // Privacy Mode Switcher & Modals
  document.getElementById('privacyModeSelect')?.addEventListener('change', async (e) => {
    APP_STATE = await apiRequest('/api/settings', {
      method: 'POST',
      body: JSON.stringify({ privacy_mode: e.target.value }),
    });
    renderGlobalTelemetry();
    showToast(`Privacy mode updated: ${e.target.value}`);
  });

  document.getElementById('openAuditModalBtn')?.addEventListener('click', async () => {
    APP_STATE = await apiRequest('/api/state');
    const logBox = document.getElementById('auditLogJsonBox');
    const logs = APP_STATE.audit_log || [];
    logBox.textContent = logs.length
      ? JSON.stringify(logs, null, 2)
      : 'Zero external LLM payloads sent in this session (deterministic local compiler active — 0 bytes left localhost).';
    document.getElementById('securityAuditDialog')?.showModal();
  });

  document.getElementById('openLlmSettingsBtn')?.addEventListener('click', () => {
    document.getElementById('llmConfigDialog')?.showModal();
  });

  document.getElementById('llmSettingsForm')?.addEventListener('submit', async (e) => {
    e.preventDefault();
    APP_STATE = await apiRequest('/api/settings', {
      method: 'POST',
      body: JSON.stringify({
        llm_model: document.getElementById('llmModelSelect').value,
        api_key: document.getElementById('llmApiKeyInput').value,
        api_base: document.getElementById('llmApiBaseInput').value,
      }),
    });
    closeTopLayer(document.getElementById('llmConfigDialog'));
    renderGlobalTelemetry();
    showToast('Updated AI model configuration');
  });

  // Clear Conversation Button
  document.getElementById('clearChatBtn')?.addEventListener('click', async () => {
    try {
      APP_STATE = await apiRequest('/api/chat/clear', { method: 'POST' });
      renderAllViews();
      showToast('Cleared conversation');
    } catch (err) {
      showToast(`Error clearing chat: ${err.message}`);
    }
  });

  // Live Read-Only SQL / DAX Scratchpad Modal
  document.getElementById('openScratchpadBtn')?.addEventListener('click', () => {
    document.getElementById('scratchpadDialog')?.showModal();
  });

  document.getElementById('scratchpadQueryInput')?.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      e.preventDefault();
      document.getElementById('scratchpadForm')?.requestSubmit();
    }
  });

  document.getElementById('scratchpadForm')?.addEventListener('submit', async (e) => {
    e.preventDefault();
    const query = document.getElementById('scratchpadQueryInput')?.value || '';
    const box = document.getElementById('scratchpadResultBox');
    if (!query.trim() || !box) return;
    box.innerHTML = `<div class="muted small">Executing read-only query…</div>`;
    try {
      const res = await apiRequest('/api/sql/execute', {
        method: 'POST',
        body: JSON.stringify({ query }),
      });
      const cols = res.columns || [];
      const rows = (res.rows || []).slice(0, 50);
      if (!cols.length) {
        box.innerHTML = `<div class="small" style="color:var(--accent-emerald);">Query executed (0 columns returned).</div>`;
        return;
      }
      box.innerHTML = `
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
          <span class="pii-badge pii-safe"><svg class="icon" style="width:11px;height:11px"><use href="#i-check"/></svg>Verified read-only · ${res.row_count} rows</span>
        </div>
        <div class="table-responsive" style="max-height:280px;">
          <table class="data-table">
            <thead>
              <tr>${cols.map((c) => `<th>${escapeHtml(c)}</th>`).join('')}</tr>
            </thead>
            <tbody>
              ${rows
                .map(
                  (r) =>
                    `<tr>${cols
                      .map((c) => `<td class="mono">${escapeHtml(formatCellVal(r[c]))}</td>`)
                      .join('')}</tr>`
                )
                .join('')}
            </tbody>
          </table>
        </div>
      `;
    } catch (err) {
      box.innerHTML = `<div class="anomaly-banner" style="border-radius:6px;"><svg class="icon"><use href="#i-alert"/></svg><span><strong>Blocked:</strong> ${escapeHtml(err.message)}</span></div>`;
    }
  });

  // Global Command Palette (Ctrl+K / Cmd+K) with Keyboard Navigation
  const cmdDialog = document.getElementById('commandPaletteDialog');
  const cmdInput = document.getElementById('cmdSearchInput');

  function openCommandPalette() {
    if (!cmdDialog) return;
    cmdDialog.showModal();
    if (cmdInput) {
      cmdInput.value = '';
      CMD_ACTIVE_INDEX = 0;
      renderCommandPalette('');
      cmdInput.focus();
    }
  }

  document.getElementById('openCmdPaletteBtn')?.addEventListener('click', openCommandPalette);

  window.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') {
      e.preventDefault();
      if (cmdDialog?.open) {
        closeTopLayer(cmdDialog);
      } else {
        openCommandPalette();
      }
    }
  });

  cmdInput?.addEventListener('input', (e) => {
    CMD_ACTIVE_INDEX = 0;
    renderCommandPalette(e.target.value);
  });

  cmdInput?.addEventListener('keydown', (e) => {
    if (!CMD_FILTERED_ITEMS.length) return;
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      CMD_ACTIVE_INDEX = (CMD_ACTIVE_INDEX + 1) % CMD_FILTERED_ITEMS.length;
      highlightCommandItem();
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      CMD_ACTIVE_INDEX = (CMD_ACTIVE_INDEX - 1 + CMD_FILTERED_ITEMS.length) % CMD_FILTERED_ITEMS.length;
      highlightCommandItem();
    } else if (e.key === 'Enter') {
      e.preventDefault();
      const item = CMD_FILTERED_ITEMS[CMD_ACTIVE_INDEX];
      closeTopLayer(cmdDialog);
      if (item && item.action) item.action();
    }
  });

  // Initial state load
  try {
    APP_STATE = await apiRequest('/api/state');
    if (!(APP_STATE.workspace?.tables || []).length) {
      APP_STATE = await apiRequest('/api/connect/demo', { method: 'POST' });
    }
    renderAllViews();
    if (!(APP_STATE.chat_history || []).length) {
      await submitChatQuestion('Show net revenue, profit, and gross margin % by product category');
    }
  } catch (err) {
    showToast(`Initialization error: ${err.message}`);
  }
});

function highlightCommandItem() {
  const listEl = document.getElementById('cmdResultsList');
  if (!listEl) return;
  listEl.querySelectorAll('.cmd-item').forEach((btn, i) => {
    btn.classList.toggle('active', i === CMD_ACTIVE_INDEX);
    if (i === CMD_ACTIVE_INDEX) {
      btn.scrollIntoView({ block: 'nearest' });
    }
  });
}

function renderCommandPalette(queryText = '') {
  const listEl = document.getElementById('cmdResultsList');
  const cmdDialog = document.getElementById('commandPaletteDialog');
  if (!listEl) return;

  const q = queryText.trim().toLowerCase();
  const ws = APP_STATE?.workspace || {};
  const items = [
    { category: 'Navigate', label: 'Ask Data — Conversational BI', action: () => switchView('chat') },
    { category: 'Navigate', label: 'Pinboard & What-If Simulator', action: () => switchView('dashboard') },
    { category: 'Setup · Step 1', label: 'Data sources & schema profiler', action: () => switchView('connect') },
    { category: 'Setup · Step 2', label: 'Relationships & star-schema graph', action: () => switchView('joins') },
    { category: 'Setup · Step 3', label: 'Business rules & guided Q&A', action: () => switchView('interview') },
    { category: 'Setup · Step 4', label: 'Measures & DAX ↔ SQL studio', action: () => switchView('measures') },
    {
      category: 'Tool',
      label: 'Open read-only SQL scratchpad',
      action: () => document.getElementById('scratchpadDialog')?.showModal(),
    },
    {
      category: 'Tool',
      label: 'Toggle light / dark theme',
      action: () => toggleTheme(),
    },
    {
      category: 'Tool',
      label: 'Inspect privacy & outbound LLM audit log',
      action: () => document.getElementById('openAuditModalBtn')?.click(),
    },
    {
      category: 'Export',
      label: 'Export executive BI report (HTML / PDF)',
      action: () => window.open('/api/export/report', '_blank'),
    },
    {
      category: 'Data',
      label: 'Reload 4-table demo warehouse',
      action: () => loadDemoDataset(),
    },
  ];

  (ws.tables || []).forEach((t) => {
    items.push({
      category: `Table · ${t.table_role}`,
      label: `${t.name} (${Number(t.row_count || 0).toLocaleString()} rows, ${(t.columns || []).length} cols)`,
      action: () => {
        FOCAL_TABLE_FILTER = t.name;
        switchView('joins');
        renderJoinsView();
      },
    });
  });

  (ws.measures || []).forEach((m) => {
    items.push({
      category: 'Measure',
      label: `${m.table_name}[${m.name}] = ${m.dax_expression}`,
      action: () => switchView('measures'),
    });
  });

  const filtered = items
    .filter((it) => !q || it.label.toLowerCase().includes(q) || it.category.toLowerCase().includes(q))
    .slice(0, 12);

  if (q && !filtered.length) {
    filtered.push({
      category: 'Ask AI',
      label: `Ask Data: "${queryText.trim()}"`,
      action: async () => {
        switchView('chat');
        await submitChatQuestion(queryText.trim());
      },
    });
  }

  CMD_FILTERED_ITEMS = filtered;

  listEl.innerHTML = filtered
    .map(
      (it, idx) => `
      <button type="button" class="cmd-item ${idx === CMD_ACTIVE_INDEX ? 'active' : ''}" data-cmd-idx="${idx}" role="option">
        <span>${escapeHtml(it.label)}</span>
        <span class="cmd-item-category">${escapeHtml(it.category)}</span>
      </button>
    `
    )
    .join('');

  listEl.querySelectorAll('.cmd-item').forEach((btn) => {
    btn.addEventListener('click', () => {
      const idx = Number(btn.dataset.cmdIdx);
      const item = CMD_FILTERED_ITEMS[idx];
      closeTopLayer(cmdDialog);
      if (item && item.action) item.action();
    });
  });
}

// Auto-resize Plotly charts on viewport / window resize
window.addEventListener('resize', () => {
  if (window.Plotly) {
    document.querySelectorAll('.chart-box').forEach((el) => {
      if (el && el.data) {
        try {
          window.Plotly.Plots.resize(el);
        } catch (_) {}
      }
    });
  }
});
