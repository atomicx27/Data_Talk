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
  verified_queries: 'Query Memory',
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

  // Adapt for multi-table university / higher-education dataset (colleges, departments, enrollments_marks, fee_invoices, faculty)
  if (tableNames.some((n) => n.includes('college') || n.includes('department') || n.includes('faculty'))) {
    return [
      {
        prompt: 'Show student count by College and Gender',
        title: 'Students by College & Gender',
        sub: 'Demographic enrollment breakdown across colleges',
        icon: 'i-bar',
      },
      {
        prompt: 'Show average marks for each College',
        title: 'Average marks by College',
        sub: 'Academic assessment score benchmark across colleges',
        icon: 'i-trend-up',
      },
      {
        prompt: 'Show total fee due by College',
        title: 'Tuition fees by College',
        sub: 'Aggregate tuition billing and net fee due per college',
        icon: 'i-pie',
      },
      {
        prompt: 'Show faculty salary trend year on year',
        title: 'Faculty salary trend YoY',
        sub: 'Annual faculty compensation trajectory over time',
        icon: 'i-schema',
      },
    ];
  }

  // Adapt for single-table education/students dataset
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

// ==================== DATA INGESTION LOADING CONTROLLER ====================
let ingestionTimerInterval = null;
let ingestionPhaseInterval = null;
let ingestionStartTime = 0;

function showIngestionLoading(title = 'Ingesting & Profiling Data', subtitle = 'Preparing your analytical workspace...') {
  const overlay = document.getElementById('ingestionLoadingOverlay');
  if (!overlay) return;

  const titleEl = document.getElementById('ingestionTitle');
  const subtitleEl = document.getElementById('ingestionSubtitle');
  const progressBar = document.getElementById('ingestionProgressBar');
  const phaseLabel = document.getElementById('ingestionCurrentPhase');
  const timerEl = document.getElementById('ingestionElapsedTime');

  if (titleEl) titleEl.textContent = title;
  if (subtitleEl) subtitleEl.textContent = subtitle;
  if (progressBar) progressBar.style.width = '18%';
  if (phaseLabel) phaseLabel.textContent = 'Stage 1 of 4: Reading and validating files';
  if (timerEl) timerEl.textContent = 'Elapsed: 0s';

  for (let i = 1; i <= 4; i++) {
    const s = document.getElementById(`ingestStep${i}`);
    if (s) s.className = i === 1 ? 'ingest-step active' : 'ingest-step pending';
  }

  overlay.classList.remove('hidden');

  ingestionStartTime = Date.now();
  if (ingestionTimerInterval) clearInterval(ingestionTimerInterval);
  ingestionTimerInterval = setInterval(() => {
    const elapsedSec = Math.floor((Date.now() - ingestionStartTime) / 1000);
    if (timerEl) timerEl.textContent = `Elapsed: ${elapsedSec}s`;
  }, 500);

  const stages = [
    { step: 1, width: '25%', label: 'Stage 1 of 4: Validating source files & schema' },
    { step: 2, width: '55%', label: 'Stage 2 of 4: Materializing DuckDB columnar storage' },
    { step: 3, width: '78%', label: 'Stage 3 of 4: Profiling columns & detecting PII' },
    { step: 4, width: '92%', label: 'Stage 4 of 4: Inferring relationships & domain graph' },
  ];
  let curStageIdx = 0;
  if (ingestionPhaseInterval) clearInterval(ingestionPhaseInterval);
  ingestionPhaseInterval = setInterval(() => {
    curStageIdx++;
    if (curStageIdx < stages.length) {
      const st = stages[curStageIdx];
      if (progressBar) progressBar.style.width = st.width;
      if (phaseLabel) phaseLabel.textContent = st.label;
      for (let i = 1; i <= 4; i++) {
        const stepEl = document.getElementById(`ingestStep${i}`);
        if (!stepEl) continue;
        if (i < st.step) {
          stepEl.className = 'ingest-step completed';
        } else if (i === st.step) {
          stepEl.className = 'ingest-step active';
        } else {
          stepEl.className = 'ingest-step pending';
        }
      }
    }
  }, 1100);
}

function hideIngestionLoading() {
  const overlay = document.getElementById('ingestionLoadingOverlay');
  if (!overlay) return;

  const progressBar = document.getElementById('ingestionProgressBar');
  const phaseLabel = document.getElementById('ingestionCurrentPhase');
  if (progressBar) progressBar.style.width = '100%';
  if (phaseLabel) phaseLabel.textContent = 'Ingestion complete!';

  for (let i = 1; i <= 4; i++) {
    const stepEl = document.getElementById(`ingestStep${i}`);
    if (stepEl) stepEl.className = 'ingest-step completed';
  }

  if (ingestionTimerInterval) {
    clearInterval(ingestionTimerInterval);
    ingestionTimerInterval = null;
  }
  if (ingestionPhaseInterval) {
    clearInterval(ingestionPhaseInterval);
    ingestionPhaseInterval = null;
  }

  setTimeout(() => {
    overlay.classList.add('hidden');
  }, 400);
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
    // In dark theme, ensure seamless transparent canvas and automargins
    return {
      ...baseLayout,
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      xaxis: {
        ...(baseLayout.xaxis || {}),
        automargin: true,
        gridcolor: 'rgba(255, 255, 255, 0.08)',
      },
      yaxis: {
        ...(baseLayout.yaxis || {}),
        automargin: true,
        gridcolor: 'rgba(255, 255, 255, 0.08)',
      },
    };
  }
  // In light theme, transparent canvas blends seamlessly with card surface, avoiding cut-off gray rectangles
  return {
    ...baseLayout,
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: {
      ...(baseLayout.font || {}),
      color: '#0f172a',
    },
    xaxis: {
      ...(baseLayout.xaxis || {}),
      automargin: true,
      gridcolor: '#e2e8f0',
    },
    yaxis: {
      ...(baseLayout.yaxis || {}),
      automargin: true,
      gridcolor: '#e2e8f0',
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
    } else if (viewName === 'verified_queries') {
      renderVerifiedQueriesView();
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
  const domainBadge = document.getElementById('headerDomainBadge');
  if (domainBadge) {
    const domainTitle = APP_STATE.domain_info?.domain_title || ws.domain_title;
    if (domainTitle) {
      domainBadge.style.display = 'inline-flex';
      domainBadge.innerHTML = `<svg class="icon" style="width:12px;height:12px"><use href="#i-sparkles"/></svg> ${escapeHtml(domainTitle)}`;
    } else {
      domainBadge.style.display = 'none';
    }
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
  const vqList = ws.verified_queries || [];
  const vqBadge = document.getElementById('badgeVerifiedQueries');
  if (vqBadge) vqBadge.textContent = `${vqList.length}`;

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
              <td class="mono" style="color:var(--text-primary); font-weight:600;">
                ${escapeHtml(c.name)}
                ${c.business_alias ? `<div style="font-size:11px; color:var(--accent-primary); font-weight:600;">[${escapeHtml(c.business_alias)}]</div>` : ''}
              </td>
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
              ${tbl.business_name ? `<span style="color:var(--accent-indigo); font-weight:600; font-size:13px;">— ${escapeHtml(tbl.business_name)}</span>` : ''}
              ${
                tbl.source_name || tbl.source_id
                  ? `<span class="source-badge" title="Source: ${escapeHtml(tbl.source_name || tbl.source_id)}"><svg class="icon" style="width:10px;height:10px;vertical-align:-1px;margin-right:2px;"><use href="#i-database"/></svg>${escapeHtml(tbl.source_name || tbl.source_id)}</span>`
                  : ''
              }
              <span class="muted" style="font-size:12px;">${Number(tbl.row_count || 0).toLocaleString()} rows · ${(tbl.columns || []).length} cols</span>
            </div>
            <div style="display:flex; align-items:center; gap:10px;">
              <button type="button" class="btn btn-secondary btn-xs btn-edit-tbl-connect" data-table="${escapeHtml(tbl.name)}" title="Edit Business Semantics">
                <svg class="icon" style="width:10px;height:10px;display:inline;vertical-align:-1px;margin-right:2px;"><use href="#i-edit"/></svg>Semantics
              </button>
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

  container.querySelectorAll('.btn-edit-tbl-connect').forEach((btn) => {
    btn.addEventListener('click', (e) => {
      e.preventDefault();
      e.stopPropagation();
      const t = tables.find((x) => x.name === btn.dataset.table);
      if (t) openTableEditModal(t);
    });
  });

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

  // 1. Domain Overview Banner with Human Customization
  const domainBanner = document.getElementById('domainOverviewBanner');
  if (domainBanner) {
    const dInfo = APP_STATE.domain_info || {};
    const dTitle = ws.domain_title || dInfo.domain_title || 'Enterprise Operations & Analytics';
    const dIndustry = dInfo.industry || 'General Business';
    const dSummary = dInfo.summary || 'Operational business dataset';
    domainBanner.innerHTML = `
      <div style="display:flex; align-items:center; gap:12px; flex:1; min-width:240px;">
        <span class="domain-badge-lg">
          <svg class="icon" style="width:12px;height:12px"><use href="#i-book"/></svg>
          ${escapeHtml(dTitle)}
        </span>
        <div>
          <div style="font-weight:600; font-size:13px; color:var(--text-primary);">${escapeHtml(dIndustry)}</div>
          <div style="font-size:11.5px; color:var(--text-muted);">${escapeHtml(dSummary)}</div>
        </div>
      </div>
      <button type="button" id="btnOpenDomainEdit" class="btn btn-secondary btn-sm" style="display:flex;align-items:center;gap:6px;">
        <svg class="icon" style="width:12px;height:12px"><use href="#i-edit"/></svg> Customize Domain
      </button>
    `;
    document.getElementById('btnOpenDomainEdit')?.addEventListener('click', () => {
      openDomainEditModal();
    });
  }

  // 2. Interview Progress Banner
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

  // 3. Multi-source question filter chips
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

  // 4. Questions container
  const container = document.getElementById('interviewQuestionsContainer');
  if (container) {
    if (!qs.length) {
      container.innerHTML = `<div class="panel-card" style="text-align:center; padding:18px; color:var(--text-muted); font-size:12.5px;">Connect a dataset in Step 1 to generate schema-tailored business rule questions.</div>`;
    } else {
      const filteredQs = qs.filter((q) => {
        if (!INTERVIEW_SOURCE_FILTER) return true;
        if (INTERVIEW_SOURCE_FILTER === '__cross__') {
          return q.id.startsWith('q_cross_link_') || q.category === 'CROSS_DATABASE_LINK';
        }
        return (q.source_id || 'default') === INTERVIEW_SOURCE_FILTER;
      });

      if (!filteredQs.length) {
        container.innerHTML = `<div class="panel-card" style="text-align:center; padding:18px; color:var(--text-muted); font-size:12.5px;">No questions found matching the selected filter.</div>`;
      } else {
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
    }
  }

  // 5. Active Business Rules & Governance Container
  const rulesContainer = document.getElementById('activeBusinessRulesContainer');
  if (rulesContainer) {
    const rules = ws.business_rules || [];
    if (!rules.length) {
      rulesContainer.innerHTML = `
        <div class="panel-card" style="grid-column: 1 / -1; text-align:center; padding:18px; color:var(--text-muted); font-size:12.5px;">
          No business rules indexed yet. Answer questions above or add custom rules below.
        </div>
      `;
    } else {
      rulesContainer.innerHTML = rules
        .map((r) => {
          const sourceLabel = r.source === 'manual' ? 'Custom' : (r.source === 'enriched' || r.source === 'auto_enricher') ? '✨ Auto-Enriched' : 'Interview';
          const filterHtml = r.sql_filter_hint ? `<div class="rule-item-sql">SQL: ${escapeHtml(r.sql_filter_hint)}</div>` : '';
          return `
            <div class="rule-item-card">
              <div class="rule-item-header">
                <div class="rule-item-term">${escapeHtml(r.term)}</div>
                <div style="display:flex; align-items:center; gap:6px;">
                  <span class="meta-chip">${escapeHtml(sourceLabel)}</span>
                  <button type="button" class="btn btn-ghost btn-icon-only btn-edit-rule" data-rid="${escapeHtml(r.id)}" title="Edit rule" style="width:26px;height:26px;">
                    <svg class="icon" style="width:12px;height:12px"><use href="#i-edit"/></svg>
                  </button>
                  <button type="button" class="btn btn-ghost btn-icon-only btn-del-rule" data-rid="${escapeHtml(r.id)}" title="Delete rule" style="width:26px;height:26px;color:var(--accent-rose);">
                    <svg class="icon" style="width:12px;height:12px"><use href="#i-trash"/></svg>
                  </button>
                </div>
              </div>
              <div class="rule-item-def">${escapeHtml(r.definition)}</div>
              ${filterHtml}
            </div>
          `;
        })
        .join('');

      rulesContainer.querySelectorAll('.btn-edit-rule').forEach((btn) => {
        btn.addEventListener('click', () => {
          const r = rules.find((x) => x.id === btn.dataset.rid);
          if (r) openRuleEditModal(r);
        });
      });

      rulesContainer.querySelectorAll('.btn-del-rule').forEach((btn) => {
        btn.addEventListener('click', async () => {
          const r = rules.find((x) => x.id === btn.dataset.rid);
          if (!r) return;
          if (confirm(`Delete rule "${r.term}" from knowledge base?`)) {
            try {
              APP_STATE = await apiRequest(`/api/interview/rule/${encodeURIComponent(r.id)}`, { method: 'DELETE' });
              renderAllViews();
              showToast(`Deleted rule "${r.term}"`);
            } catch (err) {
              showToast(`Delete failed: ${err.message}`);
            }
          }
        });
      });
    }
  }

  // 6. Semantic Domain Glossary (Tables & Columns)
  const glossaryContainer = document.getElementById('semanticGlossaryContainer');
  if (glossaryContainer) {
    const tables = (ws.tables || []).filter((t) => !t.is_blocked);
    if (!tables.length) {
      glossaryContainer.innerHTML = `<div class="panel-card" style="text-align:center; padding:18px; color:var(--text-muted); font-size:12.5px;">No tables available. Ingest data in Step 1.</div>`;
    } else {
      glossaryContainer.innerHTML = tables
        .map((tbl) => {
          const colRows = (tbl.columns || [])
            .filter((c) => !c.is_blocked)
            .map((c) => {
              let meaningsHtml = '—';
              if (c.value_meanings && Object.keys(c.value_meanings).length > 0) {
                meaningsHtml = Object.entries(c.value_meanings)
                  .map(([k, v]) => `<span class="value-meaning-pill"><strong>${escapeHtml(k)}</strong>: ${escapeHtml(v)}</span>`)
                  .join(' ');
              }
              return `
                <tr>
                  <td class="mono" style="font-weight:600; color:var(--text-primary);">${escapeHtml(c.name)}</td>
                  <td style="color:var(--accent-primary); font-weight:600;">${escapeHtml(c.business_alias || '—')}</td>
                  <td style="font-size:12px; color:var(--text-secondary); max-width:280px;">${escapeHtml(c.description || '—')}</td>
                  <td style="max-width:240px;">${meaningsHtml}</td>
                  <td style="text-align:right;">
                    <button type="button" class="btn btn-secondary btn-xs btn-edit-col" data-table="${escapeHtml(tbl.name)}" data-col="${escapeHtml(c.name)}">
                      <svg class="icon" style="width:10px;height:10px;display:inline;vertical-align:-1px;margin-right:2px;"><use href="#i-edit"/></svg>Edit
                    </button>
                  </td>
                </tr>
              `;
            })
            .join('');

          return `
            <div class="semantic-glossary-card">
              <div class="semantic-glossary-header">
                <div style="display:flex; align-items:center; gap:10px; flex-wrap:wrap;">
                  <span class="role-badge role-${escapeHtml(tbl.table_role)}">${escapeHtml(tbl.table_role)}</span>
                  <strong class="mono" style="font-size:13.5px; color:var(--text-primary);">${escapeHtml(tbl.name)}</strong>
                  ${tbl.business_name ? `<span style="color:var(--accent-indigo); font-weight:600; font-size:13px;">— ${escapeHtml(tbl.business_name)}</span>` : ''}
                  <span class="muted" style="font-size:11.5px;">(${Number(tbl.row_count || 0).toLocaleString()} rows)</span>
                </div>
                <button type="button" class="btn btn-secondary btn-xs btn-edit-tbl" data-table="${escapeHtml(tbl.name)}">
                  <svg class="icon" style="width:11px;height:11px;display:inline;vertical-align:-1px;margin-right:3px;"><use href="#i-edit"/></svg>Edit Table Semantics
                </button>
              </div>
              <div class="semantic-glossary-body">
                <p style="font-size:12.5px; color:var(--text-secondary); margin-bottom:12px;">
                  ${escapeHtml(tbl.description || 'No operational description provided yet.')}
                </p>
                <div class="table-responsive">
                  <table class="data-table">
                    <thead>
                      <tr>
                        <th>Physical Column</th>
                        <th>Business Alias</th>
                        <th>Operational Meaning</th>
                        <th>Status Code Decoders</th>
                        <th style="text-align:right;">Action</th>
                      </tr>
                    </thead>
                    <tbody>${colRows}</tbody>
                  </table>
                </div>
              </div>
            </div>
          `;
        })
        .join('');

      glossaryContainer.querySelectorAll('.btn-edit-tbl').forEach((btn) => {
        btn.addEventListener('click', () => {
          const t = tables.find((x) => x.name === btn.dataset.table);
          if (t) openTableEditModal(t);
        });
      });

      glossaryContainer.querySelectorAll('.btn-edit-col').forEach((btn) => {
        btn.addEventListener('click', () => {
          const t = tables.find((x) => x.name === btn.dataset.table);
          if (!t) return;
          const c = (t.columns || []).find((x) => x.name === btn.dataset.col);
          if (c) openColumnEditModal(t.name, c);
        });
      });
    }
  }
}

function openDomainEditModal() {
  const ws = APP_STATE.workspace || {};
  const dInfo = APP_STATE.domain_info || {};
  const titleInput = document.getElementById('domainTitleInput');
  const indInput = document.getElementById('domainIndustryInput');
  const tagInput = document.getElementById('domainTagInput');
  if (titleInput) titleInput.value = ws.domain_title || dInfo.domain_title || '';
  if (indInput) indInput.value = dInfo.industry || '';
  if (tagInput) tagInput.value = ws.domain_tag || dInfo.domain_key || '';
  document.getElementById('domainEditDialog')?.showModal();
}

function openRuleEditModal(rule) {
  const idInput = document.getElementById('editRuleIdInput');
  const termInput = document.getElementById('editRuleTermInput');
  const defInput = document.getElementById('editRuleDefInput');
  const hintInput = document.getElementById('editRuleSqlHintInput');
  if (idInput) idInput.value = rule.id || '';
  if (termInput) termInput.value = rule.term || '';
  if (defInput) defInput.value = rule.definition || '';
  if (hintInput) hintInput.value = rule.sql_filter_hint || '';
  document.getElementById('ruleEditDialog')?.showModal();
}

function openTableEditModal(table) {
  const nameInput = document.getElementById('editTableNameInput');
  const bnameInput = document.getElementById('editTableBusinessNameInput');
  const roleSelect = document.getElementById('editTableRoleSelect');
  const descInput = document.getElementById('editTableDescriptionInput');
  if (nameInput) nameInput.value = table.name || '';
  if (bnameInput) bnameInput.value = table.business_name || '';
  if (roleSelect) roleSelect.value = (table.table_role || 'unknown').toLowerCase();
  if (descInput) descInput.value = table.description || '';
  document.getElementById('tableEditDialog')?.showModal();
}

function openColumnEditModal(tableName, col) {
  const tblInput = document.getElementById('editColTableInput');
  const colInput = document.getElementById('editColNameInput');
  const aliasInput = document.getElementById('editColAliasInput');
  const descInput = document.getElementById('editColDescInput');
  const meaningsInput = document.getElementById('editColMeaningsInput');
  if (tblInput) tblInput.value = tableName || '';
  if (colInput) colInput.value = col.name || '';
  if (aliasInput) aliasInput.value = col.business_alias || '';
  if (descInput) descInput.value = col.description || '';
  if (meaningsInput) {
    if (col.value_meanings && Object.keys(col.value_meanings).length > 0) {
      meaningsInput.value = Object.entries(col.value_meanings)
        .map(([k, v]) => `${k}: ${v}`)
        .join('\n');
    } else {
      meaningsInput.value = '';
    }
  }
  document.getElementById('columnEditDialog')?.showModal();
}

function openDeleteProjectModal(targetWsId) {
  const select = document.getElementById('deleteProjectSelect');
  const confirmInput = document.getElementById('deleteConfirmInput');
  const submitBtn = document.getElementById('btnConfirmProjectDelete');
  if (select) {
    const list = APP_STATE?.workspaces || [];
    select.innerHTML = list
      .map(
        (id) =>
          `<option value="${escapeHtml(id)}" ${(targetWsId || APP_STATE?.active_workspace_id) === id ? 'selected' : ''}>${escapeHtml(id)}</option>`
      )
      .join('');
  }
  if (confirmInput) confirmInput.value = '';
  if (submitBtn) submitBtn.disabled = true;
  document.getElementById('deleteProjectDialog')?.showModal();
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
// 8. VIEW: ASK DATA (CONVERSATIONAL BI, ROOT-CAUSE & CHART THREAD STUDIO)
// ============================================================================
const OPEN_THREAD_CARD_INDICES = new Set();
const CARD_THREAD_MODES = {};
let ACTIVE_SPLIT_THREAD_IDX = null;

function getCardThreadStarterPrompts(entry) {
  const cols = entry?.columns || [];
  const rows = entry?.rows || [];
  const dimCol = cols.find((c) => typeof rows?.[0]?.[c] === 'string') || cols[0] || 'segment';
  const seg1 = rows?.[0]?.[dimCol] ? String(rows[0][dimCol]).trim() : '';
  const seg2 = rows?.[1]?.[dimCol] ? String(rows[1][dimCol]).trim() : '';

  const starters = ['Summarize key takeaways from this chart'];
  if (seg1 && seg2 && seg1 !== seg2) {
    starters.push(`What is the difference between ${seg1} and ${seg2}?`);
  } else {
    starters.push('Compare top vs bottom segment');
  }
  starters.push('Which segments are above average?');
  if (seg1) {
    starters.push(`What percentage share does ${seg1} represent?`);
  } else {
    starters.push('Which segment is lowest and why?');
  }
  return starters;
}

function renderThreadTurnsHtml(threadList, cardIdx, prefix = 'inline') {
  if (!threadList || !threadList.length) {
    return `
      <div class="thread-empty-state">
        💬 Ask any question about this visual's numbers, compare specific bars/segments, or drill down further.
        By default, answers are returned in <strong>concise text</strong> unless you toggle <strong>With Chart</strong>.
      </div>
    `;
  }
  return threadList
    .map((turn, tIdx) => {
      const isSql = turn.source_mode === 'sql_query';
      const sourceBadge = isSql
        ? `<span class="meta-chip" style="background:rgba(16,185,129,0.14); color:#10b981; border:1px solid rgba(16,185,129,0.3); font-size:10px;">⚡ Warehouse SQL Drill-Down</span>`
        : `<span class="meta-chip" style="background:rgba(59,130,246,0.14); color:#3b82f6; border:1px solid rgba(59,130,246,0.3); font-size:10px;">📊 Direct Chart Analysis</span>`;
      const modeBadge = turn.include_chart
        ? `<span class="meta-chip" style="font-size:10px;">📊 Visual</span>`
        : `<span class="meta-chip" style="font-size:10px;">💬 Text Only</span>`;
      const subPlotHtml =
        turn.include_chart && turn.plotly_spec
          ? `<div id="plotly_thread_${prefix}_${cardIdx}_${tIdx}" class="thread-subchart-box"></div>`
          : '';
      return `
        <div class="thread-turn" data-turn-idx="${tIdx}">
          <div class="thread-q-row">
            <span>Q: ${escapeHtml(turn.question || '')}</span>
            <div style="display:flex; gap:4px; align-items:center;">
              ${sourceBadge}
              ${modeBadge}
            </div>
          </div>
          <div class="thread-a-body">${formatMarkdownBold(turn.answer || '')}</div>
          ${subPlotHtml}
        </div>
      `;
    })
    .join('');
}

function renderThreadSubCharts(threadList, cardIdx, prefix = 'inline') {
  if (!threadList || !threadList.length || !window.Plotly) return;
  threadList.forEach((turn, tIdx) => {
    if (!turn.include_chart || !turn.plotly_spec) return;
    const plotEl = document.getElementById(`plotly_thread_${prefix}_${cardIdx}_${tIdx}`);
    if (!plotEl) return;
    try {
      const themedLayout = applyPlotlyThemeLayout(turn.plotly_spec.layout);
      themedLayout.height = 240;
      if (themedLayout.title) themedLayout.title = { text: '' };
      window.Plotly.react(plotEl, turn.plotly_spec.data, themedLayout, {
        responsive: true,
        displayModeBar: false,
      });
    } catch (err) {
      console.warn('Thread subchart render failed:', err);
    }
  });
}

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
      const chartLabels = {
        kpi: 'KPI Summary Card',
        bar: 'Grouped Bar Chart',
        horizontal_bar: 'Horizontal Bar Chart',
        stacked_bar: 'Stacked Bar Chart',
        stacked_bar_100: '100% Stacked Bar Chart',
        stacked_horizontal_bar: 'Stacked Horizontal Bar',
        combo: 'Dual-Axis Combo (Bar + Line)',
        line: 'Line Trend Chart',
        area: 'Stacked Area Chart',
        donut: 'Donut Ring Chart',
        pie: 'Pie Chart',
        waterfall: 'Variance Waterfall Bridge',
        rate_mix_scatter: 'Rate vs. Mix Quadrant Scatter',
        dimension_divergence: 'Dimension Divergence Ranking',
        treemap: 'Hierarchical Treemap',
        sunburst: 'Interactive Sunburst',
        heatmap: '2D Cross-Tab Heatmap',
        radar: 'Radar / Spider Chart',
        box: 'Box & Whisker Distribution',
        funnel: 'Funnel Stage Pipeline',
        scatter: 'Correlation Scatter Plot',
        table: 'Data Catalog Table',
      };
      const baseSupported = APP_STATE.supported_chart_types || [
        'waterfall',
        'bar',
        'horizontal_bar',
        'rate_mix_scatter',
        'dimension_divergence',
        'stacked_bar',
        'stacked_bar_100',
        'stacked_horizontal_bar',
        'combo',
        'line',
        'area',
        'donut',
        'pie',
        'treemap',
        'sunburst',
        'heatmap',
        'radar',
        'box',
        'funnel',
        'scatter',
        'table',
      ];
      const chartTypes = isSingleRow ? ['kpi', 'table'] : [...baseSupported];
      if (entry.chart_type && !chartTypes.includes(entry.chart_type)) {
        chartTypes.unshift(entry.chart_type);
      }
      const chartOpts = chartTypes
        .map(
          (ct) =>
            `<option value="${escapeHtml(ct)}" ${ct === entry.chart_type ? 'selected' : ''}>${escapeHtml(chartLabels[ct] || ('Chart: ' + ct.replace(/_/g, ' ').toUpperCase()))}</option>`
        )
        .join('');

      const isDiagnostic = Boolean(resp.is_diagnostic || resp.audit_receipt);
      const isVerifiedMemory = resp.compiler_mode === 'verified_memory';
      const isLlmCompiled = resp.compiler_mode === 'llm';
      const compilerBadge = isDiagnostic
        ? `<span class="meta-chip" style="background:rgba(59,130,246,0.14); color:#3b82f6; border:1px solid rgba(59,130,246,0.35); font-weight:600;" title="${escapeHtml(resp.compiler_detail || 'Diagnostic Root-Cause Engine')}"><svg class="icon" style="width:11px;height:11px"><use href="#i-trend-down"/></svg> ${escapeHtml(resp.compiler_detail || 'Diagnostic Engine')}</span>`
        : isVerifiedMemory
        ? `<span class="pii-badge" style="background:rgba(14,165,233,0.16); color:#0284c7; border:1px solid rgba(14,165,233,0.4); font-weight:600;" title="${escapeHtml(resp.compiler_detail || 'Executed from Verified Query Memory')}"><svg class="icon" style="width:11px;height:11px"><use href="#i-zap"/></svg> Verified Memory</span>`
        : isLlmCompiled
        ? `<span class="pii-badge pii-safe" style="background:rgba(16,185,129,0.14); color:#10b981; border:1px solid rgba(16,185,129,0.35);" title="${escapeHtml(resp.compiler_detail || 'Compiled via LLM')}"><svg class="icon" style="width:11px;height:11px"><use href="#i-sparkles"/></svg> ${escapeHtml(resp.compiler_detail || 'LLM Compiler')}</span>`
        : `<span class="meta-chip" style="background:rgba(245,158,11,0.14); color:#d97706; border:1px solid rgba(245,158,11,0.35); font-weight:600;" title="${escapeHtml(resp.compiler_detail || 'Solved using built-in deterministic heuristic compiler')}"><svg class="icon" style="width:11px;height:11px"><use href="#i-cpu"/></svg> Heuristic Engine</span>`;

      const balanceBadge = (resp.audit_receipt?.balance_check?.status === 'VERIFIED_EXACT_BALANCE' || resp.audit_receipt?.balance_check?.status === 'EXACT_RECONCILIATION_VERIFIED')
        ? `<span class="pii-badge pii-safe" style="background:rgba(16,185,129,0.14); color:#10b981; border:1px solid rgba(16,185,129,0.35);" title="Zero residual mathematically verified"><svg class="icon" style="width:11px;height:11px"><use href="#i-check"/></svg> Verified Balance</span>`
        : '';

      const simpsonsBadge = resp.kpi_ribbon?.simpsons_paradox
        ? `<span class="meta-chip" style="background:rgba(239,68,68,0.15); color:#ef4444; border:1px solid rgba(239,68,68,0.35); font-weight:700;" title="Simpson's Paradox: Segment performance masked by mix shift"><svg class="icon" style="width:11px;height:11px"><use href="#i-alert-circle"/></svg> Simpson's Paradox</span>`
        : '';

      const fanTrapBadge = resp.fan_trap_mitigated
        ? `<span class="pii-badge pii-safe" title="Pre-aggregated CTEs prevented multi-fact fan trap"><svg class="icon" style="width:11px;height:11px"><use href="#i-shield"/></svg>Multi-fact CTE</span>`
        : '';

      const followUpBadge = resp.is_follow_up
        ? `<span class="meta-chip meta-chip-accent" title="${escapeHtml(resp.follow_up_context || 'Refined from prior context')}"><svg class="icon" style="width:11px;height:11px"><use href="#i-arrow-right"/></svg> Refined Follow-Up</span>`
        : '';

      const resolvedContextBanner =
        resp.follow_up_context || (resp.resolved_question && resp.resolved_question !== resp.user_question)
          ? `<div class="resolved-context-pill" title="${escapeHtml(resp.resolved_question || resp.follow_up_context || '')}">
               <svg class="icon" style="width:11px;height:11px;color:var(--accent-primary)"><use href="#i-sparkles"/></svg>
               <span>${escapeHtml(resp.follow_up_context || `Interpreted as: ${resp.resolved_question}`)}</span>
             </div>`
          : '';

      const cols = entry.columns || [];
      const firstDimCol = cols.find((c) => typeof (entry.rows?.[0]?.[c]) === 'string') || cols[0];
      const topSegmentVal =
        firstDimCol && (entry.rows || []).length >= 2 && typeof entry.rows[0]?.[firstDimCol] === 'string'
          ? String(entry.rows[0][firstDimCol]).trim()
          : '';

      const drillDownBtn = (resp.drill_down_dimensions || []).length
        ? `
          <div class="drilldown-dropdown-wrap">
            <button type="button" class="btn btn-secondary btn-xs btn-drilldown-toggle" data-idx="${idx}" title="Drill down into this result">
              <svg class="icon" style="width:13px;height:13px"><use href="#i-cpu"/></svg> Drill Down
            </button>
            <div class="drilldown-menu" id="drilldown_menu_${idx}">
              <div class="drilldown-section-label">Break All Down By</div>
              ${resp.drill_down_dimensions
                .map(
                  (d) => `
                <button type="button" class="drilldown-opt-btn" data-idx="${idx}" data-mode="breakdown" data-dim="${escapeHtml(d.dimension)}" data-label="${escapeHtml(d.label)}">
                  <svg class="icon" style="width:12px;height:12px;color:var(--accent-primary)"><use href="#i-arrow-right"/></svg>
                  <span>All by <strong>${escapeHtml(d.label)}</strong></span>
                </button>
              `
                )
                .join('')}
              ${
                topSegmentVal
                  ? `
                <div class="drilldown-section-label" style="margin-top:4px; border-top:1px solid var(--border-subtle, rgba(255,255,255,0.08));">Drill Into Top (${escapeHtml(topSegmentVal.slice(0, 22))}${topSegmentVal.length > 22 ? '…' : ''})</div>
                ${resp.drill_down_dimensions
                  .slice(0, 3)
                  .map(
                    (d) => `
                  <button type="button" class="drilldown-opt-btn" data-idx="${idx}" data-mode="segment" data-segment="${escapeHtml(topSegmentVal)}" data-dim="${escapeHtml(d.dimension)}" data-label="${escapeHtml(d.label)}">
                    <svg class="icon" style="width:12px;height:12px;color:var(--accent-emerald, #10b981)"><use href="#i-sparkles"/></svg>
                    <span><strong>${escapeHtml(topSegmentVal.slice(0, 18))}${topSegmentVal.length > 18 ? '…' : ''}</strong> by ${escapeHtml(d.label)}</span>
                  </button>
                `
                  )
                  .join('')}
              `
                  : ''
              }
            </div>
          </div>
        `
        : '';

      const tablePreviewRows = (entry.rows || []).slice(0, 10);
      const totalRowsCount = entry.row_count || (entry.rows || []).length;
      const isTableChart = entry.chart_type === 'table' || !entry.plotly_spec || Boolean(entry.hide_chart);
      const chartBoxHtml = isTableChart
        ? `<div id="plotly_chat_${idx}" class="chart-box" style="display:none;"></div>`
        : `<div id="plotly_chat_${idx}" class="chart-box"></div>`;

      const threadList = entry.thread || [];
      const isThreadOpen = OPEN_THREAD_CARD_INDICES.has(idx);
      const cardThreadMode = CARD_THREAD_MODES[idx] || 'text_only';
      const threadStarters = getCardThreadStarterPrompts(entry);
      const inlineThreadPanelHtml = `
        <div class="chart-thread-panel ${isThreadOpen ? '' : 'hidden'}" id="chart_thread_panel_${idx}">
          <div class="chart-thread-header">
            <div class="chart-thread-title">
              <svg class="icon" style="width:14px;height:14px;color:var(--accent-primary)"><use href="#i-chat"/></svg>
              <span>Chart Discussion Thread (${threadList.length})</span>
            </div>
            <div style="display:flex; gap:6px; align-items:center;">
              <button type="button" class="btn btn-secondary btn-xs btn-open-tab-thread" data-idx="${idx}" title="Open chart & thread in full-screen Studio tab or window">
                <svg class="icon" style="width:12px;height:12px"><use href="#i-external-link"/></svg> Open in New Tab
              </button>
              <button type="button" class="btn btn-secondary btn-xs btn-split-thread" data-idx="${idx}" title="Pop out side-by-side Split View">
                ⤢ Split View
              </button>
              <button type="button" class="btn btn-ghost btn-xs btn-thread-close" data-idx="${idx}" title="Collapse thread">✕</button>
            </div>
          </div>
          <div class="thread-starter-chips">
            ${threadStarters
              .map(
                (sp) =>
                  `<button type="button" class="thread-chip inline-thread-chip" data-idx="${idx}" data-prompt="${escapeHtml(sp)}">💡 ${escapeHtml(sp)}</button>`
              )
              .join('')}
          </div>
          <div class="chart-thread-stream" id="chart_thread_stream_${idx}">
            ${renderThreadTurnsHtml(threadList, idx, 'inline')}
          </div>
          <form class="chart-thread-composer inline-thread-form" data-idx="${idx}">
            <div class="thread-composer-top">
              <div class="response-mode-group" role="radiogroup" aria-label="Thread answer format">
                <span style="font-size:11px; color:var(--text-muted); font-weight:600;">Answer:</span>
                <button type="button" class="mode-pill inline-thread-mode-pill ${cardThreadMode === 'text_only' ? 'active' : ''}" data-idx="${idx}" data-mode="text_only">💬 Text Only</button>
                <button type="button" class="mode-pill inline-thread-mode-pill ${cardThreadMode === 'auto' ? 'active' : ''}" data-idx="${idx}" data-mode="auto">✨ Auto</button>
                <button type="button" class="mode-pill inline-thread-mode-pill ${cardThreadMode === 'with_chart' ? 'active' : ''}" data-idx="${idx}" data-mode="with_chart">📊 With Chart</button>
              </div>
            </div>
            <div class="thread-composer-input-row">
              <input
                type="text"
                class="input-text inline-thread-input"
                id="inline_thread_input_${idx}"
                placeholder="Ask a question about this chart (e.g. 'Compare top 2', 'Which are above average?')…"
                autocomplete="off"
                required
              />
              <button type="submit" class="btn btn-primary btn-xs" id="inline_thread_send_${idx}">
                <svg class="icon" style="width:13px;height:13px"><use href="#i-send"/></svg> Ask
              </button>
            </div>
          </form>
        </div>
      `;

      const miniTableHtml =
        cols.length && tablePreviewRows.length
          ? `
          <div class="result-table-wrapper" style="margin-top:14px;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px; gap:8px; flex-wrap:wrap;">
              <span style="font-size:12px; font-weight:600; color:var(--text-secondary); display:flex; align-items:center; gap:6px;">
                <svg class="icon" style="width:14px;height:14px"><use href="#i-table"/></svg>
                ${isTableChart ? 'Schema & Table Catalog' : 'Result Table'} (<span class="table-total-count">${totalRowsCount}</span> ${totalRowsCount === 1 ? 'row' : 'rows'})
              </span>
              <div style="display:flex; align-items:center; gap:8px;">
                <input type="text" class="table-filter-input" data-idx="${idx}" placeholder="🔍 Search table..." />
                <span class="table-showing-count" style="font-size:11px; color:var(--text-muted)">Showing ${tablePreviewRows.length} ${tablePreviewRows.length === 1 ? 'row' : 'rows'}</span>
              </div>
            </div>
            <div class="table-responsive" style="max-height:320px; border:1px solid var(--border-color); border-radius:var(--radius-md); overflow:auto;">
              <table class="data-table" data-idx="${idx}">
                <thead>
                  <tr>${cols.map((c) => `<th>${escapeHtml(c.replace(/_/g, ' ').toUpperCase())}</th>`).join('')}</tr>
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
          </div>
        `
          : '';

      const suggestionsHtml = (resp.suggested_follow_ups || []).length
        ? `
          <div class="chat-suggestions-box">
            <div style="font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:0.5px; color:var(--text-muted); display:flex; align-items:center; gap:6px; margin-bottom:8px;">
              <svg class="icon" style="width:13px;height:13px;color:var(--primary-color)"><use href="#i-sparkles"/></svg>
              Suggested Follow-ups & Deep Analysis
            </div>
            <div style="display:flex; gap:8px; flex-wrap:wrap;">
              ${resp.suggested_follow_ups
                .map(
                  (q) => `
                <button type="button" class="btn-suggestion-chip" data-query="${escapeHtml(q)}">
                  <svg class="icon" style="width:11px;height:11px;color:var(--primary-color)"><use href="#i-arrow-right"/></svg>
                  <span>${escapeHtml(q)}</span>
                </button>
              `
                )
                .join('')}
            </div>
          </div>
        `
        : '';

      let kpiRibbonHtml = '';
      if (resp.kpi_ribbon && (resp.kpi_ribbon.prior_total !== undefined || resp.kpi_ribbon.top_dimension)) {
        const kpi = resp.kpi_ribbon;
        const deltaColor = (kpi.net_delta || 0) >= 0 ? '#10B981' : '#EF4444';
        const simpsonsAlert = kpi.simpsons_paradox
          ? `<div class="badge badge-error" style="grid-column: 1 / -1; justify-content: center; font-size: 12px; padding: 6px; margin-bottom: 2px;">🚨 Simpson's Paradox Detected: Segment Performance Masked by Cohort Mix Shift</div>`
          : '';
        kpiRibbonHtml = `
          <div class="rc-kpi-ribbon" style="margin: 10px 0;">
            ${simpsonsAlert}
            <div class="rc-kpi-card">
              <div class="rc-kpi-label">Prior Baseline</div>
              <div class="rc-kpi-val">${typeof kpi.prior_total === 'number' ? kpi.prior_total.toLocaleString() : (kpi.prior_total ?? 'N/A')}</div>
            </div>
            <div class="rc-kpi-card">
              <div class="rc-kpi-label">Current Period</div>
              <div class="rc-kpi-val">${typeof kpi.curr_total === 'number' ? kpi.curr_total.toLocaleString() : (kpi.curr_total ?? 'N/A')}</div>
            </div>
            <div class="rc-kpi-card">
              <div class="rc-kpi-label">Net Delta</div>
              <div class="rc-kpi-val" style="color: ${deltaColor};">${(kpi.net_delta || 0) > 0 ? '+' : ''}${typeof kpi.net_delta === 'number' ? kpi.net_delta.toLocaleString() : (kpi.net_delta ?? 'N/A')}${kpi.pct_delta !== undefined ? ` (${kpi.pct_delta > 0 ? '+' : ''}${kpi.pct_delta}%)` : ''}</div>
            </div>
            <div class="rc-kpi-card">
              <div class="rc-kpi-label">${kpi.top_dimension ? 'Top Dimension' : 'Top Driver'}</div>
              <div class="rc-kpi-val" style="font-size: 13px; color: var(--accent-primary);">${escapeHtml(kpi.top_dimension || kpi.top_driver || 'N/A')}</div>
            </div>
          </div>
        `;
      }

      let auditDrawerHtml = '';
      if (resp.audit_receipt && (resp.audit_receipt.executed_sql || resp.audit_receipt.formula_latex)) {
        const audit = resp.audit_receipt;
        const balanceBadgeStatus = (audit.balance_check?.status === 'VERIFIED_EXACT_BALANCE' || audit.balance_check?.status === 'EXACT_RECONCILIATION_VERIFIED')
          ? '<span class="rc-rec-badge">✓ Exact 0.00 Balance Verified</span>'
          : (audit.balance_check?.status ? '<span class="badge badge-warning">Rounding Diff</span>' : '');

        auditDrawerHtml = `
          <details class="rc-audit-details" style="margin-top: 10px;">
            <summary class="rc-audit-summary">
              <span>🔍 How this was calculated: Audit Receipts, SQL &amp; Integrity Proof</span>
              ${balanceBadgeStatus}
            </summary>
            <div class="rc-audit-body">
              <div style="font-size: 12px; margin-bottom: 8px;">
                <strong>Algorithm:</strong> ${escapeHtml(audit.algorithm || 'Diagnostic Engine')} &middot;
                <strong>Evaluated Rows:</strong> ${audit.rows_evaluated || 0} &middot;
                <strong>Date Lineage:</strong> ${escapeHtml(audit.date_source || 'Default Anchor')}
              </div>
              ${audit.formula_latex ? `<div style="font-size: 12px; font-family: monospace; background: var(--bg-surface-elevated, rgba(255,255,255,0.03)); padding: 8px; border-radius: 4px; margin-bottom: 8px;">📐 Formula: ${escapeHtml(audit.formula_latex)}</div>` : ''}
              ${audit.join_lineage ? `<div style="font-size: 11px; color: var(--text-muted); margin-bottom: 6px;">🔗 Join Lineage: ${escapeHtml(audit.join_lineage)}</div>` : ''}
            </div>
          </details>
        `;
      }

      const tablesUsed = (resp.involved_tables || []).length ? resp.involved_tables : [];

      return `
        <article class="chat-card ${isDiagnostic ? 'chat-card-diagnostic' : ''}" data-entry-idx="${idx}">
          <div class="chat-question-header">
            <div class="chat-q-left" style="flex-direction:column; align-items:flex-start; gap:2px;">
              <div style="display:flex; align-items:center; gap:8px;">
                <span class="chat-q-avatar" style="${isDiagnostic ? 'background:rgba(59,130,246,0.15); color:var(--accent-primary);' : ''}"><svg class="icon" style="width:14px;height:14px"><use href="${isDiagnostic ? '#i-trend-down' : '#i-chat'}"/></svg></span>
                <h3 class="chat-question-title">${escapeHtml(resp.user_question || '')}</h3>
              </div>
              ${resolvedContextBanner}
            </div>
            <div class="chat-q-actions">
              ${compilerBadge}
              ${balanceBadge}
              ${simpsonsBadge}
              ${fanTrapBadge}
              ${followUpBadge}
              ${drillDownBtn}
              <button type="button" class="btn btn-secondary btn-xs btn-thread-toggle ${threadList.length ? 'has-messages' : ''}" data-idx="${idx}" title="Ask questions about this chart inside a scoped thread">
                💬 ${threadList.length ? `Thread (${threadList.length})` : 'Ask Chart'}
              </button>
              <button type="button" class="btn btn-secondary btn-xs btn-open-tab-thread" data-idx="${idx}" title="Open chart and thread in a dedicated full-screen Studio tab or window">
                <svg class="icon" style="width:12px;height:12px"><use href="#i-external-link"/></svg> Open in New Tab
              </button>
              <button type="button" class="btn btn-secondary btn-xs btn-split-thread" data-idx="${idx}" title="Open Chart + Thread in Split-View Studio modal">
                ⤢ Split View
              </button>
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
              <button type="button" class="btn btn-secondary btn-xs btn-save-verified-query" data-idx="${idx}" title="Save as verified query memory">
                <svg class="icon" style="width:13px;height:13px"><use href="#i-zap"/></svg> Remember
              </button>
              ${isVerifiedMemory ? `
              <button type="button" class="btn btn-ghost btn-xs btn-recompile-ai" data-idx="${idx}" title="Re-synthesize query using AI model" style="color:var(--accent-primary);">
                <svg class="icon" style="width:12px;height:12px"><use href="#i-sparkles"/></svg> Re-run with AI
              </button>` : ''}
            </div>
          </div>

          ${kpiRibbonHtml}

          <div class="chat-summary-text">${formatMarkdownBold(resp.executive_summary || '')}</div>

          <div class="chat-toolbar">
            <div class="chat-toolbar-left" style="display:flex; align-items:center; gap:8px; flex-wrap:wrap;">
              <select class="input-select input-sm chart-type-switcher" data-idx="${idx}" aria-label="Chart type" style="width:190px;">
                ${chartOpts}
              </select>
              ${
                entry.hide_chart
                  ? `<button type="button" class="btn btn-secondary btn-xs btn-show-hidden-chart" data-idx="${idx}">📊 Show Visual Chart</button>`
                  : ''
              }
            </div>
            <div style="display:flex; gap:6px; align-items:center; flex-wrap:wrap;">
              ${tablesUsed.map((t) => `<span class="meta-chip">${escapeHtml(t)}</span>`).join('')}
            </div>
          </div>

          ${chartBoxHtml}
          ${inlineThreadPanelHtml}
          ${miniTableHtml}
          ${auditDrawerHtml}
          ${suggestionsHtml}
        </article>
      `;
    })
    .join('');

  // Render Plotly charts with theme-aware layout
  history.forEach((entry, idx) => {
    renderThreadSubCharts(entry.thread || [], idx, 'inline');
    const plotEl = document.getElementById(`plotly_chat_${idx}`);
    if (plotEl && entry.hide_chart) {
      plotEl.style.display = 'none';
    } else if (plotEl && entry.plotly_spec && window.Plotly) {
      try {
        const themedLayout = applyPlotlyThemeLayout(entry.plotly_spec.layout);
        // Remove duplicate inner title that collides with horizontal legend since card header displays it
        if (themedLayout.title) {
          themedLayout.title = { text: '' };
        }
        window.Plotly.react(plotEl, entry.plotly_spec.data, themedLayout, {
          responsive: true,
          displayModeBar: false,
        });

        if (typeof plotEl.on === 'function') {
          plotEl.removeAllListeners?.('plotly_click');
          plotEl.on('plotly_click', (data) => {
            if (!data || !data.points || !data.points.length) return;
            const pt = data.points[0];
            const clickedVal = (pt.x !== undefined && typeof pt.x === 'string') ? pt.x : (pt.label || (typeof pt.y === 'string' ? pt.y : ''));
            if (!clickedVal) return;

            let bannerEl = document.getElementById(`plotly_click_banner_${idx}`);
            if (!bannerEl) {
              bannerEl = document.createElement('div');
              bannerEl.id = `plotly_click_banner_${idx}`;
              bannerEl.className = 'plotly-click-banner';
              plotEl.insertAdjacentElement('beforebegin', bannerEl);
            }
            const drillDims = (entry.response && entry.response.drill_down_dimensions) || [];
            const drillButtonsHtml = (drillDims.length ? drillDims.slice(0, 4) : [{ label: 'Subcategory' }])
              .map(
                (d, dIdx) => `
                <button type="button" class="btn ${dIdx === 0 ? 'btn-primary' : 'btn-secondary'} btn-xs btn-click-drill-dim" data-dim-label="${escapeHtml(d.label)}" style="border-radius:14px;">
                  🔍 By ${escapeHtml(d.label)}
                </button>
              `
              )
              .join('');
            bannerEl.style.display = 'flex';
            bannerEl.innerHTML = `
              <span>📌 Selected: <strong>${escapeHtml(clickedVal)}</strong></span>
              <div style="display:flex; gap:6px; align-items:center; flex-wrap:wrap;">
                ${drillButtonsHtml}
                <button type="button" class="btn btn-secondary btn-xs btn-click-trend" style="border-radius:14px;">
                  📈 Trend Over Time
                </button>
                <button type="button" class="btn btn-ghost btn-xs" onclick="this.closest('.plotly-click-banner').style.display='none'">✕</button>
              </div>
            `;
            bannerEl.querySelectorAll('.btn-click-drill-dim').forEach((drillBtn) => {
              drillBtn.addEventListener('click', () => {
                const targetLabel = drillBtn.dataset.dimLabel || 'Subcategory';
                const q = `Drill down into ${clickedVal} by ${targetLabel}`;
                const inputEl = document.getElementById('chatQuestionInput');
                if (inputEl) inputEl.value = q;
                submitChatQuestion(q);
              });
            });
            bannerEl.querySelector('.btn-click-trend')?.addEventListener('click', () => {
              const q = `Monthly trend for ${clickedVal}`;
              const inputEl = document.getElementById('chatQuestionInput');
              if (inputEl) inputEl.value = q;
              submitChatQuestion(q);
            });
          });
        }
      } catch (plotErr) {
        console.warn(`Plotly render failed for chat #${idx}:`, plotErr);
        plotEl.innerHTML = `<div style="padding:12px; color:var(--text-muted); font-size:12px;">Visual chart temporarily unavailable.</div>`;
      }
    } else if (plotEl && entry.chart_type === 'table') {
      plotEl.style.display = 'none';
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
            title: entry.response.resolved_question || entry.response.user_question,
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

  stream.querySelectorAll('.btn-save-verified-query').forEach((btn) => {
    btn.addEventListener('click', () => {
      const idx = Number(btn.dataset.idx);
      const entry = history[idx];
      const resp = entry.response;
      openEditVerifiedQueryModal({
        id: '',
        question_pattern: resp.user_question || resp.resolved_question,
        executed_query: resp.executed_query,
        dax_equivalent: resp.compiled_dax_equivalent || '',
        chart_type: entry.chart_type || 'bar',
        query_dialect: resp.query_dialect || 'duckdb',
      });
    });
  });

  stream.querySelectorAll('.btn-recompile-ai').forEach((btn) => {
    btn.addEventListener('click', () => {
      const idx = Number(btn.dataset.idx);
      const entry = history[idx];
      const q = entry.response.user_question || entry.response.resolved_question;
      if (q) {
        const inputEl = document.getElementById('chatQuestionInput');
        if (inputEl) inputEl.value = q;
        submitChatQuestion(q, { force_recompile: true });
      }
    });
  });

  // 1. Suggestion chips 1-click execution
  stream.querySelectorAll('.btn-suggestion-chip').forEach((btn) => {
    btn.addEventListener('click', (e) => {
      const q = e.currentTarget.dataset.query;
      if (q) {
        const inputEl = document.getElementById('chatQuestionInput');
        if (inputEl) inputEl.value = q;
        submitChatQuestion(q);
      }
    });
  });

  // 2. Interactive drill-down dropdown toggles & option selection
  stream.querySelectorAll('.btn-drilldown-toggle').forEach((btn) => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      const idx = btn.dataset.idx;
      const menu = document.getElementById(`drilldown_menu_${idx}`);
      document.querySelectorAll('.drilldown-menu').forEach((m) => {
        if (m !== menu) m.classList.remove('open');
      });
      if (menu) menu.classList.toggle('open');
    });
  });

  stream.querySelectorAll('.drilldown-opt-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      const dimLabel = btn.dataset.label;
      const mode = btn.dataset.mode || 'breakdown';
      const segment = btn.dataset.segment || '';
      const menu = btn.closest('.drilldown-menu');
      if (menu) menu.classList.remove('open');
      const drillQuery =
        mode === 'segment' && segment
          ? `Drill down into ${segment} by ${dimLabel}`
          : `Break that down by ${dimLabel}`;
      const inputEl = document.getElementById('chatQuestionInput');
      if (inputEl) inputEl.value = drillQuery;
      submitChatQuestion(drillQuery);
    });
  });

  // 3. Real-time client-side table filter
  stream.querySelectorAll('.table-filter-input').forEach((input) => {
    input.addEventListener('input', (e) => {
      const searchVal = e.target.value.toLowerCase().trim();
      const card = input.closest('.chat-card');
      if (!card) return;
      const rows = card.querySelectorAll('table.data-table tbody tr');
      let visibleCount = 0;
      rows.forEach((tr) => {
        const text = tr.innerText.toLowerCase();
        if (!searchVal || text.includes(searchVal)) {
          tr.style.display = '';
          visibleCount++;
        } else {
          tr.style.display = 'none';
        }
      });
      const showingEl = card.querySelector('.table-showing-count');
      if (showingEl) {
        showingEl.textContent = `Showing ${visibleCount} ${visibleCount === 1 ? 'row' : 'rows'}`;
      }
    });
  });

  // 4. Inline Chart Thread Toggles, Mode Pills, Starter Chips & Form Submit
  stream.querySelectorAll('.btn-thread-toggle').forEach((btn) => {
    btn.addEventListener('click', () => {
      const idx = Number(btn.dataset.idx);
      const panel = document.getElementById(`chart_thread_panel_${idx}`);
      if (!panel) return;
      const isHidden = panel.classList.contains('hidden');
      if (isHidden) {
        OPEN_THREAD_CARD_INDICES.add(idx);
        panel.classList.remove('hidden');
        const inputEl = document.getElementById(`inline_thread_input_${idx}`);
        if (inputEl) inputEl.focus();
      } else {
        OPEN_THREAD_CARD_INDICES.delete(idx);
        panel.classList.add('hidden');
      }
    });
  });

  stream.querySelectorAll('.btn-thread-close').forEach((btn) => {
    btn.addEventListener('click', () => {
      const idx = Number(btn.dataset.idx);
      OPEN_THREAD_CARD_INDICES.delete(idx);
      const panel = document.getElementById(`chart_thread_panel_${idx}`);
      if (panel) panel.classList.add('hidden');
    });
  });

  stream.querySelectorAll('.btn-open-tab-thread').forEach((btn) => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      const idx = Number(btn.dataset.idx);
      window.open(`/thread-studio?idx=${idx}`, '_blank');
    });
  });

  stream.querySelectorAll('.btn-split-thread').forEach((btn) => {
    btn.addEventListener('click', () => {
      const idx = Number(btn.dataset.idx);
      openSplitThreadModal(idx);
    });
  });

  stream.querySelectorAll('.btn-show-hidden-chart').forEach((btn) => {
    btn.addEventListener('click', async () => {
      const idx = Number(btn.dataset.idx);
      const entry = history[idx];
      if (!entry) return;
      entry.hide_chart = false;
      if (!entry.plotly_spec || entry.chart_type === 'none') {
        try {
          APP_STATE = await apiRequest('/api/chat/chart', {
            method: 'POST',
            body: JSON.stringify({ entry_index: idx, chart_type: 'bar' }),
          });
          if (APP_STATE.chat_history?.[idx]) {
            APP_STATE.chat_history[idx].hide_chart = false;
          }
        } catch (err) {
          showToast(`Chart error: ${err.message}`);
        }
      }
      renderChatView();
    });
  });

  stream.querySelectorAll('.inline-thread-mode-pill').forEach((pill) => {
    pill.addEventListener('click', () => {
      const idx = Number(pill.dataset.idx);
      const mode = pill.dataset.mode || 'text_only';
      CARD_THREAD_MODES[idx] = mode;
      const group = pill.closest('.response-mode-group');
      if (group) {
        group.querySelectorAll('.inline-thread-mode-pill').forEach((p) => {
          p.classList.toggle('active', p === pill);
        });
      }
    });
  });

  stream.querySelectorAll('.inline-thread-chip').forEach((chip) => {
    chip.addEventListener('click', async () => {
      const idx = Number(chip.dataset.idx);
      const promptText = chip.dataset.prompt || '';
      if (!promptText) return;
      const mode = CARD_THREAD_MODES[idx] || 'text_only';
      await submitChartThreadQuestion(idx, promptText, mode);
    });
  });

  stream.querySelectorAll('.inline-thread-form').forEach((form) => {
    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      const idx = Number(form.dataset.idx);
      const inputEl = document.getElementById(`inline_thread_input_${idx}`);
      const q = (inputEl?.value || '').trim();
      if (!q) return;
      if (inputEl) inputEl.value = '';
      const mode = CARD_THREAD_MODES[idx] || 'text_only';
      await submitChartThreadQuestion(idx, q, mode);
    });
  });
}

async function submitChartThreadQuestion(entryIdx, questionText, responseMode = 'text_only') {
  const trimmed = (questionText || '').trim();
  if (!trimmed) return;

  OPEN_THREAD_CARD_INDICES.add(entryIdx);
  const inlineSendBtn = document.getElementById(`inline_thread_send_${entryIdx}`);
  const splitSendBtn = document.getElementById('splitThreadSendBtn');
  if (inlineSendBtn) inlineSendBtn.disabled = true;
  if (splitSendBtn) splitSendBtn.disabled = true;

  // Optimistic loading indicator in active thread stream
  const inlineStream = document.getElementById(`chart_thread_stream_${entryIdx}`);
  if (inlineStream) {
    const emptyEl = inlineStream.querySelector('.thread-empty-hint');
    if (emptyEl) emptyEl.remove();
    const loadingTurn = document.createElement('div');
    loadingTurn.className = 'thread-turn thread-turn-loading';
    loadingTurn.innerHTML = `
      <div class="thread-q-bubble">
        <span class="thread-q-label">You</span>
        <div class="thread-q-text">${escapeHtml(trimmed)}</div>
      </div>
      <div class="thread-a-bubble">
        <span class="meta-chip meta-chip-accent">Analyzing chart data…</span>
      </div>
    `;
    inlineStream.appendChild(loadingTurn);
    inlineStream.scrollTop = inlineStream.scrollHeight;
  }

  if (ACTIVE_SPLIT_THREAD_IDX === entryIdx) {
    const splitStream = document.getElementById('splitThreadMessages');
    if (splitStream) {
      const emptyEl = splitStream.querySelector('.thread-empty-hint');
      if (emptyEl) emptyEl.remove();
      const loadingTurn = document.createElement('div');
      loadingTurn.className = 'thread-turn thread-turn-loading';
      loadingTurn.innerHTML = `
        <div class="thread-q-bubble">
          <span class="thread-q-label">You</span>
          <div class="thread-q-text">${escapeHtml(trimmed)}</div>
        </div>
        <div class="thread-a-bubble">
          <span class="meta-chip meta-chip-accent">Analyzing chart data…</span>
        </div>
      `;
      splitStream.appendChild(loadingTurn);
      splitStream.scrollTop = splitStream.scrollHeight;
    }
  }

  try {
    const res = await apiRequest('/api/chat/thread', {
      method: 'POST',
      body: JSON.stringify({
        entry_index: entryIdx,
        question: trimmed,
        response_mode: responseMode,
      }),
    });
    APP_STATE = res.state;
    renderChatView();
    const updatedInlineStream = document.getElementById(`chart_thread_stream_${entryIdx}`);
    if (updatedInlineStream) {
      updatedInlineStream.scrollTop = updatedInlineStream.scrollHeight;
    }
    if (ACTIVE_SPLIT_THREAD_IDX === entryIdx) {
      renderSplitThreadModalContent(entryIdx);
    }
  } catch (err) {
    showToast(`Thread error: ${err.message}`);
    renderChatView();
    if (ACTIVE_SPLIT_THREAD_IDX === entryIdx) {
      renderSplitThreadModalContent(entryIdx);
    }
  } finally {
    if (inlineSendBtn) inlineSendBtn.disabled = false;
    if (splitSendBtn) splitSendBtn.disabled = false;
  }
}

function openSplitThreadModal(entryIdx) {
  ACTIVE_SPLIT_THREAD_IDX = entryIdx;
  const dialog = document.getElementById('chartThreadDialog');
  if (!dialog) return;
  if (!dialog.open) {
    dialog.showModal();
  }
  renderSplitThreadModalContent(entryIdx);
  const inputEl = document.getElementById('splitThreadInput');
  if (inputEl) inputEl.focus();
}

function renderSplitThreadModalContent(entryIdx) {
  const history = APP_STATE?.chat_history || [];
  const entry = history[entryIdx];
  if (!entry) return;

  const resp = entry.response || {};
  const titleEl = document.getElementById('splitThreadModalTitle');
  if (titleEl) {
    titleEl.textContent = resp.user_question || 'Chart Deep-Dive Thread';
  }

  const summaryEl = document.getElementById('splitThreadSummary');
  if (summaryEl) {
    summaryEl.innerHTML = formatMarkdownBold(resp.executive_summary || '');
  }

  const plotBox = document.getElementById('splitThreadPlotBox');
  if (plotBox) {
    if (entry.plotly_spec && window.Plotly) {
      plotBox.style.display = 'block';
      try {
        const themedLayout = applyPlotlyThemeLayout(entry.plotly_spec.layout);
        if (themedLayout.title) themedLayout.title = { text: '' };
        window.Plotly.react(plotBox, entry.plotly_spec.data, themedLayout, {
          responsive: true,
          displayModeBar: false,
        });
      } catch (err) {
        console.warn('Split modal Plotly error:', err);
      }
    } else {
      plotBox.style.display = 'none';
    }
  }

  const tableWrap = document.getElementById('splitThreadTableWrap');
  if (tableWrap) {
    const cols = entry.columns || [];
    const rows = (entry.rows || []).slice(0, 12);
    if (cols.length && rows.length) {
      tableWrap.innerHTML = `
        <div style="font-size:11.5px; font-weight:600; color:var(--text-secondary); margin-bottom:6px;">
          Underlying Data (${entry.row_count || entry.rows.length} rows)
        </div>
        <div class="table-responsive" style="max-height:220px; border:1px solid var(--border-color); border-radius:var(--radius-md); overflow:auto;">
          <table class="data-table">
            <thead>
              <tr>${cols.map((c) => `<th>${escapeHtml(c.replace(/_/g, ' ').toUpperCase())}</th>`).join('')}</tr>
            </thead>
            <tbody>
              ${rows
                .map(
                  (r) =>
                    `<tr>${cols.map((c) => `<td class="mono">${escapeHtml(formatCellVal(r[c]))}</td>`).join('')}</tr>`
                )
                .join('')}
            </tbody>
          </table>
        </div>
      `;
    } else {
      tableWrap.innerHTML = '';
    }
  }

  // Starter chips in Split View
  const chipsContainer = document.getElementById('splitThreadStarterChips');
  if (chipsContainer) {
    const starters = getCardThreadStarterPrompts(entry);
    chipsContainer.innerHTML = starters
      .map(
        (sp) =>
          `<button type="button" class="thread-chip split-starter-chip" data-prompt="${escapeHtml(sp)}">💡 ${escapeHtml(sp)}</button>`
      )
      .join('');
    chipsContainer.querySelectorAll('.split-starter-chip').forEach((chip) => {
      chip.addEventListener('click', async () => {
        const promptText = chip.dataset.prompt || '';
        const mode = document.getElementById('splitThreadResponseMode')?.value || 'text_only';
        await submitChartThreadQuestion(entryIdx, promptText, mode);
      });
    });
  }

  // Thread messages in Split View
  const msgContainer = document.getElementById('splitThreadMessages');
  if (msgContainer) {
    const threadList = entry.thread || [];
    msgContainer.innerHTML = renderThreadTurnsHtml(threadList, entryIdx, 'split');
    renderThreadSubCharts(threadList, entryIdx, 'split');
    msgContainer.scrollTop = msgContainer.scrollHeight;
  }
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
  if (resp.is_diagnostic || resp.audit_receipt) {
    badges.push(
      `<span class="meta-chip" style="background:rgba(59,130,246,0.14); color:#3b82f6; border:1px solid rgba(59,130,246,0.35); font-weight:600;"><svg class="icon" style="width:11px;height:11px"><use href="#i-trend-down"/></svg> ${escapeHtml(resp.compiler_detail || 'Diagnostic Engine')}</span>`
    );
    if (resp.audit_receipt?.balance_check?.status === 'VERIFIED_EXACT_BALANCE' || resp.audit_receipt?.balance_check?.status === 'EXACT_RECONCILIATION_VERIFIED') {
      badges.push(
        `<span class="pii-badge pii-safe" style="background:rgba(16,185,129,0.14); color:#10b981; border:1px solid rgba(16,185,129,0.35);"><svg class="icon" style="width:11px;height:11px"><use href="#i-check"/></svg>Verified 0.00 Balance</span>`
      );
    }
  } else if (resp.compiler_mode === 'llm') {
    badges.push(
      `<span class="pii-badge pii-safe" style="background:rgba(16,185,129,0.14); color:#10b981; border:1px solid rgba(16,185,129,0.35);"><svg class="icon" style="width:11px;height:11px"><use href="#i-sparkles"/></svg> ${escapeHtml(resp.compiler_detail || 'LLM Compiler')}</span>`
    );
  } else {
    badges.push(
      `<span class="meta-chip" style="background:rgba(245,158,11,0.14); color:#d97706; border:1px solid rgba(245,158,11,0.35); font-weight:600;"><svg class="icon" style="width:11px;height:11px"><use href="#i-cpu"/></svg> ${escapeHtml(resp.compiler_detail || 'Deterministic Heuristic Engine')}</span>`
    );
  }
  if (resp.fan_trap_mitigated) {
    badges.push(
      `<span class="pii-badge pii-safe"><svg class="icon" style="width:11px;height:11px"><use href="#i-shield"/></svg>Multi-fact chasm/fan trap mitigated via pre-aggregated CTEs</span>`
    );
  }
  badges.push(`<span class="meta-chip">Attempts: ${resp.attempts_used || 1}</span>`);
  document.getElementById('lineageBadgesRow').innerHTML = badges.join(' ');

  if (resp.audit_receipt) {
    const audit = resp.audit_receipt;
    document.getElementById('lineageMetaDetails').innerHTML = `
      <div><strong>Diagnostic Algorithm:</strong> <code class="mono">${escapeHtml(audit.algorithm || resp.compiler_detail || 'Diagnostic')}</code></div>
      <div style="margin-top:4px;"><strong>Tables evaluated:</strong> <code class="mono">${escapeHtml((resp.involved_tables || []).join(', ') || 'Connected fact/dim tables')}</code></div>
      <div style="margin-top:4px;"><strong>Temporal Lineage:</strong> <code class="mono">${escapeHtml(audit.date_source || 'Default Anchor')}</code></div>
      ${audit.join_lineage ? `<div style="margin-top:4px;"><strong>Join Traversal:</strong> <code class="mono">${escapeHtml(audit.join_lineage)}</code></div>` : ''}
      ${audit.balance_check?.status ? `<div style="margin-top:4px;"><strong>Reconciliation Status:</strong> <code class="mono">${escapeHtml(audit.balance_check.status)}</code></div>` : ''}
    `;
  } else {
    document.getElementById('lineageMetaDetails').innerHTML = `
      <div><strong>Compilation engine:</strong> <code class="mono">${escapeHtml(resp.compiler_detail || (resp.compiler_mode === 'llm' ? 'LLM' : 'Deterministic Heuristic Engine'))}</code></div>
      <div style="margin-top:4px;"><strong>Tables used:</strong> <code class="mono">${escapeHtml((resp.involved_tables || []).join(', '))}</code></div>
      <div style="margin-top:4px;"><strong>Approved join paths:</strong> <code class="mono">${escapeHtml((resp.join_clauses_used || []).join(' ; ') || 'Single-table query')}</code></div>
    `;
  }

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
// 9.5 VERIFIED QUERY MEMORY (GOLDEN QUERY CACHE)
// ============================================================================
function renderVerifiedQueriesView() {
  if (!APP_STATE) return;
  const ws = APP_STATE.workspace || {};
  const queries = ws.verified_queries || [];
  const container = document.getElementById('verifiedQueriesGrid');
  if (!container) return;

  const searchInput = document.getElementById('vqSearchInput');
  const term = (searchInput?.value || '').toLowerCase().trim();

  const filtered = queries.filter((q) => {
    if (!term) return true;
    return (
      (q.question_pattern || '').toLowerCase().includes(term) ||
      (q.executed_query || '').toLowerCase().includes(term) ||
      (q.dax_equivalent || '').toLowerCase().includes(term) ||
      (q.chart_type || '').toLowerCase().includes(term)
    );
  });

  if (filtered.length === 0) {
    if (term) {
      container.innerHTML = `
        <div class="panel-card" style="text-align:center; padding:32px 20px;">
          <svg class="icon" style="width:32px; height:32px; color:var(--text-muted); margin-bottom:8px;"><use href="#i-search"/></svg>
          <div style="font-weight:600; font-size:15px;">No matching verified queries</div>
          <p class="muted" style="margin-top:4px;">No saved queries match your search term "${escapeHtml(term)}".</p>
        </div>`;
    } else {
      container.innerHTML = `
        <div class="panel-card" style="text-align:center; padding:48px 24px; border:2px dashed var(--border-color); background:transparent;">
          <div style="width:48px; height:48px; border-radius:50%; background:rgba(14,165,233,0.12); color:#0284c7; display:flex; align-items:center; justify-content:center; margin:0 auto 14px;">
            <svg class="icon" style="width:24px; height:24px;"><use href="#i-zap"/></svg>
          </div>
          <h3 style="font-size:17px; font-weight:600; margin-bottom:6px;">No Verified Queries Saved Yet</h3>
          <p class="muted" style="max-width:500px; margin:0 auto 16px; font-size:13.5px; line-height:1.5;">
            Whenever you ask a question in <strong>Ask Data</strong> chat and the AI generates SQL/DAX, click the <strong>Remember</strong> button to bookmark that recipe into memory. Next time you ask, it runs instantly from memory!
          </p>
          <button type="button" class="btn btn-secondary btn-sm" id="btnCreateFirstVq">
            <svg class="icon"><use href="#i-plus"/></svg> Add Manual Query
          </button>
        </div>`;
      document.getElementById('btnCreateFirstVq')?.addEventListener('click', () => {
        openEditVerifiedQueryModal();
      });
    }
    return;
  }

  container.innerHTML = filtered
    .map((vq) => {
      const chartBadge = `<span class="meta-chip"><svg class="icon" style="width:11px;height:11px;"><use href="#i-bar"/></svg> Visual: ${escapeHtml(vq.chart_type || 'bar')}</span>`;
      const dialectBadge = `<span class="meta-chip mono" style="font-size:11px;"><svg class="icon" style="width:11px;height:11px;"><use href="#i-database"/></svg> ${escapeHtml(vq.query_dialect || 'duckdb')}</span>`;
      const countBadge = `<span class="meta-chip" style="background:rgba(14,165,233,0.12); color:#0284c7; border:1px solid rgba(14,165,233,0.3); font-weight:600;"><svg class="icon" style="width:11px;height:11px;"><use href="#i-zap"/></svg> Used ${vq.execution_count || 1} time${(vq.execution_count || 1) === 1 ? '' : 's'}</span>`;
      const dateFormatted = vq.updated_at ? new Date(vq.updated_at).toLocaleDateString() : 'Recent';

      return `
        <div class="vq-card" data-id="${escapeHtml(vq.id)}">
          <div class="vq-header">
            <div class="vq-title-group">
              <div class="vq-question">
                <svg class="icon" style="width:16px; height:16px; color:#0284c7;"><use href="#i-zap"/></svg>
                <span>${escapeHtml(vq.question_pattern)}</span>
              </div>
              <div class="vq-meta-row">
                ${countBadge}
                ${dialectBadge}
                ${chartBadge}
                <span class="muted" style="font-size:11.5px;">Updated ${escapeHtml(dateFormatted)}</span>
              </div>
            </div>
            <div class="vq-actions">
              <button type="button" class="btn btn-secondary btn-xs btn-test-vq" data-id="${escapeHtml(vq.id)}" title="Run live test on active connector">
                <svg class="icon" style="width:12px;height:12px;"><use href="#i-play"/></svg> Test
              </button>
              <button type="button" class="btn btn-secondary btn-xs btn-edit-vq" data-id="${escapeHtml(vq.id)}" title="Edit question pattern or SQL/DAX">
                <svg class="icon" style="width:12px;height:12px;"><use href="#i-edit"/></svg> Edit
              </button>
              <button type="button" class="btn btn-ghost btn-xs btn-delete-vq" data-id="${escapeHtml(vq.id)}" title="Delete from memory" style="color:var(--accent-rose);">
                <svg class="icon" style="width:12px;height:12px;"><use href="#i-trash"/></svg> Delete
              </button>
            </div>
          </div>

          <details class="clean-disclosure" style="margin-top:2px;">
            <summary style="font-size:12.5px; font-weight:500; cursor:pointer;">
              <svg class="icon" style="width:13px;height:13px;"><use href="#i-code"/></svg> Inspect Verified SQL &amp; DAX
            </summary>
            <div style="display:flex; flex-direction:column; gap:8px; margin-top:8px;">
              <div>
                <div class="code-header" style="font-size:11px; margin-bottom:4px;">SQL (${escapeHtml(vq.query_dialect || 'duckdb')})</div>
                <pre class="vq-code-preview">${escapeHtml(vq.executed_query || '')}</pre>
              </div>
              ${vq.dax_equivalent ? `
              <div>
                <div class="code-header" style="font-size:11px; margin-bottom:4px;">DAX Equivalent</div>
                <pre class="vq-code-preview">${escapeHtml(vq.dax_equivalent || '')}</pre>
              </div>` : ''}
            </div>
          </details>

          <div id="testResultBox_${escapeHtml(vq.id)}" class="vq-test-preview" style="display:none;"></div>
        </div>
      `;
    })
    .join('');

  // Event handlers for action buttons in cards
  container.querySelectorAll('.btn-test-vq').forEach((btn) => {
    btn.addEventListener('click', async () => {
      const id = btn.dataset.id;
      const resBox = document.getElementById(`testResultBox_${id}`);
      if (!resBox) return;
      const origHtml = btn.innerHTML;
      try {
        btn.disabled = true;
        btn.innerHTML = `<svg class="icon spin" style="width:12px;height:12px;"><use href="#i-refresh"/></svg> Running...`;
        const res = await apiRequest(`/api/verified-queries/${id}/test`, { method: 'POST' });
        resBox.style.display = 'block';
        if (res.success) {
          resBox.style.background = 'rgba(16, 185, 129, 0.05)';
          resBox.style.borderColor = 'rgba(16, 185, 129, 0.25)';
          const sampleRows = (res.rows || []).slice(0, 3);
          const sampleHtml = sampleRows.length
            ? `<div style="overflow-x:auto; margin-top:6px;"><table class="data-table" style="font-size:11.5px; width:100%;">
                <thead><tr>${res.columns.map((c) => `<th>${escapeHtml(c)}</th>`).join('')}</tr></thead>
                <tbody>${sampleRows.map((r) => `<tr>${res.columns.map((c) => `<td>${escapeHtml(String(r[c] ?? ''))}</td>`).join('')}</tr>`).join('')}</tbody>
              </table></div>`
            : '<div class="muted">Query executed successfully (0 rows returned).</div>';
          resBox.innerHTML = `
            <div style="display:flex; justify-content:space-between; align-items:center; color:var(--accent-emerald); font-weight:600;">
              <span>✓ Test Passed: ${res.row_count} rows returned in ${res.elapsed_ms}ms</span>
              <button type="button" class="btn btn-ghost btn-xs" onclick="this.parentElement.parentElement.style.display='none';">✕ Close</button>
            </div>
            ${sampleHtml}
          `;
        } else {
          resBox.style.background = 'rgba(239, 68, 68, 0.08)';
          resBox.style.borderColor = 'rgba(239, 68, 68, 0.3)';
          resBox.innerHTML = `
            <div style="display:flex; justify-content:space-between; align-items:center; color:var(--accent-rose); font-weight:600;">
              <span>✗ Test Failed</span>
              <button type="button" class="btn btn-ghost btn-xs" onclick="this.parentElement.parentElement.style.display='none';">✕ Close</button>
            </div>
            <div style="color:var(--text-secondary); margin-top:4px;">${escapeHtml(res.error || 'Unknown error')}</div>
          `;
        }
      } catch (err) {
        showToast(`Test error: ${err.message}`);
      } finally {
        btn.disabled = false;
        btn.innerHTML = origHtml;
      }
    });
  });

  container.querySelectorAll('.btn-edit-vq').forEach((btn) => {
    btn.addEventListener('click', () => {
      const id = btn.dataset.id;
      const target = queries.find((q) => q.id === id);
      if (target) {
        openEditVerifiedQueryModal(target);
      }
    });
  });

  container.querySelectorAll('.btn-delete-vq').forEach((btn) => {
    btn.addEventListener('click', async () => {
      const id = btn.dataset.id;
      const target = queries.find((q) => q.id === id);
      const pat = target?.question_pattern || 'this query';
      if (!confirm(`Are you sure you want to remove "${pat}" from verified memory?`)) return;
      try {
        APP_STATE = await apiRequest(`/api/verified-queries/${id}`, { method: 'DELETE' });
        showToast('Query removed from memory');
        renderApp();
        renderVerifiedQueriesView();
      } catch (err) {
        showToast(`Failed to delete: ${err.message}`);
      }
    });
  });
}

function openEditVerifiedQueryModal(queryObj = null) {
  const dialog = document.getElementById('editVerifiedQueryDialog');
  if (!dialog) return;

  const idIn = document.getElementById('editVqId');
  const qIn = document.getElementById('editVqQuestion');
  const sqlIn = document.getElementById('editVqSql');
  const daxIn = document.getElementById('editVqDax');
  const chartIn = document.getElementById('editVqChartType');
  const dialectIn = document.getElementById('editVqDialect');
  const testRes = document.getElementById('editVqTestResult');
  const titleEl = document.getElementById('editVqDialogTitle');
  const saveBtn = document.getElementById('saveEditVqBtn');

  if (testRes) testRes.style.display = 'none';

  if (queryObj && queryObj.id) {
    if (titleEl) titleEl.textContent = 'Edit Verified Query';
    if (saveBtn) saveBtn.textContent = 'Save Changes';
    if (idIn) idIn.value = queryObj.id;
    if (qIn) qIn.value = queryObj.question_pattern || '';
    if (sqlIn) sqlIn.value = queryObj.executed_query || '';
    if (daxIn) daxIn.value = queryObj.dax_equivalent || '';
    if (chartIn) chartIn.value = queryObj.chart_type || 'bar';
    if (dialectIn) dialectIn.value = queryObj.query_dialect || APP_STATE?.workspace?.dialect || 'duckdb';
  } else {
    if (titleEl) titleEl.textContent = 'Save to Query Memory';
    if (saveBtn) saveBtn.textContent = 'Save to Memory';
    if (idIn) idIn.value = '';
    if (qIn) qIn.value = queryObj?.question_pattern || '';
    if (sqlIn) sqlIn.value = queryObj?.executed_query || '';
    if (daxIn) daxIn.value = queryObj?.dax_equivalent || '';
    if (chartIn) chartIn.value = queryObj?.chart_type || 'bar';
    if (dialectIn) dialectIn.value = queryObj?.query_dialect || APP_STATE?.workspace?.dialect || 'duckdb';
  }

  dialog.showModal();
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
  if (!text) return '';
  let html = escapeHtml(text)
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*([^*]+)\*/g, '<em>$1</em>')
    .replace(/`([^`]+)`/g, '<code class="mono">$1</code>');

  // Convert markdown headers
  html = html.replace(/^### (.+)$/gm, '<h4 style="margin:14px 0 6px 0; font-size:14px; color:var(--text-primary); font-weight:700;">$1</h4>');
  html = html.replace(/^#### (.+)$/gm, '<h5 style="margin:10px 0 4px 0; font-size:13px; color:var(--text-primary); font-weight:600;">$1</h5>');

  // Convert bullet lists
  html = html.replace(/^\s*-\s+(.+)$/gm, '<li style="margin-left:18px; margin-bottom:4px; font-size:12.5px; line-height:1.5;">$1</li>');
  html = html.replace(/(<li.*<\/li>\n?)+/g, '<ul style="margin:6px 0; padding-left:4px;">$&</ul>');

  // Convert paragraphs / double newlines
  html = html.replace(/\n\n+/g, '<div style="margin:8px 0;"></div>');
  html = html.replace(/\n/g, '<br/>');

  return html;
}

function renderAllViews() {
  try { renderGlobalTelemetry(); } catch (e) { console.error('Error rendering telemetry:', e); }
  try { renderConnectView(); } catch (e) { console.error('Error rendering connect view:', e); }
  try { renderJoinsView(); } catch (e) { console.error('Error rendering joins view:', e); }
  try { renderInterviewView(); } catch (e) { console.error('Error rendering interview view:', e); }
  try { renderMeasuresView(); } catch (e) { console.error('Error rendering measures view:', e); }
  try { renderVerifiedQueriesView(); } catch (e) { console.error('Error rendering verified queries view:', e); }
  try { renderChatView(); } catch (e) { console.error('Error rendering chat view:', e); }
  try { if (window.loadRootCauseOptions) window.loadRootCauseOptions(true); } catch (e) {}
}

async function loadDemoDataset() {
  showIngestionLoading(
    'Loading Enterprise Demo Warehouse',
    'Materializing customers, products, orders, and returns tables into local DuckDB columnar storage...'
  );
  try {
    showToast('Loading 4-table demo warehouse…');
    APP_STATE = await apiRequest('/api/connect/demo', { method: 'POST' });
    renderAllViews();
    showToast('Demo warehouse profiled and ready');
  } catch (err) {
    showToast(`Error loading demo: ${err.message}`);
  } finally {
    hideIngestionLoading();
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

async function submitChatQuestion(questionText, options = {}) {
  const trimmed = (questionText || '').trim();
  if (!trimmed) return;
  const forceRecompile = Boolean(options.force_recompile);

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
        <span class="meta-chip meta-chip-accent">${forceRecompile ? 'Re-compiling via AI Model…' : 'Compiling SQL &amp; DAX…'}</span>
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
  const mainMode = document.getElementById('mainChatResponseMode')?.value || 'auto';
  try {
    const res = await apiRequest('/api/chat/query', {
      method: 'POST',
      body: JSON.stringify({
        question: trimmed,
        revenue_shock_pct: Number(revSlider?.value || 0),
        cost_shock_pct: Number(costSlider?.value || 0),
        response_mode: mainMode,
        force_recompile: forceRecompile,
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

  // Main Chat Response Mode Selector (Auto / Text Only / With Chart)
  document.querySelectorAll('#mainChatModeGroup .mode-pill').forEach((pill) => {
    pill.addEventListener('click', () => {
      const mode = pill.getAttribute('data-mode') || 'auto';
      const hiddenIn = document.getElementById('mainChatResponseMode');
      if (hiddenIn) hiddenIn.value = mode;
      document.querySelectorAll('#mainChatModeGroup .mode-pill').forEach((p) => {
        p.classList.toggle('active', p === pill);
      });
    });
  });

  // Split-View Thread Modal Mode Selector & Form Submit
  document.querySelectorAll('#splitThreadModeGroup .mode-pill').forEach((pill) => {
    pill.addEventListener('click', () => {
      const mode = pill.getAttribute('data-mode') || 'text_only';
      const hiddenIn = document.getElementById('splitThreadResponseMode');
      if (hiddenIn) hiddenIn.value = mode;
      document.querySelectorAll('#splitThreadModeGroup .mode-pill').forEach((p) => {
        p.classList.toggle('active', p === pill);
      });
    });
  });

  document.getElementById('splitThreadForm')?.addEventListener('submit', async (e) => {
    e.preventDefault();
    if (ACTIVE_SPLIT_THREAD_IDX === null || ACTIVE_SPLIT_THREAD_IDX === undefined) return;
    const inputEl = document.getElementById('splitThreadInput');
    const q = (inputEl?.value || '').trim();
    if (!q) return;
    if (inputEl) inputEl.value = '';
    const mode = document.getElementById('splitThreadResponseMode')?.value || 'text_only';
    await submitChartThreadQuestion(ACTIVE_SPLIT_THREAD_IDX, q, mode);
  });

  document.getElementById('splitModalOpenTabBtn')?.addEventListener('click', () => {
    if (ACTIVE_SPLIT_THREAD_IDX !== null && ACTIVE_SPLIT_THREAD_IDX !== undefined) {
      window.open(`/thread-studio?idx=${ACTIVE_SPLIT_THREAD_IDX}`, '_blank');
    }
  });

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
    const isOpening = rcDrawer?.classList.contains('hidden');
    rcDrawer?.classList.toggle('hidden');
    if (isOpening) {
      loadRootCauseOptions(true);
    }
  });
  document.getElementById('closeRootCauseBtn')?.addEventListener('click', () => {
    rcDrawer?.classList.add('hidden');
  });

  // Algorithm Selection Pills & Metric Change Listener
  document.querySelectorAll('#rcAlgoPillGroup .rc-algo-pill').forEach((pill) => {
    pill.addEventListener('click', () => {
      const algo = pill.getAttribute('data-algo');
      selectRootCauseAlgorithm(algo, true);
    });
  });

  document.getElementById('rcMetricSelect')?.addEventListener('change', () => {
    updateRootCauseRecommendation();
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
    const count = fileIn.files.length;
    Array.from(fileIn.files).forEach((f) => fd.append('files', f));
    showIngestionLoading(
      `Ingesting ${count} File${count > 1 ? 's' : ''}`,
      'Reading source data, materializing DuckDB columnar storage, and scanning for PII...'
    );
    try {
      showToast('Ingesting & profiling uploaded files…');
      APP_STATE = await apiRequest('/api/connect/upload', { method: 'POST', body: fd });
      renderAllViews();
      showToast('Files ingested and profiled');
    } catch (err) {
      showToast(`Upload error: ${err.message}`);
    } finally {
      hideIngestionLoading();
    }
  });

  // SQL Connect Submit
  document.getElementById('sqlConnectForm')?.addEventListener('submit', async (e) => {
    e.preventDefault();
    const dialect = document.getElementById('sqlDialectInput')?.value || 'SQL';
    showIngestionLoading(
      `Connecting to ${dialect.toUpperCase()} Database`,
      'Attaching database, extracting table catalogs, profiling columns, and inferring join relationships...'
    );
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
    } finally {
      hideIngestionLoading();
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
    showIngestionLoading(
      'Connecting to Power BI Semantic Model',
      'Establishing DAX endpoint session, discovering measures, and building schema model...'
    );
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
    } finally {
      hideIngestionLoading();
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

  // Auto-Enrich Business Knowledge
  document.getElementById('btnAutoEnrichKnowledge')?.addEventListener('click', async () => {
    const btn = document.getElementById('btnAutoEnrichKnowledge');
    try {
      if (btn) {
        btn.disabled = true;
        btn.innerHTML = `<svg class="icon spin" style="width:14px;height:14px"><use href="#i-refresh"/></svg> Enriching...`;
      }
      const res = await apiRequest('/api/interview/auto-enrich', { method: 'POST' });
      APP_STATE = res.state;
      renderAllViews();
      const rep = res.report || {};
      const dTitle = rep.domain?.domain_title || 'Enterprise Operations';
      showToast(`✨ Domain detected: ${dTitle}. Enriched ${rep.tables_enriched || 0} tables & ${rep.columns_enriched || 0} columns with business semantics.`);
    } catch (err) {
      showToast(`Enrichment failed: ${err.message}`);
    } finally {
      if (btn) {
        btn.disabled = false;
        btn.innerHTML = `<svg class="icon"><use href="#i-sparkles"/></svg> Auto-Enrich Knowledge`;
      }
    }
  });

  // Human-in-the-Loop Business Knowledge Customization Listeners
  document.getElementById('domainEditForm')?.addEventListener('submit', async (e) => {
    e.preventDefault();
    try {
      APP_STATE = await apiRequest('/api/interview/domain-override', {
        method: 'POST',
        body: JSON.stringify({
          domain_title: document.getElementById('domainTitleInput').value.trim(),
          industry: document.getElementById('domainIndustryInput').value.trim(),
          domain_tag: document.getElementById('domainTagInput').value.trim(),
        }),
      });
      document.getElementById('domainEditDialog')?.close();
      renderAllViews();
      showToast('Updated business domain & re-indexed knowledge');
    } catch (err) {
      showToast(`Domain update failed: ${err.message}`);
    }
  });

  document.getElementById('ruleEditForm')?.addEventListener('submit', async (e) => {
    e.preventDefault();
    const ruleId = document.getElementById('editRuleIdInput').value;
    try {
      APP_STATE = await apiRequest(`/api/interview/rule/${encodeURIComponent(ruleId)}`, {
        method: 'PUT',
        body: JSON.stringify({
          term: document.getElementById('editRuleTermInput').value.trim(),
          definition: document.getElementById('editRuleDefInput').value.trim(),
          sql_filter_hint: document.getElementById('editRuleSqlHintInput').value.trim(),
        }),
      });
      document.getElementById('ruleEditDialog')?.close();
      renderAllViews();
      showToast('Updated business rule & RAG memory');
    } catch (err) {
      showToast(`Rule update failed: ${err.message}`);
    }
  });

  document.getElementById('tableEditForm')?.addEventListener('submit', async (e) => {
    e.preventDefault();
    const tableName = document.getElementById('editTableNameInput').value;
    try {
      APP_STATE = await apiRequest('/api/schema/table-config', {
        method: 'POST',
        body: JSON.stringify({
          table_name: tableName,
          business_name: document.getElementById('editTableBusinessNameInput').value.trim(),
          description: document.getElementById('editTableDescriptionInput').value.trim(),
          table_role: document.getElementById('editTableRoleSelect').value,
        }),
      });
      document.getElementById('tableEditDialog')?.close();
      renderAllViews();
      showToast(`Updated semantics for table "${tableName}"`);
    } catch (err) {
      showToast(`Table update failed: ${err.message}`);
    }
  });

  document.getElementById('columnEditForm')?.addEventListener('submit', async (e) => {
    e.preventDefault();
    const tableName = document.getElementById('editColTableInput').value;
    const colName = document.getElementById('editColNameInput').value;
    const rawMeanings = document.getElementById('editColMeaningsInput').value.trim();
    const valueMeanings = {};
    if (rawMeanings) {
      rawMeanings.split('\n').forEach((line) => {
        const parts = line.split(/[:=]/);
        if (parts.length >= 2) {
          const k = parts[0].trim();
          const v = parts.slice(1).join(':').trim();
          if (k) valueMeanings[k] = v;
        }
      });
    }
    try {
      APP_STATE = await apiRequest('/api/schema/column-config', {
        method: 'POST',
        body: JSON.stringify({
          table_name: tableName,
          column_name: colName,
          business_alias: document.getElementById('editColAliasInput').value.trim(),
          description: document.getElementById('editColDescInput').value.trim(),
          value_meanings: valueMeanings,
        }),
      });
      document.getElementById('columnEditDialog')?.close();
      renderAllViews();
      showToast(`Updated semantics for column "${colName}"`);
    } catch (err) {
      showToast(`Column update failed: ${err.message}`);
    }
  });

  // Global click dismiss for drill-down menus
  document.addEventListener('click', () => {
    document.querySelectorAll('.drilldown-menu.open').forEach((m) => m.classList.remove('open'));
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

  // Improve Prompt Button (schema-grounded entity & column resolution)
  const improveBtn = document.getElementById('chatImprovePromptBtn');
  improveBtn?.addEventListener('click', async () => {
    if (!chatInput) return;
    const rawQ = (chatInput.value || '').trim();
    if (!rawQ) {
      showToast('Type a question first to improve it');
      chatInput.focus();
      return;
    }
    const origHtml = improveBtn.innerHTML;
    try {
      improveBtn.disabled = true;
      improveBtn.innerHTML = `<svg class="icon spin" style="width:13px;height:13px"><use href="#i-refresh"/></svg><span>Improving…</span>`;
      const res = await apiRequest('/api/chat/improve-prompt', {
        method: 'POST',
        body: JSON.stringify({ question: rawQ }),
      });
      if (res && res.improved_question) {
        chatInput.value = res.improved_question;
        chatInput.style.height = 'auto';
        chatInput.style.height = `${Math.min(140, chatInput.scrollHeight)}px`;
        chatInput.focus();
        const details = (res.resolutions || []).length
          ? `✨ ${res.resolutions.join(' · ')}`
          : '✨ Prompt refined for schema-grounded SQL compilation';
        showToast(details);
      }
    } catch (err) {
      showToast(`Prompt improve failed: ${err.message}`);
    } finally {
      improveBtn.disabled = false;
      improveBtn.innerHTML = origHtml;
    }
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

  // Dynamic Root-Cause Options State & Helpers
  let CURRENT_RC_SCHEMA = null;

  window.loadRootCauseOptions = async function(forceRefresh = false) {
    if (CURRENT_RC_SCHEMA && !forceRefresh) return;
    try {
      const data = await apiRequest('/api/root-cause/options');
      CURRENT_RC_SCHEMA = data;

      // 1. Populate Metrics
      const metricSelect = document.getElementById('rcMetricSelect');
      if (metricSelect && data.metrics) {
        metricSelect.innerHTML = data.metrics.map((m) => {
          const selected = m.column === data.default_metric ? 'selected' : '';
          return `<option value="${escapeHtml(m.column)}" data-agg="${m.agg}" ${selected}>${escapeHtml(m.label)} (${m.agg})</option>`;
        }).join('');
      }

      // 2. Populate Dimensions
      const dimSelect = document.getElementById('rcDimensionSelect');
      if (dimSelect && data.dimensions) {
        dimSelect.innerHTML = data.dimensions.map((d) => {
          const selected = d.id === data.default_dimension ? 'selected' : '';
          return `<option value="${escapeHtml(d.id)}" ${selected}>${escapeHtml(d.label)}</option>`;
        }).join('');
      }

      // 3. Populate Comparison Horizons
      const winSelect = document.getElementById('rcWindowSelect');
      if (winSelect && data.temporal_options) {
        winSelect.innerHTML = data.temporal_options.map((w, idx) => {
          return `<option value="${escapeHtml(w.id)}" ${idx === 0 ? 'selected' : ''}>${escapeHtml(w.label)}</option>`;
        }).join('');
      }

      // Pre-fill Custom Date Range if max_date is available
      if (data.temporal_metadata && data.temporal_metadata.max_date) {
        try {
          const maxD = new Date(data.temporal_metadata.max_date.split(' ')[0]);
          if (!isNaN(maxD.getTime())) {
            const t1End = maxD.toISOString().slice(0, 10);
            const t1StartD = new Date(maxD);
            t1StartD.setDate(t1StartD.getDate() - 90);
            const t1Start = t1StartD.toISOString().slice(0, 10);
            const t0EndD = new Date(t1StartD);
            t0EndD.setDate(t0EndD.getDate() - 1);
            const t0End = t0EndD.toISOString().slice(0, 10);
            const t0StartD = new Date(t0EndD);
            t0StartD.setDate(t0StartD.getDate() - 90);
            const t0Start = t0StartD.toISOString().slice(0, 10);

            const t0s = document.getElementById('rcT0Start');
            const t0e = document.getElementById('rcT0End');
            const t1s = document.getElementById('rcT1Start');
            const t1e = document.getElementById('rcT1End');
            if (t0s && !t0s.value) t0s.value = t0Start;
            if (t0e && !t0e.value) t0e.value = t0End;
            if (t1s && !t1s.value) t1s.value = t1Start;
            if (t1e && !t1e.value) t1e.value = t1End;
          }
        } catch (_) {}
      }

      // 4. Update Lineage Badge
      const lineageBadge = document.getElementById('rcDateLineageBadge');
      if (lineageBadge) {
        if (data.has_temporal && data.temporal_metadata) {
          const anchor = data.temporal_metadata.selected_anchor || '';
          const isJoined = data.temporal_metadata.is_joined;
          const joinInfo = isJoined ? ` <span class="badge badge-accent">Joined Table</span>` : '';
          lineageBadge.innerHTML = `<span>📅 Temporal Anchor: <strong>${escapeHtml(anchor)}</strong>${joinInfo}</span>`;
        } else {
          lineageBadge.innerHTML = `<span class="tone-warning">ℹ️ No date column found in connected tables. Switched to Cohort Comparison mode.</span>`;
        }
      }

      updateRootCauseRecommendation();
    } catch (err) {
      console.warn('Failed to load root cause options:', err);
    }
  };

  // Toggle Custom Date Inputs when Comparison Horizon changes
  document.getElementById('rcWindowSelect')?.addEventListener('change', (e) => {
    const isCustom = e.target.value === 'custom';
    const customFields = document.getElementById('rcCustomDateFields');
    if (customFields) {
      customFields.style.display = isCustom ? 'flex' : 'none';
    }
  });

  window.updateRootCauseRecommendation = function() {
    const metricSelect = document.getElementById('rcMetricSelect');
    const selectedOpt = metricSelect?.selectedOptions[0];
    const agg = selectedOpt?.getAttribute('data-agg') || 'SUM';
    const badge = document.getElementById('rcAlgoRecBadge');

    if (agg === 'AVG' || agg === 'RATIO') {
      if (badge) badge.textContent = "✨ Recommended for Rates (Simpson's Paradox Solver)";
      selectRootCauseAlgorithm('rate_mix', false);
    } else {
      if (badge) badge.textContent = "✨ Recommended for Totals (Additive Waterfall)";
      selectRootCauseAlgorithm('additive', false);
    }
  };

  window.selectRootCauseAlgorithm = function(algoId, isManual = true) {
    const algoInput = document.getElementById('rcAlgorithmInput');
    if (algoInput) algoInput.value = algoId;

    document.querySelectorAll('#rcAlgoPillGroup .rc-algo-pill').forEach((pill) => {
      pill.classList.toggle('active', pill.getAttribute('data-algo') === algoId);
    });

    const dimGroup = document.getElementById('rcDimensionGroup');
    if (dimGroup) {
      dimGroup.style.display = algoId === 'auto_scan' ? 'none' : 'block';
    }
  };

  // Modernized Root-Cause Diagnostic Form - Dispatches directly to Chat Stream
  document.getElementById('rootCauseForm')?.addEventListener('submit', async (e) => {
    e.preventDefault();
    const metric_col = document.getElementById('rcMetricSelect')?.value;
    const dimension = document.getElementById('rcDimensionSelect')?.value;
    const algorithm = document.getElementById('rcAlgorithmInput')?.value || 'auto';
    const window_type = document.getElementById('rcWindowSelect')?.value || 'last_90_vs_prior_90';

    let t0_start = '';
    let t0_end = '';
    let t1_start = '';
    let t1_end = '';

    if (window_type === 'custom') {
      t0_start = document.getElementById('rcT0Start')?.value || '';
      t0_end = document.getElementById('rcT0End')?.value || '';
      t1_start = document.getElementById('rcT1Start')?.value || '';
      t1_end = document.getElementById('rcT1End')?.value || '';

      if (!t0_start || !t0_end || !t1_start || !t1_end) {
        showToast('Please enter all 4 dates for custom period comparison (Baseline T0 and Current T1).');
        return;
      }
      if (t0_start > t0_end) {
        showToast('Prior Baseline Start Date must be before End Date.');
        return;
      }
      if (t1_start > t1_end) {
        showToast('Current Period Start Date must be before End Date.');
        return;
      }
    }

    const submitBtn = document.getElementById('rcSubmitBtn');
    if (submitBtn) {
      submitBtn.disabled = true;
      submitBtn.textContent = 'Diagnosing…';
    }

    try {
      const res = await apiRequest('/api/root-cause', {
        method: 'POST',
        body: JSON.stringify({
          metric_col,
          dimension,
          algorithm,
          window_type,
          t0_start,
          t0_end,
          t1_start,
          t1_end,
        }),
      });

      if (res.state) {
        APP_STATE = res.state;
      }

      // Close the drawer so user sees the diagnostic result directly in the chat view
      const drawer = document.getElementById('rootCauseDrawer');
      if (drawer) {
        drawer.classList.add('hidden');
      }

      renderChatView();

      // Smooth scroll chat stream to the newly rendered card
      const stream = document.getElementById('chatMessagesStream');
      if (stream) {
        const lastCard = stream.querySelector('.chat-card:last-child');
        if (lastCard) {
          lastCard.scrollIntoView({ behavior: 'smooth', block: 'end' });
        }
      }

      showToast('Diagnostic complete! Visual and analysis generated in chat.');
    } catch (err) {
      showToast(`Variance error: ${err.message}`);
    } finally {
      if (submitBtn) {
        submitBtn.disabled = false;
        submitBtn.textContent = '🔍 Diagnose';
      }
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

  // Delete Workspace / Project Listeners
  document.getElementById('deleteWorkspaceBtn')?.addEventListener('click', () => {
    openDeleteProjectModal();
  });

  document.getElementById('btnDeleteCurrentProjectHeader')?.addEventListener('click', () => {
    openDeleteProjectModal();
  });

  document.getElementById('deleteProjectSelect')?.addEventListener('change', () => {
    const confirmInput = document.getElementById('deleteConfirmInput');
    if (confirmInput) confirmInput.value = '';
    const submitBtn = document.getElementById('btnConfirmProjectDelete');
    if (submitBtn) submitBtn.disabled = true;
  });

  document.getElementById('deleteConfirmInput')?.addEventListener('input', (e) => {
    const val = (e.target.value || '').trim();
    const select = document.getElementById('deleteProjectSelect');
    const selectedWs = select ? select.value : '';
    const submitBtn = document.getElementById('btnConfirmProjectDelete');
    if (submitBtn) {
      const isConfirmed = val.toUpperCase() === 'DELETE' || (selectedWs && val.toLowerCase() === selectedWs.toLowerCase());
      submitBtn.disabled = !isConfirmed;
    }
  });

  document.getElementById('deleteProjectForm')?.addEventListener('submit', async (e) => {
    e.preventDefault();
    const select = document.getElementById('deleteProjectSelect');
    const wsId = select ? select.value : APP_STATE?.active_workspace_id;
    if (!wsId) return;

    const submitBtn = document.getElementById('btnConfirmProjectDelete');
    try {
      if (submitBtn) {
        submitBtn.disabled = true;
        submitBtn.innerHTML = `<svg class="icon spin" style="width:14px;height:14px"><use href="#i-refresh"/></svg> Deleting files...`;
      }
      APP_STATE = await apiRequest(`/api/workspace/${encodeURIComponent(wsId)}`, {
        method: 'DELETE',
      });
      document.getElementById('deleteProjectDialog')?.close();
      renderAllViews();
      showToast(`🗑️ Permanently deleted project "${wsId}" and all its files from disk.`);
    } catch (err) {
      showToast(`Delete failed: ${err.message}`);
    } finally {
      if (submitBtn) {
        submitBtn.disabled = false;
        submitBtn.innerHTML = `<svg class="icon"><use href="#i-trash"/></svg> Permanently Delete Project`;
      }
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

  // Verified Query Memory Event Listeners
  document.getElementById('addVerifiedQueryBtn')?.addEventListener('click', () => {
    openEditVerifiedQueryModal();
  });

  document.getElementById('vqSearchInput')?.addEventListener('input', () => {
    renderVerifiedQueriesView();
  });

  document.getElementById('cancelEditVqBtn')?.addEventListener('click', () => {
    document.getElementById('editVerifiedQueryDialog')?.close();
  });

  document.getElementById('btnTestVqSql')?.addEventListener('click', async () => {
    const sql = (document.getElementById('editVqSql')?.value || '').trim();
    const resBox = document.getElementById('editVqTestResult');
    if (!sql || !resBox) {
      showToast('Enter SQL code to test');
      return;
    }
    const testBtn = document.getElementById('btnTestVqSql');
    const origHtml = testBtn.innerHTML;
    try {
      testBtn.disabled = true;
      testBtn.innerHTML = `<svg class="icon spin" style="width:12px;height:12px"><use href="#i-refresh"/></svg> Testing...`;
      const res = await apiRequest('/api/verified-queries/test-raw', {
        method: 'POST',
        body: JSON.stringify({ query: sql }),
      });
      resBox.style.display = 'block';
      if (res.success) {
        resBox.style.background = 'rgba(16, 185, 129, 0.08)';
        resBox.style.borderColor = 'rgba(16, 185, 129, 0.3)';
        resBox.style.color = 'var(--accent-emerald)';
        resBox.innerHTML = `✓ Query valid! Returned ${res.row_count} rows in ${res.elapsed_ms}ms (${(res.columns || []).join(', ')})`;
      } else {
        resBox.style.background = 'rgba(239, 68, 68, 0.08)';
        resBox.style.borderColor = 'rgba(239, 68, 68, 0.3)';
        resBox.style.color = 'var(--accent-rose)';
        resBox.innerHTML = `✗ Query error: ${escapeHtml(res.error || 'Execution failed')}`;
      }
    } catch (err) {
      resBox.style.display = 'block';
      resBox.style.color = 'var(--accent-rose)';
      resBox.innerHTML = `✗ Test error: ${escapeHtml(err.message)}`;
    } finally {
      testBtn.disabled = false;
      testBtn.innerHTML = origHtml;
    }
  });

  document.getElementById('editVerifiedQueryForm')?.addEventListener('submit', async (e) => {
    e.preventDefault();
    const id = (document.getElementById('editVqId')?.value || '').trim();
    const question_pattern = (document.getElementById('editVqQuestion')?.value || '').trim();
    const executed_query = (document.getElementById('editVqSql')?.value || '').trim();
    const dax_equivalent = (document.getElementById('editVqDax')?.value || '').trim();
    const chart_type = document.getElementById('editVqChartType')?.value || 'bar';
    const dialect = document.getElementById('editVqDialect')?.value || 'duckdb';
    const submitBtn = document.getElementById('saveEditVqBtn');
    const origHtml = submitBtn.innerHTML;

    try {
      submitBtn.disabled = true;
      submitBtn.innerHTML = `<svg class="icon spin" style="width:13px;height:13px"><use href="#i-refresh"/></svg> Saving...`;
      if (id) {
        APP_STATE = await apiRequest(`/api/verified-queries/${id}`, {
          method: 'PUT',
          body: JSON.stringify({
            question_pattern,
            executed_query,
            dax_equivalent,
            chart_type,
            query_dialect: dialect,
          }),
        });
        showToast('Verified query updated');
      } else {
        APP_STATE = await apiRequest('/api/verified-queries', {
          method: 'POST',
          body: JSON.stringify({
            question_pattern,
            executed_query,
            dax_equivalent,
            chart_type,
            query_dialect: dialect,
          }),
        });
        showToast('Saved query to verified memory');
      }
      document.getElementById('editVerifiedQueryDialog')?.close();
      renderApp();
      renderVerifiedQueriesView();
    } catch (err) {
      showToast(`Save failed: ${err.message}`);
    } finally {
      submitBtn.disabled = false;
      submitBtn.innerHTML = origHtml;
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
    {
      category: 'Project',
      label: 'Delete project / workspace and all files',
      action: () => openDeleteProjectModal(),
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
