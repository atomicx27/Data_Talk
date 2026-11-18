/**
 * er3d_engine.js — Interactive ER Constellation & 2.5D Isometric Stack Engine.
 * - 'iso' (DEFAULT): Zero-dependency CSS 3D Isometric stack with drag-orbit, zoom,
 *   FACT vs DIM depth planes, cross-DB indicators, glowing join paths, and focus-filtering.
 * - 'orbit3d': Lazy-loaded 3d-force-graph@1.73.3 WebGL constellation on explicit user click,
 *   with safe fallback and strict destroy() WebGL cleanup.
 * - '2d': Delegates cleanly back to the native SVG blueprint renderer.
 */

import { isReducedMotionPreferred } from './spatial_tilt.js';

function esc(str) {
  return String(str ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

const FORCE_GRAPH_CDN = 'https://esm.sh/3d-force-graph@1.73.3?bundle';

let _currentMode = 'iso';
let _orbitState = { rotX: 34, rotZ: -18, zoom: 1 };
let _active3DGraph = null;
let _renderSeq = 0;

export function getERViewMode() {
  return _currentMode;
}

export function setERViewMode(mode) {
  if (mode === '2d' || mode === 'iso' || mode === 'orbit3d') {
    _currentMode = mode;
  }
  return _currentMode;
}

export function destroyER3D() {
  if (_active3DGraph) {
    try {
      if (typeof _active3DGraph._destructor === 'function') {
        _active3DGraph._destructor();
      } else if (typeof _active3DGraph.pauseAnimation === 'function') {
        _active3DGraph.pauseAnimation();
      }
    } catch (_) {
      // ignore teardown issues
    }
    _active3DGraph = null;
  }
}

function getVisibleSubgraph(allTables, rels, focalTable) {
  let visibleTables = allTables;
  if (focalTable) {
    const neighborSet = new Set([focalTable]);
    rels.forEach((r) => {
      if (r.source_table === focalTable) neighborSet.add(r.target_table);
      if (r.target_table === focalTable) neighborSet.add(r.source_table);
    });
    visibleTables = allTables.filter((t) => neighborSet.has(t.name));
  }
  return visibleTables;
}

function renderIsoStack(viewport, ctx) {
  destroyER3D();
  const { tables, relationships, focalTable, onSelectFocal } = ctx;
  const visibleTables = getVisibleSubgraph(tables, relationships, focalTable);
  const facts = visibleTables.filter((t) => t.table_role === 'FACT');
  const dims = visibleTables.filter((t) => t.table_role !== 'FACT');

  const leftList = facts.length ? facts : visibleTables.slice(0, Math.ceil(visibleTables.length / 2));
  const rightList = facts.length ? dims : visibleTables.slice(Math.ceil(visibleTables.length / 2));

  const cardW = 244;
  const cardH = 148;
  const gapY = 34;
  const stageW = 900;
  const maxRows = Math.max(leftList.length, rightList.length, 1);
  const stageH = Math.max(360, maxRows * (cardH + gapY) + 48);

  const positions = {};
  leftList.forEach((t, idx) => {
    positions[t.name] = {
      x: 48,
      y: 26 + idx * (cardH + gapY),
      z: t.table_role === 'FACT' ? 38 : 16,
      table: t,
    };
  });
  rightList.forEach((t, idx) => {
    positions[t.name] = {
      x: stageW - cardW - 48,
      y: 26 + idx * (cardH + gapY),
      z: 10,
      table: t,
    };
  });

  const visibleRels = relationships.filter((r) => positions[r.source_table] && positions[r.target_table]);

  const edgesSvg = visibleRels
    .map((r, i) => {
      const p1 = positions[r.source_table];
      const p2 = positions[r.target_table];
      const sameCol = p1.x === p2.x;
      const x1 = sameCol ? p1.x + cardW : p1.x < p2.x ? p1.x + cardW : p1.x;
      const y1 = p1.y + cardH / 2;
      const x2 = sameCol ? p2.x + cardW : p1.x < p2.x ? p2.x : p2.x + cardW;
      const y2 = p2.y + cardH / 2;
      const cx1 = sameCol ? x1 + 96 : (x1 + x2) / 2;
      const cx2 = sameCol ? x2 + 96 : (x1 + x2) / 2;
      const approved = r.status === 'approved';
      const stroke = approved ? '#10b981' : '#f59e0b';
      const cls = approved ? 'er3d-edge-approved' : 'er3d-edge-pending';
      const mx = sameCol ? (x1 + cx1) / 2 : (x1 + x2) / 2;
      const my = (y1 + y2) / 2;
      const d = `M ${x1} ${y1} C ${cx1} ${y1}, ${cx2} ${y2}, ${x2} ${y2}`;

      return `
        <g class="er3d-edge-group" data-edge-idx="${i}">
          <path d="${d}" fill="none" stroke="${stroke}" stroke-width="${approved ? 2.6 : 2}" class="${cls}" ${
        approved ? '' : 'stroke-dasharray="6,4"'
      } />
          <circle cx="${x1}" cy="${y1}" r="4.5" fill="${stroke}" />
          <circle cx="${x2}" cy="${y2}" r="4.5" fill="${stroke}" />
          <g transform="translate(${mx}, ${my})">
            <rect x="-78" y="-12" width="156" height="24" rx="6" class="er3d-edge-pill-bg" stroke="${stroke}" stroke-width="1" />
            <text x="0" y="4" text-anchor="middle" class="er3d-edge-pill-text">${esc(
              `${r.source_column} = ${r.target_column}`
            )}</text>
          </g>
        </g>
      `;
    })
    .join('');

  const cardsHtml = Object.values(positions)
    .map((pos) => {
      const t = pos.table;
      const isFact = t.table_role === 'FACT';
      const isFocal = focalTable === t.name;
      const cols = (t.columns || []).slice(0, 4);
      const extraCount = Math.max(0, (t.columns || []).length - 4);
      const colRows = cols
        .map(
          (c) => `
          <div class="er3d-col-row">
            <span class="er3d-col-name">${esc(c.name)}</span>
            <span class="er3d-col-type mono">${esc((c.data_type || '').toLowerCase())}</span>
          </div>`
        )
        .join('');

      return `
        <div
          class="er3d-table-plate ${isFact ? 'role-fact' : 'role-dim'} ${isFocal ? 'is-focal' : ''}"
          data-table-name="${esc(t.name)}"
          style="left:${pos.x}px; top:${pos.y}px; width:${cardW}px; min-height:${cardH}px; --plate-z:${pos.z}px;"
          title="Click to filter joins for ${esc(t.name)}"
        >
          <div class="er3d-plate-header">
            <span class="er3d-role-pill ${isFact ? 'fact' : 'dim'}">${esc(t.table_role || 'DIM')}</span>
            ${t.source_db ? `<span class="er3d-db-pill">${esc(t.source_db)}</span>` : ''}
            <span class="er3d-rows-pill">${Number(t.row_count || 0).toLocaleString()} r</span>
          </div>
          <div class="er3d-plate-title">${esc(t.name)}</div>
          <div class="er3d-plate-cols">${colRows}</div>
          ${extraCount > 0 ? `<div class="er3d-plate-more">+ ${extraCount} more columns</div>` : ''}
        </div>
      `;
    })
    .join('');

  const reduced = isReducedMotionPreferred();
  const initialTransform = reduced
    ? `scale(${_orbitState.zoom})`
    : `perspective(1200px) rotateX(${_orbitState.rotX}deg) rotateZ(${_orbitState.rotZ}deg) scale(${_orbitState.zoom})`;

  viewport.innerHTML = `
    <div class="er3d-iso-shell">
      <div class="er3d-iso-hud">
        <span class="er3d-hud-hint">Drag anywhere to orbit · Wheel to zoom · Click any plate to focus joins</span>
        <div class="er3d-hud-btns">
          <button type="button" class="btn btn-ghost btn-xs" data-er3d-action="zoom-out" title="Zoom out">-</button>
          <button type="button" class="btn btn-ghost btn-xs" data-er3d-action="reset" title="Reset 3D angle">Reset 2.5D</button>
          <button type="button" class="btn btn-ghost btn-xs" data-er3d-action="zoom-in" title="Zoom in">+</button>
        </div>
      </div>
      <div class="er3d-iso-stage-scroller">
        <div
          id="er3dIsoPlane"
          class="er3d-iso-plane ${reduced ? 'reduced-motion' : ''}"
          style="width:${stageW}px; height:${stageH}px; transform:${initialTransform};"
        >
          <div class="er3d-grid-floor"></div>
          <svg class="er3d-edges-svg" viewBox="0 0 ${stageW} ${stageH}" width="${stageW}" height="${stageH}" aria-hidden="true">
            ${edgesSvg}
          </svg>
          ${cardsHtml}
        </div>
      </div>
    </div>
  `;

  const plane = viewport.querySelector('#er3dIsoPlane');
  const applyTransform = () => {
    if (!plane) return;
    if (isReducedMotionPreferred()) {
      plane.style.transform = `scale(${_orbitState.zoom.toFixed(2)})`;
      return;
    }
    plane.style.transform = `perspective(1200px) rotateX(${_orbitState.rotX.toFixed(1)}deg) rotateZ(${_orbitState.rotZ.toFixed(
      1
    )}deg) scale(${_orbitState.zoom.toFixed(2)})`;
  };

  let dragging = false;
  let moved = false;
  let lastX = 0;
  let lastY = 0;

  const scroller = viewport.querySelector('.er3d-iso-stage-scroller');
  scroller?.addEventListener('pointerdown', (e) => {
    if (e.target.closest('button')) return;
    dragging = true;
    moved = false;
    lastX = e.clientX;
    lastY = e.clientY;
    scroller.classList.add('is-dragging');
  });

  window.addEventListener(
    'pointermove',
    (e) => {
      if (!dragging) return;
      const dx = e.clientX - lastX;
      const dy = e.clientY - lastY;
      if (Math.abs(dx) > 2 || Math.abs(dy) > 2) moved = true;
      lastX = e.clientX;
      lastY = e.clientY;
      _orbitState.rotZ = Math.max(-55, Math.min(55, _orbitState.rotZ + dx * 0.28));
      _orbitState.rotX = Math.max(8, Math.min(62, _orbitState.rotX - dy * 0.25));
      applyTransform();
    },
    { passive: true }
  );

  window.addEventListener('pointerup', () => {
    if (!dragging) return;
    dragging = false;
    scroller?.classList.remove('is-dragging');
  });

  scroller?.addEventListener(
    'wheel',
    (e) => {
      if (!e.ctrlKey && Math.abs(e.deltaY) < 4) return;
      e.preventDefault();
      const delta = e.deltaY > 0 ? -0.07 : 0.07;
      _orbitState.zoom = Math.max(0.65, Math.min(1.45, _orbitState.zoom + delta));
      applyTransform();
    },
    { passive: false }
  );

  viewport.querySelectorAll('[data-er3d-action]').forEach((btn) => {
    btn.addEventListener('click', () => {
      const act = btn.getAttribute('data-er3d-action');
      if (act === 'zoom-in') _orbitState.zoom = Math.min(1.45, _orbitState.zoom + 0.12);
      else if (act === 'zoom-out') _orbitState.zoom = Math.max(0.65, _orbitState.zoom - 0.12);
      else _orbitState = { rotX: 34, rotZ: -18, zoom: 1 };
      applyTransform();
    });
  });

  viewport.querySelectorAll('.er3d-table-plate').forEach((plate) => {
    plate.addEventListener('click', () => {
      if (moved) return;
      const name = plate.getAttribute('data-table-name');
      if (name && typeof onSelectFocal === 'function') {
        onSelectFocal(name);
      }
    });
  });
}

async function renderOrbit3D(viewport, ctx) {
  const seq = ++_renderSeq;
  destroyER3D();

  const { tables, relationships, focalTable, theme, onSelectFocal, onFallbackIso, onDiagnostic } = ctx;
  const visibleTables = getVisibleSubgraph(tables, relationships, focalTable);

  viewport.innerHTML = `
    <div class="er3d-loading-banner">
      <div class="dt-pulse-dot"></div>
      <span>Lazy-loading WebGL 3D Constellation engine (zero page-load penalty)…</span>
    </div>
  `;

  let ForceGraphFactory = typeof window !== 'undefined' ? window.ForceGraph3D : null;
  if (!ForceGraphFactory) {
    try {
      const mod = await import(/* @vite-ignore */ FORCE_GRAPH_CDN);
      ForceGraphFactory = mod.default || mod.ForceGraph3D || window.ForceGraph3D;
    } catch (err) {
      if (seq !== _renderSeq) return;
      const diag = {
        code: 'ERR_WEBGL_ORBIT_LOAD',
        category: 'render',
        failed_node: 'er3d_orbit_constellation',
        technical_detail: err?.message || String(err),
        remediation_hint:
          '3D Orbit Constellation requires esm.sh CDN / WebGL connectivity. Switch back to zero-dependency 2.5D Isometric Stack.',
        retry_strategy: 'open_joins',
      };
      if (typeof onDiagnostic === 'function') onDiagnostic(diag);
      viewport.innerHTML = `
        <div class="er3d-fallback-box">
          <div class="er3d-fallback-title">3D Orbit Constellation Offline / Unavailable</div>
          <p class="muted small">${esc(diag.technical_detail)}</p>
          <button type="button" class="btn btn-primary btn-sm" id="btnFallbackToIso">Switch to 2.5D Isometric Stack</button>
        </div>
      `;
      viewport.querySelector('#btnFallbackToIso')?.addEventListener('click', () => {
        setERViewMode('iso');
        if (typeof onFallbackIso === 'function') onFallbackIso();
      });
      return;
    }
  }

  if (seq !== _renderSeq || !ForceGraphFactory) return;

  const nodeSet = new Set(visibleTables.map((t) => t.name));
  const nodes = visibleTables.map((t) => ({
    id: t.name,
    role: t.table_role || 'DIM',
    rows: t.row_count || 0,
    cols: (t.columns || []).length,
    sourceDb: t.source_db || 'warehouse',
    val: t.table_role === 'FACT' ? 14 : 7,
  }));
  const links = relationships
    .filter((r) => nodeSet.has(r.source_table) && nodeSet.has(r.target_table))
    .map((r) => ({
      source: r.source_table,
      target: r.target_table,
      label: `${r.source_column} = ${r.target_column}`,
      approved: r.status === 'approved',
    }));

  viewport.innerHTML = '<div id="er3dWebGLCanvas" class="er3d-webgl-canvas"></div>';
  const canvasHost = viewport.querySelector('#er3dWebGLCanvas');
  if (!canvasHost) return;

  try {
    const isLight = theme === 'light';
    const graph = ForceGraphFactory()(canvasHost)
      .width(canvasHost.clientWidth || 780)
      .height(440)
      .backgroundColor(isLight ? '#f8fafc' : '#0b0f17')
      .graphData({ nodes, links })
      .nodeLabel((n) => `${n.id} (${n.role} · ${Number(n.rows).toLocaleString()} rows · ${n.cols} cols)`)
      .nodeColor((n) => (n.role === 'FACT' ? '#3b82f6' : focalTable === n.id ? '#a855f7' : '#10b981'))
      .nodeRelSize(6)
      .linkColor((l) => (l.approved ? '#10b981' : '#f59e0b'))
      .linkWidth((l) => (l.approved ? 2.2 : 1.2))
      .linkDirectionalParticles((l) => (l.approved ? 2 : 0))
      .linkDirectionalParticleSpeed(0.006)
      .onNodeClick((node) => {
        if (node?.id && typeof onSelectFocal === 'function') {
          onSelectFocal(node.id);
        }
      });

    _active3DGraph = graph;
  } catch (err) {
    if (typeof onDiagnostic === 'function') {
      onDiagnostic({
        code: 'ERR_WEBGL_INIT',
        category: 'render',
        failed_node: 'er3d_webgl_canvas',
        technical_detail: err?.message || String(err),
        remediation_hint: 'WebGL context failed to initialize. Switching cleanly to 2.5D Isometric Stack.',
        retry_strategy: 'open_joins',
      });
    }
    setERViewMode('iso');
    renderIsoStack(viewport, ctx);
  }
}

/**
 * Main dispatcher called by renderERDiagram() in app.js.
 * Returns true if handled ('iso' or 'orbit3d'), or false so caller executes native 2D SVG render.
 */
export function renderER(viewport, ctx) {
  if (!viewport) return false;
  if (_currentMode === '2d') {
    destroyER3D();
    return false;
  }
  if (_currentMode === 'orbit3d') {
    renderOrbit3D(viewport, ctx);
    return true;
  }
  renderIsoStack(viewport, ctx);
  return true;
}

export { renderER as render };
