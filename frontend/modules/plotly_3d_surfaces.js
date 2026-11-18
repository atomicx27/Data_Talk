/**
 * plotly_3d_surfaces.js — Interactive Plotly WebGL 3D Sensitivity Surface (What-If)
 * and Chat Card 3D Scatter / 3D Surface visuals with strict Plotly.purge() lifecycle.
 */

function purgeContainer(container) {
  if (!container || typeof window === 'undefined' || !window.Plotly) return;
  try {
    window.Plotly.purge(container);
  } catch (_) {
    // ignore if container was not yet initialized by Plotly
  }
}

function attachWebGLHandler(container, onDiagnostic) {
  if (!container || container.dataset.dtWebglBound === '1') return;
  container.dataset.dtWebglBound = '1';
  container.addEventListener(
    'webglcontextlost',
    (e) => {
      e.preventDefault();
      if (typeof onDiagnostic === 'function') {
        onDiagnostic({
          error_code: 'ERR_PLOTLY_WEBGL_LOST',
          error_category: 'internal',
          failed_node: 'plotly_3d_surface',
          technical_detail: 'Plotly WebGL 3D scene context lost.',
          remediation_hint: 'Switch back to a 2D chart type or reload if GPU memory is exhausted.',
        });
      }
    },
    true
  );
}

/**
 * Check whether an entry's result rows support interactive Plotly 3D views.
 */
export function supportsChat3D(entry) {
  if (!entry || !Array.isArray(entry.rows) || entry.rows.length < 2) {
    return { scatter3d: false, surface3d: false };
  }
  const cols = Array.isArray(entry.columns) && entry.columns.length ? entry.columns : Object.keys(entry.rows[0] || {});
  if (cols.length < 3) return { scatter3d: false, surface3d: false };

  const sample = entry.rows.slice(0, 20);
  const numericCols = cols.filter((c) => {
    let numCount = 0;
    for (const row of sample) {
      const v = row[c];
      if (v !== null && v !== undefined && v !== '' && !Number.isNaN(Number(v))) numCount++;
    }
    return numCount >= Math.max(1, Math.floor(sample.length * 0.6));
  });

  return {
    scatter3d: cols.length >= 3 && numericCols.length >= 2,
    surface3d: (numericCols.length >= 3 && entry.rows.length >= 4) || (numericCols.length >= 1 && cols.length >= 3 && entry.rows.length >= 4),
  };
}

/**
 * Render Chat Card 'scatter3d' or 'surface3d' using Plotly WebGL.
 */
export function renderChatPlotly3D(container, entry, chartType, options = {}) {
  if (!container || typeof window === 'undefined' || !window.Plotly) return false;
  const rows = entry?.rows || [];
  const cols = Array.isArray(entry?.columns) && entry.columns.length ? entry.columns : Object.keys(rows[0] || {});
  if (rows.length < 2 || cols.length < 3) return false;

  const sample = rows.slice(0, 25);
  const numericCols = [];
  const labelCols = [];
  for (const c of cols) {
    const isNum = sample.some((r) => r[c] !== null && r[c] !== undefined && r[c] !== '' && !Number.isNaN(Number(r[c])));
    if (isNum) numericCols.push(c);
    else labelCols.push(c);
  }
  if (numericCols.length < 2 && chartType === 'scatter3d') return false;

  const isLight = typeof options.getActiveTheme === 'function' ? options.getActiveTheme() === 'light' : false;
  const axisColor = isLight ? '#334155' : '#cbd5e1';
  const gridColor = isLight ? '#cbd5e1' : 'rgba(255,255,255,0.12)';
  const themeLayoutFn = options.applyPlotlyThemeLayout || ((l) => l);

  purgeContainer(container);
  attachWebGLHandler(container, options.onDiagnostic);

  let traces = [];
  let sceneLabels = {};

  if (chartType === 'surface3d') {
    if (labelCols.length >= 2 && numericCols.length >= 1) {
      const xCol = labelCols[0];
      const yCol = labelCols[1];
      const zCol = numericCols[0];
      const xCats = [...new Set(rows.map((r) => String(r[xCol] ?? '')))].slice(0, 25);
      const yCats = [...new Set(rows.map((r) => String(r[yCol] ?? '')))].slice(0, 25);
      const lookup = new Map();
      rows.forEach((r) => {
        const key = `${r[xCol]}__${r[yCol]}`;
        lookup.set(key, (lookup.get(key) || 0) + Number(r[zCol] || 0));
      });
      const zMatrix = yCats.map((yv) => xCats.map((xv) => lookup.get(`${xv}__${yv}`) || 0));
      traces = [
        {
          type: 'surface',
          x: xCats,
          y: yCats,
          z: zMatrix,
          colorscale: 'Viridis',
          showscale: true,
        },
      ];
      sceneLabels = { xaxis: { title: xCol }, yaxis: { title: yCol }, zaxis: { title: zCol } };
    } else {
      // Build smoothed grid surface across first 2 numeric dimensions + Z value
      const xCol = numericCols[0];
      const yCol = numericCols[1];
      const zCol = numericCols[2] || numericCols[0];
      const xs = rows.slice(0, 20).map((r) => Number(r[xCol] || 0));
      const ys = rows.slice(0, 20).map((r) => Number(r[yCol] || 0));
      const zs = rows.slice(0, 20).map((r, ri) =>
        xs.map((_, ci) => {
          const rA = Number(rows[ri % rows.length][zCol] || 0);
          const rB = Number(rows[ci % rows.length][zCol] || 0);
          return Number(((rA + rB) / 2).toFixed(2));
        })
      );
      traces = [{ type: 'surface', x: xs, y: ys, z: zs, colorscale: 'Portland', showscale: true }];
      sceneLabels = { xaxis: { title: xCol }, yaxis: { title: yCol }, zaxis: { title: zCol } };
    }
  } else {
    // scatter3d
    const xCol = labelCols[0] || numericCols[0];
    const yCol = labelCols[0] ? numericCols[0] : numericCols[1];
    const zCol = labelCols[0] ? numericCols[1] : numericCols[2] || numericCols[0];
    const labels = rows.map((r) => cols.map((c) => `${c}: ${r[c]}`).join('<br>'));
    traces = [
      {
        type: 'scatter3d',
        mode: 'markers',
        x: rows.map((r) => r[xCol]),
        y: rows.map((r) => Number(r[yCol] || 0)),
        z: rows.map((r) => Number(r[zCol] || 0)),
        text: labels,
        hoverinfo: 'text',
        marker: {
          size: 7,
          color: rows.map((r) => Number(r[zCol] || 0)),
          colorscale: 'Turbo',
          opacity: 0.9,
          line: { width: 1, color: isLight ? '#0f172a' : '#f8fafc' },
        },
      },
    ];
    sceneLabels = { xaxis: { title: xCol }, yaxis: { title: yCol }, zaxis: { title: zCol } };
  }

  const baseLayout = themeLayoutFn({
    margin: { t: 28, r: 16, b: 16, l: 16 },
    height: 360,
    uirevision: `chat-3d-${chartType}`,
    scene: {
      bgcolor: 'rgba(0,0,0,0)',
      xaxis: { ...sceneLabels.xaxis, color: axisColor, gridcolor: gridColor },
      yaxis: { ...sceneLabels.yaxis, color: axisColor, gridcolor: gridColor },
      zaxis: { ...sceneLabels.zaxis, color: axisColor, gridcolor: gridColor },
      camera: { eye: { x: 1.45, y: 1.45, z: 1.15 } },
    },
  });

  window.Plotly.newPlot(container, traces, baseLayout, {
    responsive: true,
    displayModeBar: true,
    displaylogo: false,
  });
  return true;
}

/**
 * Render interactive 3D Sensitivity Response Surface inside What-If Simulator drawer.
 */
export function renderWhatIfSurface3D(container, res, options = {}) {
  if (!container || typeof window === 'undefined' || !window.Plotly) return false;

  const k = res?.kpis || {};
  const revShock = Number(options.revShock ?? 0);
  const costShock = Number(options.costShock ?? 0);

  const baseRev = Number(k.base_revenue || k.sim_revenue || 100000);
  const baseCost = Number(k.base_cost || 0);
  const hasCostTerm = baseCost > 0;

  const baseZ = hasCostTerm ? baseRev - baseCost : baseRev;
  const activeSimZ = hasCostTerm
    ? Number(k.sim_profit !== undefined ? k.sim_profit : baseRev * (1 + revShock / 100) - baseCost * (1 + costShock / 100))
    : Number(k.sim_revenue !== undefined ? k.sim_revenue : baseRev * (1 + revShock / 100));

  const gridPct = [-25, -18, -12, -6, 0, 6, 12, 18, 25];
  const zGrid = gridPct.map((cPct) =>
    gridPct.map((rPct) => {
      if (hasCostTerm) {
        return Math.round(baseRev * (1 + rPct / 100) - baseCost * (1 + cPct / 100));
      }
      return Math.round(baseRev * (1 + rPct / 100 - (cPct / 100) * 0.45));
    })
  );

  const isLight = typeof options.getActiveTheme === 'function' ? options.getActiveTheme() === 'light' : false;
  const axisColor = isLight ? '#334155' : '#cbd5e1';
  const gridColor = isLight ? '#cbd5e1' : 'rgba(255,255,255,0.12)';
  const themeLayoutFn = options.applyPlotlyThemeLayout || ((l) => l);

  purgeContainer(container);
  attachWebGLHandler(container, options.onDiagnostic);

  const isAtBaseline = revShock === 0 && costShock === 0;
  const traces = [
    {
      type: 'surface',
      name: 'Sensitivity Manifold',
      x: gridPct,
      y: gridPct,
      z: zGrid,
      colorscale: 'Viridis',
      opacity: 0.88,
      hovertemplate: 'Revenue Shock: %{x}%<br>Cost Shock: %{y}%<br>Projected: %{z:,.0f}<extra></extra>',
      showscale: false,
    },
    {
      type: 'scatter3d',
      mode: isAtBaseline ? 'markers' : 'markers+text',
      name: 'Baseline (0%, 0%)',
      x: [0],
      y: [0],
      z: [baseZ],
      text: isAtBaseline ? undefined : ['Baseline'],
      textposition: 'bottom center',
      marker: { size: 6, color: '#94a3b8' },
    },
    {
      type: 'scatter3d',
      mode: 'markers+text',
      name: 'Active Scenario',
      x: [revShock],
      y: [costShock],
      z: [activeSimZ],
      text: [
        isAtBaseline
          ? 'Baseline (0%, 0%)'
          : `Selected (${revShock >= 0 ? '+' : ''}${revShock}%, ${costShock >= 0 ? '+' : ''}${costShock}%)`,
      ],
      textposition: 'top center',
      marker: { size: 9, color: '#10b981', line: { width: 2, color: '#ffffff' } },
    },
  ];

  const layout = themeLayoutFn({
    title: {
      text: `3D Sensitivity Manifold (${res?.target_metric || 'Projected Outcome'})`,
      font: { size: 13 },
    },
    margin: { t: 40, r: 24, b: 28, l: 24 },
    height: 360,
    uirevision: 'whatif-surface3d',
    showlegend: false,
    scene: {
      bgcolor: 'rgba(0,0,0,0)',
      aspectmode: 'manual',
      aspectratio: { x: 1, y: 1, z: 0.78 },
      xaxis: { title: 'Rev Δ%', color: axisColor, gridcolor: gridColor },
      yaxis: { title: 'Cost Δ%', color: axisColor, gridcolor: gridColor },
      zaxis: { title: hasCostTerm ? 'Projected Net' : 'Outcome', color: axisColor, gridcolor: gridColor },
      camera: { eye: { x: 2.1, y: 1.75, z: 1.38 } },
    },
  });

  window.Plotly.newPlot(container, traces, layout, {
    responsive: true,
    displayModeBar: false,
  });
  return true;
}
