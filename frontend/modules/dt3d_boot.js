/**
 * dt3d_boot.js — Deferred ES Module bootstrapper exposing Data_Talk's 2026 3D & Diagnostic stack on window.DT3D.
 * Zero heavy CDN imports on initial page load (<100ms critical path preserved).
 */

import * as tilt from './spatial_tilt.js';
import * as errorBoundary from './error_boundary.js';
import * as er3d from './er3d_engine.js';
import * as langgraph from './langgraph_3d_trace.js';
import * as plotly3d from './plotly_3d_surfaces.js';

function dispatchDiagnosticRecoveryAction(action, ctx = {}) {
  const question = ctx.question || '';
  if (action === 'retry-recompile' && question && typeof window.submitChatQuestion === 'function') {
    window.submitChatQuestion(question, { force_recompile: true });
  } else if (action === 'force-heuristic' && question && typeof window.submitChatQuestion === 'function') {
    window.submitChatQuestion(question, { force_heuristic: true });
  } else if (action === 'retry-sync' && question && typeof window.submitChatQuestion === 'function') {
    window.submitChatQuestion(question, { disable_stream: true });
  } else if (action === 'open-joins' && typeof window.switchView === 'function') {
    window.switchView('joins');
  } else if (action === 'open-llm') {
    document.getElementById('openLlmModalBtn')?.click();
  } else if (action === 'open-scratchpad') {
    document.getElementById('scratchpadDialog')?.showModal();
  }
}

window.DT3D = {
  version: '2.11.0',
  tilt,
  errorBoundary,
  er3d,
  langgraph,
  plotly3d,
  dispatchDiagnosticRecoveryAction,
};

tilt.initSpatialTilt();
tilt.refreshTiltCards(document);
errorBoundary.installGlobalErrorBoundary({
  onAction: dispatchDiagnosticRecoveryAction,
});

window.dispatchEvent(new CustomEvent('dt3d:ready', { detail: window.DT3D }));
