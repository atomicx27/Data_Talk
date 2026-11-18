/**
 * spatial_tilt.js — Lightweight delegated 2026 spatial 3D tilt & specular glare micro-interactions.
 * Uses a single delegated document listener + requestAnimationFrame; zero overhead on dynamic DOM updates.
 */

const REDUCED_MOTION_QUERY =
  typeof window !== 'undefined' && window.matchMedia
    ? window.matchMedia('(prefers-reduced-motion: reduce)')
    : null;

let _activeCard = null;
let _rafId = 0;
let _pendingPointer = null;
let _installed = false;

export function isReducedMotionPreferred() {
  if (typeof document !== 'undefined' && document.documentElement.getAttribute('data-motion') === 'reduced') {
    return true;
  }
  return Boolean(REDUCED_MOTION_QUERY && REDUCED_MOTION_QUERY.matches);
}

function _applyTiltFrame() {
  _rafId = 0;
  if (!_activeCard || !_pendingPointer || isReducedMotionPreferred()) return;
  const rect = _activeCard.getBoundingClientRect();
  if (!rect.width || !rect.height) return;

  const px = Math.min(1, Math.max(0, (_pendingPointer.x - rect.left) / rect.width));
  const py = Math.min(1, Math.max(0, (_pendingPointer.y - rect.top) / rect.height));
  const maxDeg = Number(_activeCard.dataset.tiltMax || 5.5);

  const rx = (0.5 - py) * maxDeg;
  const ry = (px - 0.5) * maxDeg;

  _activeCard.style.setProperty('--dt-rx', `${rx.toFixed(2)}deg`);
  _activeCard.style.setProperty('--dt-ry', `${ry.toFixed(2)}deg`);
  _activeCard.style.setProperty('--dt-mx', `${(px * 100).toFixed(1)}%`);
  _activeCard.style.setProperty('--dt-my', `${(py * 100).toFixed(1)}%`);
  _activeCard.classList.add('dt-tilting');
}

function _resetCard(card) {
  if (!card) return;
  card.classList.remove('dt-tilting');
  card.style.removeProperty('--dt-rx');
  card.style.removeProperty('--dt-ry');
  card.style.removeProperty('--dt-mx');
  card.style.removeProperty('--dt-my');
}

export function initSpatialTilt() {
  if (_installed || typeof document === 'undefined') return;
  _installed = true;

  const setup = () => {
    document.addEventListener(
      'pointermove',
      (e) => {
        if (e.pointerType === 'touch' || isReducedMotionPreferred()) return;
        const target = e.target && e.target.closest ? e.target.closest('.dt-tilt-card') : null;
        if (target !== _activeCard) {
          _resetCard(_activeCard);
          _activeCard = target;
        }
        if (!_activeCard) return;
        _pendingPointer = { x: e.clientX, y: e.clientY };
        if (!_rafId) {
          _rafId = window.requestAnimationFrame(_applyTiltFrame);
        }
      },
      { passive: true }
    );

    document.addEventListener(
      'pointerleave',
      () => {
        if (_activeCard) {
          _resetCard(_activeCard);
          _activeCard = null;
        }
      },
      { passive: true }
    );

    window.addEventListener(
      'blur',
      () => {
        if (_activeCard) {
          _resetCard(_activeCard);
          _activeCard = null;
        }
      },
      { passive: true }
    );
  };

  if (typeof window.requestIdleCallback === 'function') {
    window.requestIdleCallback(setup, { timeout: 250 });
  } else {
    setup();
  }
}

export function refreshTiltCards(root = document) {
  if (!root || isReducedMotionPreferred()) return;
  // Ensures opted-in selector cards participate cleanly without touching large data-table containers.
  root.querySelectorAll('.starter-card, .kpi-bento-card, .workbench-card, .stat-tile').forEach((el) => {
    el.classList.add('dt-tilt-card');
  });
}
