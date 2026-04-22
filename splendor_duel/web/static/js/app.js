// app.js — Game interaction logic, API calls, main loop

import { GEM_NAMES, ABILITY_ICON } from './constants.js';
import { renderBoard } from './board.js';
import { renderCard, renderRoyal } from './cards.js';
import { renderPlayer } from './player.js';

// ── State ────────────────────────────────────────────────────
let G = null;           // current server response
let selectedCell = null; // {r,c} or null (for TakeTokens first click)
let reserveMode = false; // true after clicking gold
let busy = false;        // prevent double-clicks during async
let useImages = false;   // toggle card images (set via UI or config)

const AI_THINK_DELAY = 2000; // ms to show "AI thinking" before requesting AI turn

// ── API helpers ──────────────────────────────────────────────
async function api(url, body) {
  const opts = body !== undefined
    ? { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }
    : { method: 'GET' };
  const r = await fetch(url, opts);
  if (!r.ok) {
    const e = await r.json().catch(() => ({}));
    showToast(e.detail || 'Server error', 2500);
    return null;
  }
  return r.json();
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

// ── Toast ────────────────────────────────────────────────────
function showToast(msg, duration = 1800) {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.classList.add('show');
  clearTimeout(t._timer);
  t._timer = setTimeout(() => t.classList.remove('show'), duration);
}

// ── Game flow ────────────────────────────────────────────────
async function loadAgents() {
  const data = await api('/api/agents');
  if (!data) return;
  const sel = document.getElementById('agentSelect');
  sel.innerHTML = '';
  for (const a of data.agents) {
    const o = document.createElement('option');
    o.value = a; o.textContent = a;
    sel.appendChild(o);
  }
}

async function newGame() {
  const agent = document.getElementById('agentSelect').value;
  resetInteraction();
  busy = true;
  G = await api('/api/new_game', { agent, player_side: 0 });
  if (!G) { busy = false; return; }
  // If AI goes first, trigger AI turn
  if (G.is_ai_turn) {
    render();
    await sleep(1000);
    await runAiTurn();
  }
  // Auto-skip trivial phases
  while (await autoSkipIfTrivial()) {}
  busy = false;
  render();
}

async function sendAction(idx) {
  if (busy) return;
  busy = true;
  resetInteraction();

  G = await api('/api/action', { action_index: idx });
  if (!G) { busy = false; return; }

  // Auto-skip trivial phases (e.g. OPTIONAL with only ProceedToMain)
  while (await autoSkipIfTrivial()) { /* keep skipping */ }

  if (G.is_ai_turn) {
    // Show board after our move, then AI thinking
    render();
    await sleep(AI_THINK_DELAY);
    await runAiTurn();
    // After AI, auto-skip again if needed
    while (await autoSkipIfTrivial()) {}
  }

  // Ready for next human interaction
  busy = false;
  render();
}

async function runAiTurn() {
  setThinking(true);
  G = await api('/api/ai_turn', {});
  setThinking(false);
  if (G) render();
}

function resetInteraction() {
  selectedCell = null;
  reserveMode = false;
}

function setThinking(on) {
  document.getElementById('turnInfo').textContent = on ? 'AI is thinking...' : '';
  const bar = document.getElementById('actionBar');
  if (on) {
    bar.innerHTML = `<div class="thinking-bar">
      <span class="hint-text">Opponent is thinking</span>
      <div class="thinking-dots"><span></span><span></span><span></span></div>
    </div>`;
  }
}

/**
 * Auto-skip trivial phases (e.g. OPTIONAL with only ProceedToMain).
 * Returns true if an auto-action was sent.
 */
async function autoSkipIfTrivial() {
  if (!G || !G.is_human_turn || G.state.is_game_over) return false;
  if (G.legal_actions.length === 1 && G.legal_actions[0].type === 'ProceedToMain') {
    G = await api('/api/action', { action_index: 0 });
    return true;
  }
  return false;
}

// ── Board interaction ────────────────────────────────────────
function onCellClick(r, c) {
  if (!G || !G.is_human_turn || busy) return;
  const phase = G.state.phase;
  const val = G.state.board[r][c];

  // Gold → enter reserve mode
  if (val === 6) {
    if (phase === 'MAIN' && G.legal_actions.some(a => a.type === 'ReserveCard')) {
      reserveMode = true;
      selectedCell = null;
      render();
    } else {
      showToast('No reserve actions available', 1500);
    }
    return;
  }

  // UseScroll in OPTIONAL
  if (phase === 'OPTIONAL') {
    const idx = G.legal_actions.findIndex(a =>
      a.type === 'UseScroll' && a.position[0] === r && a.position[1] === c);
    if (idx >= 0) { sendAction(idx); return; }
  }

  // EffectTakeSameGem in EFFECT
  if (phase === 'EFFECT') {
    const idx = G.legal_actions.findIndex(a =>
      a.type === 'EffectTakeSameGem' && a.position[0] === r && a.position[1] === c);
    if (idx >= 0) { sendAction(idx); return; }
  }

  // TakeTokens — two-click line
  if (phase === 'MAIN' && val >= 0) {
    reserveMode = false;
    if (!selectedCell) {
      selectedCell = { r, c };
      render();
      return;
    }

    const line = buildLine(selectedCell.r, selectedCell.c, r, c);
    if (line) {
      const idx = findTakeAction(line);
      if (idx >= 0) { sendAction(idx); return; }
    }

    showToast('Invalid selection', 1000);
    selectedCell = { r, c };
    render();
  }
}

function buildLine(r1, c1, r2, c2) {
  if (r1 === r2 && c1 === c2) return [[r1, c1]];
  const dr = Math.sign(r2 - r1), dc = Math.sign(c2 - c1);
  const diffR = Math.abs(r2 - r1), diffC = Math.abs(c2 - c1);
  if (diffR !== diffC && diffR !== 0 && diffC !== 0) return null;
  const steps = Math.max(diffR, diffC);
  if (steps > 2) return null;
  const line = [];
  for (let i = 0; i <= steps; i++) line.push([r1 + i * dr, c1 + i * dc]);
  return line;
}

function findTakeAction(line) {
  const reversed = [...line].reverse();
  return G.legal_actions.findIndex(a => {
    if (a.type !== 'TakeTokens' || a.positions.length !== line.length) return false;
    const fwd = line.every((p, i) => p[0] === a.positions[i][0] && p[1] === a.positions[i][1]);
    if (fwd) return true;
    return reversed.every((p, i) => p[0] === a.positions[i][0] && p[1] === a.positions[i][1]);
  });
}

// ── Card interaction ─────────────────────────────────────────
function onPyramidCard(level, index) {
  if (!G || !G.is_human_turn || G.state.phase !== 'MAIN' || busy) return;
  if (reserveMode) {
    // Reserve this card
    const idx = G.legal_actions.findIndex(a =>
      a.type === 'ReserveCard' && a.source === 'pyramid' && a.level === level && a.index === index);
    if (idx >= 0) sendAction(idx);
    else showToast('Cannot reserve this card', 1200);
    return;
  }
  // Direct buy (no popup)
  const idx = G.legal_actions.findIndex(a =>
    a.type === 'BuyCard' && a.source === 'pyramid' && a.level === level && a.index === index);
  if (idx >= 0) sendAction(idx);
  else showToast('Cannot afford this card', 1200);
}
// Exposed to onclick strings in rendered HTML
window._buyPyramid = (lvl, idx) => onPyramidCard(lvl, idx);
window._reserveDeck = (lvl) => {
  const i = G.legal_actions.findIndex(a =>
    a.type === 'ReserveCard' && a.source === 'deck' && a.level === lvl);
  if (i >= 0) sendAction(i);
};
window._onBuyReserved = (idx) => {
  const i = G.legal_actions.findIndex(a =>
    a.type === 'BuyCard' && a.source === 'reserve' && a.index === idx);
  if (i >= 0) sendAction(i);
};
window._onDiscard = (gemIdx) => {
  const gemName = GEM_NAMES[gemIdx];
  const i = G.legal_actions.findIndex(a => a.type === 'DiscardToken' && a.gem === gemName);
  if (i >= 0) sendAction(i);
};
window._chooseRoyal = (idx) => {
  const i = G.legal_actions.findIndex(a => a.type === 'ChooseRoyal' && a.index === idx);
  if (i >= 0) sendAction(i);
};
window._sendAction = (idx) => sendAction(idx);
window._cancelReserve = () => { reserveMode = false; render(); };
window._cancelSelection = () => { selectedCell = null; render(); };

// ── Main render ──────────────────────────────────────────────
function render() {
  if (!G) return;
  const st = G.state;
  const hp = G.human_player;
  const ai = 1 - hp;

  // Header
  document.getElementById('phaseBadge').textContent = st.phase;
  document.getElementById('turnInfo').textContent =
    st.is_game_over ? 'Game over' :
    G.is_human_turn ? 'Your move' : 'AI turn';

  // Board
  const validCells = new Set();
  const hintCells = new Set();
  if (G.is_human_turn && !busy) {
    for (const a of G.legal_actions) {
      if (a.type === 'TakeTokens') a.positions.forEach(p => validCells.add(`${p[0]},${p[1]}`));
      if (a.type === 'UseScroll') validCells.add(`${a.position[0]},${a.position[1]}`);
      if (a.type === 'EffectTakeSameGem') validCells.add(`${a.position[0]},${a.position[1]}`);
    }
    // Compute hint cells when a cell is selected
    if (selectedCell) {
      for (let r = 0; r < 5; r++) for (let c = 0; c < 5; c++) {
        if (st.board[r][c] >= 0 && st.board[r][c] !== 6) {
          const line = buildLine(selectedCell.r, selectedCell.c, r, c);
          if (line && findTakeAction(line) >= 0) hintCells.add(`${r},${c}`);
        }
      }
    }
  }

  const hasReserveActions = G.is_human_turn && G.legal_actions.some(a => a.type === 'ReserveCard');
  renderBoard(document.getElementById('boardGrid'), st.board, {
    validCells, selectedCell, hintCells,
    goldClickable: hasReserveActions && st.phase === 'MAIN' && !busy,
    goldSelected: reserveMode,
    onCellClick,
    useImages,
  });

  // Board meta
  const sc = st.scrolls_center;
  document.getElementById('scrollsInfo').textContent = `Scrolls: ${'⚜'.repeat(sc)}${'·'.repeat(3 - sc)}`;
  document.getElementById('bagInfo').textContent = `Bag: ${st.bag_total}`;

  // Players
  renderPlayer(document.getElementById('panelLeft'), st.players[hp], {
    label: 'You', isHuman: true, useImages,
    isActive: hp === st.current_player && !st.is_game_over,
    canDiscard: (gem) => G.is_human_turn && st.phase === 'DISCARD'
      && G.legal_actions.some(a => a.type === 'DiscardToken' && a.gem === GEM_NAMES[gem]),
    onDiscard: window._onDiscard,
    canBuyReserved: (idx) => G.is_human_turn && st.phase === 'MAIN'
      && G.legal_actions.some(a => a.type === 'BuyCard' && a.source === 'reserve' && a.index === idx),
    onBuyReserved: window._onBuyReserved,
  });
  renderPlayer(document.getElementById('panelRight'), st.players[ai], {
    label: `AI (${G.agent_name})`, isHuman: false, useImages,
    isActive: ai === st.current_player && !st.is_game_over,
  });

  // Pyramid
  renderPyramidSection(st);

  // Royals
  renderRoyalsSection(st);

  // Action bar
  renderActionBar(st);

  // Log
  renderLog();

  // Game over
  renderGameOver(st);
}

function renderPyramidSection(st) {
  const area = document.getElementById('pyramidArea');
  let h = '';
  for (const lvl of [3, 2, 1]) {
    const cards = st.pyramid[String(lvl)] || [];
    const deckSize = st.deck_sizes[String(lvl)] || 0;
    h += `<div class="pyr-row"><span class="pyr-label">L${lvl}</span>`;
    cards.forEach((c, idx) => {
      const canBuy = !reserveMode && G.is_human_turn && st.phase === 'MAIN' && !busy
        && G.legal_actions.some(a => a.type === 'BuyCard' && a.source === 'pyramid' && a.level === lvl && a.index === idx);
      const canRes = reserveMode && G.is_human_turn && st.phase === 'MAIN' && !busy
        && G.legal_actions.some(a => a.type === 'ReserveCard' && a.source === 'pyramid' && a.level === lvl && a.index === idx);
      const cls = canRes ? 'reservable' : (canBuy ? 'buyable' : '');
      const onclick = (canBuy || canRes)
        ? `event.stopPropagation();window._onPyramidCard(${lvl},${idx})` : '';

      h += renderCard(c, { onclick, extraClass: cls, useImage: useImages });
    });
    // Deck
    const canResDeck = reserveMode && G.is_human_turn && st.phase === 'MAIN' && !busy
      && G.legal_actions.some(a => a.type === 'ReserveCard' && a.source === 'deck' && a.level === lvl);
    if (deckSize > 0) {
      const deckAttr = canResDeck
        ? `onclick="window._reserveDeck(${lvl})" class="deck-count reservable" style="cursor:pointer"`
        : 'class="deck-count"';
      h += `<span ${deckAttr}>(${deckSize}${canResDeck ? ' ⬅' : ''})</span>`;
    }
    h += '</div>';
  }
  area.innerHTML = h;
}
window._onPyramidCard = onPyramidCard;

function renderRoyalsSection(st) {
  const row = document.getElementById('royalsRow');
  let h = '';
  st.royal_cards.forEach((r, idx) => {
    const canChoose = G.is_human_turn && st.phase === 'ROYAL' && !busy
      && G.legal_actions.some(a => a.type === 'ChooseRoyal' && a.index === idx);
    h += renderRoyal(r, {
      onclick: canChoose ? `window._chooseRoyal(${idx})` : '',
      extraClass: canChoose ? 'clickable' : '',
      useImage: useImages,
    });
  });
  if (!st.royal_cards.length) h = '<span class="empty-hint">—</span>';
  row.innerHTML = h;
}

function renderActionBar(st) {
  const bar = document.getElementById('actionBar');
  if (!G.is_human_turn || busy) {
    if (st.is_game_over) bar.innerHTML = '';
    else if (!busy) bar.innerHTML = '<span class="hint-text">Waiting...</span>';
    return;
  }
  const phase = st.phase;
  let h = '';

  if (phase === 'OPTIONAL') {
    h += '<span class="hint-text">Optional: </span>';
    const pi = G.legal_actions.findIndex(a => a.type === 'ProceedToMain');
    if (pi >= 0) h += `<button class="action-btn" onclick="window._sendAction(${pi})">Proceed →</button>`;
    const ri = G.legal_actions.findIndex(a => a.type === 'RefillBoard');
    if (ri >= 0) h += `<button class="action-btn" onclick="window._sendAction(${ri})">Refill Board</button>`;
    if (G.legal_actions.some(a => a.type === 'UseScroll'))
      h += '<span class="hint-text"> · click board to use scroll</span>';
  } else if (phase === 'MAIN') {
    if (reserveMode) {
      h += '<span class="hint-text">Reserve mode — click a card to reserve</span>';
      h += '<button class="action-btn" onclick="window._cancelReserve()">Cancel</button>';
    } else {
      h += '<span class="hint-text">Click board to take tokens · Gold to reserve · Cards to buy</span>';
      if (selectedCell) h += '<button class="action-btn" onclick="window._cancelSelection()">Cancel</button>';
    }
  } else if (phase === 'EFFECT') {
    const t = G.legal_actions[0]?.type || '';
    if (t === 'EffectTakeSameGem') h += '<span class="hint-text">Click a matching gem on the board</span>';
    else if (t === 'EffectTakeOpponentGem') {
      h += '<span class="hint-text">Steal from opponent: </span>';
      G.legal_actions.forEach((a, i) => {
        if (a.type === 'EffectTakeOpponentGem')
          h += `<button class="action-btn" onclick="window._sendAction(${i})">${a.gem}</button>`;
      });
    } else if (t === 'EffectChooseWildcard') {
      h += '<span class="hint-text">Place wildcard on: </span>';
      G.legal_actions.forEach((a, i) => {
        if (a.type === 'EffectChooseWildcard')
          h += `<button class="action-btn" onclick="window._sendAction(${i})">${a.target_card_id}</button>`;
      });
    } else if (t === 'EffectSkip') {
      h += `<button class="action-btn" onclick="window._sendAction(0)">Skip (no targets)</button>`;
    }
  } else if (phase === 'ROYAL') {
    h += '<span class="hint-text">Choose a royal card above</span>';
  } else if (phase === 'DISCARD') {
    h += '<span class="hint-text">Click a token in your panel to discard</span>';
  }
  bar.innerHTML = h;
}

function renderLog() {
  const area = document.getElementById('logArea');
  const entries = G.log || [];
  area.innerHTML = entries.map(e => `<div class="log-entry">${e}</div>`).join('');
  area.parentElement.scrollTop = area.parentElement.scrollHeight;
}

function renderGameOver(st) {
  const el = document.getElementById('gameOver');
  if (!st.is_game_over) { el.classList.add('hidden'); return; }
  el.classList.remove('hidden');
  const winner = st.winner;
  const isHumanWin = winner === G.human_player;
  el.innerHTML = `
    <h2>${isHumanWin ? '🎉 You win!' : '😔 AI wins'}</h2>
    <p>Score: ${st.players[G.human_player].points} — ${st.players[1 - G.human_player].points}</p>
    <button onclick="window._newGame()" class="new-game-btn">Play Again</button>`;
}

// Toggle images
window._toggleImages = () => { useImages = !useImages; if (G) render(); };

// ── Init ─────────────────────────────────────────────────────
window._newGame = newGame;
loadAgents();