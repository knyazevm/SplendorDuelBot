"""
html_renderer.py — Generate a standalone HTML replay of a game.

Usage:
    from splendor_duel.viz.html_renderer import render_html
    render_html(game_log, "replay.html")
    # open replay.html in browser — step through with arrow keys
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .replay import GameLog, log_to_json_data


def render_html(
        log: GameLog,
        output_path: str = "replay.html",
        title: str = "Splendor Duel Replay",
) -> str:
    """
    Generate a self-contained HTML file with a visual game replay.

    Returns the output file path.
    """
    steps_json = json.dumps(log_to_json_data(log), ensure_ascii=False)

    winner = log.winner
    winner_text = f"Player {winner} wins!" if winner is not None else "Game in progress"
    n_steps = log.n_steps

    html = _HTML_TEMPLATE.replace('__STEPS_JSON__', steps_json)
    html = html.replace('__TITLE__', title)
    html = html.replace('__WINNER__', winner_text)
    html = html.replace('__N_STEPS__', str(n_steps))

    Path(output_path).write_text(html, encoding='utf-8')
    return output_path


# ══════════════════════════════════════════════════════════════════════════════

_HTML_TEMPLATE = r'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>__TITLE__</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #1a1a2e; color: #e0e0e0; min-height: 100vh; }
.app { max-width: 1100px; margin: 0 auto; padding: 16px; }

/* Header */
.header { display: flex; justify-content: space-between; align-items: center; padding: 12px 20px; background: #16213e; border-radius: 12px; margin-bottom: 12px; }
.header h1 { font-size: 18px; font-weight: 600; color: #e8b84b; }
.turn-info { font-size: 14px; color: #a0a0b8; }
.phase-badge { display: inline-block; padding: 3px 10px; border-radius: 6px; font-size: 12px; font-weight: 600; background: #0f3460; color: #53a8ff; }

/* Main grid */
.main-grid { display: grid; grid-template-columns: 1fr 280px 1fr; gap: 12px; margin-bottom: 12px; }

/* Player panel */
.player-panel { background: #16213e; border-radius: 12px; padding: 14px; border: 2px solid transparent; }
.player-panel.active { border-color: #e8b84b; }
.player-name { font-size: 15px; font-weight: 600; margin-bottom: 10px; }
.player-name .star { color: #e8b84b; }
.stat-row { display: flex; gap: 16px; margin-bottom: 6px; font-size: 13px; }
.stat-label { color: #888; }
.stat-val { font-weight: 600; }
.token-row { display: flex; gap: 4px; flex-wrap: wrap; margin: 6px 0; }
.token-pill { display: inline-flex; align-items: center; gap: 3px; padding: 2px 8px; border-radius: 10px; font-size: 12px; font-weight: 600; }
.scroll-icon { color: #e8b84b; font-size: 16px; }
.card-count { font-size: 12px; color: #888; margin-top: 6px; }
.card-list { font-size: 11px; color: #aaa; margin-top: 4px; line-height: 1.5; max-height: 120px; overflow-y: auto; }
.royal-badge { display: inline-block; padding: 2px 8px; background: #4a1942; color: #d946ef; border-radius: 6px; font-size: 11px; margin: 2px; }

/* Board */
.board-wrap { background: #16213e; border-radius: 12px; padding: 14px; text-align: center; }
.board-title { font-size: 13px; color: #888; margin-bottom: 8px; }
.board-grid { display: inline-grid; grid-template-columns: repeat(5, 48px); gap: 4px; }
.cell { width: 48px; height: 48px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 16px; border: 2px solid transparent; }
.cell.empty { background: #252540; border: 1px dashed #333; }
.cell-white { background: #d4d4c8; color: #333; border-color: #aaa; }
.cell-black { background: #333; color: #ccc; border-color: #555; }
.cell-red { background: #c0392b; color: #fff; border-color: #e74c3c; }
.cell-blue { background: #2c6fbb; color: #fff; border-color: #3d8bfd; }
.cell-green { background: #27ae60; color: #fff; border-color: #2ecc71; }
.cell-pearl { background: #7ec8e3; color: #1a1a2e; border-color: #a8d8ea; }
.cell-gold { background: #d4a017; color: #1a1a2e; border-color: #f1c40f; }
.board-meta { margin-top: 10px; font-size: 12px; color: #888; }
.scrolls-display { display: inline-flex; gap: 4px; }
.scroll-pip { width: 14px; height: 14px; border-radius: 50%; display: inline-block; }
.scroll-pip.on { background: #e8b84b; }
.scroll-pip.off { background: #333; border: 1px solid #555; }

/* Pyramid */
.pyramid-wrap { background: #16213e; border-radius: 12px; padding: 14px; margin-bottom: 12px; }
.pyramid-title { font-size: 13px; color: #888; margin-bottom: 8px; }
.pyramid-row { display: flex; gap: 6px; margin-bottom: 6px; align-items: center; }
.level-label { width: 24px; font-size: 12px; font-weight: 600; text-align: center; padding: 2px 4px; border-radius: 4px; }
.level-1 { background: #1b4332; color: #52b788; }
.level-2 { background: #5c4b17; color: #e8b84b; }
.level-3 { background: #1b2a5c; color: #6ea8fe; }
.card-chip { display: inline-flex; align-items: center; gap: 4px; padding: 4px 8px; background: #252540; border-radius: 6px; font-size: 11px; border: 1px solid #333; cursor: default; position: relative; }
.card-chip:hover { border-color: #666; background: #2a2a50; }
.card-chip .pts { font-weight: 700; color: #e8b84b; }
.card-chip .cr { color: #d946ef; }
.card-chip .bon { font-weight: 600; }
.card-chip .eff { color: #53a8ff; }
.deck-count { font-size: 11px; color: #555; margin-left: 4px; }

/* Royals row */
.royals-wrap { display: flex; gap: 8px; align-items: center; margin-bottom: 12px; }
.royals-label { font-size: 13px; color: #888; }
.royal-chip { padding: 6px 12px; background: #2a1a3e; border: 1px solid #5a3d7a; border-radius: 8px; font-size: 12px; color: #d4a0ff; }

/* Navigation */
.nav { display: flex; align-items: center; gap: 12px; padding: 12px 20px; background: #16213e; border-radius: 12px; }
.nav button { padding: 8px 18px; border: 1px solid #333; background: #252540; color: #ddd; border-radius: 8px; cursor: pointer; font-size: 13px; }
.nav button:hover { background: #333; }
.nav button:active { transform: scale(0.97); }
.nav button.primary { background: #0f3460; border-color: #1a5276; color: #53a8ff; }
.progress-wrap { flex: 1; }
.progress-bar { height: 4px; background: #252540; border-radius: 2px; overflow: hidden; }
.progress-fill { height: 100%; background: #e8b84b; transition: width 0.15s; }
.step-text { font-size: 12px; color: #888; margin-top: 4px; }
.action-desc { font-size: 13px; color: #e8b84b; margin-top: 8px; padding: 8px 12px; background: #1a1a2e; border-left: 3px solid #e8b84b; border-radius: 0 6px 6px 0; }
.winner-banner { text-align: center; padding: 16px; background: linear-gradient(135deg, #1b4332, #16213e); border-radius: 12px; margin-top: 12px; font-size: 18px; font-weight: 600; color: #52b788; }
</style>
</head>
<body>
<div class="app" id="app">
  <div class="header">
    <h1>__TITLE__</h1>
    <div class="turn-info" id="turnInfo"></div>
  </div>
  <div class="main-grid">
    <div class="player-panel" id="panel0"></div>
    <div class="board-wrap" id="boardWrap"></div>
    <div class="player-panel" id="panel1"></div>
  </div>
  <div class="pyramid-wrap" id="pyramidWrap"></div>
  <div class="royals-wrap" id="royalsWrap"></div>
  <div class="nav" id="nav">
    <button onclick="goTo(0)">⏮</button>
    <button onclick="step(-1)">◀ Prev</button>
    <div class="progress-wrap">
      <div class="progress-bar"><div class="progress-fill" id="progFill"></div></div>
      <div class="step-text" id="stepText"></div>
    </div>
    <button onclick="step(1)">Next ▶</button>
    <button onclick="goTo(steps.length-1)">⏭</button>
    <button class="primary" id="playBtn" onclick="togglePlay()">▶ Play</button>
  </div>
  <div id="actionDesc" class="action-desc" style="display:none"></div>
  <div id="winnerBanner" style="display:none"></div>
</div>

<script>
const steps = __STEPS_JSON__;
let cur = 0;
let playing = false;
let playTimer = null;

const GEM_NAMES = ['white','black','red','blue','green','pearl','gold'];
const GEM_LETTERS = {white:'W',black:'K',red:'R',blue:'B',green:'G',pearl:'P',gold:'$'};
const GEM_CSS = {white:'cell-white',black:'cell-black',red:'cell-red',blue:'cell-blue',green:'cell-green',pearl:'cell-pearl',gold:'cell-gold'};
const BON_CSS = {white:'#d4d4c8',black:'#666',red:'#e74c3c',blue:'#3d8bfd',green:'#2ecc71',pearl:'#7ec8e3',gold:'#f1c40f'};

function render() {
  const d = steps[cur];
  const s = d.state;

  document.getElementById('turnInfo').innerHTML =
    `Turn ${s.turn} &middot; <span class="phase-badge">${s.phase}</span>`;

  renderBoard(s);
  renderPlayer(0, s);
  renderPlayer(1, s);
  renderPyramid(s);
  renderRoyals(s);

  const pct = steps.length > 1 ? (cur / (steps.length - 1) * 100) : 100;
  document.getElementById('progFill').style.width = pct + '%';
  document.getElementById('stepText').textContent = `Step ${cur} / ${steps.length - 1}`;

  const ad = document.getElementById('actionDesc');
  if (d.description) { ad.style.display = 'block'; ad.textContent = `▸ ${d.description}`; }
  else { ad.style.display = 'none'; }

  const wb = document.getElementById('winnerBanner');
  if (s.phase === 'GAME_OVER') {
    wb.style.display = 'block';
    const w = s.players[0].points > s.players[1].points ? 0 : 1;
    wb.textContent = `Player ${w} wins!`;
    wb.className = 'winner-banner';
  } else { wb.style.display = 'none'; }
}

function renderBoard(s) {
  let h = '<div class="board-title">Board</div><div class="board-grid">';
  for (let r = 0; r < 5; r++) for (let c = 0; c < 5; c++) {
    const v = s.board[r][c];
    if (v === -1) h += '<div class="cell empty"></div>';
    else { const name = GEM_NAMES[v]; h += `<div class="cell ${GEM_CSS[name]}">${GEM_LETTERS[name]}</div>`; }
  }
  h += '</div><div class="board-meta"><span>Scrolls: </span><span class="scrolls-display">';
  for (let i = 0; i < 3; i++) h += `<span class="scroll-pip ${i < s.scrolls_center ? 'on' : 'off'}"></span>`;
  h += `</span> &middot; Bag: ${s.bag_total}</div>`;
  document.getElementById('boardWrap').innerHTML = h;
}

function tokenPills(tokens) {
  let h = '';
  GEM_NAMES.forEach(name => {
    const v = tokens[name] || 0;
    if (v > 0) h += `<span class="token-pill" style="background:${BON_CSS[name]}33;color:${BON_CSS[name]}">${GEM_LETTERS[name]}:${v}</span>`;
  });
  return h;
}

function renderPlayer(idx, s) {
  const p = s.players[idx];
  const active = s.current_player === idx;
  const el = document.getElementById('panel' + idx);
  el.className = 'player-panel' + (active ? ' active' : '');

  let h = `<div class="player-name">Player ${idx}${active ? ' <span class="star">★</span>' : ''}</div>`;
  h += `<div class="stat-row"><span class="stat-label">Points:</span> <span class="stat-val">${p.points}</span>`;
  h += ` &nbsp; <span class="stat-label">Crowns:</span> <span class="stat-val">${p.crowns}</span></div>`;
  h += `<div style="font-size:12px;color:#888;margin:4px 0">Tokens (${p.total_tokens})</div>`;
  h += `<div class="token-row">${tokenPills(p.tokens)}</div>`;
  h += `<div style="font-size:12px;color:#888;margin:4px 0">Bonuses</div>`;
  h += `<div class="token-row">${tokenPills(p.bonuses)}</div>`;
  h += `<div class="card-count">Scrolls: ${'⚜'.repeat(p.scrolls)} &middot; Cards: ${p.cards.length} &middot; Reserved: ${p.reserved.length}</div>`;

  if (p.royals.length) {
    h += '<div style="margin-top:4px">';
    p.royals.forEach(r => { h += `<span class="royal-badge">${r.id} (${r.points}pts)</span>`; });
    h += '</div>';
  }

  if (p.cards.length) {
    h += '<div class="card-list">';
    p.cards.forEach(c => {
      let bon = '';
      if (c.is_wildcard) bon = '★';
      else if (c.gem_bonus) { for (const[k,v] of Object.entries(c.gem_bonus)) if (v>0) bon = `<span style="color:${BON_CSS[k]}">${GEM_LETTERS[k].repeat(v)}</span>`; }
      h += `<span style="margin-right:6px">${c.id}(${c.points}pt${bon})</span>`;
    });
    h += '</div>';
  }

  el.innerHTML = h;
}

function renderPyramid(s) {
  let h = '<div class="pyramid-title">Pyramid</div>';
  [3,2,1].forEach(lvl => {
    const cards = s.pyramid[lvl] || [];
    const deck = s.deck_sizes[lvl] || 0;
    h += `<div class="pyramid-row"><span class="level-label level-${lvl}">L${lvl}</span>`;
    cards.forEach(c => {
      let bon = '';
      if (c.is_wildcard) bon = '<span class="bon" style="color:#e8b84b">★</span>';
      else if (c.gem_bonus) { for (const[k,v] of Object.entries(c.gem_bonus)) if (v>0) bon = `<span class="bon" style="color:${BON_CSS[k]}">${GEM_LETTERS[k].repeat(v)}</span>`; }
      const costTotal = Object.values(c.cost).reduce((a,b)=>a+b,0);
      const costStr = GEM_NAMES.filter(g=>c.cost[g]>0).map(g=>`<span style="color:${BON_CSS[g]}">${c.cost[g]}</span>`).join('+');
      h += `<div class="card-chip" title="${c.id}: cost ${costTotal}">`;
      h += `<span class="pts">${c.points}</span>${bon}`;
      if (c.crowns) h += `<span class="cr">${'♛'.repeat(c.crowns)}</span>`;
      if (c.ability) h += `<span class="eff">⚡</span>`;
      h += `<span style="font-size:10px;color:#666">${costStr}</span>`;
      h += '</div>';
    });
    h += `<span class="deck-count">(${deck})</span></div>`;
  });
  document.getElementById('pyramidWrap').innerHTML = h;
}

function renderRoyals(s) {
  if (!s.royal_cards.length) { document.getElementById('royalsWrap').innerHTML = ''; return; }
  let h = '<span class="royals-label">Royals:</span>';
  s.royal_cards.forEach(r => {
    h += `<span class="royal-chip">${r.id}: ${r.points}pts${r.ability ? ' ⚡' + r.ability : ''}</span>`;
  });
  document.getElementById('royalsWrap').innerHTML = h;
}

function step(dir) { goTo(cur + dir); }
function goTo(i) { cur = Math.max(0, Math.min(steps.length - 1, i)); render(); }
function togglePlay() {
  playing = !playing;
  document.getElementById('playBtn').textContent = playing ? '⏸ Pause' : '▶ Play';
  if (playing) playTimer = setInterval(() => { if (cur < steps.length - 1) step(1); else togglePlay(); }, 400);
  else clearInterval(playTimer);
}

document.addEventListener('keydown', e => {
  if (e.key === 'ArrowRight' || e.key === ' ') { e.preventDefault(); step(1); }
  if (e.key === 'ArrowLeft') { e.preventDefault(); step(-1); }
  if (e.key === 'Home') goTo(0);
  if (e.key === 'End') goTo(steps.length - 1);
});

render();
</script>
</body>
</html>
'''
