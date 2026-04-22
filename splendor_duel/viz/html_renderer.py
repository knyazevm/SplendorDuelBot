"""
html_renderer.py — Generate a standalone HTML replay of a game.

Two rendering modes:
  Digital (default): Cards drawn with CSS — points, crowns, gem bonus, cost, ability.
  Image:             Card photos from images/cards/ directory.

Usage:
    from splendor_duel.viz.html_renderer import render_html

    render_html(log, "replay.html")                          # digital
    render_html(log, "replay.html", images_dir="../images")  # with photos
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .replay import GameLog, log_to_json_data


def render_html(
        log: GameLog,
        output_path: str,
        images_dir: Optional[str] = None,
) -> None:
    """
    Write a self-contained HTML replay file.

    Args:
        log:         Recorded GameLog.
        output_path: Where to save the HTML file.
        images_dir:  Relative path to images/ directory (enables photo mode).
                     If None, uses digital card rendering.
    """
    steps_json = json.dumps(log_to_json_data(log), ensure_ascii=False)
    use_images = images_dir is not None
    images_dir_js = json.dumps(images_dir or '')

    html = _HTML_TEMPLATE.replace('__STEPS_DATA__', steps_json)
    html = html.replace('__USE_IMAGES__', json.dumps(use_images))
    html = html.replace('__IMAGES_DIR__', images_dir_js)

    Path(output_path).write_text(html, encoding='utf-8')


# ══════════════════════════════════════════════════════════════════════════════

_HTML_TEMPLATE = r'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Splendor Duel — Game Replay</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
       background: #1a1a2e; color: #e0e0e0; min-height: 100vh; padding: 16px; }

/* ── Layout ────────────────────────────────────────────────────────────── */
.container { max-width: 1200px; margin: 0 auto; display: flex; flex-direction: column; gap: 12px; }

.header { display: flex; align-items: center; justify-content: space-between;
          background: #16213e; border-radius: 10px; padding: 12px 20px; }
.header-left { display: flex; flex-direction: column; gap: 2px; }
.turn-info { font-size: 18px; font-weight: 600; }
.action-desc { font-size: 13px; color: #8899aa; min-height: 18px; }
.nav { display: flex; align-items: center; gap: 10px; }
.nav button { background: #0f3460; border: none; color: #e0e0e0; padding: 8px 16px;
              border-radius: 6px; cursor: pointer; font-size: 14px; }
.nav button:hover { background: #1a5276; }
.step-counter { font-size: 13px; color: #8899aa; min-width: 80px; text-align: center; }

.main-area { display: grid; grid-template-columns: auto 1fr; gap: 12px; }

/* ── Board ─────────────────────────────────────────────────────────────── */
.board-panel { background: #16213e; border-radius: 10px; padding: 16px; }
.board-grid { display: grid; grid-template-columns: repeat(5, 48px); gap: 4px; }
.board-cell { width: 48px; height: 48px; border-radius: 50%; display: flex;
              align-items: center; justify-content: center; font-weight: 700;
              font-size: 16px; border: 2px solid transparent; transition: all 0.15s; }
.board-cell.empty { background: #2a2a40; border: 2px dashed #3a3a50; }

.side-info { margin-top: 14px; font-size: 13px; color: #8899aa; }
.side-info .label { color: #667; }
.scrolls-display { margin-top: 6px; font-size: 18px; letter-spacing: 4px; }

/* ── Pyramid ───────────────────────────────────────────────────────────── */
.right-panel { display: flex; flex-direction: column; gap: 12px; }
.pyramid-panel { background: #16213e; border-radius: 10px; padding: 14px; }
.pyramid-panel h3 { font-size: 13px; color: #667; margin-bottom: 8px; text-transform: uppercase;
                    letter-spacing: 1px; }
.pyramid-row { display: flex; gap: 6px; margin-bottom: 6px; align-items: center; }
.pyramid-label { font-size: 11px; color: #667; width: 24px; text-align: right; margin-right: 4px; }

/* ── Royal cards ───────────────────────────────────────────────────────── */
.royals-row { display: flex; gap: 6px; flex-wrap: wrap; }

/* ── Players ───────────────────────────────────────────────────────────── */
.players-area { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
.player-panel { background: #16213e; border-radius: 10px; padding: 14px; }
.player-panel.active { border: 2px solid #e9b44c; }
.player-panel.inactive { border: 2px solid transparent; }
.player-header { display: flex; justify-content: space-between; align-items: center;
                 margin-bottom: 10px; }
.player-name { font-size: 15px; font-weight: 600; }
.player-name .star { color: #e9b44c; }
.player-stats { display: flex; gap: 16px; font-size: 13px; }
.stat { display: flex; align-items: center; gap: 4px; }
.stat-val { font-weight: 600; font-size: 15px; }

.tokens-row { display: flex; gap: 6px; margin: 8px 0; flex-wrap: wrap; }
.token-badge { display: flex; align-items: center; gap: 3px; background: #1a1a30;
               border-radius: 14px; padding: 3px 8px 3px 4px; }
.token-dot { width: 18px; height: 18px; border-radius: 50%; display: flex;
             align-items: center; justify-content: center; font-size: 10px; font-weight: 700; }
.token-count { font-size: 12px; font-weight: 600; }

.section-label { font-size: 11px; color: #556; text-transform: uppercase;
                 letter-spacing: 1px; margin: 8px 0 4px; }
.cards-grid { display: flex; gap: 5px; flex-wrap: wrap; }

/* ── Gem colors ────────────────────────────────────────────────────────── */
.gem-white  { background: #e8e0d0; color: #333; }
.gem-black  { background: #3a3a3a; color: #fff; }
.gem-red    { background: #d44040; color: #fff; }
.gem-blue   { background: #4080d0; color: #fff; }
.gem-green  { background: #40a060; color: #fff; }
.gem-pearl  { background: #b0d0e8; color: #334; }
.gem-gold   { background: #d4a020; color: #333; }

/* ── Card (digital mode) ──────────────────────────────────────────────── */
.card { width: 88px; min-height: 110px; border-radius: 6px; padding: 5px 6px;
        display: flex; flex-direction: column; position: relative;
        border: 2px solid; font-size: 11px; cursor: default; flex-shrink: 0; }
.card:hover { transform: scale(1.08); z-index: 10; transition: transform 0.1s; }

.card-l1 { background: #1b3a25; border-color: #3a7a4a; }
.card-l2 { background: #3a3218; border-color: #9a8030; }
.card-l3 { background: #1a2a42; border-color: #3a6aa0; }
.card-royal { background: #2a1a3a; border-color: #7a4a9a; width: 80px; min-height: 64px; }

.card-top { display: flex; justify-content: space-between; align-items: flex-start; }
.card-points { font-size: 18px; font-weight: 700; min-width: 16px; }
.card-points.zero { visibility: hidden; }
.card-crowns { font-size: 12px; color: #e9b44c; letter-spacing: 1px; flex-shrink: 0; }
.card-bonus-area { display: flex; flex-direction: column; gap: 2px; align-items: flex-end; }
.gem-circle { width: 20px; height: 20px; border-radius: 50%; border: 1.5px solid rgba(255,255,255,0.3); }
.gem-circle.wildcard { background: linear-gradient(135deg, #d44040, #4080d0, #40a060, #e8e0d0);
                       border-color: rgba(255,255,255,0.5); }

.card-ability { font-size: 10px; margin-top: 2px; color: #aab; }
.card-cost { margin-top: auto; display: flex; flex-direction: column; gap: 2px; padding-top: 4px; }
.cost-pip { display: flex; align-items: center; gap: 3px; }
.cost-dot { width: 16px; height: 16px; border-radius: 50%; display: flex;
            align-items: center; justify-content: center; font-size: 9px; font-weight: 700;
            border: 1px solid rgba(255,255,255,0.2); }

/* ── Card (image mode) ─────────────────────────────────────────────────── */
.card-img-wrap { flex-shrink: 0; position: relative; }
.card-img { width: 88px; height: 120px; border-radius: 6px; object-fit: cover;
            border: 2px solid #444; cursor: default; display: block; }
.card-img-wrap:hover { transform: scale(1.5); z-index: 10; transition: transform 0.15s; }

/* ── Royal card ────────────────────────────────────────────────────────── */
.royal-pts { font-size: 16px; font-weight: 700; }
.royal-ability { font-size: 9px; color: #aab; margin-top: 2px; }

/* ── Winner banner ─────────────────────────────────────────────────────── */
.winner-banner { background: #e9b44c; color: #1a1a2e; text-align: center;
                 padding: 12px; border-radius: 10px; font-size: 18px; font-weight: 700; }

/* ── Keyboard hint ─────────────────────────────────────────────────────── */
.hint { text-align: center; font-size: 11px; color: #445; margin-top: 4px; }
</style>
</head>
<body>
<div class="container" id="app"></div>
<div class="hint">← → arrow keys to navigate · Home / End for first / last step</div>

<script>
const STEPS = __STEPS_DATA__;
const USE_IMAGES = __USE_IMAGES__;
const IMAGES_DIR = __IMAGES_DIR__;
let current = 0;

const GEM_NAMES = ['white','black','red','blue','green','pearl','gold'];
const GEM_CSS   = ['gem-white','gem-black','gem-red','gem-blue','gem-green','gem-pearl','gem-gold'];
const GEM_CHAR  = ['W','K','R','B','G','P','$'];
const ABILITY_ICONS = {
    'extra_turn':'\u21bb', 'take_same_gem':'\u25ce',
    'take_scroll':'\ud83d\udcdc', 'take_opponent_gem':'\u21c4',
};
const PHASE_LABELS = {
    'OPTIONAL':'Optional','MAIN':'Main','EFFECT':'Effect',
    'ROYAL':'Royal','DISCARD':'Discard','GAME_OVER':'Game Over',
};

function gemCssClass(idx) { return idx >= 0 && idx < 7 ? GEM_CSS[idx] : ''; }

/* ── Card rendering ─────────────────────────────────────────────────────── */

function cardDigital(card) {
    const lvl = card.level ? 'card-l'+card.level : 'card-royal';
    const ptsClass = card.points ? '' : ' zero';
    const pts = '<span class="card-points'+ptsClass+'">'+card.points+'</span>';
    const crowns = card.crowns ? '<span class="card-crowns">'+ '\u265b'.repeat(card.crowns)+'</span>' : '';

    let bonusHtml = '';
    if (card.is_wildcard) {
        bonusHtml = '<div class="gem-circle wildcard"></div>';
    } else if (card.gem_bonus) {
        for (const nm of GEM_NAMES) {
            const cnt = card.gem_bonus[nm] || 0;
            if (cnt > 0) {
                const idx = GEM_NAMES.indexOf(nm);
                for (let j = 0; j < cnt; j++)
                    bonusHtml += '<div class="gem-circle '+GEM_CSS[idx]+'"></div>';
            }
        }
    }

    const abilIcon = card.ability ? (ABILITY_ICONS[card.ability]||card.ability) : '';
    const abilHtml = abilIcon ? '<div class="card-ability">'+abilIcon+'</div>' : '';

    let costHtml = '';
    if (card.cost) {
        for (let i = 0; i < 7; i++) {
            const v = card.cost[GEM_NAMES[i]] || 0;
            if (v > 0) costHtml += '<div class="cost-pip"><div class="cost-dot '+GEM_CSS[i]+'">'+v+'</div></div>';
        }
    }

    return '<div class="card '+lvl+'" title="'+card.id+'">'
        +'<div class="card-top">'+pts+crowns+'<div class="card-bonus-area">'+bonusHtml+'</div></div>'
        +abilHtml
        +'<div class="card-cost">'+costHtml+'</div></div>';
}

function cardImage(card) {
    const src = IMAGES_DIR+'/cards/'+card.id+'.jpg';
    const fallback = cardDigital(card).replace(/"/g,'&quot;');
    return '<div class="card-img-wrap"><img class="card-img" src="'+src+'" alt="'+card.id+'" title="'+card.id+'"'
        +' onerror="this.parentElement.outerHTML=\''+fallback.replace(/'/g,"\\'")+'\';"></div>';
}

function renderCard(card) {
    return USE_IMAGES ? cardImage(card) : cardDigital(card);
}

function renderRoyalCard(card) {
    const abilIcon = card.ability ? (ABILITY_ICONS[card.ability]||card.ability) : '';
    return '<div class="card card-royal" title="'+card.id+'">'
        +'<div class="card-top"><span class="royal-pts">'+card.points+'</span></div>'
        +(abilIcon ? '<div class="royal-ability">'+abilIcon+'</div>' : '')
        +'</div>';
}

/* ── Board ────────────────────────────────────────────────────────────── */

function renderBoard(board) {
    let h = '<div class="board-grid">';
    for (let r = 0; r < 5; r++)
        for (let c = 0; c < 5; c++) {
            const v = board[r][c];
            h += v < 0
                ? '<div class="board-cell empty"></div>'
                : '<div class="board-cell '+GEM_CSS[v]+'">'+GEM_CHAR[v]+'</div>';
        }
    return h+'</div>';
}

/* ── Tokens / bonuses ─────────────────────────────────────────────────── */

function renderTokens(tokens) {
    let h = '';
    for (let i = 0; i < 7; i++) {
        const v = tokens[GEM_NAMES[i]]||0;
        if (v > 0) h += '<div class="token-badge"><div class="token-dot '+GEM_CSS[i]+'">'+GEM_CHAR[i]+'</div><span class="token-count">\u00d7'+v+'</span></div>';
    }
    return h || '<span style="color:#445;font-size:12px">none</span>';
}

function renderBonuses(bonuses) {
    let h = '';
    for (let i = 0; i < 7; i++) {
        const v = bonuses[GEM_NAMES[i]]||0;
        if (v > 0) h += '<div class="token-badge"><div class="token-dot '+GEM_CSS[i]+'" style="border:1.5px solid rgba(255,255,255,0.4)">'+v+'</div></div>';
    }
    return h || '<span style="color:#445;font-size:12px">\u2014</span>';
}

function renderScrolls(count, max) {
    let s = '';
    for (let i = 0; i < max; i++) s += i < count ? '\u269c' : '\u00b7';
    return s;
}

/* ── Full step render ────────────────────────────────────────────────── */

function renderStep(si) {
    const step = STEPS[si], st = step.state;
    const isOver = st.phase === 'GAME_OVER';
    const desc = step.description || (si === 0 ? 'Game start' : '');
    const ap = step.player_acted;
    let h = '';

    h += '<div class="header"><div class="header-left">'
       + '<div class="turn-info">Turn '+st.turn+' \u00b7 Player '+st.current_player+' \u00b7 '+(PHASE_LABELS[st.phase]||st.phase)+'</div>'
       + '<div class="action-desc">'+(ap!==undefined?'P'+ap+': ':'')+desc+'</div>'
       + '</div><div class="nav">'
       + '<button onclick="go(0)">\u23ee</button>'
       + '<button onclick="go(current-1)">\u25c0</button>'
       + '<span class="step-counter">'+si+' / '+(STEPS.length-1)+'</span>'
       + '<button onclick="go(current+1)">\u25b6</button>'
       + '<button onclick="go(STEPS.length-1)">\u23ed</button>'
       + '</div></div>';

    if (isOver) {
        let winner = '?';
        for (let i = 0; i < 2; i++) {
            const p = st.players[i];
            if (p.points >= 20 || p.crowns >= 10) { winner = i; break; }
            const byCol = {};
            for (const c of p.cards) {
                if (c.is_wildcard) continue;
                if (c.gem_bonus) for (const nm of GEM_NAMES) {
                    if ((c.gem_bonus[nm]||0) > 0) { byCol[nm] = (byCol[nm]||0)+c.points; break; }
                }
            }
            for (const pts of Object.values(byCol)) if (pts >= 10) { winner = i; break; }
            if (winner !== '?') break;
        }
        h += '<div class="winner-banner">\ud83c\udfc6 Player '+winner+' wins!</div>';
    }

    h += '<div class="main-area"><div class="board-panel">'+renderBoard(st.board)
       + '<div class="side-info"><div><span class="label">Scrolls:</span> <span class="scrolls-display">'+renderScrolls(st.scrolls_center,3)+'</span></div>'
       + '<div style="margin-top:6px"><span class="label">Bag:</span> '+st.bag_total+' tokens</div></div></div>'
       + '<div class="right-panel"><div class="pyramid-panel"><h3>Pyramid</h3>';

    for (const lvl of ['3','2','1']) {
        const cards = st.pyramid[lvl]||[], ds = st.deck_sizes[lvl]||0;
        h += '<div class="pyramid-row"><span class="pyramid-label">L'+lvl+'</span>'
           + cards.map(c=>renderCard(c)).join('')
           + '<span style="font-size:10px;color:#556;margin-left:4px">('+ds+')</span></div>';
    }

    h += '<h3 style="margin-top:10px">Royal cards</h3><div class="royals-row">'
       + (st.royal_cards.length ? st.royal_cards.map(r=>renderRoyalCard(r)).join('') : '<span style="color:#445;font-size:12px">none</span>')
       + '</div></div></div></div>';

    h += '<div class="players-area">';
    for (let pi = 0; pi < 2; pi++) {
        const p = st.players[pi], act = pi===st.current_player && !isOver;
        h += '<div class="player-panel '+(act?'active':'inactive')+'">'
           + '<div class="player-header"><span class="player-name">Player '+pi+(act?' <span class="star">\u2605</span>':'')+'</span>'
           + '<div class="player-stats">'
           + '<div class="stat">\ud83c\udfc5 <span class="stat-val">'+p.points+'</span></div>'
           + '<div class="stat">\u265b <span class="stat-val">'+p.crowns+'</span></div>'
           + '<div class="stat">\u269c <span class="stat-val">'+p.scrolls+'</span></div>'
           + '</div></div>'
           + '<div class="section-label">Tokens ('+p.total_tokens+'/10)</div>'
           + '<div class="tokens-row">'+renderTokens(p.tokens)+'</div>'
           + '<div class="section-label">Bonuses</div>'
           + '<div class="tokens-row">'+renderBonuses(p.bonuses)+'</div>'
           + '<div class="section-label">Cards ('+p.cards.length+')</div>'
           + '<div class="cards-grid">'+p.cards.map(c=>renderCard(c)).join('')+'</div>';
        if (p.reserved.length)
            h += '<div class="section-label">Reserved ('+p.reserved.length+')</div>'
               + '<div class="cards-grid">'+p.reserved.map(c=>renderCard(c)).join('')+'</div>';
        if (p.royals.length)
            h += '<div class="section-label">Royal cards</div>'
               + '<div class="cards-grid">'+p.royals.map(r=>renderRoyalCard(r)).join('')+'</div>';
        h += '</div>';
    }
    h += '</div>';

    document.getElementById('app').innerHTML = h;
}

function go(idx) {
    current = Math.max(0, Math.min(STEPS.length-1, idx));
    renderStep(current);
}

document.addEventListener('keydown', function(e) {
    if (e.key==='ArrowRight'||e.key===' ') go(current+1);
    else if (e.key==='ArrowLeft') go(current-1);
    else if (e.key==='Home') go(0);
    else if (e.key==='End') go(STEPS.length-1);
});

renderStep(0);
</script>
</body>
</html>'''
