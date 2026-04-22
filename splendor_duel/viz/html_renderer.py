"""
html_renderer.py — Generate a standalone HTML replay of a game.

Image mode explained:
    Images are loaded via relative URL from the HTML file location.
    If images_dir="../images", the browser looks for:
        ../images/cards/L1_01.jpg
    So if your HTML is at:
        replays/game.html
    Put card images at:
        images/cards/L1_01.jpg  (sibling of replays/)

    If an image fails to load (file missing), it falls back to digital rendering.

Usage:
    render_html(log, "replays/game.html")                      # digital only
    render_html(log, "replays/game.html", images_dir="../images")  # with photos
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
        output_path: Where to write the HTML file.
        images_dir:  Relative path from the HTML file to the images/ folder.
                     Example: "../images" means one level up, then images/.
                     Card photos expected at {images_dir}/cards/{card_id}.jpg
                     If None → digital card rendering (no photos needed).
    """
    steps_data = log_to_json_data(log)
    steps_json = json.dumps(steps_data, ensure_ascii=False)
    use_images = json.dumps(images_dir is not None)
    images_base = json.dumps(images_dir or '')

    html = _TEMPLATE
    html = html.replace('/*STEPS*/', steps_json)
    html = html.replace('/*USE_IMAGES*/', use_images)
    html = html.replace('/*IMAGES_DIR*/', images_base)

    Path(output_path).write_text(html, encoding='utf-8')


_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Splendor Duel Replay</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',system-ui,sans-serif;background:#1a1a2e;color:#e0e0e0;padding:14px}
.wrap{max-width:1200px;margin:0 auto;display:flex;flex-direction:column;gap:10px}

/* Header */
.hdr{display:flex;align-items:center;justify-content:space-between;background:#16213e;border-radius:10px;padding:10px 18px}
.hdr-left{display:flex;flex-direction:column;gap:2px}
.turn{font-size:17px;font-weight:600}
.desc{font-size:12px;color:#8899aa;min-height:16px}
.nav{display:flex;align-items:center;gap:8px}
.nav button{background:#0f3460;border:none;color:#e0e0e0;padding:7px 14px;border-radius:6px;cursor:pointer;font-size:14px}
.nav button:hover{background:#1a5276}
.ctr{font-size:12px;color:#8899aa;min-width:80px;text-align:center}

/* Winner */
.winner{background:#e9b44c;color:#1a1a2e;text-align:center;padding:10px;border-radius:10px;font-size:17px;font-weight:700}

/* Main area */
.main{display:grid;grid-template-columns:260px 1fr;gap:10px}

/* Board */
.board-panel{background:#16213e;border-radius:10px;padding:14px}
.board-grid{display:grid;grid-template-columns:repeat(5,46px);gap:4px}
.cell{width:46px;height:46px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:15px}
.cell.empty{background:#252538;border:2px dashed #383858}
.side-meta{margin-top:12px;font-size:12px;color:#8899aa;display:flex;flex-direction:column;gap:4px}
.scrolls{font-size:16px;letter-spacing:3px}

/* Pyramid panel */
.right{display:flex;flex-direction:column;gap:10px}
.pyr-panel{background:#16213e;border-radius:10px;padding:12px}
.sec-title{font-size:11px;color:#556;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px}
.pyr-row{display:flex;gap:6px;align-items:center;margin-bottom:6px}
.pyr-lbl{font-size:11px;color:#556;width:22px;text-align:right;margin-right:4px;flex-shrink:0}
.deck-cnt{font-size:10px;color:#445;margin-left:4px}
.royals-row{display:flex;gap:6px;flex-wrap:wrap}

/* Players */
.players{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.pl{background:#16213e;border-radius:10px;padding:12px;border:2px solid transparent}
.pl.active{border-color:#e9b44c}
.pl-hdr{display:flex;justify-content:space-between;align-items:center;margin-bottom:8px}
.pl-name{font-size:14px;font-weight:600}
.star{color:#e9b44c}
.pl-stats{display:flex;gap:12px;font-size:12px}
.stat-val{font-weight:700;font-size:14px}
.sec-lbl{font-size:10px;color:#445;text-transform:uppercase;letter-spacing:1px;margin:7px 0 4px}
.tokens-row{display:flex;gap:5px;flex-wrap:wrap}
.tok{display:flex;align-items:center;gap:2px;background:#1a1a30;border-radius:12px;padding:2px 7px 2px 3px}
.tok-dot{width:17px;height:17px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:9px;font-weight:700}
.tok-n{font-size:11px;font-weight:600}
.bon-dot{width:17px;height:17px;border-radius:50%;border:1.5px solid rgba(255,255,255,0.35);display:flex;align-items:center;justify-content:center;font-size:10px;font-weight:700;margin-right:2px}

/* Card columns */
.cards-columns{display:flex;gap:8px;flex-wrap:wrap;align-items:flex-start}
.card-col{display:flex;flex-direction:column;gap:4px;align-items:center}
.col-label{font-size:9px;color:#445;text-transform:uppercase;letter-spacing:1px;margin-bottom:2px}

/* Gem colors */
.g0{background:#e8e0d0;color:#333} /* white  */
.g1{background:#3a3a3a;color:#fff} /* black  */
.g2{background:#d44040;color:#fff} /* red    */
.g3{background:#4080d0;color:#fff} /* blue   */
.g4{background:#40a060;color:#fff} /* green  */
.g5{background:#b0d0e8;color:#334} /* pearl  */
.g6{background:#d4a020;color:#333} /* gold   */

/* Digital card */
.card{width:82px;min-height:108px;border-radius:6px;padding:5px 5px 4px;display:flex;flex-direction:column;
      position:relative;border:2px solid;font-size:11px;cursor:default;flex-shrink:0}
.card:hover{transform:scale(1.1);z-index:20;transition:transform 0.1s}
.cl1{background:#1b3a25;border-color:#3a7a4a}
.cl2{background:#3a3218;border-color:#9a8030}
.cl3{background:#1a2a42;border-color:#3a6aa0}
.croy{background:#2a1a3a;border-color:#7a4a9a;width:76px;min-height:60px}
.card-top{display:flex;justify-content:space-between;align-items:flex-start}
.card-pts{font-size:17px;font-weight:700;line-height:1}
.card-pts.zero{opacity:0}
.card-crowns{font-size:11px;color:#e9b44c}
.card-bonus-col{display:flex;flex-direction:column;gap:2px;align-items:flex-end}
.gem-circ{width:18px;height:18px;border-radius:50%;border:1.5px solid rgba(255,255,255,0.25)}
.gem-circ.wc{background:conic-gradient(#d44040 0%,#4080d0 25%,#40a060 50%,#e8e0d0 75%);border-color:rgba(255,255,255,0.5)}
.card-abil{font-size:10px;color:#9ab;margin-top:1px}
.card-cost{margin-top:auto;padding-top:4px;display:flex;flex-direction:column;gap:2px}
.cost-pip{display:flex;align-items:center;gap:2px}
.cost-dot{width:15px;height:15px;border-radius:50%;display:flex;align-items:center;justify-content:center;
          font-size:8px;font-weight:700;border:1px solid rgba(255,255,255,0.15)}

/* Image card */
.card-img-wrap{position:relative;flex-shrink:0;cursor:default}
.card-img-wrap:hover{transform:scale(1.5);z-index:20;transition:transform 0.15s;transform-origin:center top}
.card-img{width:82px;height:114px;border-radius:6px;object-fit:cover;border:2px solid #444;display:block}

.hint{text-align:center;font-size:11px;color:#333;margin-top:6px}
</style>
</head>
<body>
<div class="wrap" id="app"></div>
<div class="hint">← → navigate · Home/End first/last</div>

<script>
const STEPS = /*STEPS*/;
const USE_IMAGES = /*USE_IMAGES*/;
const IMAGES_DIR = /*IMAGES_DIR*/;
let cur = 0;

const GN = ['white','black','red','blue','green','pearl','gold'];
const GC = ['g0','g1','g2','g3','g4','g5','g6'];
const GH = ['W','K','R','B','G','P','$'];
const LEVEL_COLORS = ['','#3a7a4a','#9a8030','#3a6aa0'];  // L1=green, L2=gold, L3=blue
const ABIL = {extra_turn:'↻',take_same_gem:'◎',take_scroll:'📜',take_opponent_gem:'⇄'};
const PHASE = {OPTIONAL:'Optional',MAIN:'Main',EFFECT:'Effect',ROYAL:'Royal',DISCARD:'Discard',GAME_OVER:'Game Over'};

// ── Digital card ─────────────────────────────────────────────────────────
function cardDigital(c) {
    const lvl = c.level ? 'cl'+c.level : 'croy';
    const pts = '<span class="card-pts'+(c.points?'':' zero')+'">'+c.points+'</span>';
    const crowns = c.crowns ? '<span class="card-crowns">'+'♛'.repeat(c.crowns)+'</span>' : '';

    let bonus = '';
    if (c.is_wildcard) {
        bonus = '<div class="gem-circ wc"></div>';
    } else if (c.gem_bonus) {
        for (let i=0;i<7;i++) {
            const v=c.gem_bonus[GN[i]]||0;
            for(let j=0;j<v;j++) bonus+='<div class="gem-circ '+GC[i]+'"></div>';
        }
    }

    const abil = c.ability ? '<div class="card-abil">'+(ABIL[c.ability]||c.ability)+'</div>' : '';

    let cost='';
    if(c.cost) for(let i=0;i<7;i++){const v=c.cost[GN[i]]||0;if(v>0)cost+='<div class="cost-pip"><div class="cost-dot '+GC[i]+'">'+v+'</div></div>';}

    return '<div class="card '+lvl+'" title="'+c.id+'">'
        +'<div class="card-top">'+pts+crowns+'<div class="card-bonus-col">'+bonus+'</div></div>'
        +abil
        +'<div class="card-cost">'+cost+'</div></div>';
}

// ── Image card ───────────────────────────────────────────────────────────
// Images are served as relative URLs from the HTML file's location.
// If images_dir = "../images", browser resolves: ../images/cards/L1_01.jpg
// The onerror replaces the <img> with a digital card fallback.
function cardImage(c) {
    const src = IMAGES_DIR + '/cards/' + c.id + '.png';
    // Encode the digital fallback as a data attribute to avoid escaping hell
    const fallbackId = 'fb_' + c.id.replace(/[^a-zA-Z0-9]/g,'_');
    return '<div class="card-img-wrap" title="'+c.id+'">'
        + '<img class="card-img" src="'+src+'" alt="'+c.id+'"'
        + ' onerror="this.parentElement.outerHTML=document.getElementById(\''+fallbackId+'\').innerHTML">'
        + '<template id="'+fallbackId+'">'+cardDigital(c)+'</template>'
        + '</div>';
}

function renderCard(c) { return USE_IMAGES ? cardImage(c) : cardDigital(c); }

function renderRoyal(c) {
    const abil = c.ability ? '<div class="card-abil">'+(ABIL[c.ability]||c.ability)+'</div>' : '';
    return '<div class="card croy" title="'+c.id+'"><div class="card-top"><span class="card-pts">'+c.points+'</span></div>'+abil+'</div>';
}

// ── Board ─────────────────────────────────────────────────────────────────
function renderBoard(b) {
    let h='<div class="board-grid">';
    for(let r=0;r<5;r++) for(let c=0;c<5;c++){
        const v=b[r][c];
        h += v<0 ? '<div class="cell empty"></div>'
                 : '<div class="cell '+GC[v]+'">'+GH[v]+'</div>';
    }
    return h+'</div>';
}

// ── Tokens / bonuses ──────────────────────────────────────────────────────
function renderTokens(t) {
    let h='';
    for(let i=0;i<7;i++){const v=t[GN[i]]||0;if(v>0)h+='<div class="tok"><div class="tok-dot '+GC[i]+'">'+GH[i]+'</div><span class="tok-n">×'+v+'</span></div>';}
    return h||'<span style="color:#445;font-size:11px">—</span>';
}

function renderBonuses(b) {
    let h='';
    for(let i=0;i<7;i++){const v=b[GN[i]]||0;if(v>0)h+='<div class="tok"><div class="bon-dot '+GC[i]+'">'+v+'</div></div>';}
    return h||'<span style="color:#445;font-size:11px">—</span>';
}

function scrollHtml(n,max){let s='';for(let i=0;i<max;i++)s+=i<n?'⚜':'·';return s;}

// ── Cards in colour columns ───────────────────────────────────────────────
// Groups bought cards by their bonus colour.
// Wildcard cards go into the column of their assigned colour (from wildcard_assignments).
// Cards with no bonus go into a separate "no bonus" column.
// Each column is labelled with a coloured dot + gem name.
function renderCardColumns(player) {
    // Build columns: key = gem index (0-6) or 'none'
    const cols = {}; // index or 'none' → [cardHtml, ...]

    for (const c of player.cards) {
        let key = 'none';
        if (c.is_wildcard) {
            // Look up wildcard assignment if available
            // The assignment is stored in player.wildcard_assignments if serialized
            // We encode it via the card column rendering logic:
            // since engine serialized cards in order, we need assignment from player data
            // For simplicity: wildcard goes to 'wc' column if not assigned
            key = 'wc';
        } else if (c.gem_bonus) {
            for (let i=0; i<7; i++) {
                if ((c.gem_bonus[GN[i]]||0) > 0) { key = i; break; }
            }
        }
        if (!cols[key]) cols[key] = [];
        cols[key].push(renderCard(c));
    }

    if (Object.keys(cols).length === 0) return '<span style="color:#445;font-size:11px">—</span>';

    let h = '<div class="cards-columns">';
    // Render gem colour columns in fixed order
    for (let i=0; i<7; i++) {
        if (!cols[i]) continue;
        h += '<div class="card-col">'
           + '<div class="col-label"><span class="tok-dot '+GC[i]+'" style="display:inline-flex;width:14px;height:14px;border-radius:50%;align-items:center;justify-content:center;font-size:8px;font-weight:700">'+GH[i]+'</span></div>'
           + cols[i].join('')
           + '</div>';
    }
    // Wildcard column
    if (cols['wc']) {
        h += '<div class="card-col">'
           + '<div class="col-label"><span style="font-size:10px">✦</span></div>'
           + cols['wc'].join('')
           + '</div>';
    }
    // No-bonus column
    if (cols['none']) {
        h += '<div class="card-col">'
           + '<div class="col-label">—</div>'
           + cols['none'].join('')
           + '</div>';
    }
    h += '</div>';
    return h;
}

// ── Full render ───────────────────────────────────────────────────────────
function render(si) {
    const step=STEPS[si], st=step.state;
    const over=st.phase==='GAME_OVER';
    const desc=step.description||(si===0?'Game start':'');
    const ap=step.player_acted;
    let h='';

    // Header
    h+='<div class="hdr"><div class="hdr-left">'
      +'<div class="turn">Turn '+st.turn+' · Player '+st.current_player+' · '+(PHASE[st.phase]||st.phase)+'</div>'
      +'<div class="desc">'+(ap!==undefined?'P'+ap+': ':'')+desc+'</div>'
      +'</div><div class="nav">'
      +'<button onclick="go(0)">⏮</button>'
      +'<button onclick="go(cur-1)">◀</button>'
      +'<span class="ctr">'+si+' / '+(STEPS.length-1)+'</span>'
      +'<button onclick="go(cur+1)">▶</button>'
      +'<button onclick="go(STEPS.length-1)">⏭</button>'
      +'</div></div>';

    // Winner
    if (over) {
        let w='?';
        for(let i=0;i<2;i++){
            const p=st.players[i];
            if(p.points>=20||p.crowns>=10){w=i;break;}
            const byCol={};
            for(const c of p.cards) if(!c.is_wildcard&&c.gem_bonus)
                for(const nm of GN) if((c.gem_bonus[nm]||0)>0){byCol[nm]=(byCol[nm]||0)+c.points;break;}
            for(const pts of Object.values(byCol)) if(pts>=10){w=i;break;}
            if(w!=='?')break;
        }
        h+='<div class="winner">🏆 Player '+w+' wins!</div>';
    }

    // Main: board + pyramid
    h+='<div class="main"><div class="board-panel">'+renderBoard(st.board)
      +'<div class="side-meta">'
      +'<div>Scrolls: <span class="scrolls">'+scrollHtml(st.scrolls_center,3)+'</span></div>'
      +'<div>Bag: '+st.bag_total+' tokens</div></div></div>'
      +'<div class="right"><div class="pyr-panel"><div class="sec-title">Pyramid</div>';

    for(const lvl of ['3','2','1']){
        const cards=st.pyramid[lvl]||[], ds=st.deck_sizes[lvl]||0;
        h+='<div class="pyr-row"><span class="pyr-lbl">L'+lvl+'</span>'
          +cards.map(c=>renderCard(c)).join('')
          +'<span class="deck-cnt">('+ds+')</span></div>';
    }
    h+='<div class="sec-title" style="margin-top:10px">Royal cards</div>'
      +'<div class="royals-row">'
      +(st.royal_cards.length ? st.royal_cards.map(r=>renderRoyal(r)).join('') : '<span style="color:#445;font-size:11px">none</span>')
      +'</div></div></div></div>';

    // Players
    h+='<div class="players">';
    for(let pi=0;pi<2;pi++){
        const p=st.players[pi],act=pi===st.current_player&&!over;
        h+='<div class="pl'+(act?' active':'')+'"><div class="pl-hdr">'
          +'<span class="pl-name">Player '+pi+(act?' <span class="star">★</span>':'')+'</span>'
          +'<div class="pl-stats">'
          +'<div>🏅 <span class="stat-val">'+p.points+'</span></div>'
          +'<div>♛ <span class="stat-val">'+p.crowns+'</span></div>'
          +'<div>⚜ <span class="stat-val">'+p.scrolls+'</span></div>'
          +'</div></div>'
          +'<div class="sec-lbl">Tokens ('+p.total_tokens+'/10)</div>'
          +'<div class="tokens-row">'+renderTokens(p.tokens)+'</div>'
          +'<div class="sec-lbl">Bonuses</div>'
          +'<div class="tokens-row">'+renderBonuses(p.bonuses)+'</div>'
          +'<div class="sec-lbl">Cards ('+p.cards.length+')</div>'
          +renderCardColumns(p);
        if(p.reserved.length)
            h+='<div class="sec-lbl">Reserved ('+p.reserved.length+')</div>'
              +'<div style="display:flex;gap:5px;flex-wrap:wrap">'+p.reserved.map(c=>renderCard(c)).join('')+'</div>';
        if(p.royals.length)
            h+='<div class="sec-lbl">Royal cards</div>'
              +'<div style="display:flex;gap:5px;flex-wrap:wrap">'+p.royals.map(r=>renderRoyal(r)).join('')+'</div>';
        h+='</div>';
    }
    h+='</div>';

    document.getElementById('app').innerHTML=h;
}

function go(idx){cur=Math.max(0,Math.min(STEPS.length-1,idx));render(cur);}
document.addEventListener('keydown',function(e){
    if(e.key==='ArrowRight'||e.key===' ')go(cur+1);
    else if(e.key==='ArrowLeft')go(cur-1);
    else if(e.key==='Home')go(0);
    else if(e.key==='End')go(STEPS.length-1);
});
render(0);
</script>
</body>
</html>"""
