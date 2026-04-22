"""
html_renderer.py — Generate a standalone HTML replay of a game.

Layout: Player 0 | Board + Pyramid | Player 1

Image mode:
    render_html(log, "replays/game.html", images_dir="../images")
    Browser resolves: ../images/cards/L1_01.png
    Falls back to digital rendering if image not found.
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
    steps_data = log_to_json_data(log)
    steps_json = json.dumps(steps_data, ensure_ascii=False)
    use_images = json.dumps(images_dir is not None)
    images_base = json.dumps(images_dir or '')

    html = _TEMPLATE
    html = html.replace('/*STEPS*/', steps_json)
    html = html.replace('/*USE_IMAGES*/', use_images)
    html = html.replace('/*IMAGES_DIR*/', images_base)

    Path(output_path).write_text(html, encoding='utf-8')


_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Splendor Duel Replay</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',system-ui,sans-serif;background:#1a1a2e;color:#e0e0e0;padding:12px}
.wrap{max-width:100%;margin:0 auto;display:flex;flex-direction:column;gap:10px}

/* Header */
.hdr{display:flex;align-items:center;justify-content:space-between;background:#16213e;border-radius:10px;padding:10px 18px}
.hdr-left{display:flex;flex-direction:column;gap:2px}
.turn{font-size:17px;font-weight:600}
.desc{font-size:12px;color:#8899aa;min-height:16px}
.nav{display:flex;align-items:center;gap:8px}
.nav button{background:#0f3460;border:none;color:#e0e0e0;padding:7px 14px;border-radius:6px;cursor:pointer;font-size:14px}
.nav button:hover{background:#1a5276}
.ctr{font-size:12px;color:#8899aa;min-width:80px;text-align:center}
.winner{background:#e9b44c;color:#1a1a2e;text-align:center;padding:10px;border-radius:10px;font-size:17px;font-weight:700}

/* 3-column game layout: player0 | center | player1 */
.game-layout{display:grid;grid-template-columns:minmax(180px,1fr) minmax(520px,2fr) minmax(180px,1fr);gap:10px;align-items:start}
.center-col{display:flex;flex-direction:column;gap:10px;min-width:0}

/* Board */
.board-panel{background:#16213e;border-radius:10px;padding:14px}
.board-grid{display:grid;grid-template-columns:repeat(5,46px);gap:4px}
.cell{width:46px;height:46px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:15px}
.cell.empty{background:#252538;border:2px dashed #383858}
.side-meta{margin-top:10px;font-size:12px;color:#8899aa;display:flex;flex-direction:column;gap:4px}
.scrolls{font-size:16px;letter-spacing:3px}

/* Pyramid */
.pyr-panel{background:#16213e;border-radius:10px;padding:12px}
.sec-title{font-size:11px;color:#556;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px}
.pyr-row{display:flex;gap:5px;align-items:center;margin-bottom:5px;flex-wrap:nowrap}
.pyr-lbl{font-size:11px;color:#556;width:22px;text-align:right;margin-right:4px;flex-shrink:0}
.deck-cnt{font-size:10px;color:#445;margin-left:4px}
.royals-row{display:flex;gap:6px;flex-wrap:wrap}

/* Player panel */
.pl{background:#16213e;border-radius:10px;padding:12px;border:2px solid transparent}
.pl.active{border-color:#e9b44c}
/* Cards inside player panels smaller to save space */
.pl .card{width:62px;min-height:86px;padding:3px 4px}
.pl .card-pts{font-size:14px}
.pl .gem-circ{width:14px;height:14px}
.pl .cost-dot{width:12px;height:12px;font-size:7px}
.pl .card-img{width:62px;height:88px}
.pl-hdr{display:flex;justify-content:space-between;align-items:center;margin-bottom:8px}
.pl-name{font-size:14px;font-weight:600}
.star{color:#e9b44c}
.pl-stats{display:flex;gap:10px;font-size:12px}
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
.col-header{display:flex;flex-direction:column;align-items:center;gap:2px;margin-bottom:2px}
.col-gem{width:18px;height:18px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:9px;font-weight:700}
.col-pts{font-size:11px;font-weight:700;color:#e9b44c}
.col-pts.low{color:#8899aa}

/* Gem colors */
.g0{background:#e8e0d0;color:#333}
.g1{background:#3a3a3a;color:#fff}
.g2{background:#d44040;color:#fff}
.g3{background:#4080d0;color:#fff}
.g4{background:#40a060;color:#fff}
.g5{background:#b0d0e8;color:#334}
.g6{background:#d4a020;color:#333}

/* Digital card */
.card{width:78px;min-height:106px;border-radius:6px;padding:4px 5px;display:flex;flex-direction:column;
      border:2px solid;font-size:11px;cursor:default;flex-shrink:0}
.card:hover{transform:scale(1.12);z-index:20;transition:transform 0.1s}
.cl1{background:#1b3a25;border-color:#3a7a4a}
.cl2{background:#3a3218;border-color:#9a8030}
.cl3{background:#1a2a42;border-color:#3a6aa0}
.croy{background:#2a1a3a;border-color:#7a4a9a;width:72px;min-height:56px}
.card-top{display:flex;justify-content:space-between;align-items:flex-start}
.card-pts{font-size:17px;font-weight:700;line-height:1}
.card-pts.zero{opacity:0}
.card-crowns{font-size:11px;color:#e9b44c}
.card-bonus-col{display:flex;flex-direction:column;gap:2px;align-items:flex-end}
.gem-circ{width:17px;height:17px;border-radius:50%;border:1.5px solid rgba(255,255,255,0.25)}
.gem-circ.wc{background:conic-gradient(#d44040 0%,#4080d0 25%,#40a060 50%,#e8e0d0 75%);border-color:rgba(255,255,255,0.5)}
.card-abil{font-size:10px;color:#9ab;margin-top:1px}
.card-cost{margin-top:auto;padding-top:4px;display:flex;flex-direction:column;gap:2px}
.cost-pip{display:flex;align-items:center;gap:2px}
.cost-dot{width:14px;height:14px;border-radius:50%;display:flex;align-items:center;justify-content:center;
          font-size:8px;font-weight:700;border:1px solid rgba(255,255,255,0.15)}

/* Image card */
.card-img-wrap{position:relative;flex-shrink:0;cursor:default}
.card-img-wrap:hover{transform:scale(1.5);z-index:20;transition:transform 0.15s;transform-origin:center top}
.card-img{width:78px;height:110px;border-radius:6px;object-fit:cover;border:2px solid #444;display:block}

.hint{text-align:center;font-size:11px;color:#333;margin-top:6px}
</style>
</head>
<body>
<div class="wrap" id="app"></div>
<div class="hint">&#8592; &#8594; navigate &middot; Home/End first/last</div>

<script>
const STEPS = /*STEPS*/;
const USE_IMAGES = /*USE_IMAGES*/;
const IMAGES_DIR = /*IMAGES_DIR*/;
let cur = 0;

const GN=['white','black','red','blue','green','pearl','gold'];
const GC=['g0','g1','g2','g3','g4','g5','g6'];
const GH=['W','K','R','B','G','P','$'];
const ABIL={extra_turn:'\\u21bb',take_same_gem:'\\u25ce',take_scroll:'\\ud83d\\udcdc',take_opponent_gem:'\\u21c4'};
const PHASE={OPTIONAL:'Optional',MAIN:'Main',EFFECT:'Effect',ROYAL:'Royal',DISCARD:'Discard',GAME_OVER:'Game Over'};

/* ── Digital card ────────────────────────────────────────────────────── */
function cardDigital(c) {
    const lvl = c.level ? 'cl'+c.level : 'croy';
    const pts = '<span class="card-pts'+(c.points?'':' zero')+'">'+c.points+'</span>';
    const crowns = c.crowns ? '<span class="card-crowns">'+'\\u265b'.repeat(c.crowns)+'</span>' : '';
    let bonus='';
    if(c.is_wildcard){bonus='<div class="gem-circ wc"></div>';}
    else if(c.gem_bonus){for(let i=0;i<7;i++){const v=c.gem_bonus[GN[i]]||0;for(let j=0;j<v;j++)bonus+='<div class="gem-circ '+GC[i]+'"></div>';}}
    const abil=c.ability?'<div class="card-abil">'+(ABIL[c.ability]||c.ability)+'</div>':'';
    let cost='';
    if(c.cost)for(let i=0;i<7;i++){const v=c.cost[GN[i]]||0;if(v>0)cost+='<div class="cost-pip"><div class="cost-dot '+GC[i]+'">'+v+'</div></div>';}
    return '<div class="card '+lvl+'" title="'+c.id+'">'
        +'<div class="card-top">'+pts+crowns+'<div class="card-bonus-col">'+bonus+'</div></div>'
        +abil+'<div class="card-cost">'+cost+'</div></div>';
}

/* ── Image card ──────────────────────────────────────────────────────── */
function cardImage(c) {
    const src=IMAGES_DIR+'/cards/'+c.id+'.png';
    const fbId='fb_'+c.id.replace(/[^a-zA-Z0-9]/g,'_');
    return '<div class="card-img-wrap" title="'+c.id+'">'
        +'<img class="card-img" src="'+src+'" alt="'+c.id+'"'
        +' onerror="this.parentElement.outerHTML=document.getElementById(\\''+fbId+'\\').innerHTML">'
        +'<template id="'+fbId+'">'+cardDigital(c)+'</template>'
        +'</div>';
}

function renderCard(c){return USE_IMAGES?cardImage(c):cardDigital(c);}

function renderRoyal(c){
    const abil=c.ability?'<div class="card-abil">'+(ABIL[c.ability]||c.ability)+'</div>':'';
    return '<div class="card croy" title="'+c.id+'"><div class="card-top"><span class="card-pts">'+c.points+'</span></div>'+abil+'</div>';
}

/* ── Board ───────────────────────────────────────────────────────────── */
function renderBoard(b){
    let h='<div class="board-grid">';
    for(let r=0;r<5;r++)for(let c=0;c<5;c++){
        const v=b[r][c];
        h+=v<0?'<div class="cell empty"></div>':'<div class="cell '+GC[v]+'">'+GH[v]+'</div>';
    }
    return h+'</div>';
}

/* ── Tokens / bonuses ────────────────────────────────────────────────── */
function renderTokens(t){
    let h='';
    for(let i=0;i<7;i++){const v=t[GN[i]]||0;if(v>0)h+='<div class="tok"><div class="tok-dot '+GC[i]+'">'+GH[i]+'</div><span class="tok-n">\\u00d7'+v+'</span></div>';}
    return h||'<span style="color:#445;font-size:11px">\\u2014</span>';
}
function renderBonuses(b){
    let h='';
    for(let i=0;i<7;i++){const v=b[GN[i]]||0;if(v>0)h+='<div class="tok"><div class="bon-dot '+GC[i]+'">'+v+'</div></div>';}
    return h||'<span style="color:#445;font-size:11px">\\u2014</span>';
}
function scrollHtml(n,max){let s='';for(let i=0;i<max;i++)s+=i<n?'\\u269c':'\\u00b7';return s;}

/* ── Card columns with points sum ───────────────────────────────────── */
function renderCardColumns(player){
    // cols[key] = { cards: [html,...], points: number }
    const cols={};

    // wildcard_assignments: {card_id -> gem_index} from server
    const wca = player.wildcard_assignments || {};
    for(const c of player.cards){
        let key='none';
        if(c.is_wildcard){
            // Place wildcard in the column of its assigned colour if known
            const assigned = wca[c.id];
            key = (assigned !== undefined) ? assigned : 'wc';
        } else if(c.gem_bonus){
            for(let i=0;i<7;i++){if((c.gem_bonus[GN[i]]||0)>0){key=i;break;}}
        }
        if(!cols[key])cols[key]={cards:[],points:0};
        cols[key].cards.push(renderCard(c));
        cols[key].points+=c.points||0;
    }

    if(Object.keys(cols).length===0)return '<span style="color:#445;font-size:11px">\\u2014</span>';

    function colHeader(dotHtml, pts){
        const cls=pts>=10?'col-pts':'col-pts low';
        const ptsHtml=pts>0?'<div class="'+cls+'">'+pts+'</div>':'<div class="col-pts low">0</div>';
        return '<div class="col-header">'+dotHtml+ptsHtml+'</div>';
    }

    let h='<div class="cards-columns">';
    for(let i=0;i<7;i++){
        if(!cols[i])continue;
        const dot='<div class="col-gem '+GC[i]+'">'+GH[i]+'</div>';
        h+='<div class="card-col">'+colHeader(dot,cols[i].points)+cols[i].cards.join('')+'</div>';
    }
    if(cols['wc']){
        const dot='<div class="col-gem" style="border:1.5px solid #aab;font-size:11px">\\u2756</div>';
        h+='<div class="card-col">'+colHeader(dot,cols['wc'].points)+cols['wc'].cards.join('')+'</div>';
    }
    if(cols['none']){
        const dot='<div class="col-gem" style="border:1.5px solid #445;font-size:10px">\\u2014</div>';
        h+='<div class="card-col">'+colHeader(dot,cols['none'].points)+cols['none'].cards.join('')+'</div>';
    }
    return h+'</div>';
}

/* ── Player panel ────────────────────────────────────────────────────── */
function renderPlayer(p, pi, isActive){
    let h='<div class="pl'+(isActive?' active':'')+'"><div class="pl-hdr">'
        +'<span class="pl-name">Player '+pi+(isActive?' <span class="star">\\u2605</span>':'')+'</span>'
        +'<div class="pl-stats">'
        +'<div>\\ud83c\\udfc5 <span class="stat-val">'+p.points+'</span></div>'
        +'<div>\\u265b <span class="stat-val">'+p.crowns+'</span></div>'
        +'<div>\\u269c <span class="stat-val">'+p.scrolls+'</span></div>'
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
        h+='<div class="sec-lbl">Royals</div>'
          +'<div style="display:flex;gap:5px;flex-wrap:wrap">'+p.royals.map(r=>renderRoyal(r)).join('')+'</div>';
    return h+'</div>';
}

/* ── Full render ─────────────────────────────────────────────────────── */
function render(si){
    const step=STEPS[si],st=step.state;
    const over=st.phase==='GAME_OVER';
    const desc=step.description||(si===0?'Game start':'');
    const ap=step.player_acted;
    let h='';

    h+='<div class="hdr"><div class="hdr-left">'
      +'<div class="turn">Turn '+st.turn+' \\u00b7 Player '+st.current_player+' \\u00b7 '+(PHASE[st.phase]||st.phase)+'</div>'
      +'<div class="desc">'+(ap!==undefined?'P'+ap+': ':'')+desc+'</div>'
      +'</div><div class="nav">'
      +'<button onclick="go(0)">\\u23ee</button>'
      +'<button onclick="go(cur-1)">\\u25c0</button>'
      +'<span class="ctr">'+si+' / '+(STEPS.length-1)+'</span>'
      +'<button onclick="go(cur+1)">\\u25b6</button>'
      +'<button onclick="go(STEPS.length-1)">\\u23ed</button>'
      +'</div></div>';

    if(over){
        let w='?';
        for(let i=0;i<2;i++){
            const p=st.players[i];
            if(p.points>=20||p.crowns>=10){w=i;break;}
            const byCol={};
            for(const c of p.cards)if(!c.is_wildcard&&c.gem_bonus)
                for(const nm of GN)if((c.gem_bonus[nm]||0)>0){byCol[nm]=(byCol[nm]||0)+c.points;break;}
            for(const pts of Object.values(byCol))if(pts>=10){w=i;break;}
            if(w!=='?')break;
        }
        h+='<div class="winner">\\ud83c\\udfc6 Player '+w+' wins!</div>';
    }

    // Center column: board + pyramid + royals
    let center='<div class="center-col"><div class="board-panel">'+renderBoard(st.board)
        +'<div class="side-meta">'
        +'<div>Scrolls: <span class="scrolls">'+scrollHtml(st.scrolls_center,3)+'</span></div>'
        +'<div>Bag: '+st.bag_total+' tokens</div></div></div>'
        +'<div class="pyr-panel"><div class="sec-title">Pyramid</div>';
    for(const lvl of ['3','2','1']){
        const cards=st.pyramid[lvl]||[],ds=st.deck_sizes[lvl]||0;
        center+='<div class="pyr-row"><span class="pyr-lbl">L'+lvl+'</span>'
               +cards.map(c=>renderCard(c)).join('')
               +'<span class="deck-cnt">('+ds+')</span></div>';
    }
    center+='<div class="sec-title" style="margin-top:10px">Royal cards</div>'
           +'<div class="royals-row">'
           +(st.royal_cards.length?st.royal_cards.map(r=>renderRoyal(r)).join(''):'<span style="color:#445;font-size:11px">none</span>')
           +'</div></div></div>';

    const p0=st.players[0],p1=st.players[1];
    const a0=0===st.current_player&&!over,a1=1===st.current_player&&!over;

    h+='<div class="game-layout">'
      +renderPlayer(p0,0,a0)
      +center
      +renderPlayer(p1,1,a1)
      +'</div>';

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
