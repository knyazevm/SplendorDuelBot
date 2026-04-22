// cards.js — Card and Royal card rendering (shared between game and replay)

import { GEM_NAMES, GEM_CSS, GEM_CHARS, ABILITY_ICON, IMAGES_BASE } from './constants.js';

/**
 * Get the CSS class for a card's bonus gem color.
 */
export function cardGemClass(card) {
  if (card.is_wildcard) return 'g3';
  if (card.gem_bonus) {
    for (let i = 0; i < 7; i++) {
      if ((card.gem_bonus[GEM_NAMES[i]] || 0) > 0) return GEM_CSS[i];
    }
  }
  return 'g1';
}

/**
 * Render a card as HTML string (digital style).
 * @param {object} card - card data
 * @param {object} opts - { onclick, extraClass, useImage }
 */
export function renderCard(card, opts = {}) {
  const { onclick = '', extraClass = '', useImage = false } = opts;

  if (useImage) {
    return renderCardImage(card, opts);
  }
  return renderCardDigital(card, opts);
}

function renderCardDigital(card, { onclick = '', extraClass = '' } = {}) {
  const lvl = card.level ? 'cl' + card.level : 'cl1';
  const pts = `<span class="card-pts${card.points ? '' : ' zero'}">${card.points}</span>`;
  const crowns = card.crowns ? `<span class="card-crowns">${'♛'.repeat(card.crowns)}</span>` : '';

  let bonus = '';
  if (card.is_wildcard) {
    bonus = '<div class="gem-circ wc"></div>';
  } else if (card.gem_bonus) {
    for (let i = 0; i < 7; i++) {
      const v = card.gem_bonus[GEM_NAMES[i]] || 0;
      for (let j = 0; j < v; j++) bonus += `<div class="gem-circ ${GEM_CSS[i]}"></div>`;
    }
  }

  const abil = card.ability
    ? `<div class="card-abil">${ABILITY_ICON[card.ability] || card.ability}</div>`
    : '';

  let cost = '';
  if (card.cost) {
    for (let i = 0; i < 7; i++) {
      const v = card.cost[GEM_NAMES[i]] || 0;
      if (v > 0) cost += `<div class="cost-pip"><div class="cost-dot ${GEM_CSS[i]}">${v}</div></div>`;
    }
  }

  const attr = onclick ? `onclick="${onclick}"` : '';
  return `<div class="card ${lvl} ${extraClass}" ${attr} title="${card.id}">
    <div class="card-top">${pts}${crowns}
      <div class="card-bonus-col">${bonus}</div>
    </div>
    ${abil}
    <div class="card-cost">${cost}</div>
  </div>`;
}

function renderCardImage(card, { onclick = '', extraClass = '' } = {}) {
  const src = `${IMAGES_BASE}/cards/${card.id}.png`;
  const fbId = 'fb_' + card.id.replace(/[^a-zA-Z0-9]/g, '_');
  const attr = onclick ? `onclick="${onclick}"` : '';
  const fallback = renderCardDigital(card, { onclick, extraClass });

  return `<div class="card-img-wrap ${extraClass}" ${attr} title="${card.id}">
    <img class="card-img" src="${src}" alt="${card.id}"
         onerror="this.parentElement.outerHTML=document.getElementById('${fbId}').innerHTML">
    <template id="${fbId}">${fallback}</template>
  </div>`;
}

/**
 * Render a royal card.
 */
export function renderRoyal(royal, { onclick = '', extraClass = '', useImage = false } = {}) {
  if (useImage) {
    return renderRoyalImage(royal, { onclick, extraClass });
  }
  const abil = royal.ability
    ? `<div class="royal-abil">${ABILITY_ICON[royal.ability] || royal.ability}</div>`
    : '';
  const attr = onclick ? `onclick="${onclick}"` : '';
  return `<div class="royal ${extraClass}" ${attr}>
    <div class="royal-pts">${royal.points}</div>
    ${abil}
    <div style="font-size:9px;color:var(--text2)">${royal.id}</div>
  </div>`;
}

function renderRoyalImage(royal, { onclick = '', extraClass = '' } = {}) {
  const src = `${IMAGES_BASE}/cards/${royal.id}.png`;
  const fbId = 'fb_' + royal.id.replace(/[^a-zA-Z0-9]/g, '_');
  const attr = onclick ? `onclick="${onclick}"` : '';
  const fallback = renderRoyal(royal, { onclick, extraClass, useImage: false });
  return `<div class="royal-img-wrap ${extraClass}" ${attr} title="${royal.id}">
    <img class="royal-img" src="${src}" alt="${royal.id}"
         onerror="this.parentElement.outerHTML=document.getElementById('${fbId}').innerHTML">
    <template id="${fbId}">${fallback}</template>
  </div>`;
}

/**
 * Render a mini card (for bought cards in player panel).
 */
export function renderMiniCard(card, wca = {}) {
  let gc;
  if (card.is_wildcard) {
    const assigned = wca[card.id];
    gc = assigned !== undefined ? GEM_CSS[assigned] : 'wc-mini';
  } else {
    gc = cardGemClass(card);
  }
  const crown = card.crowns ? '♛' : '';
  const abil = card.ability ? ' ⚡' : '';
  return `<div class="mini-card ${gc}" title="${card.id}: ${card.points}pts ${crown}${abil}">
    ${card.points || '·'}
  </div>`;
}