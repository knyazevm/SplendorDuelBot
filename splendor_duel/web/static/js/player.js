// player.js — Player panel rendering

import { GEM_NAMES, GEM_CSS, GEM_CHARS, gemImageUrl } from './constants.js';
import { renderCard, renderMiniCard } from './cards.js';

/**
 * Render a player panel.
 *
 * @param {HTMLElement} el - container
 * @param {object} player - player data from state
 * @param {object} opts
 *   label: string
 *   isHuman: boolean
 *   isActive: boolean
 *   canDiscard: function(gemIndex) → boolean
 *   onDiscard: function(gemIndex)
 *   canBuyReserved: function(index) → boolean
 *   onBuyReserved: function(index)
 */
export function renderPlayer(el, player, opts = {}) {
  const {
    label = 'Player',
    isHuman = false,
    isActive = false,
    canDiscard = () => false,
    onDiscard = null,
    canBuyReserved = () => false,
    onBuyReserved = null,
  } = opts;

  el.className = 'player'
    + (isActive ? ' active' : '')
    + (isHuman ? ' human' : '');

  let h = `<div class="pl-header">
    <span class="pl-name">${label}</span>
    <span class="pl-tag ${isHuman ? 'you' : 'ai'}">${isHuman ? 'YOU' : 'AI'}</span>
  </div>`;

  // Stats
  h += `<div class="pl-stats">
    <div class="stat">🏅 <b>${player.points}</b></div>
    <div class="stat">♛ <b>${player.crowns}</b></div>
    <div class="stat">⚜ <b>${player.scrolls}</b></div>
  </div>`;

  const imgMode = opts.useImages;

  // Tokens
  h += `<div class="sec-label">Tokens (${player.total_tokens}/10)</div><div class="tokens-row">`;
  let hasTokens = false;
  for (let i = 0; i < 7; i++) {
    const v = player.tokens[GEM_NAMES[i]] || 0;
    if (v > 0) {
      hasTokens = true;
      const clickable = canDiscard(i);
      const cls = clickable ? 'tok clickable' : 'tok';
      const click = clickable && onDiscard ? `onclick="window._onDiscard(${i})"` : '';
      const dot = imgMode
        ? `<img class="tok-gem-img" src="${gemImageUrl(i)}" alt="${GEM_CHARS[i]}">`
        : `<div class="tok-dot ${GEM_CSS[i]}">${GEM_CHARS[i]}</div>`;
      h += `<div class="${cls}" ${click}>${dot}<span class="tok-n">×${v}</span></div>`;
    }
  }
  if (!hasTokens) h += '<span class="empty-hint">—</span>';
  h += '</div>';

  // Bonuses
  h += `<div class="sec-label">Bonuses</div><div class="tokens-row">`;
  let hasBonuses = false;
  for (let i = 0; i < 7; i++) {
    const v = player.bonuses[GEM_NAMES[i]] || 0;
    if (v > 0) {
      hasBonuses = true;
      const dot = imgMode
        ? `<img class="tok-gem-img" src="${gemImageUrl(i)}" alt="${GEM_CHARS[i]}">`
        : `<div class="tok-dot ${GEM_CSS[i]}">${v}</div>`;
      h += `<div class="tok">${dot}<span class="tok-n">${v}</span></div>`;
    }
  }
  if (!hasBonuses) h += '<span class="empty-hint">—</span>';
  h += '</div>';

  // Cards grouped by bonus color
  const wca = player.wildcard_assignments || {};
  h += `<div class="sec-label">Cards (${player.cards.length})</div><div class="pl-cards">`;
  if (player.cards.length > 0) {
    for (const c of player.cards) {
      h += renderCard(c, { extraClass: 'card-sm', useImage: opts.useImages });
    }
  } else {
    h += '<span class="empty-hint">—</span>';
  }
  h += '</div>';

  // Reserved
  if (player.reserved.length > 0) {
    h += `<div class="sec-label">Reserved (${player.reserved.length})</div><div class="reserved-cards">`;
    player.reserved.forEach((c, idx) => {
      const buyable = canBuyReserved(idx);
      h += renderCard(c, {
        onclick: buyable ? `window._onBuyReserved(${idx})` : '',
        extraClass: buyable ? 'buyable' : '',
        useImage: opts.useImages,
      });
    });
    h += '</div>';
  }

  // Royals
  if (player.royals.length > 0) {
    h += `<div class="sec-label">Royals</div><div class="tokens-row">`;
    for (const r of player.royals) {
      h += `<span style="font-size:12px;margin-right:6px">${r.id} (${r.points}pts)</span>`;
    }
    h += '</div>';
  }

  el.innerHTML = h;

  // Attach global callbacks for onclick strings
  if (onDiscard) window._onDiscard = onDiscard;
  if (onBuyReserved) window._onBuyReserved = onBuyReserved;
}