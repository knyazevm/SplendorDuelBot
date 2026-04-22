// board.js — 5×5 game board rendering

import { GEM_CSS, GEM_CHARS } from './constants.js';

/**
 * Render the 5×5 board into a container element.
 *
 * @param {HTMLElement} el - container
 * @param {number[][]} board - 5×5 grid of gem indices (-1 = empty)
 * @param {object} opts
 *   validCells: Set of "r,c" strings that are clickable
 *   selectedCell: {r,c} or null
 *   hintCells: Set of "r,c" to highlight as hint
 *   goldClickable: boolean — can gold cells be clicked
 *   goldSelected: boolean — is gold currently selected (reserve mode)
 *   onCellClick: function(r, c)
 */
export function renderBoard(el, board, opts = {}) {
  const {
    validCells = new Set(),
    selectedCell = null,
    hintCells = new Set(),
    goldClickable = false,
    goldSelected = false,
    onCellClick = null,
  } = opts;

  el.innerHTML = '';
  for (let r = 0; r < 5; r++) {
    for (let c = 0; c < 5; c++) {
      const v = board[r][c];
      const cell = document.createElement('div');
      cell.className = 'cell';

      if (v < 0) {
        cell.classList.add('empty');
      } else {
        cell.classList.add('filled', GEM_CSS[v]);
        cell.textContent = GEM_CHARS[v];

        if (v === 6) {
          // Gold cell
          if (goldClickable && onCellClick) {
            cell.style.cursor = 'pointer';
            cell.addEventListener('click', () => onCellClick(r, c));
            if (goldSelected) cell.classList.add('selected');
            else cell.classList.add('hint');
          }
        } else if (validCells.has(`${r},${c}`) && onCellClick) {
          cell.addEventListener('click', () => onCellClick(r, c));
        }
      }

      if (selectedCell && selectedCell.r === r && selectedCell.c === c) {
        cell.classList.add('selected');
      }
      if (hintCells.has(`${r},${c}`)) {
        cell.classList.add('hint');
      }

      el.appendChild(cell);
    }
  }
}
