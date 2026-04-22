// board.js — 5×5 game board rendering

import { GEM_CSS, GEM_CHARS, gemImageUrl } from './constants.js';

/**
 * Render the 5×5 board into a container element.
 */
export function renderBoard(el, board, opts = {}) {
  const {
    validCells = new Set(),
    selectedCell = null,
    hintCells = new Set(),
    goldClickable = false,
    goldSelected = false,
    onCellClick = null,
    useImages = false,
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
        if (useImages) {
          const img = document.createElement('img');
          img.src = gemImageUrl(v);
          img.className = 'cell-gem-img';
          img.alt = GEM_CHARS[v];
          img.onerror = () => { img.remove(); cell.textContent = GEM_CHARS[v]; };
          cell.appendChild(img);
        } else {
          cell.textContent = GEM_CHARS[v];
        }

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