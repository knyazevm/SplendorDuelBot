"""
terminal.py — Rich-based terminal visualization.

Usage:
    from splendor_duel.viz.terminal import TerminalRenderer
    renderer = TerminalRenderer()
    renderer.render_state(state)
    renderer.render_game(log)             # step-through replay
    renderer.render_game(log, auto=True)  # auto-play
"""
from __future__ import annotations

import time
from typing import Optional

from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from splendor_duel.game.actions import Phase
from splendor_duel.game.constants import BOARD_SIZE, GEM_NAMES, Gem, N_GEMS
from splendor_duel.game.state import GameState
from .replay import GameLog, describe_action

console = Console()

# ── Gem rendering ─────────────────────────────────────────────────────────────

GEM_RICH_STYLES = {
    -1: ('·', 'dim'),
    Gem.WHITE: ('W', 'bold white'),
    Gem.BLACK: ('K', 'bold grey37'),
    Gem.RED: ('R', 'bold red'),
    Gem.BLUE: ('B', 'bold blue'),
    Gem.GREEN: ('G', 'bold green'),
    Gem.PEARL: ('P', 'bold cyan'),
    Gem.GOLD: ('$', 'bold yellow'),
}

GEM_COMPACT = {
    Gem.WHITE: ('W', 'white'), Gem.BLACK: ('K', 'grey37'),
    Gem.RED: ('R', 'red'), Gem.BLUE: ('B', 'blue'),
    Gem.GREEN: ('G', 'green'), Gem.PEARL: ('P', 'cyan'),
    Gem.GOLD: ('$', 'yellow'),
}


class TerminalRenderer:

    def __init__(self, clear: bool = True) -> None:
        self.clear = clear

    # ── Main render ───────────────────────────────────────────────────────────

    def render_state(
            self,
            state: GameState,
            action_desc: str = '',
            step: int = 0,
            total_steps: int = 0,
    ) -> None:
        if self.clear:
            console.clear()

        # Header
        phase_name = state.phase.name
        header = f"Turn {state.turn}  ·  Player {state.current_player}  ·  {phase_name}"
        if total_steps:
            header += f"  ·  Step {step}/{total_steps}"
        console.print(Panel(header, style="bold", border_style="blue"))

        # Board + scrolls
        console.print(self._render_board(state))

        # Players side by side
        cols = Table.grid(padding=(0, 4))
        cols.add_column(width=36)
        cols.add_column(width=36)
        marker = ['', '']
        marker[state.current_player] = ' ★'
        cols.add_row(
            self._render_player(state.players[0], f"Player 0{marker[0]}"),
            self._render_player(state.players[1], f"Player 1{marker[1]}"),
        )
        console.print(cols)

        # Pyramid
        console.print(self._render_pyramid(state))

        # Royal cards
        if state.royal_cards:
            royals_text = '  '.join(
                f"[bold]{r.id}[/] ({r.points}pts"
                + (f", {r.ability}" if r.ability else '')
                + ')'
                for r in state.royal_cards
            )
            console.print(Panel(royals_text, title="Royal cards", border_style="magenta"))

        # Action
        if action_desc:
            console.print(f"  [yellow]▸ {action_desc}[/]")
            console.print()

        # Game over
        if state.is_game_over:
            winner = state.winner
            vic = state.players[winner].check_victory() if winner is not None else '?'
            console.print(
                Panel(
                    f"[bold green]Player {winner} wins! ({vic})[/]",
                    border_style="green",
                )
            )

    # ── Board ─────────────────────────────────────────────────────────────────

    def _render_board(self, state: GameState) -> Panel:
        board = state.board.grid
        lines: list[Text] = []

        # Column headers
        header = Text("     ")
        for c in range(BOARD_SIZE):
            header.append(f" {c} ", style="dim")
        lines.append(header)

        for r in range(BOARD_SIZE):
            row = Text(f"  {r}  ")
            for c in range(BOARD_SIZE):
                val = int(board[r, c])
                char, style = GEM_RICH_STYLES.get(val, ('?', ''))
                row.append(f" {char} ", style=style)
            lines.append(row)

        scrolls_str = '⚜' * state.scrolls_center + '·' * (3 - state.scrolls_center)
        lines.append(Text(f"\n  Scrolls: {scrolls_str}  |  Bag: {sum(state.bag.values())} tokens", style="dim"))

        content = Text('\n').join(lines)
        return Panel(content, title="Board", border_style="blue")

    # ── Player info ───────────────────────────────────────────────────────────

    def _render_player(self, player, title: str) -> Panel:
        lines: list[str] = []

        # Points & crowns
        lines.append(f"Points: [bold]{player.points}[/]  Crowns: [bold]{player.crowns}[/]")

        # Tokens
        tok = Text("Tokens: ")
        for g in range(N_GEMS):
            if player.tokens[g] > 0:
                char, style = GEM_COMPACT[g]
                tok.append(f"{char}:{player.tokens[g]} ", style=style)
        tok.append(f"({player.total_tokens})", style="dim")
        lines.append(tok)

        # Bonuses
        bon = Text("Bonus:  ")
        for g in range(N_GEMS):
            if player.bonuses[g] > 0:
                char, style = GEM_COMPACT[g]
                bon.append(f"{char}:{player.bonuses[g]} ", style=style)
        lines.append(bon)

        # Scrolls
        lines.append(f"Scrolls: {'⚜' * player.scrolls}")

        # Cards count
        lines.append(f"Cards: {len(player.cards)}  Reserved: {len(player.reserved)}")

        # Royals
        if player.royals:
            lines.append(f"Royals: {', '.join(r.id for r in player.royals)}")

        content = Text('\n')
        for line in lines:
            if isinstance(line, str):
                content.append_text(Text.from_markup(line))
            else:
                content.append_text(line)
            content.append('\n')

        style = "green" if '★' in title else "white"
        return Panel(content, title=title, border_style=style)

    # ── Pyramid ───────────────────────────────────────────────────────────────

    def _render_pyramid(self, state: GameState) -> Panel:
        lines: list[Text] = []
        for lvl in (3, 2, 1):
            cards = state.pyramid.get(lvl, [])
            deck_size = len(state.decks.get(lvl, []))
            row = Text(f"  L{lvl} ({deck_size:2d} in deck):  ")
            for card in cards:
                # Compact card display
                bonus_str = ''
                if card.is_wildcard:
                    bonus_str = '★'
                elif card.gem_bonus:
                    for i, v in enumerate(card.gem_bonus):
                        if v > 0:
                            char, _ = GEM_COMPACT[i]
                            bonus_str = char * v
                            break

                cost_total = sum(card.cost)
                effect = ''
                if card.ability:
                    effect = ' ⚡'
                crown_str = '♛' * card.crowns

                card_text = f"[{card.id} {card.points}pt {bonus_str}{crown_str}{effect}]  "
                row.append(card_text, style="dim" if cost_total > 6 else "")
            lines.append(row)

        content = Text('\n').join(lines)
        return Panel(content, title="Pyramid", border_style="yellow")

    # ── Game replay ───────────────────────────────────────────────────────────

    def render_game(
            self,
            log: GameLog,
            auto: bool = False,
            delay: float = 0.5,
            skip_minor: bool = True,
    ) -> None:
        """
        Step through a game log.

        auto=True: play automatically with delay
        skip_minor=True: skip ProceedToMain phases in display
        """
        for i, state in enumerate(log.states):
            # Action description
            desc = ''
            if i > 0:
                action = log.actions[i - 1]
                prev = log.states[i - 1]
                if skip_minor and isinstance(action, type) and \
                        action.__class__.__name__ == 'ProceedToMain':
                    continue
                desc = describe_action(action, prev)

            self.render_state(
                state,
                action_desc=desc,
                step=i,
                total_steps=log.n_steps,
            )

            if state.is_game_over:
                break

            if auto:
                time.sleep(delay)
            else:
                try:
                    inp = input("  [Enter] next  [q] quit  [j] jump 10: ")
                    if inp.strip().lower() == 'q':
                        break
                    if inp.strip().lower() == 'j':
                        # Skip ahead (handled by caller if needed)
                        pass
                except (KeyboardInterrupt, EOFError):
                    break
