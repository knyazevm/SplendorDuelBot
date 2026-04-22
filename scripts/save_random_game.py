from splendor_duel.viz.replay import GameLog
from splendor_duel.viz.terminal import TerminalRenderer
from splendor_duel.viz.html_renderer import render_html

# Записать рандомную игру
log = GameLog.play_random_game("data/cards.json", seed=42)

# Терминальная визуализация (пошагово)
renderer = TerminalRenderer()
renderer.render_game(log)

# HTML replay
render_html(log, "replays/game_001.html")