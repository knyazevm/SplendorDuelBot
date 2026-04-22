from splendor_duel.viz.replay import GameLog
from splendor_duel.viz.terminal import TerminalRenderer
from splendor_duel.viz.html_renderer import render_html

# Записать рандомную игру
log = GameLog.play_random_game("data/cards.json", seed=42)

# Терминальная визуализация (пошагово)
renderer = TerminalRenderer()
renderer.render_game(log)

# Без фото — CSS-карточки
render_html(log, "replays/replay.html")

# С фото — показывает images/cards/L1_01.jpg, fallback на CSS если фото нет
render_html(log, "replays/replay_with_images.html", images_dir="../images")
