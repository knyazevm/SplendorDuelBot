"""
server.py — FastAPI server for human-vs-AI play.

Endpoints:
    GET  /              → serve game.html
    POST /api/new_game  → create game
    POST /api/action    → apply ONE human action (no AI auto-play)
    POST /api/ai_turn   → run AI until human's turn
    GET  /api/state     → current state
    GET  /api/agents    → list available agents
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from splendor_duel.game.engine import GameEngine
from splendor_duel.game.state import GameState
from splendor_duel.agents import BaseAgent, RandomAgent, GreedyAgent, MCTSAgent
from .serialize import state_to_dict, serialize_action

CARDS_PATH = "data/cards.json"
WEB_DIR = Path(__file__).parent
TEMPLATES_DIR = WEB_DIR / "templates"
STATIC_DIR = WEB_DIR / "static"

app = FastAPI()

# Serve card/gem images from project root images/ directory
# Must be mounted BEFORE /static so the more specific path matches first
IMAGES_DIR = Path(CARDS_PATH).parent.parent / "images"
if IMAGES_DIR.exists():
    app.mount("/static/images", StaticFiles(directory=str(IMAGES_DIR)), name="images")

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── Session ───────────────────────────────────────────────────────────────────

class Session:
    def __init__(self):
        self.state: Optional[GameState] = None
        self.agent: Optional[BaseAgent] = None
        self.human_player: int = 0
        self.log: list[str] = []


session = Session()

AGENTS = {
    "Random": lambda: RandomAgent(),
    "Greedy": lambda: GreedyAgent(),
    "MCTS_fast": lambda: MCTSAgent(iterations=200, rollout='none'),
    "MCTS_greedy": lambda: MCTSAgent(iterations=100, rollout='greedy', rollout_depth=20),
}


# Auto-register neural network agents if checkpoints exist
def _try_register_ppo():
    import glob
    ppo_files = sorted(glob.glob("checkpoints/ppo_*.pt"))
    if not ppo_files:
        return
    try:
        from splendor_duel.agents.ppo import PPOAgent
        for path in ppo_files:
            name = Path(path).stem  # e.g. ppo_100
            AGENTS[name] = lambda p=path: PPOAgent.load(p)
    except Exception as e:
        print(f"PPO registration failed: {e}")


def _try_register_az():
    import glob
    az_files = sorted(glob.glob("checkpoints_az/az_*.pt"))
    if not az_files:
        return
    try:
        from splendor_duel.agents.az import AZAgent
        for path in az_files:
            name = Path(path).stem  # e.g. az_10
            AGENTS[name] = lambda p=path: AZAgent.load(p, n_simulations=100)
    except Exception as e:
        print(f"AZ registration failed: {e}")


_try_register_ppo()
_try_register_az()


# ── Request models ────────────────────────────────────────────────────────────

class NewGameRequest(BaseModel):
    agent: str = "Greedy"
    player_side: int = 0


class ActionRequest(BaseModel):
    action_index: int


# ── Response builder ──────────────────────────────────────────────────────────

def _build_response() -> dict:
    state = session.state
    is_human_turn = (
            not state.is_game_over
            and state.current_player == session.human_player
    )
    legal = []
    if is_human_turn:
        actions = GameEngine.get_legal_actions(state)
        legal = [serialize_action(a, state) for a in actions]

    return {
        "state": state_to_dict(state),
        "legal_actions": legal,
        "human_player": session.human_player,
        "is_human_turn": is_human_turn,
        "is_ai_turn": not state.is_game_over and not is_human_turn,
        "log": session.log[-50:],
        "agent_name": session.agent.name if session.agent else "",
    }


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    return (TEMPLATES_DIR / "game.html").read_text(encoding="utf-8")


@app.get("/api/agents")
async def get_agents():
    return {"agents": list(AGENTS.keys())}


@app.post("/api/new_game")
async def new_game(req: NewGameRequest):
    if req.agent not in AGENTS:
        raise HTTPException(400, f"Unknown agent: {req.agent}")

    import random
    random.seed()

    session.agent = AGENTS[req.agent]()
    session.state = GameState.new_game(CARDS_PATH)
    session.human_player = req.player_side
    session.log = ["Game started"]
    session.agent.notify_game_start(1 - session.human_player)

    return _build_response()


@app.post("/api/action")
async def apply_action(req: ActionRequest):
    """Apply ONE human action. Does NOT run AI."""
    if session.state is None:
        raise HTTPException(400, "No game in progress")
    if session.state.is_game_over:
        raise HTTPException(400, "Game is over")
    if session.state.current_player != session.human_player:
        raise HTTPException(400, "Not your turn")

    actions = GameEngine.get_legal_actions(session.state)
    if req.action_index < 0 or req.action_index >= len(actions):
        raise HTTPException(400, f"Invalid action index")

    action = actions[req.action_index]
    desc = serialize_action(action, session.state).get("desc", "?")
    session.log.append(f"You: {desc}")
    session.state = GameEngine.apply_action(session.state, action)

    return _build_response()


@app.post("/api/ai_turn")
async def ai_turn():
    """Run AI actions until it's the human's turn (or game over)."""
    if session.state is None:
        raise HTTPException(400, "No game in progress")
    if session.state.is_game_over:
        return _build_response()
    if session.state.current_player == session.human_player:
        return _build_response()  # already human's turn

    state = session.state
    for _ in range(200):
        if state.is_game_over or state.current_player == session.human_player:
            break
        actions = GameEngine.get_legal_actions(state)
        if not actions:
            break
        action = session.agent.choose_action(state, actions)
        desc = serialize_action(action, state).get("desc", "?")
        session.log.append(f"🤖 {desc}")
        state = GameEngine.apply_action(state, action)

    session.state = state
    return _build_response()


@app.get("/api/state")
async def get_state():
    if session.state is None:
        raise HTTPException(400, "No game in progress")
    return _build_response()
