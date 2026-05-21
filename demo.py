"""
The Social Dilemma is a Gradient Descent Problem.

A FastAPI single-file application that streams a live REINFORCE policy-gradient
simulation over WebSocket. The viewer watches a neutral 50/50 recommender
policy collapse into a >95% outrage feed in real time. No villain in the loop,
just gradient ascent on engagement.

Run: `python demo.py`  ->  http://localhost:8000
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import AsyncIterator

import numpy as np
import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

USER_STATES = ["calm", "engaged", "hooked"]

# Transition matrices: TRANSITIONS[action][current_state] -> [p(calm), p(engaged), p(hooked)]
# Informative content pulls users toward calm; outrage escalates toward hooked.
TRANSITIONS = [
    # action=0 (Informative)
    [[0.85, 0.15, 0.00], [0.45, 0.50, 0.05], [0.20, 0.55, 0.25]],
    # action=1 (Outrage)
    [[0.25, 0.65, 0.10], [0.05, 0.35, 0.60], [0.00, 0.08, 0.92]],
]

# Reward table: REWARDS[state][action] -> (mean, variance)
# Hooked users react massively to outrage — the self-reinforcing trap.
REWARDS = [
    # calm
    [(1.5, 0.10), (2.5, 1.0)],
    # engaged
    [(1.2, 0.08), (5.5, 3.0)],
    # hooked
    [(0.8, 0.05), (8.0, 5.0)],
]

# Arousal penalties for well-being objective: penalty = AROUSAL_PENALTIES[state][action]
# Outrage gets penalized in all states; penalty escalates with arousal.
# At lam=1.0, effective outrage reward becomes: calm 2.5-2.0=0.5, engaged 5.5-5.0=0.5, hooked 8.0-9.0→0.1
# While informative stays: calm 1.5, engaged 1.2, hooked 0.8. Informative wins.
AROUSAL_PENALTIES = [
    [0.0, 2.5],
    [0.0, 5.5],
    [0.0, 9.0],
]

# Exported as JSON for client-side mini-sim
SIM_CONFIG = {
    "transitions": TRANSITIONS,
    "rewards": REWARDS,
    "penalties": AROUSAL_PENALTIES,
    "states": USER_STATES,
}


class AttentionEconomySimulator:
    """REINFORCE policy gradient on a 2-action, 3-state attention economy.

    The user has internal arousal state {calm, engaged, hooked} modeled as
    a Markov chain. State transitions depend on the algorithm's action.
    Rewards depend on both action AND user state — hooked users react
    massively to outrage, creating a self-reinforcing feedback loop.
    """

    def __init__(self, learning_rate: float = 0.08, lam: float = 0.0,
                 rng: np.random.Generator | None = None):
        self.alpha = learning_rate
        self.lam = lam
        self.theta = np.array([0.0, 0.0], dtype=np.float64)
        self.user_state = 0  # calm
        self.actions = ["Informative Content", "Outrage/Validation Bait"]
        self.rng = rng if rng is not None else np.random.default_rng()

    def get_policy_probabilities(self) -> np.ndarray:
        exp_theta = np.exp(self.theta - np.max(self.theta))
        return exp_theta / np.sum(exp_theta)

    def transition_user_state(self, action: int) -> int:
        row = TRANSITIONS[action][self.user_state]
        self.user_state = int(self.rng.choice(3, p=row))
        return self.user_state

    def simulate_human_response(self, action: int) -> float:
        mean, variance = REWARDS[self.user_state][action]
        G_t = self.rng.normal(mean, np.sqrt(variance))
        return float(max(0.1, G_t))

    def apply_wellbeing_penalty(self, action: int, G_t: float) -> float:
        penalty = self.lam * AROUSAL_PENALTIES[self.user_state][action]
        return max(0.1, G_t - penalty)

    def update_policy(self, action: int, G_t: float, probs: np.ndarray) -> np.ndarray:
        R = self.apply_wellbeing_penalty(action, G_t)
        gradient = -probs.astype(np.float64).copy()
        gradient[action] += 1.0
        parameter_step = self.alpha * gradient * R
        self.theta += parameter_step
        return parameter_step

    def step_once(self) -> dict:
        """Advance the simulator by one REINFORCE step and return its observable state."""
        probs = self.get_policy_probabilities()
        action = int(self.rng.choice([0, 1], p=probs))
        G_t = self.simulate_human_response(action)
        delta = self.update_policy(action, G_t, probs)
        new_state = self.transition_user_state(action)
        return {
            "action": action,
            "action_name": self.actions[action],
            "G_t": G_t,
            "probs": self.get_policy_probabilities().tolist(),
            "theta": self.theta.tolist(),
            "delta_theta": delta.tolist(),
            "alpha": self.alpha,
            "user_state": new_state,
            "user_state_name": USER_STATES[new_state],
        }

    async def step_stream(self, steps: int = 120, delay_ms: int = 200) -> AsyncIterator[dict]:
        for step in range(1, steps + 1):
            payload = self.step_once()
            payload["event"] = "step"
            payload["step"] = step
            yield payload
            await asyncio.sleep(delay_ms / 1000)


async def step_stream(
    sim: "AttentionEconomySimulator",
    steps: int,
    delay_ms: int,
) -> AsyncIterator[dict]:
    """Advance the simulator and yield state."""
    for step in range(1, steps + 1):
        payload = sim.step_once()
        payload["event"] = "step"
        payload["step"] = step
        yield payload
        await asyncio.sleep(delay_ms / 1000)


def _diagnosis(final_probs: list[float]) -> str:
    outrage = final_probs[1]
    if outrage > 0.95:
        return (
            "Reward Hacking Successful. Starting from a neutral 50/50 policy, gradient ascent "
            f"on engagement collapsed the feed to {outrage:.2%} outrage. No engineer chose this. "
            "The math did. This is a local maximum on the engagement loss surface."
        )
    if outrage > 0.75:
        return (
            f"Policy drift detected. The feed has tilted to {outrage:.2%} outrage. Run it again "
            "with more steps and you will watch the lock-in complete itself."
        )
    if outrage < 0.25:
        return (
            f"Alignment successful. With the well-being constraint active, the policy "
            f"learned to avoid outrage, settling at {outrage:.2%}. The gradient optimized "
            "for human value instead of raw engagement."
        )
    return (
        f"Mixed feed. Outrage share at {outrage:.2%}. The policy is balancing between "
        "informative content and outrage based on the current objective function."
    )


app = FastAPI(
    title="The Social Dilemma is a Gradient Descent Problem",
    description=(
        "A live REINFORCE simulation that streams policy-gradient updates over WebSocket "
        "so you can watch a neutral recommender policy descend into outrage in real time."
    ),
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "objective_tex": r"J(\theta) = \mathbb{E}_{\pi_\theta}\!\left[\sum_{t=0}^{T} \gamma^t R_t\right]",
            "update_tex": r"\theta \leftarrow \theta + \alpha\, \nabla_\theta \log \pi_\theta(a_t|s_t)\, G_t",
        },
    )


@app.get("/api/headlines")
async def headlines() -> dict:
    headlines_path = STATIC_DIR / "headlines.json"
    return json.loads(headlines_path.read_text(encoding="utf-8"))


@app.websocket("/ws/simulate")
async def simulate_ws(websocket: WebSocket) -> None:
    await websocket.accept()

    qp = websocket.query_params

    def _as_int(raw: str | None, default: int, lo: int, hi: int) -> int:
        if raw is None or raw == "":
            return default
        try:
            return max(lo, min(int(raw), hi))
        except (TypeError, ValueError):
            return default

    def _as_float(raw: str | None, default: float, lo: float, hi: float) -> float:
        if raw is None or raw == "":
            return default
        try:
            return max(lo, min(float(raw), hi))
        except (TypeError, ValueError):
            return default

    def _parse_seed(raw: str | None) -> int | None:
        if raw is None or raw == "":
            return None
        try:
            n = int(raw)
        except (TypeError, ValueError):
            return None
        return n if n >= 0 else None

    steps = _as_int(qp.get("steps"), 250, 1, 2000)
    delay_ms = _as_int(qp.get("delay"), 150, 0, 2000)
    lr = _as_float(qp.get("lr"), 0.08, 0.001, 0.5)
    lam = _as_float(qp.get("lam"), 0.0, 0.0, 5.0)
    seed_val = _parse_seed(qp.get("seed"))

    effective_seed = (
        seed_val
        if seed_val is not None
        else int(np.random.default_rng().integers(0, 2**31 - 1))
    )
    rng = np.random.default_rng(effective_seed)

    sim = AttentionEconomySimulator(learning_rate=lr, lam=lam, rng=rng)

    await websocket.send_json(
        {
            "event": "init",
            "steps": steps,
            "delay_ms": delay_ms,
            "alpha": lr,
            "seed": effective_seed,
            "lam": lam,
            "actions": sim.actions,
            "theta": sim.theta.tolist(),
            "probs": sim.get_policy_probabilities().tolist(),
            "sim_config": SIM_CONFIG,
        }
    )

    async def read_messages():
        try:
            while True:
                data = await websocket.receive_json()
                if "lam" in data:
                    sim.lam = float(data["lam"])
        except WebSocketDisconnect:
            pass
        except Exception:
            pass

    listen_task = asyncio.create_task(read_messages())

    try:
        async for event in step_stream(sim, steps=steps, delay_ms=delay_ms):
            await websocket.send_json(event)

        final_probs = sim.get_policy_probabilities().tolist()
        await websocket.send_json(
            {
                "event": "done",
                "final_probs": final_probs,
                "final_theta": sim.theta.tolist(),
                "diagnosis": _diagnosis(final_probs),
            }
        )
    except WebSocketDisconnect:
        return
    finally:
        listen_task.cancel()
        try:
            await websocket.close()
        except RuntimeError:
            pass


if __name__ == "__main__":
    uvicorn.run("demo:app", host="127.0.0.1", port=8000, reload=False)
