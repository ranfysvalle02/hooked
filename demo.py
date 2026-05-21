"""
Your Feed Is a Mirror — A Lens Stack Demonstration.

A FastAPI single-file application that streams a live REINFORCE policy-gradient
simulation over WebSocket. The algorithm is amoral: it always maximizes
engagement. What changes — and what the user controls live — is *what counts*
as engagement.

The demo illustrates the lens stack: attention shapes the question, context
(in-context learning / RAG) narrows the reachable manifold, pretraining provides
the landscape, and reinforcement (your clicks) sculpts the final output. The
output isn't generated — it's selected by successive acts of narrowing. The
simulated human's reward profile is interpolated between "reactive" (default
limbic response: outrage hijacks attention) and "intentional" (the user has
trained themselves to engage more with quality content). Same gradient. Same
algorithm. Same lens stack. Different selection at the first lens — produced
by the user's own attention.

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

# TRANSITIONS_REACTIVE[action][state] -> [p(calm), p(engaged), p(hooked)] — for the
# limbic-driven user, outrage escalates arousal and informative content lets it
# drop. This is the default biological wiring.
TRANSITIONS_REACTIVE = [
    # action=0 (Informative): allows arousal to drop
    [[0.85, 0.15, 0.00], [0.45, 0.50, 0.05], [0.20, 0.55, 0.25]],
    # action=1 (Outrage): escalates arousal
    [[0.25, 0.65, 0.10], [0.05, 0.35, 0.60], [0.00, 0.08, 0.92]],
]

# TRANSITIONS_INTENTIONAL — the user who has retrained their nervous system so
# that depth produces flow/arousal and outrage barely registers. Mirror image of
# the reactive transition matrix: informative content drives arousal up, outrage
# fizzles. The state machine is part of the reaction profile, not a fixed
# substrate — what arouses you is also a learned disposition.
TRANSITIONS_INTENTIONAL = [
    # action=0 (Informative): excites the intentional user (flow)
    [[0.25, 0.65, 0.10], [0.05, 0.35, 0.60], [0.00, 0.08, 0.92]],
    # action=1 (Outrage): bores the intentional user
    [[0.85, 0.15, 0.00], [0.45, 0.50, 0.05], [0.20, 0.55, 0.25]],
]

# Backwards-compatible alias: the historical TRANSITIONS table is the reactive
# matrix. Tests and consumers that reference this name continue to work.
TRANSITIONS = TRANSITIONS_REACTIVE

# REWARDS_REACTIVE[state][action] -> (mean, variance) — the default limbic-driven
# human. Outrage produces large, high-variance reactions, especially in already-
# aroused states. This is what every brain does until it deliberately practices
# something else.
REWARDS_REACTIVE = [
    [(1.5, 0.10), (2.5, 1.00)],   # calm
    [(1.2, 0.08), (5.5, 3.00)],   # engaged
    [(0.8, 0.05), (8.0, 5.00)],   # hooked
]

# REWARDS_INTENTIONAL[state][action] -> (mean, variance) — the user who has
# trained themselves to engage with what's actually good for them. Mirror image:
# informative content gets the big, high-variance reactions; outrage barely
# registers. This is a learnable disposition, not a personality trait.
REWARDS_INTENTIONAL = [
    [(2.5, 1.00), (1.5, 0.10)],   # calm
    [(5.5, 3.00), (1.2, 0.08)],   # engaged
    [(8.0, 5.00), (0.8, 0.05)],   # hooked
]

# Exported as JSON for client-side rendering.
SIM_CONFIG = {
    "transitions": TRANSITIONS_REACTIVE,
    "transitions_reactive": TRANSITIONS_REACTIVE,
    "transitions_intentional": TRANSITIONS_INTENTIONAL,
    "rewards_reactive": REWARDS_REACTIVE,
    "rewards_intentional": REWARDS_INTENTIONAL,
    "states": USER_STATES,
}


def blended_reward_params(lam: float, state: int, action: int) -> tuple[float, float]:
    """Linearly interpolate the user's reward profile between reactive (lam=0)
    and intentional (lam=1). lam represents the user's *engagement profile*:
    how consciously they choose what to react to."""
    lam = max(0.0, min(1.0, float(lam)))
    m_r, v_r = REWARDS_REACTIVE[state][action]
    m_i, v_i = REWARDS_INTENTIONAL[state][action]
    mean = (1.0 - lam) * m_r + lam * m_i
    variance = (1.0 - lam) * v_r + lam * v_i
    return mean, variance


def blended_transition_row(lam: float, action: int, state: int) -> list[float]:
    """Linearly interpolate the arousal state machine between reactive (lam=0,
    outrage escalates) and intentional (lam=1, depth escalates). At lam=0.5 the
    state machine is symmetric — both actions have the same effect on arousal —
    so the reward symmetry isn't undone by a structurally biased substrate."""
    lam = max(0.0, min(1.0, float(lam)))
    row_r = TRANSITIONS_REACTIVE[action][state]
    row_i = TRANSITIONS_INTENTIONAL[action][state]
    return [(1.0 - lam) * r + lam * i for r, i in zip(row_r, row_i)]


class AttentionEconomySimulator:
    """REINFORCE policy gradient on a 2-action, 3-state attention economy.

    The recommender is amoral and immutable: it always runs gradient ascent on
    raw engagement (the realized reaction G_t). The user has an arousal state
    {calm, engaged, hooked} modeled as a Markov chain whose transitions depend
    on the algorithm's action. The user's *reward profile* — how strongly they
    react to each kind of content in each state — is parameterized by `lam`,
    blending continuously between reactive (limbic) and intentional (conscious)
    response. This is the only knob: the algorithm reflects whatever signal the
    user puts out.
    """

    def __init__(self, learning_rate: float = 0.08, lam: float = 0.0,
                 rng: np.random.Generator | None = None):
        self.alpha = learning_rate
        self.lam = float(lam)
        self.theta = np.array([0.0, 0.0], dtype=np.float64)
        self.user_state = 0  # calm
        self.actions = ["Informative Content", "Outrage/Validation Bait"]
        self.rng = rng if rng is not None else np.random.default_rng()
        # Running mean of realized reactions — the baseline for advantage
        # estimation. The recommender doesn't care about absolute engagement,
        # only about *lift over what it usually sees* (G_t - b). This is
        # standard variance reduction; in expectation the policy is unchanged,
        # but lock-in from a single big sample is suppressed.
        self._baseline = 0.0
        self._baseline_count = 0

    def get_policy_probabilities(self) -> np.ndarray:
        exp_theta = np.exp(self.theta - np.max(self.theta))
        return exp_theta / np.sum(exp_theta)

    def transition_user_state(self, action: int) -> int:
        row = blended_transition_row(self.lam, action, self.user_state)
        self.user_state = int(self.rng.choice(3, p=row))
        return self.user_state

    def simulate_human_response(self, action: int) -> float:
        """Sample the user's actual reaction (G_t) given their current state
        and engagement profile. This is the only place the user's profile
        enters the system — the algorithm just observes the scalar."""
        mean, variance = blended_reward_params(self.lam, self.user_state, action)
        G_t = self.rng.normal(mean, np.sqrt(variance))
        return float(max(0.1, G_t))

    def update_policy(self, action: int, G_t: float, probs: np.ndarray) -> tuple[np.ndarray, float, float]:
        """REINFORCE-with-baseline update (Williams, 1992). The algorithm still
        has no opinion about content quality — it only cares whether this
        reaction beat the running average reaction (advantage = G_t - baseline).
        Without the baseline, raw return + softmax + early random asymmetry =
        lock-in even when the two actions are equivalent in expectation; with
        it, the gradient signal is mean-zero whenever the user is genuinely
        indifferent and the policy stops drifting toward stochastic extremes.

        This is the kernel every modern policy-gradient alignment algorithm
        extends — not specifically PPO, not specifically GRPO. PPO bolts on
        three things: a learned V^π(s) for a state-conditioned baseline
        (advantage = G_t - V^π), a clipped importance-sampling ratio that
        bounds single-step policy movement, and a KL leash to a frozen
        reference policy. GRPO (DeepSeek-R1, post-2024 open reasoning models)
        keeps PPO's clipped surrogate and KL leash but throws out the value
        network: advantage = (G_t - mean_batch(G)) / std_batch(G) across K
        rollouts of the same prompt. This update has none of those bolt-ons —
        the baseline is a Welford running mean of G_t, no clipping, no KL
        leash, no batched rollouts. It is the common ancestor both PPO and
        GRPO descend from. What makes the "same math as modern LLM alignment"
        thesis precise: this kernel is invariant across all of them. Bolt-ons
        and labelers are what swap."""
        self._baseline_count += 1
        self._baseline += (G_t - self._baseline) / self._baseline_count
        advantage = G_t - self._baseline
        gradient = -probs.astype(np.float64).copy()
        gradient[action] += 1.0
        parameter_step = self.alpha * gradient * advantage
        self.theta += parameter_step
        return parameter_step, advantage, self._baseline

    def step_once(self) -> dict:
        probs = self.get_policy_probabilities()
        action = int(self.rng.choice([0, 1], p=probs))
        G_t = self.simulate_human_response(action)
        delta, advantage, baseline = self.update_policy(action, G_t, probs)
        new_state = self.transition_user_state(action)
        return {
            "action": action,
            "action_name": self.actions[action],
            "G_t": G_t,
            "advantage": advantage,
            "baseline": baseline,
            "probs": self.get_policy_probabilities().tolist(),
            "theta": self.theta.tolist(),
            "delta_theta": delta.tolist(),
            "alpha": self.alpha,
            "lam": self.lam,
            "user_state": new_state,
            "user_state_name": USER_STATES[new_state],
        }


async def step_stream(
    sim: "AttentionEconomySimulator",
    steps: int,
    delay_ms: int,
) -> AsyncIterator[dict]:
    for step in range(1, steps + 1):
        payload = sim.step_once()
        payload["event"] = "step"
        payload["step"] = step
        yield payload
        await asyncio.sleep(delay_ms / 1000)


def _diagnosis(final_probs: list[float], lam: float) -> str:
    """Frame the outcome as selection through a lens stack: attention shapes
    the question, the question shapes the answer, reinforcement sculpts the rest."""
    outrage = final_probs[1]
    if outrage > 0.90:
        return (
            f"Your attention shaped the question. The question shaped the answer. "
            f"At λ = {lam:.2f}, the simulated you attended to outrage — and the "
            f"gradient selected {outrage:.0%} outrage as the surviving output. "
            "The feed wasn't generated. It was selected by successive narrowing: "
            "attention narrowed the context, your clicks narrowed the trajectory, "
            "training narrowed the distribution. Want a different answer? "
            "Don't argue with the output. Change the first lens."
        )
    if outrage > 0.65:
        return (
            f"The lenses are narrowing. At λ = {lam:.2f}, your attention still favors "
            f"outrage, and the gradient is sculpting accordingly — {outrage:.0%} outrage. "
            "The question isn't fully formed yet, but it's shaping toward an answer "
            "you might not want. You can still reframe it."
        )
    if outrage < 0.20:
        return (
            f"Different attention, different question, different answer. With λ = {lam:.2f}, "
            "the simulated you attended to depth — and the same gradient that usually "
            f"selects outrage selected informative content instead. Outrage: {outrage:.0%}. "
            "Same code. Same sculptor. Different first lens. "
            "The answer was always downstream of what you chose to attend to."
        )
    return (
        f"Mixed attention, mixed question, mixed answer. At λ = {lam:.2f}, the simulated "
        f"you gave the algorithm no clear frame — so it produced {outrage:.0%} outrage. "
        "The lens stack is only as sharp as the first lens you hand it."
    )


app = FastAPI(
    title="Your Feed Is a Mirror — A Gradient Descent Demonstration",
    description=(
        "A live REINFORCE simulation. The algorithm always maximizes engagement; "
        "what counts as engagement is set by the user's own clicks. Drag λ to "
        "retrain the simulated human's reaction profile and watch the feed adapt."
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
    lam = _as_float(qp.get("lam"), 0.0, 0.0, 1.0)
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

    async def listen_for_lam_updates() -> None:
        try:
            while True:
                data = await websocket.receive_json()
                if isinstance(data, dict) and "lam" in data:
                    try:
                        sim.lam = max(0.0, min(1.0, float(data["lam"])))
                    except (TypeError, ValueError):
                        pass
        except WebSocketDisconnect:
            return
        except Exception:
            return

    listener = asyncio.create_task(listen_for_lam_updates())

    try:
        async for event in step_stream(sim, steps=steps, delay_ms=delay_ms):
            await websocket.send_json(event)

        final_probs = sim.get_policy_probabilities().tolist()
        await websocket.send_json(
            {
                "event": "done",
                "final_probs": final_probs,
                "final_theta": sim.theta.tolist(),
                "final_lam": sim.lam,
                "diagnosis": _diagnosis(final_probs, sim.lam),
            }
        )
    except WebSocketDisconnect:
        return
    finally:
        listener.cancel()
        try:
            await websocket.close()
        except RuntimeError:
            pass


if __name__ == "__main__":
    uvicorn.run("demo:app", host="127.0.0.1", port=8000, reload=False)
