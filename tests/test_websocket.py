import pytest
from httpx import AsyncClient, ASGITransport
from starlette.testclient import TestClient

from demo import app


@pytest.mark.asyncio
async def test_index_returns_html():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.get("/")
        assert resp.status_code == 200
        assert "policy-chart" in resp.text
        assert "og:title" in resp.text


def test_websocket_streams_events():
    client = TestClient(app)
    with client.websocket_connect("/ws/simulate?steps=5&delay=0&seed=1&lam=0") as ws:
        init = ws.receive_json()
        assert init["event"] == "init"
        assert "sim_config" in init
        assert init["seed"] == 1
        assert init["lam"] == 0.0
        assert abs(sum(init["probs"]) - 1.0) < 1e-6
        cfg = init["sim_config"]
        assert "rewards_reactive" in cfg
        assert "rewards_intentional" in cfg

        steps_received = 0
        done_evt = None
        for _ in range(20):
            msg = ws.receive_json()
            if msg["event"] == "step":
                steps_received += 1
                assert "user_state_name" in msg
                assert msg["user_state_name"] in ("calm", "engaged", "hooked")
                assert "lam" in msg
            elif msg["event"] == "done":
                done_evt = msg
                break

        assert steps_received == 5
        assert done_evt is not None
        assert "diagnosis" in done_evt
        assert "final_lam" in done_evt
        assert abs(sum(done_evt["final_probs"]) - 1.0) < 1e-6


def test_websocket_intentional_diverges_from_reactive():
    """With identical seed, reactive (lam=0) should converge to outrage and
    intentional (lam=1) should converge to informative. Same algorithm, same
    gradient — only the simulated user's reaction profile differs."""
    client = TestClient(app)

    def run(lam):
        with client.websocket_connect(
            f"/ws/simulate?steps=250&delay=0&seed=42&lam={lam}"
        ) as ws:
            ws.receive_json()  # init
            done = None
            for _ in range(400):
                msg = ws.receive_json()
                if msg["event"] == "done":
                    done = msg
                    break
            return done

    reactive = run(0.0)
    intentional = run(1.0)
    assert reactive is not None and intentional is not None
    assert reactive["final_probs"][1] > intentional["final_probs"][1], (
        f"Reactive outrage ({reactive['final_probs'][1]:.2%}) should exceed "
        f"intentional outrage ({intentional['final_probs'][1]:.2%})."
    )


def test_websocket_respects_seed():
    client = TestClient(app)

    def collect(seed):
        with client.websocket_connect(
            f"/ws/simulate?steps=10&delay=0&seed={seed}&lam=0"
        ) as ws:
            events = []
            for _ in range(20):
                msg = ws.receive_json()
                events.append(msg)
                if msg["event"] == "done":
                    break
            return events

    run1 = collect(77)
    run2 = collect(77)
    steps1 = [e for e in run1 if e["event"] == "step"]
    steps2 = [e for e in run2 if e["event"] == "step"]
    for s1, s2 in zip(steps1, steps2):
        assert s1["action"] == s2["action"]
        assert abs(s1["G_t"] - s2["G_t"]) < 1e-10


def test_websocket_accepts_live_lam_update():
    """Simulator should accept a {lam: x} message mid-run and start applying
    the new profile to subsequent steps."""
    client = TestClient(app)
    with client.websocket_connect(
        "/ws/simulate?steps=20&delay=0&seed=99&lam=0"
    ) as ws:
        init = ws.receive_json()
        assert init["lam"] == 0.0

        # Take a few steps under reactive profile.
        for _ in range(3):
            msg = ws.receive_json()
            assert msg["event"] == "step"

        # Switch to intentional mid-run.
        ws.send_json({"lam": 1.0})

        # Drain rest of the stream and verify final lam updated.
        last_step = None
        done = None
        for _ in range(40):
            msg = ws.receive_json()
            if msg["event"] == "step":
                last_step = msg
            elif msg["event"] == "done":
                done = msg
                break
        assert done is not None
        assert done["final_lam"] == 1.0
