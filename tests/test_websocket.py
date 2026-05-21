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
    with client.websocket_connect("/ws/simulate?steps=5&delay=0&seed=1") as ws:
        init = ws.receive_json()
        assert init["event"] == "init"
        assert "sim_config" in init
        assert init["seed"] == 1
        assert "wb_probs" in init
        assert abs(sum(init["wb_probs"]) - 1.0) < 1e-6

        steps_received = 0
        done_evt = None
        for _ in range(20):
            msg = ws.receive_json()
            if msg["event"] == "step":
                steps_received += 1
                assert "user_state_name" in msg
                assert msg["user_state_name"] in ("calm", "engaged", "hooked")
                assert "wb_probs" in msg
                assert abs(sum(msg["wb_probs"]) - 1.0) < 1e-6
            elif msg["event"] == "done":
                done_evt = msg
                break

        assert steps_received == 5
        assert "counterfactual" in done_evt
        assert "wb_final_probs" in done_evt
        assert abs(sum(done_evt["final_probs"]) - 1.0) < 1e-6
        assert abs(sum(done_evt["wb_final_probs"]) - 1.0) < 1e-6


def test_websocket_wellbeing_diverges_from_engagement():
    """Across a full run with a fixed seed, the engagement objective should
    end up more outrage-heavy than the well-being objective. This is the
    entire visceral payload of the live overlay."""
    client = TestClient(app)
    with client.websocket_connect("/ws/simulate?steps=120&delay=0&seed=42") as ws:
        init = ws.receive_json()
        assert init["event"] == "init"

        done_evt = None
        for _ in range(200):
            msg = ws.receive_json()
            if msg["event"] == "done":
                done_evt = msg
                break

        assert done_evt is not None
        eng_outrage = done_evt["final_probs"][1]
        wb_outrage = done_evt["wb_final_probs"][1]
        assert eng_outrage > wb_outrage, (
            f"Expected engagement outrage ({eng_outrage:.2%}) to exceed "
            f"well-being outrage ({wb_outrage:.2%})."
        )


def test_websocket_respects_seed():
    client = TestClient(app)

    def collect(seed):
        with client.websocket_connect(f"/ws/simulate?steps=10&delay=0&seed={seed}") as ws:
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
