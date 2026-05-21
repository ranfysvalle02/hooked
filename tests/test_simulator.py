import numpy as np
import pytest

from demo import (
    AttentionEconomySimulator,
    TRANSITIONS,
    REWARDS,
    AROUSAL_PENALTIES,
    USER_STATES,
    _run_counterfactual,
)


class TestAttentionEconomySimulator:
    def test_initial_policy_is_uniform(self):
        sim = AttentionEconomySimulator(rng=np.random.default_rng(0))
        probs = sim.get_policy_probabilities()
        assert abs(probs[0] - 0.5) < 1e-10
        assert abs(probs[1] - 0.5) < 1e-10

    def test_converges_to_outrage_with_engagement_objective(self):
        rng = np.random.default_rng(42)
        sim = AttentionEconomySimulator(learning_rate=0.08, lam=0.0, rng=rng)
        for _ in range(120):
            probs = sim.get_policy_probabilities()
            action = int(sim.rng.choice([0, 1], p=probs))
            G_t = sim.simulate_human_response(action)
            sim.update_policy(action, G_t, probs)
            sim.transition_user_state(action)
        final = sim.get_policy_probabilities()
        assert final[1] > 0.95

    def test_converges_across_multiple_seeds(self):
        for seed in range(5):
            rng = np.random.default_rng(seed)
            sim = AttentionEconomySimulator(learning_rate=0.08, lam=0.0, rng=rng)
            for _ in range(120):
                probs = sim.get_policy_probabilities()
                action = int(sim.rng.choice([0, 1], p=probs))
                G_t = sim.simulate_human_response(action)
                sim.update_policy(action, G_t, probs)
                sim.transition_user_state(action)
            assert sim.get_policy_probabilities()[1] > 0.95, f"seed {seed} failed"

    def test_user_state_transitions_are_valid(self):
        rng = np.random.default_rng(0)
        sim = AttentionEconomySimulator(rng=rng)
        states_seen = set()
        for _ in range(200):
            probs = sim.get_policy_probabilities()
            action = int(sim.rng.choice([0, 1], p=probs))
            sim.simulate_human_response(action)
            sim.update_policy(action, sim.simulate_human_response(action), probs)
            new_state = sim.transition_user_state(action)
            assert new_state in (0, 1, 2)
            states_seen.add(new_state)
        assert len(states_seen) == 3

    def test_wellbeing_penalty_reduces_outrage_reward(self):
        rng = np.random.default_rng(0)
        sim = AttentionEconomySimulator(learning_rate=0.08, lam=1.0, rng=rng)
        sim.user_state = 2  # hooked
        raw_reward = 8.0
        penalized = sim.apply_wellbeing_penalty(1, raw_reward)
        assert penalized < raw_reward
        info_penalized = sim.apply_wellbeing_penalty(0, 1.5)
        assert info_penalized == 1.5  # no penalty on informative

    def test_transition_matrix_rows_sum_to_one(self):
        for action in range(2):
            for state in range(3):
                row = TRANSITIONS[action][state]
                assert abs(sum(row) - 1.0) < 1e-10


class TestCounterfactual:
    def test_counterfactual_returns_valid_probs(self):
        result = _run_counterfactual(120, 0.08, 1.0, 42)
        assert "probs" in result
        assert "theta" in result
        assert abs(sum(result["probs"]) - 1.0) < 1e-6

    def test_counterfactual_favors_informative(self):
        results = [_run_counterfactual(120, 0.08, 1.0, s)["probs"][0] for s in range(10)]
        mean_info = np.mean(results)
        assert mean_info > 0.5, f"Mean informative {mean_info:.2%} should be >50%"

    def test_counterfactual_deterministic_with_seed(self):
        r1 = _run_counterfactual(120, 0.08, 1.0, 99)
        r2 = _run_counterfactual(120, 0.08, 1.0, 99)
        assert r1["probs"] == r2["probs"]

    def test_lambda_zero_matches_engagement(self):
        result = _run_counterfactual(120, 0.08, 0.0, 42)
        assert result["probs"][1] > 0.90
