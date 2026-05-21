import numpy as np
import pytest

from demo import (
    AttentionEconomySimulator,
    TRANSITIONS,
    TRANSITIONS_REACTIVE,
    TRANSITIONS_INTENTIONAL,
    REWARDS_REACTIVE,
    REWARDS_INTENTIONAL,
    USER_STATES,
    blended_reward_params,
    blended_transition_row,
    _diagnosis,
)


class TestAttentionEconomySimulator:
    def test_initial_policy_is_uniform(self):
        sim = AttentionEconomySimulator(rng=np.random.default_rng(0))
        probs = sim.get_policy_probabilities()
        assert abs(probs[0] - 0.5) < 1e-10
        assert abs(probs[1] - 0.5) < 1e-10

    def test_reactive_profile_converges_to_outrage(self):
        """At lam=0 the simulated user reacts limbically: outrage spikes
        biggest, so the gradient finds outrage."""
        rng = np.random.default_rng(42)
        sim = AttentionEconomySimulator(learning_rate=0.08, lam=0.0, rng=rng)
        for _ in range(250):
            sim.step_once()
        final = sim.get_policy_probabilities()
        assert final[1] > 0.90, f"reactive profile should lock onto outrage, got {final}"

    def test_intentional_profile_converges_to_informative(self):
        """At lam=1 the simulated user has trained themselves: informative
        content spikes biggest, so the same gradient finds informative."""
        rng = np.random.default_rng(42)
        sim = AttentionEconomySimulator(learning_rate=0.08, lam=1.0, rng=rng)
        for _ in range(250):
            sim.step_once()
        final = sim.get_policy_probabilities()
        assert final[0] > 0.90, f"intentional profile should lock onto informative, got {final}"

    def test_reactive_converges_across_seeds(self):
        for seed in range(5):
            rng = np.random.default_rng(seed)
            sim = AttentionEconomySimulator(learning_rate=0.08, lam=0.0, rng=rng)
            for _ in range(250):
                sim.step_once()
            assert sim.get_policy_probabilities()[1] > 0.85, f"seed {seed}: reactive should favor outrage"

    def test_intentional_converges_across_seeds(self):
        for seed in range(5):
            rng = np.random.default_rng(seed)
            sim = AttentionEconomySimulator(learning_rate=0.08, lam=1.0, rng=rng)
            for _ in range(250):
                sim.step_once()
            assert sim.get_policy_probabilities()[0] > 0.85, f"seed {seed}: intentional should favor informative"

    def test_user_state_transitions_are_valid(self):
        rng = np.random.default_rng(0)
        sim = AttentionEconomySimulator(rng=rng)
        states_seen = set()
        for _ in range(300):
            new_state = sim.step_once()["user_state"]
            assert new_state in (0, 1, 2)
            states_seen.add(new_state)
        assert len(states_seen) == 3

    def test_transition_matrix_rows_sum_to_one(self):
        for matrix in (TRANSITIONS, TRANSITIONS_REACTIVE, TRANSITIONS_INTENTIONAL):
            for action in range(2):
                for state in range(3):
                    row = matrix[action][state]
                    assert abs(sum(row) - 1.0) < 1e-10

    def test_intentional_transitions_mirror_reactive(self):
        """At lam=1 the intentional user's arousal is driven by depth, not
        outrage — the state machine is the action-axis swap of the reactive
        one. This keeps the substrate symmetric under the lambda blend."""
        for state in range(3):
            assert TRANSITIONS_INTENTIONAL[0][state] == TRANSITIONS_REACTIVE[1][state]
            assert TRANSITIONS_INTENTIONAL[1][state] == TRANSITIONS_REACTIVE[0][state]

    def test_blended_transitions_are_symmetric_at_midpoint(self):
        """At lam=0.5 both actions induce identical state-transition rows: the
        substrate stops favoring outrage so the gradient sees a truly neutral
        environment."""
        for state in range(3):
            row_info = blended_transition_row(0.5, 0, state)
            row_outrage = blended_transition_row(0.5, 1, state)
            for a, b in zip(row_info, row_outrage):
                assert abs(a - b) < 1e-12

    def test_balanced_lambda_does_not_lock_into_outrage(self):
        """The pedagogical contract: at lam~0.5 a balanced user must not see a
        99% outrage feed. Across many seeds the average outrage share should be
        roughly balanced, and overwhelming outrage lock-in should be rare."""
        outrage_shares = []
        for seed in range(40):
            rng = np.random.default_rng(seed)
            sim = AttentionEconomySimulator(learning_rate=0.08, lam=0.5, rng=rng)
            for _ in range(250):
                sim.step_once()
            outrage_shares.append(sim.get_policy_probabilities()[1])
        mean = sum(outrage_shares) / len(outrage_shares)
        extreme = sum(1 for s in outrage_shares if s > 0.9) / len(outrage_shares)
        assert 0.25 < mean < 0.75, f"balanced lam should be roughly mixed, got mean={mean:.3f}"
        assert extreme < 0.25, f"balanced lam should rarely lock into outrage, got {extreme:.0%}"

    def test_lam_is_clamped_in_blend(self):
        m_lo, _ = blended_reward_params(-0.5, 0, 1)
        m_hi, _ = blended_reward_params(1.5, 0, 1)
        m0, _ = blended_reward_params(0.0, 0, 1)
        m1, _ = blended_reward_params(1.0, 0, 1)
        assert m_lo == m0
        assert m_hi == m1

    def test_blend_endpoints_match_profiles(self):
        for s in range(3):
            for a in range(2):
                m_r, v_r = blended_reward_params(0.0, s, a)
                m_i, v_i = blended_reward_params(1.0, s, a)
                assert (m_r, v_r) == REWARDS_REACTIVE[s][a]
                assert (m_i, v_i) == REWARDS_INTENTIONAL[s][a]

    def test_blend_midpoint_is_average(self):
        for s in range(3):
            for a in range(2):
                m_mid, v_mid = blended_reward_params(0.5, s, a)
                m_r, v_r = REWARDS_REACTIVE[s][a]
                m_i, v_i = REWARDS_INTENTIONAL[s][a]
                assert abs(m_mid - (m_r + m_i) / 2) < 1e-10
                assert abs(v_mid - (v_r + v_i) / 2) < 1e-10

    def test_intentional_profile_is_mirror_of_reactive(self):
        """The defining property of the model: REWARDS_INTENTIONAL is
        REWARDS_REACTIVE with the action axis swapped. This is what makes the
        same gradient produce two different feeds — the user's reaction
        asymmetry is just pointed at a different action."""
        for s in range(3):
            assert REWARDS_INTENTIONAL[s][0] == REWARDS_REACTIVE[s][1]
            assert REWARDS_INTENTIONAL[s][1] == REWARDS_REACTIVE[s][0]

    def test_step_payload_exposes_lam(self):
        sim = AttentionEconomySimulator(lam=0.42, rng=np.random.default_rng(0))
        payload = sim.step_once()
        assert payload["lam"] == 0.42


class TestDiagnosis:
    def test_diagnosis_for_outrage_lockin(self):
        msg = _diagnosis([0.04, 0.96], lam=0.0)
        assert "attention" in msg.lower() or "narrowing" in msg.lower()

    def test_diagnosis_for_quality_lockin(self):
        msg = _diagnosis([0.96, 0.04], lam=1.0)
        assert "followed your lead" in msg.lower() or "quality" in msg.lower() or "informative" in msg.lower()

    def test_diagnosis_for_mixed(self):
        msg = _diagnosis([0.5, 0.5], lam=0.5)
        assert "mixed" in msg.lower() or "balanced" in msg.lower()
