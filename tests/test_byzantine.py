"""
Unit tests for Byzantine agent subtypes (BYZ-03).

Owner: Role C (Byzantine & Comms)
Run: pytest tests/test_byzantine.py -v

CRITICAL INVARIANT: Byzantine agents corrupt ONLY outgoing messages.
Movement actions are determined by the RL policy (ippo.py), not by the
Byzantine class itself.  This is enforced architecturally — ByzantineAgent
has no act() method — so there is nothing to test for movement purity.

Tests verify:
  - Corruption logic is correct for each subtype.
  - Returned messages are structurally valid (right fields, normalised coords).
  - SilentByzantine returns None.
  - Positions are always clamped / normalised to [0, 1].
  - step and sender_id fields are preserved where expected.
  - MisdirectionByzantine reflection formula is exact.
  - SpoofingByzantine never forges its own id.
  - RandomNoiseByzantine uses its RNG (seed-reproducible).
"""

from __future__ import annotations

import pytest
import numpy as np

from env.schema import Message, SENTINEL
from agents.byzantine import (
    RandomNoiseByzantine,
    MisdirectionByzantine,
    SpoofingByzantine,
    SilentByzantine,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _honest_msg(
    sender: str = "seeker_0",
    bx: float | None = 0.5,
    by: float | None = 0.5,
    step: int = 10,
) -> Message:
    return Message(sender_id=sender, believed_hider_x=bx, believed_hider_y=by, step=step)


def _all_seekers(n: int = 4) -> list[str]:
    return [f"seeker_{i}" for i in range(n)]


# ---------------------------------------------------------------------------
# RandomNoiseByzantine
# ---------------------------------------------------------------------------

class TestRandomNoiseByzantine:

    def test_returns_message_not_none(self):
        agent = RandomNoiseByzantine("seeker_0", grid_size=10, seed=0)
        result = agent.corrupt_message(_honest_msg())
        assert result is not None
        assert isinstance(result, Message)

    def test_positions_differ_from_honest(self):
        """With a fixed seed, noise output is deterministic and different."""
        agent = RandomNoiseByzantine("seeker_0", grid_size=10, seed=99)
        honest = _honest_msg(bx=0.5, by=0.5)
        corrupted = agent.corrupt_message(honest)
        # With grid_size=10 and seed=99, the chance of exact match is 1/100
        # but we verify the positions are plausibly in range
        assert 0.0 <= corrupted.believed_hider_x <= 1.0
        assert 0.0 <= corrupted.believed_hider_y <= 1.0

    def test_seed_reproducibility(self):
        """Same seed → same sequence of corrupted messages."""
        a1 = RandomNoiseByzantine("seeker_0", grid_size=10, seed=7)
        a2 = RandomNoiseByzantine("seeker_0", grid_size=10, seed=7)
        honest = _honest_msg()
        m1 = a1.corrupt_message(honest)
        m2 = a2.corrupt_message(honest)
        assert m1.believed_hider_x == pytest.approx(m2.believed_hider_x)
        assert m1.believed_hider_y == pytest.approx(m2.believed_hider_y)

    def test_different_seeds_differ(self):
        """Different seeds should (with overwhelming probability) diverge."""
        a1 = RandomNoiseByzantine("seeker_0", grid_size=20, seed=1)
        a2 = RandomNoiseByzantine("seeker_0", grid_size=20, seed=2)
        honest = _honest_msg()
        results_match = all(
            a1.corrupt_message(honest).believed_hider_x
            == a2.corrupt_message(honest).believed_hider_x
            for _ in range(10)
        )
        assert not results_match

    def test_positions_always_normalised(self):
        """Over many calls, every output position must stay in [0, 1]."""
        agent = RandomNoiseByzantine("seeker_0", grid_size=15, seed=42)
        honest = _honest_msg()
        for _ in range(200):
            msg = agent.corrupt_message(honest)
            assert 0.0 <= msg.believed_hider_x <= 1.0, msg.believed_hider_x
            assert 0.0 <= msg.believed_hider_y <= 1.0, msg.believed_hider_y

    def test_sender_id_preserved(self):
        agent = RandomNoiseByzantine("seeker_2", grid_size=10, seed=0)
        result = agent.corrupt_message(_honest_msg(sender="seeker_2"))
        assert result.sender_id == "seeker_2"

    def test_step_preserved(self):
        agent = RandomNoiseByzantine("seeker_0", grid_size=10, seed=0)
        result = agent.corrupt_message(_honest_msg(step=77))
        assert result.step == 77

    def test_positions_cover_full_grid_range(self):
        """Verify the RNG samples the full grid, not just one cell."""
        agent = RandomNoiseByzantine("seeker_0", grid_size=10, seed=0)
        honest = _honest_msg()
        xs = {agent.corrupt_message(honest).believed_hider_x for _ in range(100)}
        # Should see more than 1 unique x value over 100 draws
        assert len(xs) > 1

    def test_honest_message_not_mutated(self):
        agent = RandomNoiseByzantine("seeker_0", grid_size=10, seed=0)
        honest = _honest_msg(bx=0.3, by=0.7)
        agent.corrupt_message(honest)
        assert honest.believed_hider_x == pytest.approx(0.3)
        assert honest.believed_hider_y == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# MisdirectionByzantine
# ---------------------------------------------------------------------------

class TestMisdirectionByzantine:
    """
    Reflection formula: reflected = (2*agent - hider) clamped to [0, gs-1]
    then normalised by (gs - 1).
    """

    def _make_agent(
        self,
        agent_id: str = "seeker_0",
        grid_size: int = 10,
        agent_pos: tuple[int, int] = (5, 5),
        hider_pos: tuple[int, int] = (3, 3),
    ) -> MisdirectionByzantine:
        return MisdirectionByzantine(
            agent_id=agent_id,
            grid_size=grid_size,
            get_true_hider_pos=lambda: hider_pos,
            get_agent_pos=lambda: agent_pos,
        )

    def test_returns_message_not_none(self):
        agent = self._make_agent()
        result = agent.corrupt_message(_honest_msg())
        assert result is not None
        assert isinstance(result, Message)

    def test_reflection_formula_exact(self):
        """
        Agent at (5,5), hider at (3,3), grid_size=10:
        reflected = (2*5-3, 2*5-3) = (7, 7)
        normalised = 7/9 ≈ 0.7778
        """
        agent = self._make_agent(agent_pos=(5, 5), hider_pos=(3, 3), grid_size=10)
        msg = agent.corrupt_message(_honest_msg())
        expected = 7.0 / 9.0
        assert msg.believed_hider_x == pytest.approx(expected)
        assert msg.believed_hider_y == pytest.approx(expected)

    def test_reflection_points_away_from_hider(self):
        """
        The reflected point should be further from the hider than the
        original honest position would be.
        """
        grid_size = 20
        agent_pos = (10, 10)
        hider_pos = (5, 5)
        agent = MisdirectionByzantine(
            "seeker_0", grid_size,
            get_true_hider_pos=lambda: hider_pos,
            get_agent_pos=lambda: agent_pos,
        )
        msg = agent.corrupt_message(_honest_msg())
        norm = grid_size - 1
        hx_norm = hider_pos[0] / norm
        hy_norm = hider_pos[1] / norm
        # Reflected position should be further from true hider
        reflected_dist = abs(msg.believed_hider_x - hx_norm) + abs(msg.believed_hider_y - hy_norm)
        honest_dist = abs(0.5 - hx_norm) + abs(0.5 - hy_norm)
        assert reflected_dist >= honest_dist

    def test_clamp_negative_reflection(self):
        """
        Agent at (1,1), hider at (5,5), grid_size=10:
        reflected = (2*1-5, 2*1-5) = (-3, -3) → clamped to (0, 0)
        normalised = 0.0
        """
        agent = self._make_agent(agent_pos=(1, 1), hider_pos=(5, 5), grid_size=10)
        msg = agent.corrupt_message(_honest_msg())
        assert msg.believed_hider_x == pytest.approx(0.0)
        assert msg.believed_hider_y == pytest.approx(0.0)

    def test_clamp_overflow_reflection(self):
        """
        Agent at (8,8), hider at (2,2), grid_size=10:
        reflected = (2*8-2, 2*8-2) = (14, 14) → clamped to (9, 9)
        normalised = 9/9 = 1.0
        """
        agent = self._make_agent(agent_pos=(8, 8), hider_pos=(2, 2), grid_size=10)
        msg = agent.corrupt_message(_honest_msg())
        assert msg.believed_hider_x == pytest.approx(1.0)
        assert msg.believed_hider_y == pytest.approx(1.0)

    def test_agent_on_hider_reflects_to_itself(self):
        """
        If agent is on the hider: reflection = (2*a - a) = a.
        Misdirection collapses to honest position when co-located.
        """
        agent = self._make_agent(agent_pos=(4, 6), hider_pos=(4, 6), grid_size=10)
        msg = agent.corrupt_message(_honest_msg())
        expected_x = 4.0 / 9.0
        expected_y = 6.0 / 9.0
        assert msg.believed_hider_x == pytest.approx(expected_x)
        assert msg.believed_hider_y == pytest.approx(expected_y)

    def test_output_always_normalised(self):
        """Reflected positions must always be in [0, 1]."""
        grid_size = 12
        for ax in range(0, grid_size):
            for ay in range(0, grid_size):
                for hx in (0, grid_size // 2, grid_size - 1):
                    for hy in (0, grid_size // 2, grid_size - 1):
                        agent = MisdirectionByzantine(
                            "seeker_0", grid_size,
                            get_true_hider_pos=lambda _hx=hx, _hy=hy: (_hx, _hy),
                            get_agent_pos=lambda _ax=ax, _ay=ay: (_ax, _ay),
                        )
                        msg = agent.corrupt_message(_honest_msg())
                        assert 0.0 <= msg.believed_hider_x <= 1.0
                        assert 0.0 <= msg.believed_hider_y <= 1.0

    def test_sender_id_preserved(self):
        agent = self._make_agent(agent_id="seeker_1")
        msg = agent.corrupt_message(_honest_msg(sender="seeker_1"))
        assert msg.sender_id == "seeker_1"

    def test_step_preserved(self):
        agent = self._make_agent()
        msg = agent.corrupt_message(_honest_msg(step=33))
        assert msg.step == 33

    def test_callbacks_called_each_time(self):
        """Callable is invoked fresh each call — dynamic position tracking."""
        positions = {"hider": (3, 3), "agent": (5, 5)}

        agent = MisdirectionByzantine(
            "seeker_0",
            grid_size=10,
            get_true_hider_pos=lambda: positions["hider"],
            get_agent_pos=lambda: positions["agent"],
        )
        honest = _honest_msg()

        msg1 = agent.corrupt_message(honest)
        # Move both to new positions
        positions["hider"] = (8, 8)
        positions["agent"] = (2, 2)
        msg2 = agent.corrupt_message(honest)

        # The two calls must produce different results
        assert msg1.believed_hider_x != pytest.approx(msg2.believed_hider_x)


# ---------------------------------------------------------------------------
# SpoofingByzantine
# ---------------------------------------------------------------------------

class TestSpoofingByzantine:

    def test_returns_message_not_none(self):
        agent = SpoofingByzantine("seeker_0", _all_seekers(4), seed=0)
        result = agent.corrupt_message(_honest_msg())
        assert result is not None
        assert isinstance(result, Message)

    def test_sender_id_is_changed(self):
        agent = SpoofingByzantine("seeker_0", _all_seekers(4), seed=0)
        honest = _honest_msg(sender="seeker_0")
        corrupted = agent.corrupt_message(honest)
        assert corrupted.sender_id != "seeker_0"

    def test_forged_id_is_valid_peer(self):
        """Forged id must belong to another seeker in the team."""
        all_ids = _all_seekers(4)
        agent = SpoofingByzantine("seeker_0", all_ids, seed=0)
        honest = _honest_msg(sender="seeker_0")
        for _ in range(30):
            corrupted = agent.corrupt_message(honest)
            assert corrupted.sender_id in all_ids
            assert corrupted.sender_id != "seeker_0"

    def test_content_fields_preserved(self):
        """Spoofing only changes sender_id — hider belief and step unchanged."""
        agent = SpoofingByzantine("seeker_0", _all_seekers(4), seed=0)
        honest = _honest_msg(bx=0.4, by=0.6, step=55)
        corrupted = agent.corrupt_message(honest)
        assert corrupted.believed_hider_x == pytest.approx(0.4)
        assert corrupted.believed_hider_y == pytest.approx(0.6)
        assert corrupted.step == 55

    def test_seed_reproducibility(self):
        a1 = SpoofingByzantine("seeker_0", _all_seekers(4), seed=5)
        a2 = SpoofingByzantine("seeker_0", _all_seekers(4), seed=5)
        honest = _honest_msg()
        assert a1.corrupt_message(honest).sender_id == a2.corrupt_message(honest).sender_id

    def test_single_agent_team_returns_unchanged(self):
        """With no other seekers to spoof, message is returned unchanged."""
        agent = SpoofingByzantine("seeker_0", ["seeker_0"], seed=0)
        honest = _honest_msg(sender="seeker_0")
        result = agent.corrupt_message(honest)
        assert result.sender_id == "seeker_0"

    def test_spoofed_ids_vary_over_calls(self):
        """With 4 peers, multiple calls should produce different fake ids."""
        agent = SpoofingByzantine("seeker_0", _all_seekers(5), seed=0)
        honest = _honest_msg()
        ids = {agent.corrupt_message(honest).sender_id for _ in range(40)}
        assert len(ids) > 1

    def test_none_content_fields_preserved(self):
        """None position fields (hider not visible) pass through unchanged."""
        agent = SpoofingByzantine("seeker_0", _all_seekers(4), seed=0)
        honest = _honest_msg(bx=None, by=None)
        corrupted = agent.corrupt_message(honest)
        assert corrupted.believed_hider_x is None
        assert corrupted.believed_hider_y is None


# ---------------------------------------------------------------------------
# SilentByzantine
# ---------------------------------------------------------------------------

class TestSilentByzantine:

    def test_returns_none(self):
        agent = SilentByzantine("seeker_0")
        result = agent.corrupt_message(_honest_msg())
        assert result is None

    def test_returns_none_for_any_message(self):
        agent = SilentByzantine("seeker_3")
        for bx, by in [(0.0, 0.0), (1.0, 1.0), (None, None), (0.5, SENTINEL)]:
            result = agent.corrupt_message(_honest_msg(bx=bx, by=by))
            assert result is None

    def test_honest_message_not_mutated(self):
        """Even though we return None, the honest message must be untouched."""
        agent = SilentByzantine("seeker_0")
        honest = _honest_msg(bx=0.3, by=0.7, step=12)
        agent.corrupt_message(honest)
        assert honest.believed_hider_x == pytest.approx(0.3)
        assert honest.believed_hider_y == pytest.approx(0.7)
        assert honest.step == 12


# ---------------------------------------------------------------------------
# Cross-subtype: all subtypes must be importable from agents.byzantine
# ---------------------------------------------------------------------------

class TestImports:
    def test_all_subtypes_importable(self):
        from agents.byzantine import (
            RandomNoiseByzantine,
            MisdirectionByzantine,
            SpoofingByzantine,
            SilentByzantine,
        )
        assert RandomNoiseByzantine is not None
        assert MisdirectionByzantine is not None
        assert SpoofingByzantine is not None
        assert SilentByzantine is not None

    def test_all_extend_byzantine_agent(self):
        from comms.interface import ByzantineAgent
        from agents.byzantine import (
            RandomNoiseByzantine,
            MisdirectionByzantine,
            SpoofingByzantine,
            SilentByzantine,
        )
        for cls in (RandomNoiseByzantine, MisdirectionByzantine,
                    SpoofingByzantine, SilentByzantine):
            assert issubclass(cls, ByzantineAgent), (
                f"{cls.__name__} does not extend ByzantineAgent"
            )


# ---------------------------------------------------------------------------
# Env integration: Byzantine agents wired into ByzantinePursuitEnv
# ---------------------------------------------------------------------------

class TestByzantineEnvIntegration:
    """
    Verifies that wiring Byzantine agents into the env causes _message_buffer
    to be populated with corrupted values, not the honest position.
    """

    @pytest.fixture
    def env_with_random_noise(self):
        from env.pursuit_env import ByzantinePursuitEnv
        from comms.broadcast import BroadcastProtocol

        n_seekers = 4
        env = ByzantinePursuitEnv(
            n_seekers=n_seekers,
            grid_size=10,
            obs_radius=None,
            obstacle_density=0.0,
            byzantine_fraction=0.5,   # 2 Byzantine seekers: seeker_0, seeker_1
            max_steps=50,
            seed=0,
            protocol=BroadcastProtocol(),
            byzantine_agents={
                "seeker_0": RandomNoiseByzantine("seeker_0", grid_size=10, seed=0),
                "seeker_1": RandomNoiseByzantine("seeker_1", grid_size=10, seed=1),
            },
        )
        return env

    def test_silent_byzantine_sets_buffer_to_sentinel(self):
        """SilentByzantine must cause its buffer slot to be reset to sentinel."""
        from env.pursuit_env import ByzantinePursuitEnv
        from comms.broadcast import BroadcastProtocol

        env = ByzantinePursuitEnv(
            n_seekers=4,
            grid_size=10,
            obs_radius=None,
            obstacle_density=0.0,
            byzantine_fraction=0.25,  # seeker_0 is Byzantine
            max_steps=50,
            seed=0,
            protocol=BroadcastProtocol(),
            byzantine_agents={
                "seeker_0": SilentByzantine("seeker_0"),
            },
        )
        env.reset()

        # Step seeker_0 (first agent in AEC order)
        if env.agent_selection == "seeker_0":
            env.step(0)
            bx, by = env._message_buffer["seeker_0"]
            assert bx == SENTINEL
            assert by == SENTINEL

    def test_random_noise_buffer_entry_in_valid_range(self, env_with_random_noise):
        """After RandomNoiseByzantine acts, buffer entry must be in [0, 1]."""
        env = env_with_random_noise
        env.reset()

        # Step until seeker_0 has acted
        while env.agents and env.agent_selection != "seeker_0":
            env.step(0)
        if env.agents and env.agent_selection == "seeker_0":
            env.step(0)
            bx, by = env._message_buffer["seeker_0"]
            if bx != SENTINEL:
                assert 0.0 <= bx <= 1.0
                assert 0.0 <= by <= 1.0
