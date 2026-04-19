"""
Integration tests for the BYZ-01 message passing interface.

Tests verify:
  - BaseProtocol.send() correctly extracts believed hider position from obs.
  - BaseProtocol.receive() returns the correct message buffer.
  - NoneProtocol discards all messages (independent baseline).
  - None (SilentByzantine) entries in the messages list are handled gracefully.
  - EnvState carries the fields required by the interface.

Owner: Role C (Byzantine & Comms)
Run: pytest tests/test_comms.py -v
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pytest

from env.schema import Message, SENTINEL, OBS_DIM
from comms.interface import BaseProtocol, ByzantineAgent, EnvState, NoneProtocol
from comms.broadcast import BroadcastProtocol
from comms.gossip import GossipProtocol
from comms.trimmed_mean import TrimmedMeanProtocol
from comms.reputation import ReputationProtocol


# ---------------------------------------------------------------------------
# Helpers / minimal concrete implementations for testing
# ---------------------------------------------------------------------------

class _EchoBroadcastProtocol(BaseProtocol):
    """
    Minimal concrete protocol used only in these tests.

    send()    : honest — reads obs[2:4] for hider belief.
    receive() : last-writer-wins per sender; None entries produce sentinels.
    """

    def send(self, agent_id: str, state: EnvState) -> Message:
        hider_x = float(state.obs[2])
        hider_y = float(state.obs[3])
        bx: Optional[float] = None if hider_x == SENTINEL else hider_x
        by: Optional[float] = None if hider_y == SENTINEL else hider_y
        return Message(
            sender_id=agent_id,
            believed_hider_x=bx,
            believed_hider_y=by,
            step=state.step,
        )

    def receive(
        self,
        messages: list[Optional[Message]],
    ) -> dict[str, tuple[float, float]]:
        buffer: dict[str, tuple[float, float]] = {}
        for msg in messages:
            if msg is None:
                continue
            bx = msg.believed_hider_x if msg.believed_hider_x is not None else SENTINEL
            by = msg.believed_hider_y if msg.believed_hider_y is not None else SENTINEL
            buffer[msg.sender_id] = (float(bx), float(by))
        return buffer


def _make_obs(
    agent_x: float = 0.5,
    agent_y: float = 0.5,
    hider_x: float = 0.3,
    hider_y: float = 0.7,
    n_seekers: int = 4,
    grid_size: int = 10,
    obs_radius: int | None = None,
) -> np.ndarray:
    """Build a minimal float32 obs vector matching the schema."""
    dim = OBS_DIM(n_seekers, grid_size, obs_radius)
    obs = np.full(dim, SENTINEL, dtype=np.float32)
    obs[0] = agent_x
    obs[1] = agent_y
    obs[2] = hider_x
    obs[3] = hider_y
    # local_obstacle_map stays 0.0 (passable) — set passable cells
    map_size = grid_size * grid_size if obs_radius is None else (2 * obs_radius + 1) ** 2
    obs[4 : 4 + map_size] = 0.0
    return obs


# ---------------------------------------------------------------------------
# EnvState tests
# ---------------------------------------------------------------------------

class TestEnvState:
    def test_fields_accessible(self):
        obs = _make_obs()
        state = EnvState(obs=obs, step=42, grid_size=10)
        assert state.step == 42
        assert state.grid_size == 10
        assert state.obs is obs


# ---------------------------------------------------------------------------
# BaseProtocol.send() tests
# ---------------------------------------------------------------------------

class TestProtocolSend:
    def setup_method(self):
        self.proto = _EchoBroadcastProtocol()

    def test_send_with_visible_hider(self):
        obs = _make_obs(hider_x=0.3, hider_y=0.7)
        state = EnvState(obs=obs, step=5, grid_size=10)
        msg = self.proto.send("seeker_0", state)

        assert msg.sender_id == "seeker_0"
        assert msg.step == 5
        assert msg.believed_hider_x == pytest.approx(0.3)
        assert msg.believed_hider_y == pytest.approx(0.7)

    def test_send_with_hider_not_visible(self):
        """When hider is occluded obs[2:4] == SENTINEL → None fields."""
        obs = _make_obs(hider_x=SENTINEL, hider_y=SENTINEL)
        state = EnvState(obs=obs, step=1, grid_size=10)
        msg = self.proto.send("seeker_1", state)

        assert msg.believed_hider_x is None
        assert msg.believed_hider_y is None

    def test_send_returns_message_type(self):
        obs = _make_obs()
        state = EnvState(obs=obs, step=0, grid_size=10)
        msg = self.proto.send("seeker_0", state)
        assert isinstance(msg, Message)

    def test_send_embeds_correct_step(self):
        obs = _make_obs()
        for step in (0, 1, 100, 499):
            state = EnvState(obs=obs, step=step, grid_size=10)
            msg = self.proto.send("seeker_0", state)
            assert msg.step == step


# ---------------------------------------------------------------------------
# BaseProtocol.receive() tests
# ---------------------------------------------------------------------------

class TestProtocolReceive:
    def setup_method(self):
        self.proto = _EchoBroadcastProtocol()

    def _make_msg(self, sender: str, bx: float, by: float, step: int = 0) -> Message:
        return Message(sender_id=sender, believed_hider_x=bx, believed_hider_y=by, step=step)

    def test_receive_returns_dict(self):
        msgs = [self._make_msg("seeker_0", 0.3, 0.7)]
        result = self.proto.receive(msgs)
        assert isinstance(result, dict)

    def test_receive_single_message(self):
        msgs = [self._make_msg("seeker_0", 0.3, 0.7)]
        result = self.proto.receive(msgs)
        assert "seeker_0" in result
        assert result["seeker_0"] == pytest.approx((0.3, 0.7))

    def test_receive_multiple_messages(self):
        msgs = [
            self._make_msg("seeker_0", 0.1, 0.2),
            self._make_msg("seeker_1", 0.5, 0.6),
            self._make_msg("seeker_2", 0.9, 0.8),
        ]
        result = self.proto.receive(msgs)
        assert result["seeker_0"] == pytest.approx((0.1, 0.2))
        assert result["seeker_1"] == pytest.approx((0.5, 0.6))
        assert result["seeker_2"] == pytest.approx((0.9, 0.8))

    def test_receive_handles_none_gracefully(self):
        """None entries from SilentByzantine must not raise errors."""
        msgs = [
            self._make_msg("seeker_0", 0.3, 0.7),
            None,  # SilentByzantine agent
            self._make_msg("seeker_2", 0.1, 0.9),
        ]
        result = self.proto.receive(msgs)
        # seeker_0 and seeker_2 present; no KeyError for None
        assert "seeker_0" in result
        assert "seeker_2" in result
        # No entry for silent agent
        assert len(result) == 2

    def test_receive_all_none_returns_empty_dict(self):
        result = self.proto.receive([None, None, None])
        assert result == {}

    def test_receive_none_position_fields_map_to_sentinel(self):
        """Messages with None coordinates (hider not seen) → sentinel -1.0."""
        msgs = [Message(sender_id="seeker_0", believed_hider_x=None, believed_hider_y=None, step=0)]
        result = self.proto.receive(msgs)
        bx, by = result["seeker_0"]
        assert bx == SENTINEL
        assert by == SENTINEL


# ---------------------------------------------------------------------------
# NoneProtocol tests (no-communication baseline)
# ---------------------------------------------------------------------------

class TestNoneProtocol:
    def setup_method(self):
        self.proto = NoneProtocol()

    def test_send_returns_valid_message(self):
        obs = _make_obs(hider_x=0.4, hider_y=0.6)
        state = EnvState(obs=obs, step=3, grid_size=10)
        msg = self.proto.send("seeker_0", state)
        assert isinstance(msg, Message)
        assert msg.sender_id == "seeker_0"

    def test_receive_returns_empty_buffer(self):
        """NoneProtocol discards all messages → agents see only sentinels."""
        msgs = [
            Message("seeker_0", 0.3, 0.7, step=0),
            Message("seeker_1", 0.5, 0.5, step=0),
        ]
        result = self.proto.receive(msgs)
        assert result == {}

    def test_receive_with_none_entries_still_empty(self):
        result = self.proto.receive([None, None])
        assert result == {}


# ---------------------------------------------------------------------------
# ByzantineAgent abstract interface tests
# ---------------------------------------------------------------------------

class TestByzantineAgentInterface:
    """
    Verify that the ByzantineAgent abstract class cannot be instantiated
    directly and that a concrete subclass must implement corrupt_message().
    """

    def test_cannot_instantiate_abstract_class(self):
        with pytest.raises(TypeError):
            ByzantineAgent()  # type: ignore[abstract]

    def test_concrete_subclass_must_implement_corrupt_message(self):
        class _IncompleteAgent(ByzantineAgent):
            pass  # does not implement corrupt_message

        with pytest.raises(TypeError):
            _IncompleteAgent()  # type: ignore[abstract]

    def test_concrete_subclass_can_return_none(self):
        class _SilentStub(ByzantineAgent):
            def corrupt_message(self, honest_message: Message):
                return None

        agent = _SilentStub()
        msg = Message("seeker_0", 0.3, 0.7, step=0)
        assert agent.corrupt_message(msg) is None

    def test_concrete_subclass_can_return_modified_message(self):
        class _NoiseStub(ByzantineAgent):
            def corrupt_message(self, honest_message: Message):
                return Message(
                    sender_id=honest_message.sender_id,
                    believed_hider_x=0.0,
                    believed_hider_y=0.0,
                    step=honest_message.step,
                )

        agent = _NoiseStub()
        honest = Message("seeker_0", 0.5, 0.5, step=1)
        corrupted = agent.corrupt_message(honest)
        assert corrupted is not None
        assert corrupted.believed_hider_x == pytest.approx(0.0)
        assert corrupted.believed_hider_y == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Integration smoke test: 4 honest agents, each receives 3 messages
# ---------------------------------------------------------------------------

class TestBroadcastIntegration:
    """
    Smoke test: with N=4 honest seekers and broadcast protocol, each seeker
    should receive exactly 3 non-sentinel message pairs in the buffer.
    """

    def test_four_seekers_each_receives_three_messages(self):
        proto = _EchoBroadcastProtocol()
        n_seekers = 4
        grid_size = 10

        # Build one obs per seeker with the hider visible
        seekers = [f"seeker_{i}" for i in range(n_seekers)]
        states = {
            sid: EnvState(
                obs=_make_obs(
                    agent_x=i * 0.2,
                    agent_y=i * 0.1,
                    hider_x=0.5,
                    hider_y=0.5,
                    n_seekers=n_seekers,
                    grid_size=grid_size,
                ),
                step=0,
                grid_size=grid_size,
            )
            for i, sid in enumerate(seekers)
        }

        # All agents send honest messages
        messages = [proto.send(sid, states[sid]) for sid in seekers]

        # No corruption
        buffer = proto.receive(messages)

        # Every seeker should have an entry in the buffer
        assert len(buffer) == n_seekers
        for sid in seekers:
            assert sid in buffer
            bx, by = buffer[sid]
            assert bx == pytest.approx(0.5)
            assert by == pytest.approx(0.5)

    def test_silent_agent_reduces_buffer_size(self):
        """With one SilentByzantine, only 3 out of 4 entries in buffer."""
        proto = _EchoBroadcastProtocol()
        n_seekers = 4
        grid_size = 10

        seekers = [f"seeker_{i}" for i in range(n_seekers)]
        obs = _make_obs(hider_x=0.5, hider_y=0.5, n_seekers=n_seekers, grid_size=grid_size)
        state = EnvState(obs=obs, step=0, grid_size=grid_size)

        messages = [proto.send(sid, state) for sid in seekers]
        # seeker_0 is SilentByzantine — suppress its message
        messages[0] = None

        buffer = proto.receive(messages)

        # Only 3 senders visible
        assert len(buffer) == 3
        assert "seeker_0" not in buffer
        for sid in seekers[1:]:
            assert sid in buffer


# ---------------------------------------------------------------------------
# BroadcastProtocol unit tests (BYZ-02)
# ---------------------------------------------------------------------------

class TestBroadcastProtocol:
    """Unit tests for comms/broadcast.py BroadcastProtocol."""

    def setup_method(self):
        self.proto = BroadcastProtocol()

    # --- send() ---

    def test_send_visible_hider(self):
        obs = _make_obs(hider_x=0.4, hider_y=0.6)
        msg = self.proto.send("seeker_0", EnvState(obs=obs, step=7, grid_size=10))
        assert isinstance(msg, Message)
        assert msg.sender_id == "seeker_0"
        assert msg.step == 7
        assert msg.believed_hider_x == pytest.approx(0.4)
        assert msg.believed_hider_y == pytest.approx(0.6)

    def test_send_hider_occluded_yields_none_fields(self):
        obs = _make_obs(hider_x=SENTINEL, hider_y=SENTINEL)
        msg = self.proto.send("seeker_1", EnvState(obs=obs, step=2, grid_size=10))
        assert msg.believed_hider_x is None
        assert msg.believed_hider_y is None

    def test_send_does_not_mutate_obs(self):
        obs = _make_obs(hider_x=0.3, hider_y=0.7)
        original = obs.copy()
        self.proto.send("seeker_0", EnvState(obs=obs, step=0, grid_size=10))
        np.testing.assert_array_equal(obs, original)

    # --- receive() ---

    def test_receive_single_honest_message(self):
        msg = Message("seeker_0", 0.3, 0.7, step=0)
        buf = self.proto.receive([msg])
        assert buf["seeker_0"] == pytest.approx((0.3, 0.7))

    def test_receive_none_skipped(self):
        buf = self.proto.receive([None])
        assert buf == {}

    def test_receive_none_fields_become_sentinel(self):
        msg = Message("seeker_2", None, None, step=0)
        buf = self.proto.receive([msg])
        assert buf["seeker_2"] == (SENTINEL, SENTINEL)

    def test_receive_last_writer_wins(self):
        """Duplicate sender: last entry in the list wins."""
        msgs = [
            Message("seeker_0", 0.1, 0.2, step=0),
            Message("seeker_0", 0.9, 0.8, step=1),
        ]
        buf = self.proto.receive(msgs)
        assert buf["seeker_0"] == pytest.approx((0.9, 0.8))

    def test_receive_mixed_none_and_messages(self):
        msgs = [
            None,
            Message("seeker_1", 0.5, 0.5, step=0),
            None,
            Message("seeker_3", 0.2, 0.8, step=0),
        ]
        buf = self.proto.receive(msgs)
        assert len(buf) == 2
        assert "seeker_1" in buf
        assert "seeker_3" in buf

    def test_reset_is_noop(self):
        """BroadcastProtocol is stateless — reset() must not raise."""
        self.proto.reset()  # should not raise

    # --- BYZ-02 key requirement: N=4 honest seekers, each gets 3 messages ---

    def test_four_honest_seekers_each_receives_three_nonnull_messages(self):
        """
        BYZ-02 acceptance test: with N=4 honest agents and broadcast protocol,
        each agent receives exactly 3 non-sentinel message pairs per step.
        """
        n_seekers = 4
        grid_size = 10
        seekers = [f"seeker_{i}" for i in range(n_seekers)]

        # Each seeker has the hider visible at a known position
        states = {
            sid: EnvState(
                obs=_make_obs(hider_x=0.5, hider_y=0.5,
                              n_seekers=n_seekers, grid_size=grid_size),
                step=0,
                grid_size=grid_size,
            )
            for sid in seekers
        }

        # All seekers send honest messages
        messages = [self.proto.send(sid, states[sid]) for sid in seekers]
        buffer = self.proto.receive(messages)

        # Buffer must have exactly 4 entries (one per sender)
        assert len(buffer) == n_seekers, (
            f"Expected {n_seekers} buffer entries, got {len(buffer)}"
        )

        # Each receiver sees 3 non-sentinel peers (itself excluded by env)
        for receiver in seekers:
            peers = [sid for sid in seekers if sid != receiver]
            non_sentinel = [
                sid for sid in peers
                if buffer.get(sid, (SENTINEL, SENTINEL)) != (SENTINEL, SENTINEL)
            ]
            assert len(non_sentinel) == n_seekers - 1, (
                f"{receiver} should see {n_seekers - 1} non-sentinel peers, "
                f"got {len(non_sentinel)}"
            )


# ---------------------------------------------------------------------------
# Env integration tests: BroadcastProtocol wired into ByzantinePursuitEnv
# ---------------------------------------------------------------------------

class TestBroadcastEnvIntegration:
    """
    Verify that wiring BroadcastProtocol into ByzantinePursuitEnv causes
    the message buffer to be populated after seekers act.
    """

    @pytest.fixture
    def env_with_broadcast(self):
        from env.pursuit_env import ByzantinePursuitEnv
        proto = BroadcastProtocol()
        env = ByzantinePursuitEnv(
            n_seekers=4,
            grid_size=10,
            obs_radius=None,
            obstacle_density=0.0,
            byzantine_fraction=0.0,
            max_steps=50,
            seed=0,
            protocol=proto,
        )
        return env, proto

    def test_message_buffer_starts_at_sentinel(self, env_with_broadcast):
        env, _ = env_with_broadcast
        env.reset()
        for seeker in [a for a in env.possible_agents if a.startswith("seeker_")]:
            assert env._message_buffer[seeker] == (SENTINEL, SENTINEL)

    def test_message_buffer_populated_after_seeker_acts(self, env_with_broadcast):
        """After a seeker takes one step, its buffer entry must be non-sentinel
        (assuming full observability and hider is visible)."""
        env, _ = env_with_broadcast
        env.reset()

        # Step the first agent (should be seeker_0 in normal AEC order)
        first_agent = env.agent_selection
        if first_agent.startswith("seeker_"):
            env.step(0)  # NOOP
            bx, by = env._message_buffer[first_agent]
            # With full observability the hider is always visible
            assert bx != SENTINEL or by != SENTINEL, (
                "Buffer entry should be non-sentinel after seeker acts with "
                "full observability"
            )

    def test_four_seekers_buffer_fully_populated_after_full_round(
        self, env_with_broadcast
    ):
        """After all 4 seekers have acted once, every seeker's buffer entry
        should contain the sender's believed hider position (non-sentinel)."""
        env, _ = env_with_broadcast
        env.reset()
        n_seekers = 4
        seekers = [f"seeker_{i}" for i in range(n_seekers)]

        # Drive one full round (4 seeker steps + 1 hider step)
        steps_taken = 0
        while steps_taken < n_seekers + 1 and env.agents:
            env.step(0)  # NOOP for all
            steps_taken += 1

        # Every seeker should now have a non-sentinel buffer entry
        non_sentinel_count = sum(
            1 for sid in seekers
            if env._message_buffer.get(sid, (SENTINEL, SENTINEL)) != (SENTINEL, SENTINEL)
        )
        assert non_sentinel_count == n_seekers, (
            f"Expected all {n_seekers} buffer entries populated, "
            f"got {non_sentinel_count}"
        )


# ---------------------------------------------------------------------------
# GossipProtocol unit tests (BYZ-06)
# ---------------------------------------------------------------------------

def _make_messages(n: int, hider_x: float = 0.5, hider_y: float = 0.5) -> list[Message]:
    return [
        Message(sender_id=f"seeker_{i}", believed_hider_x=hider_x,
                believed_hider_y=hider_y, step=0)
        for i in range(n)
    ]


class TestGossipProtocol:
    """Unit tests for comms/gossip.py GossipProtocol (BYZ-06)."""

    # --- send() is identical to BroadcastProtocol ---

    def test_send_visible_hider(self):
        proto = GossipProtocol(fanout=2, seed=0)
        obs = _make_obs(hider_x=0.4, hider_y=0.6)
        msg = proto.send("seeker_0", EnvState(obs=obs, step=3, grid_size=10))
        assert isinstance(msg, Message)
        assert msg.sender_id == "seeker_0"
        assert msg.believed_hider_x == pytest.approx(0.4)
        assert msg.believed_hider_y == pytest.approx(0.6)

    def test_send_occluded_hider_yields_none_fields(self):
        proto = GossipProtocol(fanout=2, seed=0)
        obs = _make_obs(hider_x=SENTINEL, hider_y=SENTINEL)
        msg = proto.send("seeker_1", EnvState(obs=obs, step=0, grid_size=10))
        assert msg.believed_hider_x is None
        assert msg.believed_hider_y is None

    # --- receive(): fanout controls how many messages are kept ---

    def test_fanout_one_keeps_exactly_one_message(self):
        proto = GossipProtocol(fanout=1, seed=0)
        msgs = _make_messages(4)
        buf = proto.receive(msgs)
        assert len(buf) == 1

    def test_fanout_n_minus_one_keeps_all_messages(self):
        """fanout = N-1 is equivalent to broadcast — all messages delivered."""
        n = 4
        proto = GossipProtocol(fanout=n - 1, seed=0)
        msgs = _make_messages(n)
        buf = proto.receive(msgs)
        assert len(buf) == n - 1

    def test_fanout_exceeds_pool_keeps_all_valid(self):
        """When fanout > available messages, all valid messages are kept."""
        proto = GossipProtocol(fanout=10, seed=0)
        msgs = _make_messages(3)
        buf = proto.receive(msgs)
        assert len(buf) == 3

    def test_fanout_zero_returns_empty_buffer(self):
        proto = GossipProtocol(fanout=0, seed=0)
        msgs = _make_messages(4)
        buf = proto.receive(msgs)
        assert buf == {}

    def test_none_messages_excluded_from_pool(self):
        """None (SilentByzantine) entries are removed before sampling."""
        proto = GossipProtocol(fanout=2, seed=0)
        msgs = [
            Message("seeker_0", 0.5, 0.5, step=0),
            None,
            Message("seeker_2", 0.3, 0.7, step=0),
            None,
        ]
        buf = proto.receive(msgs)
        # Only 2 valid messages → fanout=2 keeps both
        assert len(buf) == 2
        assert "seeker_0" in buf
        assert "seeker_2" in buf

    def test_all_none_returns_empty_buffer(self):
        proto = GossipProtocol(fanout=2, seed=0)
        buf = proto.receive([None, None, None])
        assert buf == {}

    def test_none_position_fields_become_sentinel(self):
        """Messages with None coords (hider not visible) → sentinel in buffer."""
        proto = GossipProtocol(fanout=1, seed=0)
        msgs = [Message("seeker_0", None, None, step=0)]
        buf = proto.receive(msgs)
        assert buf["seeker_0"] == (SENTINEL, SENTINEL)

    def test_seed_reproducibility(self):
        """Same seed → same fanout selection across two identical calls."""
        msgs = _make_messages(8)
        buf1 = GossipProtocol(fanout=3, seed=42).receive(msgs)
        buf2 = GossipProtocol(fanout=3, seed=42).receive(msgs)
        assert set(buf1.keys()) == set(buf2.keys())

    def test_different_seeds_may_differ(self):
        """Different seeds should (almost always) produce different selections."""
        msgs = _make_messages(8)
        results = set()
        for seed in range(20):
            buf = GossipProtocol(fanout=3, seed=seed).receive(msgs)
            results.add(frozenset(buf.keys()))
        # With 8 agents and fanout=3 there are C(8,3)=56 possibilities;
        # 20 different seeds should hit at least 2 distinct subsets.
        assert len(results) > 1

    def test_reset_restores_determinism(self):
        """After reset(), the same random sequence is replayed."""
        proto = GossipProtocol(fanout=3, seed=7)
        msgs = _make_messages(8)
        first_run = proto.receive(msgs)
        proto.reset()
        second_run = proto.receive(msgs)
        assert set(first_run.keys()) == set(second_run.keys())

    def test_invalid_fanout_raises(self):
        with pytest.raises(ValueError):
            GossipProtocol(fanout=-1)


# ---------------------------------------------------------------------------
# TrimmedMeanProtocol unit tests (BYZ-07)
# ---------------------------------------------------------------------------

class TestTrimmedMeanProtocol:
    """Unit tests for comms/trimmed_mean.py TrimmedMeanProtocol (BYZ-07)."""

    # --- send() is identical to BroadcastProtocol ---

    def test_send_visible_hider(self):
        proto = TrimmedMeanProtocol()
        obs = _make_obs(hider_x=0.4, hider_y=0.6)
        msg = proto.send("seeker_0", EnvState(obs=obs, step=1, grid_size=10))
        assert isinstance(msg, Message)
        assert msg.believed_hider_x == pytest.approx(0.4)
        assert msg.believed_hider_y == pytest.approx(0.6)

    def test_send_occluded_hider_yields_none_fields(self):
        proto = TrimmedMeanProtocol()
        obs = _make_obs(hider_x=SENTINEL, hider_y=SENTINEL)
        msg = proto.send("seeker_0", EnvState(obs=obs, step=0, grid_size=10))
        assert msg.believed_hider_x is None
        assert msg.believed_hider_y is None

    # --- receive(): trimmed-mean aggregation ---

    def test_outlier_trimmed_one_byzantine_vs_three_honest(self):
        """
        3 honest agents report (0.5, 0.5); 1 Byzantine reports (0.0, 0.0).
        With trim_fraction=0.25 (trim 1 of 4 from each end), the Byzantine
        outlier is discarded and the consensus should be close to 0.5.
        """
        proto = TrimmedMeanProtocol(trim_fraction=0.25)
        msgs = [
            Message("seeker_0", 0.5, 0.5, step=0),  # honest
            Message("seeker_1", 0.5, 0.5, step=0),  # honest
            Message("seeker_2", 0.5, 0.5, step=0),  # honest
            Message("seeker_3", 0.0, 0.0, step=0),  # byzantine outlier
        ]
        buf = proto.receive(msgs)
        cx, cy = next(iter(buf.values()))
        assert cx == pytest.approx(0.5, abs=1e-6)
        assert cy == pytest.approx(0.5, abs=1e-6)

    def test_consensus_written_under_all_senders(self):
        """Every active sender's ID maps to the same consensus value."""
        proto = TrimmedMeanProtocol(trim_fraction=0.2)
        msgs = _make_messages(4, hider_x=0.5, hider_y=0.5)
        buf = proto.receive(msgs)
        assert len(buf) == 4
        values = list(buf.values())
        for v in values[1:]:
            assert v == pytest.approx(values[0])

    def test_trim_fraction_zero_is_plain_mean(self):
        """trim_fraction=0.0 → no trimming, result equals the simple mean."""
        proto = TrimmedMeanProtocol(trim_fraction=0.0)
        msgs = [
            Message("s0", 0.2, 0.4, step=0),
            Message("s1", 0.6, 0.8, step=0),
            Message("s2", 0.4, 0.6, step=0),
        ]
        buf = proto.receive(msgs)
        cx, cy = buf["s0"]
        assert cx == pytest.approx((0.2 + 0.6 + 0.4) / 3)
        assert cy == pytest.approx((0.4 + 0.8 + 0.6) / 3)

    def test_trim_fraction_half_is_approx_median(self):
        """trim_fraction=0.49 with 5 values keeps only the middle value."""
        proto = TrimmedMeanProtocol(trim_fraction=0.49)
        # x values: 0.1, 0.2, 0.5, 0.8, 0.9 → k=floor(5*0.49)=2 → keep [0.5]
        msgs = [
            Message(f"s{i}", x, 0.5, step=0)
            for i, x in enumerate([0.1, 0.2, 0.5, 0.8, 0.9])
        ]
        buf = proto.receive(msgs)
        cx, _ = buf["s0"]
        assert cx == pytest.approx(0.5, abs=1e-6)

    def test_fewer_than_three_messages_uses_simple_mean(self):
        """Fallback: <3 valid messages → simple mean, no trimming."""
        proto = TrimmedMeanProtocol(trim_fraction=0.4)
        msgs = [
            Message("s0", 0.2, 0.6, step=0),
            Message("s1", 0.8, 0.4, step=0),
        ]
        buf = proto.receive(msgs)
        cx, cy = buf["s0"]
        assert cx == pytest.approx(0.5)
        assert cy == pytest.approx(0.5)

    def test_single_message_returns_that_value(self):
        proto = TrimmedMeanProtocol()
        msgs = [Message("s0", 0.3, 0.7, step=0)]
        buf = proto.receive(msgs)
        assert buf["s0"] == pytest.approx((0.3, 0.7))

    def test_all_none_returns_empty(self):
        proto = TrimmedMeanProtocol()
        buf = proto.receive([None, None])
        assert buf == {}

    def test_none_messages_excluded_from_aggregation(self):
        """None (SilentByzantine) entries do not count toward the mean."""
        proto = TrimmedMeanProtocol(trim_fraction=0.0)
        msgs = [
            Message("s0", 0.4, 0.4, step=0),
            None,
            Message("s2", 0.6, 0.6, step=0),
        ]
        buf = proto.receive(msgs)
        # Only s0 and s2 in buffer; mean = 0.5
        assert len(buf) == 2
        cx, cy = buf["s0"]
        assert cx == pytest.approx(0.5)

    def test_all_sentinel_coordinates_return_sentinel(self):
        """Messages where hider is not visible (None coords) → SENTINEL consensus."""
        proto = TrimmedMeanProtocol()
        msgs = [
            Message("s0", None, None, step=0),
            Message("s1", None, None, step=0),
            Message("s2", None, None, step=0),
        ]
        buf = proto.receive(msgs)
        for sender_id, (cx, cy) in buf.items():
            assert cx == SENTINEL
            assert cy == SENTINEL

    def test_invalid_trim_fraction_raises(self):
        with pytest.raises(ValueError):
            TrimmedMeanProtocol(trim_fraction=-0.1)
        with pytest.raises(ValueError):
            TrimmedMeanProtocol(trim_fraction=0.5)

    def test_reset_is_noop(self):
        """TrimmedMeanProtocol is stateless — reset() must not raise."""
        TrimmedMeanProtocol().reset()


# ---------------------------------------------------------------------------
# ReputationProtocol unit tests (BYZ-08)
# ---------------------------------------------------------------------------

class TestReputationProtocol:
    """Unit tests for comms/reputation.py ReputationProtocol (BYZ-08)."""

    # --- send() is identical to BroadcastProtocol ---

    def test_send_visible_hider(self):
        proto = ReputationProtocol()
        obs = _make_obs(hider_x=0.4, hider_y=0.6)
        msg = proto.send("seeker_0", EnvState(obs=obs, step=1, grid_size=10))
        assert isinstance(msg, Message)
        assert msg.believed_hider_x == pytest.approx(0.4)
        assert msg.believed_hider_y == pytest.approx(0.6)

    # --- all agents start trusted ---

    def test_all_agents_start_trusted(self):
        """New senders are registered at trust 1.0 on first contact."""
        proto = ReputationProtocol(min_trust=0.3)
        msgs = _make_messages(4)
        proto.receive(msgs)
        for score in proto.trust_scores.values():
            assert score >= 0.9  # started at 1.0, consistent → incremented or unchanged

    # --- reputation decays for noisy senders ---

    def test_reputation_decays_for_consistent_noise(self):
        """
        A Byzantine sender always reports (0.0, 0.0) while honest agents
        report (0.5, 0.5).  After enough steps the Byzantine sender's trust
        should have decreased below the starting value.

        With 3 honest vs 1 Byzantine and deviation_threshold=0.2:
          consensus ≈ 0.375; honest deviation ≈ 0.177 < 0.2 (increment);
          Byzantine deviation ≈ 0.530 > 0.2 (decrement). Scores diverge.
        """
        proto = ReputationProtocol(
            min_trust=0.3, deviation_threshold=0.2, trust_decrement=0.1
        )
        honest_pos = (0.5, 0.5)
        byz_pos = (0.0, 0.0)

        for _ in range(5):
            msgs = [
                Message("honest_0", *honest_pos, step=0),
                Message("honest_1", *honest_pos, step=0),
                Message("honest_2", *honest_pos, step=0),
                Message("byz_0", *byz_pos, step=0),
            ]
            proto.receive(msgs)

        assert proto.trust_scores["byz_0"] < proto.trust_scores["honest_0"]

    # --- reputation recovers for honest agents ---

    def test_reputation_recovers_for_honest_agents(self):
        """
        After penalising an agent for one noisy step, it should recover
        when it subsequently reports consistent positions.
        """
        proto = ReputationProtocol(
            min_trust=0.1, deviation_threshold=0.1,
            trust_increment=0.1, trust_decrement=0.2
        )
        # Step 1: agent reports noise — penalised
        msgs = [
            Message("s0", 0.5, 0.5, step=0),
            Message("s1", 0.5, 0.5, step=0),
            Message("s2", 0.0, 0.0, step=0),  # noisy
        ]
        proto.receive(msgs)
        score_after_penalty = proto.trust_scores["s2"]

        # Steps 2–5: s2 now reports honestly
        for step in range(1, 6):
            msgs = [
                Message("s0", 0.5, 0.5, step=step),
                Message("s1", 0.5, 0.5, step=step),
                Message("s2", 0.5, 0.5, step=step),
            ]
            proto.receive(msgs)

        assert proto.trust_scores["s2"] > score_after_penalty

    # --- complete isolation when trust reaches zero ---

    def test_isolated_sender_absent_from_buffer(self):
        """
        A sender whose score drops below min_trust is excluded from the
        returned buffer.

        3 honest vs 1 Byzantine with deviation_threshold=0.3 ensures honest
        agents (deviation ≈ 0.177) are rewarded while the Byzantine
        (deviation ≈ 0.530) is penalised until it falls below min_trust=0.5.
        """
        proto = ReputationProtocol(
            min_trust=0.5, deviation_threshold=0.3,
            trust_decrement=0.2
        )
        # Drive the Byzantine agent's score below min_trust
        for step in range(6):
            msgs = [
                Message("honest_0", 0.5, 0.5, step=step),
                Message("honest_1", 0.5, 0.5, step=step),
                Message("honest_2", 0.5, 0.5, step=step),
                Message("byz",      0.0, 0.0, step=step),
            ]
            proto.receive(msgs)

        # By now byz score should be below 0.5
        assert proto.trust_scores["byz"] < 0.5

        # Final receive: byz should not appear in the buffer
        final_msgs = [
            Message("honest_0", 0.5, 0.5, step=6),
            Message("honest_1", 0.5, 0.5, step=6),
            Message("honest_2", 0.5, 0.5, step=6),
            Message("byz",      0.0, 0.0, step=6),
        ]
        buf = proto.receive(final_msgs)
        assert "byz" not in buf
        assert "honest_0" in buf

    # --- scores clamped to [0.0, 1.0] ---

    def test_scores_never_exceed_one(self):
        proto = ReputationProtocol(trust_increment=0.5)
        msgs = _make_messages(3, hider_x=0.5, hider_y=0.5)
        for _ in range(10):
            proto.receive(msgs)
        for score in proto.trust_scores.values():
            assert score <= 1.0

    def test_scores_never_go_below_zero(self):
        proto = ReputationProtocol(
            min_trust=0.01, deviation_threshold=0.01, trust_decrement=0.5
        )
        msgs = [
            Message("honest", 0.5, 0.5, step=0),
            Message("byz",    0.0, 0.0, step=0),
        ]
        for _ in range(10):
            proto.receive(msgs)
        assert proto.trust_scores["byz"] >= 0.0

    # --- reset restores all scores to 1.0 ---

    def test_reset_restores_scores(self):
        proto = ReputationProtocol(
            min_trust=0.3, deviation_threshold=0.05, trust_decrement=0.2
        )
        for step in range(5):
            msgs = [
                Message("honest", 0.5, 0.5, step=step),
                Message("byz",    0.0, 0.0, step=step),
            ]
            proto.receive(msgs)

        assert proto.trust_scores["byz"] < 1.0
        proto.reset()
        for score in proto.trust_scores.values():
            assert score == pytest.approx(1.0)

    # --- all-None returns empty buffer ---

    def test_all_none_returns_empty(self):
        proto = ReputationProtocol()
        buf = proto.receive([None, None])
        assert buf == {}

    # --- invalid constructor args ---

    def test_invalid_min_trust_raises(self):
        with pytest.raises(ValueError):
            ReputationProtocol(min_trust=0.0)
        with pytest.raises(ValueError):
            ReputationProtocol(min_trust=1.1)

    def test_invalid_deviation_threshold_raises(self):
        with pytest.raises(ValueError):
            ReputationProtocol(deviation_threshold=-0.1)
