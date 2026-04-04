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
