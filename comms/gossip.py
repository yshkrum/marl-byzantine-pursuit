"""
BYZ-06: Gossip communication protocol.

Each step, every agent sends its believed hider position (honest message).
On receive(), only a random subset of ``fanout`` messages are forwarded to
the shared buffer — the rest are silently discarded.  This models
bandwidth-constrained networks where agents cannot hear every broadcast.

Protocol behaviour
------------------
send()    : Identical to BroadcastProtocol — reads obs[2:4] for hider belief.
receive() : Randomly selects min(fanout, n_valid) non-None messages to write
            into the buffer; unselected senders are absent from the returned
            dict and will be filled with sentinel -1.0 by the environment.
reset()   : Re-seeds the internal RNG so episode rollouts are reproducible.

Special cases
-------------
- fanout >= len(valid messages)  → all valid messages are kept (no drop).
- fanout = 0                     → empty buffer (no message delivered).
- All messages are None          → empty buffer.

Owner : Role C (Byzantine & Comms)
Ticket: BYZ-06
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from env.schema import Message, SENTINEL
from comms.interface import BaseProtocol, EnvState


class GossipProtocol(BaseProtocol):
    """
    Gossip protocol: each agent's message is delivered to a random subset of
    ``fanout`` peers rather than to all seekers.

    Parameters
    ----------
    fanout : int
        Number of messages randomly kept from the received pool each step.
        Defaults to 2.  Set to ``n_seekers - 1`` to match broadcast behaviour.
    seed : int
        Seed for the internal numpy RNG.  ``reset()`` restores this seed so
        every episode starts with the same random draw sequence.
    """

    def __init__(self, fanout: int = 2, seed: int = 0) -> None:
        if fanout < 0:
            raise ValueError(f"fanout must be >= 0, got {fanout}")
        self._fanout = fanout
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Re-seed the RNG so episode rollouts are deterministic."""
        self._rng = np.random.default_rng(self._seed)

    # ------------------------------------------------------------------
    # send  (identical to BroadcastProtocol)
    # ------------------------------------------------------------------

    def send(self, agent_id: str, state: EnvState) -> Message:
        """
        Build the honest outgoing message for *agent_id*.

        Reads the agent's believed hider position from ``state.obs[2:4]``.
        If either coordinate is SENTINEL (-1.0) the hider is not visible and
        both believed_hider_x/y are set to None.

        Parameters
        ----------
        agent_id : str
            Sending agent identifier, e.g. ``"seeker_0"``.
        state : EnvState
            Current environment snapshot.

        Returns
        -------
        Message
            Honest message ready to be (optionally) corrupted by a Byzantine
            agent before being passed to ``receive()``.
        """
        bx, by = state.true_hider_pos

        return Message(
            sender_id=agent_id,
            believed_hider_x=bx,
            believed_hider_y=by,
            step=state.step,
        )

    # ------------------------------------------------------------------
    # receive
    # ------------------------------------------------------------------

    def receive(
        self,
        messages: list[Optional[Message]],
    ) -> dict[str, tuple[float, float]]:
        """
        Keep a random subset of ``fanout`` messages; discard the rest.

        ``None`` entries (SilentByzantine) are removed from the pool before
        sampling so they do not consume a fanout slot.

        Parameters
        ----------
        messages : list of Message or None
            All outgoing messages for this round, possibly corrupted.

        Returns
        -------
        dict[str, tuple[float, float]]
            ``{sender_id: (believed_hider_x, believed_hider_y)}`` for the
            selected subset.  Senders absent from this dict will have their
            observation slots filled with sentinel -1.0 by the environment.
        """
        valid = [m for m in messages if m is not None]

        n_select = min(self._fanout, len(valid))
        if n_select == 0:
            return {}

        indices = self._rng.choice(len(valid), size=n_select, replace=False)
        selected = [valid[i] for i in sorted(indices)]

        buffer: dict[str, tuple[float, float]] = {}
        for msg in selected:
            bx = msg.believed_hider_x if msg.believed_hider_x is not None else SENTINEL
            by = msg.believed_hider_y if msg.believed_hider_y is not None else SENTINEL
            buffer[msg.sender_id] = (float(bx), float(by))

        return buffer
