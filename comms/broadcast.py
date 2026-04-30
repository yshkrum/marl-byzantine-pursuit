"""
BYZ-02: Honest broadcast communication protocol.

Every seeker sends its believed hider position to all other seekers each
step.  This is the honest baseline protocol — no corruption, no filtering.
All Byzantine experiments use this protocol on the send side; Byzantine
agents corrupt their outgoing message after send() returns.

Protocol behaviour
------------------
send()    : Read obs[2:4] for believed hider position; build Message.
receive() : Last-writer-wins per sender.  None entries (SilentByzantine)
            are skipped — the corresponding slot stays at its previous
            value in env._message_buffer (or sentinel if never received).

Integration into env/pursuit_env.py step()
-------------------------------------------
After env resolves each seeker's movement action:

    state = EnvState(obs=env._get_observation(agent), step=env._step_count,
                     grid_size=env.grid_size)
    msg = protocol.send(agent, state)                  # honest message
    if agent in byzantine_agents:
        msg = byzantine_agents[agent].corrupt_message(msg)   # corrupt
    if msg is None:                                    # SilentByzantine
        env._message_buffer[agent] = (SENTINEL, SENTINEL)
    else:
        env._message_buffer.update(protocol.receive([msg]))  # write buffer

Owner : Role C (Byzantine & Comms)
Ticket: BYZ-02
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from env.schema import Message, SENTINEL
from comms.interface import BaseProtocol, EnvState


class BroadcastProtocol(BaseProtocol):
    """
    Honest broadcast protocol: every seeker shares its believed hider
    position with all other seekers every step.

    This is the communication baseline for all experiments.  Running with
    ``byzantine_fraction=0.0`` gives the honest-broadcast performance
    ceiling; running with ``f > 0`` and no resilience protocol gives the
    degraded performance that Experiment 1 measures.

    Stateless — no per-episode state; ``reset()`` is a no-op.
    """

    # ------------------------------------------------------------------
    # send
    # ------------------------------------------------------------------

    def send(self, agent_id: str, state: EnvState) -> Message:
        """
        Build the honest outgoing message for *agent_id*.

        Reads the agent's believed hider position from ``state.obs[2:4]``.
        If the hider is not visible (either coordinate equals SENTINEL -1.0)
        both ``believed_hider_x`` and ``believed_hider_y`` are set to
        ``None`` — the receiver will treat the slot as unknown.

        Parameters
        ----------
        agent_id : str
            Sending agent identifier, e.g. ``"seeker_0"``.
        state : EnvState
            Current environment snapshot; must have ``obs``, ``step``,
            and ``grid_size`` fields.

        Returns
        -------
        Message
            Well-formed honest message ready to be (optionally) corrupted
            by a Byzantine agent before being passed to ``receive()``.
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
        Aggregate a batch of messages into a message buffer update.

        Last-writer-wins per sender: if the same sender appears more than
        once only the last message is kept (degenerate case; shouldn't
        occur in normal operation).

        ``None`` entries (from ``SilentByzantine``) are silently skipped.
        The caller is responsible for writing sentinel values for silent
        senders if needed (see integration notes in module docstring).

        ``None`` position fields (hider not visible at sender) map to
        ``SENTINEL`` (-1.0) so the receiving agent's observation slot is
        filled with the canonical unknown value.

        Parameters
        ----------
        messages : list of Message or None
            Outgoing messages for this round, possibly corrupted.

        Returns
        -------
        dict[str, tuple[float, float]]
            ``{sender_id: (believed_hider_x, believed_hider_y)}``.
            Write this to ``env._message_buffer``.
        """
        buffer: dict[str, tuple[float, float]] = {}

        for msg in messages:
            if msg is None:
                continue  # SilentByzantine — caller handles sentinel

            bx = msg.believed_hider_x if msg.believed_hider_x is not None else SENTINEL
            by = msg.believed_hider_y if msg.believed_hider_y is not None else SENTINEL
            buffer[msg.sender_id] = (float(bx), float(by))

        return buffer
