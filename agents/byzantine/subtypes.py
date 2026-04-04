"""
BYZ-03: Four Byzantine agent communication corruption strategies.

Each class extends the ``ByzantineAgent`` mixin from ``comms/interface.py``
and overrides ``corrupt_message()`` to substitute or suppress the outgoing
message.  Movement is NOT affected — Byzantine agents use the same RL policy
as honest seekers; this module only intercepts communication.

Subtype summary
---------------
+---------------------+------------------------------------------------------+
| Class               | What it corrupts                                     |
+=====================+======================================================+
| RandomNoiseByzantine| believed_hider_x/y → uniform random grid position   |
| MisdirectionByzantine| believed_hider_x/y → reflection of true hider pos  |
|                     | through the agent's own position (omniscient)        |
| SpoofingByzantine   | sender_id → a randomly chosen other seeker's id     |
| SilentByzantine     | returns None (no message transmitted)                |
+---------------------+------------------------------------------------------+

Constraints (from BYZ-01 interface contract)
--------------------------------------------
* ``corrupt_message()`` uses **numpy only** — no PyTorch.
* All returned position values are normalised floats in [0, 1].
* Positions computed via arithmetic are clamped to [0, grid_size-1]
  (integer grid space) **before** normalising.
* Returning ``None`` is valid (SilentByzantine); receiving code fills
  the corresponding obs slot with sentinel -1.0.

Omniscient attacker assumption (MisdirectionByzantine)
------------------------------------------------------
MisdirectionByzantine requires the true hider position at every step,
which is injected via a callable at construction time.  This models a
**worst-case omniscient adversary** — the attacker always knows where the
hider is, even when the agent's FoV would normally occlude it.
This assumption must be documented clearly in paper §3.3.

Byzantine assignment (deterministic, set at env init)
------------------------------------------------------
Agents ``"seeker_0"`` … ``"seeker_{floor(N*f)-1}"`` are Byzantine where
*f* = ``byzantine_fraction``.  Assignment never changes mid-episode.

Owner : Role C (Byzantine & Comms)
Ticket: BYZ-03
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from env.schema import Message, SENTINEL
from comms.interface import ByzantineAgent


# ---------------------------------------------------------------------------
# RandomNoiseByzantine
# ---------------------------------------------------------------------------

class RandomNoiseByzantine(ByzantineAgent):
    """
    Replaces the believed hider position with a uniformly random grid cell.

    Models a **sensor malfunction** — the agent transmits plausible-looking
    but entirely random position data, providing no useful information to
    teammates and actively misleading them with high probability.

    Parameters
    ----------
    agent_id : str
        Identifier of this Byzantine seeker, e.g. ``"seeker_0"``.
    grid_size : int
        Side length of the square grid.  Random coordinates are drawn
        uniformly from ``[0, grid_size-1]`` then normalised to [0, 1].
    seed : int or None
        RNG seed for reproducibility.  ``None`` → non-deterministic.
    """

    def __init__(
        self,
        agent_id: str,
        grid_size: int,
        seed: int | None = None,
    ) -> None:
        self.agent_id = agent_id
        self.grid_size = grid_size
        self.rng = np.random.default_rng(seed)

    def corrupt_message(self, honest_message: Message) -> Message:
        """
        Return a copy of *honest_message* with position replaced by noise.

        The ``sender_id`` and ``step`` fields are preserved so the message
        looks structurally valid to receivers.

        Returns
        -------
        Message
            Well-formed message with random ``believed_hider_x/y``.
        """
        norm = float(self.grid_size - 1)
        bx = float(self.rng.integers(0, self.grid_size)) / norm
        by = float(self.rng.integers(0, self.grid_size)) / norm
        return Message(
            sender_id=honest_message.sender_id,
            believed_hider_x=bx,
            believed_hider_y=by,
            step=honest_message.step,
        )


# ---------------------------------------------------------------------------
# MisdirectionByzantine
# ---------------------------------------------------------------------------

class MisdirectionByzantine(ByzantineAgent):
    """
    Sends the reflection of the true hider position through the agent's
    own position, actively directing teammates *away* from the hider.

    Reflection formula (integer grid space)::

        reflected_row = 2 * agent_row - hider_row
        reflected_col = 2 * agent_col - hider_col

    Both coordinates are clamped to ``[0, grid_size - 1]`` before
    normalising to [0, 1] to ensure the message is a valid grid position.

    **Omniscient attacker assumption**: the agent always knows the true
    hider position, even when the hider is occluded in the agent's field
    of view.  This is the worst-case adversary model.  Document in §3.3.

    Parameters
    ----------
    agent_id : str
        Identifier of this Byzantine seeker.
    grid_size : int
        Side length of the square grid; used for clamping and normalisation.
    get_true_hider_pos : callable → tuple[int, int]
        Zero-argument callable that returns the hider's current
        ``(row, col)`` grid position each time it is called.
        Wire this to ``lambda: env.positions["hider"]``.
    get_agent_pos : callable → tuple[int, int]
        Zero-argument callable that returns this agent's current
        ``(row, col)`` grid position.
        Wire this to ``lambda: env.positions[agent_id]``.
    """

    def __init__(
        self,
        agent_id: str,
        grid_size: int,
        get_true_hider_pos: Callable[[], tuple[int, int]],
        get_agent_pos: Callable[[], tuple[int, int]],
    ) -> None:
        self.agent_id = agent_id
        self.grid_size = grid_size
        self._get_true_hider_pos = get_true_hider_pos
        self._get_agent_pos = get_agent_pos

    def corrupt_message(self, honest_message: Message) -> Message:
        """
        Return a message pointing *away* from the hider.

        Computes the reflection of the hider's true position through the
        agent's own position and clamps to grid bounds.

        Returns
        -------
        Message
            Message with ``believed_hider_x/y`` set to the reflected,
            clamped, and normalised anti-hider position.
        """
        ax, ay = self._get_agent_pos()
        hx, hy = self._get_true_hider_pos()

        # Reflection in integer grid space, then clamp
        rx = int(np.clip(2 * ax - hx, 0, self.grid_size - 1))
        ry = int(np.clip(2 * ay - hy, 0, self.grid_size - 1))

        norm = float(self.grid_size - 1)
        return Message(
            sender_id=honest_message.sender_id,
            believed_hider_x=float(rx) / norm,
            believed_hider_y=float(ry) / norm,
            step=honest_message.step,
        )


# ---------------------------------------------------------------------------
# SpoofingByzantine
# ---------------------------------------------------------------------------

class SpoofingByzantine(ByzantineAgent):
    """
    Forges the ``sender_id`` field with a randomly chosen other seeker's id.

    Models a **Sybil-style identity attack** — the message content (believed
    hider position) is taken from the agent's honest observation, but the
    receiver attributes it to a different seeker.  This causes receivers to
    build an incorrect picture of which teammates have seen the hider, and
    can overwrite valid messages from the spoofed agent.

    Parameters
    ----------
    agent_id : str
        True identifier of this Byzantine seeker.
    all_seeker_ids : list[str]
        IDs of all seekers in the episode, including this agent.
        The other agents' IDs are used as the pool of forged identities.
    seed : int or None
        RNG seed for reproducibility.
    """

    def __init__(
        self,
        agent_id: str,
        all_seeker_ids: list[str],
        seed: int | None = None,
    ) -> None:
        self.agent_id = agent_id
        self._other_ids: list[str] = [
            sid for sid in all_seeker_ids if sid != agent_id
        ]
        self.rng = np.random.default_rng(seed)

    def corrupt_message(self, honest_message: Message) -> Message:
        """
        Return *honest_message* with ``sender_id`` replaced by a random
        other seeker's id.

        If there are no other seekers (degenerate single-agent case) the
        message is returned unchanged.

        Returns
        -------
        Message
            Message identical to the honest one except for a forged
            ``sender_id``.
        """
        if not self._other_ids:
            return honest_message  # nothing to spoof against

        idx = int(self.rng.integers(0, len(self._other_ids)))
        fake_id = self._other_ids[idx]

        return Message(
            sender_id=fake_id,
            believed_hider_x=honest_message.believed_hider_x,
            believed_hider_y=honest_message.believed_hider_y,
            step=honest_message.step,
        )


# ---------------------------------------------------------------------------
# SilentByzantine
# ---------------------------------------------------------------------------

class SilentByzantine(ByzantineAgent):
    """
    Transmits no message — returns ``None`` every step.

    Models **communication jamming or dropout**.  The receiving protocol
    leaves the corresponding observation slot at its previous sentinel
    value (-1.0), so teammates receive no updated information from this
    agent.

    Parameters
    ----------
    agent_id : str
        Identifier of this Byzantine seeker.
    """

    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id

    def corrupt_message(self, honest_message: Message) -> None:
        """
        Suppress the message entirely.

        Returns
        -------
        None
            Signals to the receiving protocol that no message was sent.
            The env resets this agent's buffer slot to sentinel -1.0.
        """
        return None
