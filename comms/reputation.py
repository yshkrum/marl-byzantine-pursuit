"""
BYZ-08: Reputation-weighted aggregation protocol.

Each peer maintains a trust score in [0, 1] that is updated every step based
on how consistent their reported hider position is with the current consensus.
Peers whose score falls below ``min_trust`` are silenced — their messages are
discarded before the consensus is computed.

Algorithm per receive() call
-----------------------------
1. Filter None messages (SilentByzantine).
2. Split senders into trusted (score >= min_trust) and untrusted.
3. Compute consensus (x, y) as the mean of trusted senders' positions.
   - If no trusted senders have a valid position, fall back to mean of all
     valid non-None senders (so the protocol degrades gracefully rather than
     returning an empty buffer on the first call before scores diverge).
4. Update every sender's trust score:
   - L2 deviation from consensus > ``deviation_threshold``  → score -= 0.1
   - L2 deviation <= ``deviation_threshold``                → score += 0.1
   - Scores are clamped to [0.0, 1.0] after each update.
5. Return the consensus under every *trusted* sender's ID.

Senders not previously seen are initialised to trust 1.0 on first contact.
``reset()`` restores all scores to 1.0 (called by env at episode start).

Owner : Role C (Byzantine & Comms)
Ticket: BYZ-08
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from env.schema import Message, SENTINEL
from comms.interface import BaseProtocol, EnvState


class ReputationProtocol(BaseProtocol):
    """
    Reputation-based resilience protocol.

    Each sender accumulates a trust score updated every step by comparing
    their reported position to the team consensus.  Consistent senders gain
    trust; outliers lose it.  Senders below ``min_trust`` are ignored when
    computing the consensus.

    Parameters
    ----------
    min_trust : float
        Reputation threshold below which a sender's message is discarded.
        Default 0.3.  Must be in (0.0, 1.0].
    deviation_threshold : float
        Maximum L2 distance (in normalised [0, 1] coordinates) between a
        sender's report and the consensus before their score is penalised.
        Default 0.1 (≈ 1 grid cell on a 10×10 grid).
    trust_increment : float
        Amount added to a sender's score when their report is consistent.
        Default 0.1.
    trust_decrement : float
        Amount subtracted when their report deviates beyond the threshold.
        Default 0.1.
    """

    def __init__(
        self,
        min_trust: float = 0.3,
        deviation_threshold: float = 0.1,
        trust_increment: float = 0.1,
        trust_decrement: float = 0.1,
    ) -> None:
        if not (0.0 < min_trust <= 1.0):
            raise ValueError(f"min_trust must be in (0.0, 1.0], got {min_trust}")
        if deviation_threshold < 0.0:
            raise ValueError(
                f"deviation_threshold must be >= 0.0, got {deviation_threshold}"
            )
        self._min_trust = min_trust
        self._deviation_threshold = deviation_threshold
        self._trust_increment = trust_increment
        self._trust_decrement = trust_decrement
        self._scores: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def trust_scores(self) -> dict[str, float]:
        """Read-only view of current trust scores (copy)."""
        return dict(self._scores)

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset all trust scores to 1.0 (called by env at episode start)."""
        self._scores = {sid: 1.0 for sid in self._scores}

    # ------------------------------------------------------------------
    # send  (identical to BroadcastProtocol)
    # ------------------------------------------------------------------

    def send(self, agent_id: str, state: EnvState) -> Message:
        """
        Build the honest outgoing message for *agent_id*.

        Reads obs[2:4] for believed hider position; sets fields to None if
        the hider is occluded (SENTINEL).

        Parameters
        ----------
        agent_id : str
            Sending agent identifier, e.g. ``"seeker_0"``.
        state : EnvState
            Current environment snapshot.

        Returns
        -------
        Message
            Honest message ready to be (optionally) corrupted before receive().
        """
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

    # ------------------------------------------------------------------
    # receive
    # ------------------------------------------------------------------

    def receive(
        self,
        messages: list[Optional[Message]],
    ) -> dict[str, tuple[float, float]]:
        """
        Aggregate trusted messages into a reputation-weighted consensus.

        Trust scores are updated in-place; the consensus is returned under
        every trusted sender's ID.

        Parameters
        ----------
        messages : list of Message or None
            All outgoing messages this round, possibly corrupted.

        Returns
        -------
        dict[str, tuple[float, float]]
            Every *trusted* sender mapped to the consensus (x, y).
            Returns an empty dict only if all messages are None.
        """
        valid = [m for m in messages if m is not None]
        if not valid:
            return {}

        # Register new senders at full trust
        for m in valid:
            if m.sender_id not in self._scores:
                self._scores[m.sender_id] = 1.0

        trusted = [m for m in valid if self._scores[m.sender_id] >= self._min_trust]
        untrusted = [m for m in valid if self._scores[m.sender_id] < self._min_trust]

        # Compute consensus from trusted senders; fall back to all valid if needed
        pool = trusted if trusted else valid
        consensus_x, consensus_y = self._mean_position(pool)

        # Update trust scores for all valid senders
        for m in valid:
            mx = m.believed_hider_x if m.believed_hider_x is not None else SENTINEL
            my = m.believed_hider_y if m.believed_hider_y is not None else SENTINEL

            if consensus_x == SENTINEL or consensus_y == SENTINEL:
                # No consensus available — treat all as consistent
                deviation = 0.0
            elif mx == SENTINEL or my == SENTINEL:
                # Sender claims no visibility; treat as inconsistent
                deviation = float("inf")
            else:
                deviation = float(np.sqrt((mx - consensus_x) ** 2 + (my - consensus_y) ** 2))

            if deviation > self._deviation_threshold:
                self._scores[m.sender_id] = max(
                    0.0, self._scores[m.sender_id] - self._trust_decrement
                )
            else:
                self._scores[m.sender_id] = min(
                    1.0, self._scores[m.sender_id] + self._trust_increment
                )

        # Return consensus under trusted senders only
        if not trusted:
            # No trusted senders yet — return consensus under all valid senders
            # so the first few steps are not completely blind
            return {m.sender_id: (consensus_x, consensus_y) for m in valid}

        return {m.sender_id: (consensus_x, consensus_y) for m in trusted}

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _mean_position(
        self, messages: list[Message]
    ) -> tuple[float, float]:
        """Return the mean (x, y) of non-SENTINEL positions in *messages*."""
        xs = [
            m.believed_hider_x
            for m in messages
            if m.believed_hider_x is not None
        ]
        ys = [
            m.believed_hider_y
            for m in messages
            if m.believed_hider_y is not None
        ]

        cx = float(np.mean(xs)) if xs else float(SENTINEL)
        cy = float(np.mean(ys)) if ys else float(SENTINEL)
        return cx, cy
