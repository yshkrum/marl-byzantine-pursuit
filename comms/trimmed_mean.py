"""
BYZ-07: Trimmed-mean aggregation protocol.

A Byzantine-robust resilience protocol (Yin et al., ICML 2018).  On each
step every agent sends its honest believed hider position (identical to
BroadcastProtocol.send()).  On receive(), the protocol aggregates all
non-None messages into a single consensus (x, y) estimate by:

  1. Collecting all non-SENTINEL coordinate values independently per axis.
  2. Sorting each axis and discarding the bottom and top ``trim_fraction``
     of values (rounded down to a whole number of values).
  3. Averaging the remaining values.  If no values survive, SENTINEL is used.

The consensus position is then written into the buffer under every active
sender's ID.  This means each peer slot in the observation vector receives
the same robust aggregate rather than a raw (possibly corrupted) individual
report.

Fallback rule
-------------
If fewer than 3 valid messages are received, trimming is skipped and a
simple mean is used instead.  This avoids degenerate cases where trimming
would discard all values.

Reference: Yin et al., "Byzantine-Robust Distributed Learning: Towards
Optimal Statistical Rates", ICML 2018.

Owner : Role C (Byzantine & Comms)
Ticket: BYZ-07
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from env.schema import Message, SENTINEL
from comms.interface import BaseProtocol, EnvState


class TrimmedMeanProtocol(BaseProtocol):
    """
    Trimmed-mean resilience protocol.

    Each receiver aggregates all peer messages into a single consensus
    position estimate by discarding the most extreme reported values before
    averaging.  The consensus is broadcast into every peer slot of the
    observation vector, replacing raw per-sender reports.

    Parameters
    ----------
    trim_fraction : float
        Fraction of messages to discard from each extreme (0.0 → plain mean,
        0.5 → median-of-two middle values).  Must be in [0.0, 0.5).
        Default is 0.2 (trim bottom 20% and top 20%).
    """

    def __init__(self, trim_fraction: float = 0.2) -> None:
        if not (0.0 <= trim_fraction < 0.5):
            raise ValueError(
                f"trim_fraction must be in [0.0, 0.5), got {trim_fraction}"
            )
        self._trim_fraction = trim_fraction

    # ------------------------------------------------------------------
    # send  (identical to BroadcastProtocol)
    # ------------------------------------------------------------------

    def send(self, agent_id: str, state: EnvState) -> Message:
        """
        Build the honest outgoing message for *agent_id*.

        Reads the agent's believed hider position from ``state.obs[2:4]``.
        If the hider is not visible (SENTINEL) both fields are set to None.

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
        Aggregate received messages into a single trimmed-mean consensus.

        The consensus (x, y) is written under every active (non-None)
        sender's ID so that all peer slots in the observation vector receive
        the same robust estimate.

        Parameters
        ----------
        messages : list of Message or None
            All outgoing messages for this round, possibly corrupted.

        Returns
        -------
        dict[str, tuple[float, float]]
            Every active sender mapped to the same consensus position.
            Returns an empty dict if all messages are None.
        """
        valid = [m for m in messages if m is not None]
        if not valid:
            return {}

        # Collect non-SENTINEL coordinate values per axis
        xs = [
            m.believed_hider_x
            for m in valid
            if m.believed_hider_x is not None
        ]
        ys = [
            m.believed_hider_y
            for m in valid
            if m.believed_hider_y is not None
        ]

        consensus_x = self._trimmed_mean(xs)
        consensus_y = self._trimmed_mean(ys)

        return {m.sender_id: (consensus_x, consensus_y) for m in valid}

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _trimmed_mean(self, values: list[float]) -> float:
        """
        Compute the trimmed mean of *values*.

        Falls back to a simple mean when fewer than 3 values are present
        (trimming would be degenerate).  Returns SENTINEL if the list is
        empty.
        """
        if not values:
            return float(SENTINEL)

        if len(values) < 3:
            return float(np.mean(values))

        k = math.floor(len(values) * self._trim_fraction)
        sorted_vals = sorted(values)
        trimmed = sorted_vals[k : len(sorted_vals) - k] if k > 0 else sorted_vals

        if not trimmed:
            return float(SENTINEL)

        return float(np.mean(trimmed))
