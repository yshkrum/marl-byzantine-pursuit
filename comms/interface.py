"""
BYZ-01: Message passing interface for Byzantine Pursuit.

Defines abstract base classes for all communication protocols (BaseProtocol)
and Byzantine agent corruption strategies (ByzantineAgent).

Send / receive timing within a single env.step() round
-------------------------------------------------------
Each "round" spans one full cycle of the AEC loop (all seekers have submitted
their movement actions and the env has resolved movement).  Role C integrates
this interface at the end of each round:

  1. All seekers submit movement actions; env.step() resolves movement.
  2. For each seeker:
         msg = protocol.send(agent_id, state)
     The honest message contains the agent's believed hider position read
     directly from its observation (obs[2], obs[3]).
  3. If the seeker is Byzantine:
         msg = agent.corrupt_message(msg)
     The Byzantine agent substitutes or suppresses the message.
     SilentByzantine returns None; all other subtypes return a Message.
  4. Collect all outgoing messages (a list that may contain None entries).
  5. Update the environment message buffer:
         env._message_buffer = protocol.receive(messages)
     The protocol aggregates messages into a sender → position mapping.
  6. env._get_observation() reads env._message_buffer to fill the
     received_messages slots in the next observation vector.

Observation slot layout (cross-reference env/schema.py OBS_FIELDS)
-------------------------------------------------------------------
  [0:4]          agent_x, agent_y, hider_x, hider_y
  [4 : 4+M]      local_obstacle_map  (M = grid_size² or (2*radius+1)²)
  [4+M : 4+M+2K] received_messages   (K = n_seekers-1 peer slots)

Each peer slot is a (believed_hider_x, believed_hider_y) float32 pair ordered
by seeker index.  Slots for silent or absent senders are filled with sentinel
-1.0 (schema.SENTINEL).

Owner: Role C (Byzantine & Comms)
Ticket: BYZ-01
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np

from env.schema import Message, SENTINEL


# ---------------------------------------------------------------------------
# EnvState
# ---------------------------------------------------------------------------

@dataclass
class EnvState:
    """
    Snapshot of per-agent environment state passed to ``BaseProtocol.send()``.

    Attributes
    ----------
    obs : np.ndarray
        The sending agent's current float32 observation vector.
        Indices 2 and 3 are ``believed_hider_x`` and ``believed_hider_y``
        (normalised to [0, 1], or ``SENTINEL`` -1.0 if the hider is not
        visible this step).
    step : int
        Current episode step count; embedded in the outgoing Message so
        recipients can discard stale messages.
    grid_size : int
        Side length of the square grid.  Passed to Byzantine subtypes so
        they can clamp reflected positions to [0, grid_size-1].
    """

    obs: np.ndarray
    step: int
    grid_size: int
    true_hider_pos: tuple[float, float]  # normalised (row/norm, col/norm), always valid


# ---------------------------------------------------------------------------
# BaseProtocol
# ---------------------------------------------------------------------------

class BaseProtocol(ABC):
    """
    Abstract base class for all inter-agent communication protocols.

    Each concrete protocol (broadcast, gossip, trimmed-mean, reputation …)
    extends this class and implements ``send()`` and ``receive()``.

    The protocol is **stateless** by default — no per-episode state is stored
    here.  The only exception is ``ReputationProtocol``, which maintains
    ``trust_score`` vectors; those are reset at the start of each episode by
    calling ``reset()``.
    """

    def reset(self) -> None:
        """
        Reset any per-episode protocol state (e.g. reputation scores).

        The default implementation is a no-op; stateful protocols override
        this and call it from ``env.reset()``.
        """

    @abstractmethod
    def send(self, agent_id: str, state: EnvState) -> Message:
        """
        Construct the honest outgoing message for *agent_id*.

        Reads ``state.obs[2]`` and ``state.obs[3]`` for the agent's believed
        hider position.  If either coordinate is ``SENTINEL`` (-1.0) the
        hider is not visible and the message fields are set to ``None``.

        Parameters
        ----------
        agent_id : str
            Identifier of the sending agent, e.g. ``"seeker_0"``.
        state : EnvState
            Current environment snapshot for this agent.

        Returns
        -------
        Message
            An honest message populated with the agent's believed hider
            position (or ``None`` fields if the hider is not visible).
        """
        ...

    @abstractmethod
    def receive(
        self,
        messages: list[Optional[Message]],
    ) -> dict[str, tuple[float, float]]:
        """
        Aggregate a batch of outgoing messages into an updated message buffer.

        Called once per round after all seekers have produced their (possibly
        corrupted) outgoing messages.  The returned dict is written directly
        to ``env._message_buffer`` so that ``env._get_observation()`` picks up
        the new values on the next step.

        ``None`` entries (from ``SilentByzantine``) must be handled gracefully:
        the corresponding sender is simply absent from the returned dict.
        ``env._get_observation()`` already fills missing entries with sentinel
        ``(-1.0, -1.0)``.

        Parameters
        ----------
        messages : list of Message or None
            All outgoing messages for this round.  May contain ``None`` for
            silent senders.

        Returns
        -------
        dict[str, tuple[float, float]]
            Mapping ``sender_id → (believed_hider_x, believed_hider_y)``.
            Coordinates are normalised floats in [0, 1] or ``SENTINEL``
            (-1.0) for unknown positions.
            Write this return value to ``env._message_buffer``.
        """
        ...


# ---------------------------------------------------------------------------
# ByzantineAgent
# ---------------------------------------------------------------------------

class ByzantineAgent(ABC):
    """
    Abstract mixin for Byzantine agent communication corruption strategies.

    Byzantine agents execute **honest movement** — their ``act()`` method is
    identical to that of an honest seeker given the same observation.  The
    only deviation is in ``corrupt_message()``, which replaces or suppresses
    the outgoing communication message.

    Subclasses
    ----------
    RandomNoiseByzantine  : Replace position with a uniform random grid coord.
    MisdirectionByzantine : Reflect true hider position through agent position
                            (requires true hider pos injected at construction —
                            omniscient attacker assumption; document in paper).
    SpoofingByzantine     : Forge the sender_id field with another agent's id.
    SilentByzantine       : Return None — transmit no message this step.

    Byzantine agent assignment (deterministic for reproducibility)
    --------------------------------------------------------------
    Agents ``"seeker_0"`` … ``"seeker_{floor(N*f)-1}"`` are Byzantine where
    *f* is ``byzantine_fraction`` set at ``env.__init__``.  Assignment never
    changes mid-episode.

    Constraint
    ----------
    ``corrupt_message()`` must use **numpy only** — no PyTorch.
    All returned positions must be clamped to [0, 1] (normalised) or to the
    valid grid range [0, grid_size-1] before normalisation.
    """

    @abstractmethod
    def corrupt_message(self, honest_message: Message) -> Optional[Message]:
        """
        Intercept and corrupt an outgoing message.

        This method is called **after** the movement action has been resolved
        by the environment.  It must not affect movement in any way — it only
        substitutes the communication payload.

        Parameters
        ----------
        honest_message : Message
            The message that an honest agent would have sent (produced by
            ``protocol.send()``).

        Returns
        -------
        Message or None
            Corrupted message to transmit in place of the honest one.
            Return ``None`` to suppress transmission entirely
            (``SilentByzantine`` behaviour).  The receiving protocol must
            fill the corresponding observation slot with sentinel ``-1.0``.
        """
        ...


# ---------------------------------------------------------------------------
# NoneProtocol  (no-communication baseline — independent agents)
# ---------------------------------------------------------------------------

class NoneProtocol(BaseProtocol):
    """
    No-communication protocol.

    ``send()`` still builds a well-formed Message (the agent computes its
    belief internally) but ``receive()`` returns an empty buffer, so the
    received_messages slots in every observation are always sentinel -1.0.

    Use this as the independent-agents baseline (Experiment 3, f=0,
    no comms) to isolate the effect of communication from the baseline
    capture performance.
    """

    def send(self, agent_id: str, state: EnvState) -> Message:
        """Build an honest message but it will be discarded by receive()."""
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
        """Discard all messages — agents receive no peer information."""
        return {}
