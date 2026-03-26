"""
ENV-04: Frozen observation / action / message schema for Byzantine Pursuit.

⚠️  FROZEN AFTER WEEK 1 SIGN-OFF ⚠️
Do NOT modify this file without explicit agreement from all 5 team members.
Any change invalidates all training runs completed before the change.

Owner: Role A (Environment Engineer), co-authored with Role C (Byzantine & Comms)
Imported by: env/pursuit_env.py, agents/, comms/, scripts/

Schema version: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np

# ---------------------------------------------------------------------------
# Schema version – bump if any field changes (requires team agreement)
# ---------------------------------------------------------------------------
schema_version: str = "1.0.0"

# ---------------------------------------------------------------------------
# ACTION_MAP
# Discrete(5) action space shared by all agents.
# ---------------------------------------------------------------------------
ACTION_MAP: dict[int, str] = {
    0: "NOOP",
    1: "UP",       # row - 1
    2: "DOWN",     # row + 1
    3: "LEFT",     # col - 1
    4: "RIGHT",    # col + 1
}

# ---------------------------------------------------------------------------
# OBS_FIELDS
# Ordered list of (name, start_index, dtype, description) tuples.
# Indices 4 onward are relative offsets — use OBS_DIM() for the full layout
# since local_map and message sizes depend on grid_size / obs_radius / n_seekers.
# ---------------------------------------------------------------------------
OBS_FIELDS: list[tuple[str, int, str, str]] = [
    # name                index  dtype     description
    ("agent_x",          0,     "float32", "Agent row position, normalised to [0, 1]"),
    ("agent_y",          1,     "float32", "Agent col position, normalised to [0, 1]"),
    ("hider_x",          2,     "float32", "Hider row, normalised; sentinel -1.0 if occluded or outside FoV"),
    ("hider_y",          3,     "float32", "Hider col, normalised; sentinel -1.0 if occluded or outside FoV"),
    ("local_obstacle_map", 4,   "float32", "Flattened obstacle patch: (2*obs_radius+1)^2 when partial obs, "
                                           "grid_size^2 when full obs. 1.0=obstacle, 0.0=passable. "
                                           "Out-of-bounds cells padded with 1.0."),
    ("received_messages", -1,   "float32", "2*(n_seekers-1) floats: per-peer (believed_hider_x, believed_hider_y). "
                                           "Sentinel -1.0 until Role C populates the buffer (BYZ-01). "
                                           "Index = 4 + local_map_size."),
]

# Sentinel value used for all unknown / occluded / not-yet-received fields
SENTINEL: float = -1.0

# ---------------------------------------------------------------------------
# OBS_DIM
# Computes the total observation vector length given environment parameters.
# Call this instead of hardcoding a number anywhere in the codebase.
# ---------------------------------------------------------------------------

def OBS_DIM(n_seekers: int, grid_size: int, obs_radius: int | None) -> int:
    """
    Return the total observation vector length.

    Parameters
    ----------
    n_seekers : int
        Number of seeker agents in the environment.
    grid_size : int
        Side length of the square grid.
    obs_radius : int or None
        FoV half-side. ``None`` = full observability (entire grid visible).

    Returns
    -------
    int
        Total number of float32 elements in one observation vector.

    Layout
    ------
    [0:4]           agent_x, agent_y, hider_x, hider_y
    [4 : 4+M]       local_obstacle_map  (M = local_map_size)
    [4+M : 4+M+2K]  received_messages   (K = n_seekers - 1)
    """
    if obs_radius is None:
        local_map_size = grid_size * grid_size
    else:
        patch_side = 2 * obs_radius + 1
        local_map_size = patch_side * patch_side

    msg_size = 2 * (n_seekers - 1)
    return 4 + local_map_size + msg_size


# ---------------------------------------------------------------------------
# Message dataclass
# Struct passed between agents via the communication layer (Role C / BYZ-01).
# Byzantine agents corrupt believed_hider_x / believed_hider_y.
# ---------------------------------------------------------------------------

@dataclass
class Message:
    """
    Inter-agent communication message.

    Fields
    ------
    sender_id : str
        Agent id of the sender (e.g. ``"seeker_0"``).
    believed_hider_x : float or None
        Sender's belief of hider row position, normalised to [0, 1].
        ``None`` if the sender has no belief (e.g. hider not seen this step).
        Byzantine agents replace this with a corrupted value.
    believed_hider_y : float or None
        Sender's belief of hider col position, normalised to [0, 1].
        ``None`` if the sender has no belief.
        Byzantine agents replace this with a corrupted value.
    step : int
        Episode step at which the message was generated.
        Recipients can use this to discard stale messages.
    """
    sender_id: str
    believed_hider_x: Optional[float]
    believed_hider_y: Optional[float]
    step: int


# ---------------------------------------------------------------------------
# validate_observation
# Call this in tests and at environment boundaries to catch schema violations.
# ---------------------------------------------------------------------------

def validate_observation(
    obs: np.ndarray,
    n_seekers: int,
    grid_size: int,
    obs_radius: int | None = None,
) -> None:
    """
    Assert that *obs* conforms to the schema.

    Checks shape, dtype, and that all values are either in a valid normalised
    range [0, 1] or equal to the sentinel -1.0.

    Parameters
    ----------
    obs : np.ndarray
        Observation vector to validate.
    n_seekers : int
        Number of seekers (determines message slot count).
    grid_size : int
        Grid side length (determines local map size).
    obs_radius : int or None
        FoV half-side; ``None`` = full observability.

    Raises
    ------
    AssertionError
        If any schema constraint is violated.
    """
    expected_dim = OBS_DIM(n_seekers, grid_size, obs_radius)

    assert isinstance(obs, np.ndarray), (
        f"obs must be np.ndarray, got {type(obs)}"
    )
    assert obs.dtype == np.float32, (
        f"obs dtype must be float32, got {obs.dtype}"
    )
    assert obs.shape == (expected_dim,), (
        f"obs shape must be ({expected_dim},), got {obs.shape}. "
        f"(n_seekers={n_seekers}, grid_size={grid_size}, obs_radius={obs_radius})"
    )

    # All values must be either sentinel or in [0, 1]
    valid_mask = (obs == SENTINEL) | ((obs >= 0.0) & (obs <= 1.0))
    assert valid_mask.all(), (
        f"obs contains out-of-range values (not sentinel {SENTINEL} and not in [0,1]): "
        f"{obs[~valid_mask]}"
    )
