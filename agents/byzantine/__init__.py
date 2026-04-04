"""
Byzantine agent subtypes. Owner: C. Ticket: BYZ-03.

Subtypes:
    - RandomNoiseByzantine     : replace message with uniform random position
    - MisdirectionByzantine    : send position opposite to true hider (omniscient)
    - SpoofingByzantine        : forge sender_id with another seeker's id
    - SilentByzantine          : transmit null message (return None)
"""

from agents.byzantine.subtypes import (
    RandomNoiseByzantine,
    MisdirectionByzantine,
    SpoofingByzantine,
    SilentByzantine,
)

__all__ = [
    "RandomNoiseByzantine",
    "MisdirectionByzantine",
    "SpoofingByzantine",
    "SilentByzantine",
]
