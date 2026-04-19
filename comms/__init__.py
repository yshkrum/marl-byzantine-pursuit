"""
Communication protocols. Owner: C. Tickets: BYZ-01, BYZ-02, BYZ-06, BYZ-07, BYZ-08.

Protocols:
    - NoneProtocol        : no communication (independent baseline)
    - BroadcastProtocol   : all agents broadcast to all others
    - GossipProtocol      : each agent forwards to k random neighbours
    - TrimmedMeanProtocol : trimmed-mean aggregation (Byzantine-robust)
    - ReputationProtocol  : trust-score filtered aggregation (Byzantine-robust)

Note: SelectiveProtocol (send only on belief delta > threshold) is not
implemented and has been removed from scope. Three resilience protocols
(broadcast, trimmed_mean, reputation) are sufficient for Experiment 2.
"""

from comms.interface import BaseProtocol, ByzantineAgent, EnvState, NoneProtocol
from comms.broadcast import BroadcastProtocol
from comms.gossip import GossipProtocol
from comms.trimmed_mean import TrimmedMeanProtocol
from comms.reputation import ReputationProtocol

__all__ = [
    "BaseProtocol",
    "ByzantineAgent",
    "EnvState",
    "NoneProtocol",
    "BroadcastProtocol",
    "GossipProtocol",
    "TrimmedMeanProtocol",
    "ReputationProtocol",
]
