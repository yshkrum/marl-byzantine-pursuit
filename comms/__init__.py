"""
Communication protocols. Owner: C. Ticket: BYZ-01, BYZ-02.

Protocols:
    - NoneProtocol       : no communication (independent baseline)
    - BroadcastProtocol  : all agents broadcast to all others
    - GossipProtocol     : each agent sends to k random neighbours
    - SelectiveProtocol  : send only if belief update > threshold delta
    - TrimmedMeanProtocol: recipients apply trimmed mean aggregation
    - ReputationProtocol : trust-weighted message aggregation
"""
