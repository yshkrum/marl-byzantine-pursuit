"""
Byzantine agent subtypes. Owner: C. Ticket: BYZ-03.

Subtypes:
    - RandomNoiseByzantine     : replace message with uniform random position
    - MisdirectionByzantine    : send position opposite to true hider
    - SpoofingByzantine        : broadcast false self-position
    - SilentByzantine          : transmit null message
"""
