# Phase 2 Tickets — Role C: Byzantine & Comms
*Paper ownership: §3.3 Byzantine Threat Model*
*Contact: Ronit*

---

> **Status note:** BYZ-01 through BYZ-05 are fully complete and merged. All four Byzantine
> subtypes, the broadcast protocol, the comms interface, the validation script, and §3.3 are
> done with 130 tests passing. The Phase 2 tickets below are the remaining items needed to
> support Experiment 2 (protocol comparison).

---

### BYZ-06 · Implement GossipProtocol
**Priority:** High · **Blocks:** EXP-02 (Experiment 2) · **Deadline:** Before Exp2 runs

**Background**

`exp2_protocol_comparison.yaml` sweeps over 5 protocols: `none, broadcast, gossip, trimmed_mean, reputation`.
`gossip` and `trimmed_mean` are referenced but not implemented in `comms/`. `reputation` is
mentioned in docstrings but not implemented either. Experiment 2 cannot run without them.

A gossip protocol randomly selects a subset of peers to forward messages to each step, rather
than broadcasting to all. This models bandwidth-constrained networks and is more realistic
than all-to-all broadcast.

**Acceptance criteria**

- [ ] `comms/gossip.py`: `GossipProtocol(BaseProtocol)` class
- [ ] Constructor: `__init__(self, fanout: int = 2, seed: int = 0)`
  - `fanout`: number of peers each agent forwards to (default 2 of N-1 peers)
- [ ] `send()`: identical to BroadcastProtocol (honest message extraction from obs[2:4])
- [ ] `receive(messages)`: only processes a random `fanout` subset of received messages;
  others discarded; SENTINEL used for undelivered slots
- [ ] `reset()`: re-seeds the RNG so episodes are reproducible
- [ ] At least 10 tests in `tests/test_comms.py` covering: fanout=1 (each agent hears exactly
  one peer), fanout=N-1 (equivalent to broadcast), None handling, SENTINEL for undelivered

---

### BYZ-07 · Implement TrimmedMeanProtocol
**Priority:** High · **Blocks:** EXP-02 · **Deadline:** Before Exp2 runs

**Background**

Trimmed-mean aggregation is a classical Byzantine-robust aggregation rule (Yin et al. 2018).
Rather than averaging all received positions, it discards the top and bottom `trim_fraction`
of values before averaging. This provides partial resilience to RandomNoiseByzantine and
MisdirectionByzantine by removing outliers.

**Acceptance criteria**

- [ ] `comms/trimmed_mean.py`: `TrimmedMeanProtocol(BaseProtocol)` class
- [ ] Constructor: `__init__(self, trim_fraction: float = 0.2)`
  - Fraction of messages trimmed from each extreme (e.g., 0.2 → trim bottom 20% and top 20%)
- [ ] `receive(messages)`: for each coordinate (x, y) independently:
  1. Collect all non-SENTINEL values
  2. Sort; discard bottom and top `trim_fraction` of values (round down)
  3. Average the remaining values; if none remain, use SENTINEL
- [ ] Edge cases: fewer than 3 messages → fall back to simple mean (no trimming)
- [ ] `send()`: identical to BroadcastProtocol
- [ ] At least 10 tests: 1 Byzantine vs 3 honest (verify outlier trimmed), all-SENTINEL input,
  trim_fraction=0.0 (equivalent to mean), trim_fraction=0.5 (median)

**Reference:** Yin et al., *Byzantine-Robust Distributed Learning*, ICML 2018.

---

### BYZ-08 · Implement ReputationProtocol (optional / stretch)
**Priority:** Medium · **Deadline:** Only if time permits before Exp2

**Background**

A reputation-based protocol maintains a per-peer trust score that decays when a peer's
reported position is inconsistent with the team's consensus. If a peer's score falls below
a threshold, their messages are ignored. This is the most sophisticated resilience mechanism
and provides the strongest Exp2 comparison point.

**Acceptance criteria**

- [ ] `comms/reputation.py`: `ReputationProtocol(BaseProtocol)` class
- [ ] Per-peer reputation score initialised to 1.0 at `reset()`
- [ ] On `receive()`: compute deviation of each message from the current consensus position;
  decrease reputation if deviation > threshold, increase if within threshold
- [ ] Messages from agents with reputation < `min_trust` (default 0.3) are discarded
- [ ] `reset()` resets all scores to 1.0 (called by env at episode start)
- [ ] At least 8 tests: reputation decays for consistent noise, recovers for honest agents,
  all agents start trusted, complete isolation when trust = 0

**Note:** If time is too short, document this as a future-work item in the paper and drop it
from Exp2 config. Three protocols (none, broadcast, trimmed_mean) are sufficient for a
meaningful comparison. Let Role B know if you are skipping this so they can update Exp2 config.
