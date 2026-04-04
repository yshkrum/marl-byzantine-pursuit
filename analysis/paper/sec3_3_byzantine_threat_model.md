# §3.3 Byzantine Threat Model

We adopt the Byzantine fault model introduced by Lamport et al. [1] to
define a communication-focused threat for cooperative pursuit. A seeker
agent $i$ is *Byzantine* if it deviates from the honest communication
protocol while executing movement actions indistinguishable from those
of an honest agent. Formally, let $\pi^{\mathrm{act}}_i$ denote agent
$i$'s movement policy and $\pi^{\mathrm{comm}}_i$ its message-generation
function. An honest agent satisfies $\pi^{\mathrm{comm}}_i(o_i) =
m^*(o_i)$, where $m^*$ extracts the agent's believed hider position from
its local observation $o_i$. A Byzantine agent $i \in \mathcal{B}$
satisfies $\pi^{\mathrm{act}}_i = \pi^{\mathrm{act}}_{\mathrm{honest}}$
but $\pi^{\mathrm{comm}}_i \neq m^*$: movement is indistinguishable from
honest behaviour, while outgoing messages are corrupted.

**Honest-movement assumption.** Restricting Byzantine deviations to the
communication channel isolates its effect on team coordination.
Practically, this models the most plausible failure modes in deployed
multi-robot systems—compromised radio transceivers, adversarial message
injection, and sensor spoofing—all of which corrupt the communication
layer while leaving locomotion unaffected. It also ensures that any
degradation in capture performance is attributable solely to information
quality rather than to sub-optimal movement strategy, providing a clean
causal signal for our experiments.

**Byzantine subtypes.** We implement four corruption strategies of
increasing adversarial sophistication. *(i) Random Noise:* the sender
replaces $\hat{p}_{\mathrm{hider}}$ with coordinates sampled uniformly
at random from the grid, modelling sensor malfunction or packet
corruption [2]. *(ii) Adversarial Misdirection:* the sender transmits
the reflection of the hider's true position through its own location,
$\hat{p} = 2p_i - p_{\mathrm{hider}}$, clamped to grid bounds. This
omniscient attacker actively directs teammates away from the hider and
represents the worst-case adversary. *(iii) Position Spoofing:* the
sender forges its identifier with a randomly chosen teammate's ID,
constituting a Sybil-style identity attack [3]. Receivers incorrectly
attribute the message to a legitimate peer, potentially suppressing that
peer's genuine contribution to the shared belief state. *(iv) Silence:*
the sender transmits no message, modelling communication jamming or
network dropout. Receiving agents retain their prior belief for the
silent sender's observation slot.

**Byzantine assignment.** Agents are assigned Byzantine roles
deterministically by index: agents $0, \ldots, \lfloor Nf \rfloor - 1$
are Byzantine, where $f \in \{0,\ 0.17,\ 0.33,\ 0.5\}$ is the
Byzantine fraction varied in Experiment 1. Deterministic assignment
eliminates sampling variance as a confound, ensuring that differences in
capture performance across conditions reflect communication degradation
alone. Prior work on Byzantine-resilient multi-agent systems has
similarly argued for fixed adversary assignment when the goal is
controlled measurement of protocol degradation rather than worst-case
robustness analysis [4].

---

### References

[1] L. Lamport, R. Shostak, and M. Pease. The Byzantine Generals
Problem. *ACM Transactions on Programming Languages and Systems*,
4(3):382–401, 1982.

[2] P. Blanchard, E. M. El Mhamdi, R. Guerraoui, and J. Stainer.
Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent.
In *Advances in Neural Information Processing Systems (NeurIPS)*, 2017.

[3] J. R. Douceur. The Sybil Attack. In *Proceedings of the 1st
International Workshop on Peer-to-Peer Systems (IPTPS)*, pp. 251–260,
2002.

[4] J. Blumenkamp and S. Albrecht. The Emergence of Adversarial
Communication in Multi-Agent Reinforcement Learning. In *Conference on
Robot Learning (CoRL)*, 2020.
