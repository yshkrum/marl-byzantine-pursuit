# Phase 2 Tickets — Role E: Analysis & Visualisation
*Paper ownership: §4 Results figures, §5 Discussion*
*Contact: (Role E)*

---

> **Status note:** `scripts/plot_byzantine_results.py` exists (201 lines, merged in BYZ commit).
> No experiment result CSVs exist yet — Role D (EXP-06, EXP-07) must run first.
> The tickets below can be partially prepared now (figure scaffolding, paper structure)
> while waiting for data.

---

### VIZ-01 · Inspect plot_byzantine_results.py and confirm figure spec
**Priority:** High · **No blockers — do now**

**Background**

`scripts/plot_byzantine_results.py` was written by Role C alongside the Byzantine
implementation. Review it now to confirm it produces the figures the paper needs, and
identify any gaps before result CSVs arrive.

**Acceptance criteria**

- [ ] Read `scripts/plot_byzantine_results.py` fully
- [ ] Confirm it generates: (1) capture rate vs Byzantine fraction line plot, (2) mean
  capture time vs Byzantine fraction bar chart, (3) per-subtype breakdown if available
- [ ] Confirm it reads from `experiments/results/exp1/` CSV files correctly
- [ ] Identify any missing figure for Exp2 (protocol comparison) and note what needs adding
- [ ] File `analysis/paper/figure_plan.md` with: figure number, caption draft, data source,
  script that generates it

---

### VIZ-02 · Generate Experiment 1 figures (after EXP-06 complete)
**Priority:** High · **Blocks:** Paper §4.1 · **Deadline:** After EXP-06 data arrives

**Figures needed for §4.1 — Byzantine Degradation**

1. **Fig 1 — Capture rate vs f:** Line plot, x-axis = byzantine_fraction {0.0, 0.125, 0.25, 0.375, 0.5}
   (= {0,1,2,3,4} Byzantine of 8), y-axis = capture rate (%), error bars = ±1 std across 5 seeds.
   Two lines: iPPO (no comms) and MAPPO (broadcast). N=8, 16×16, obs_radius=7 — cite PettingZoo
   benchmark defaults in figure caption.

2. **Fig 2 — Mean capture time vs f:** Same axes, mean capture time (steps) instead of
   capture rate. Helps distinguish "slower but eventual capture" from "complete failure to capture".

3. **Optional Fig 3 — Subtype comparison at f=0.33:** Bar chart across four Byzantine subtypes
   (random, misdirection, spoofing, silence) if Role D ran subtype variants. Otherwise skip.

**How to generate**

```bash
python scripts/plot_byzantine_results.py --exp_dir experiments/results/exp1/ --output analysis/figures/
```

**Acceptance criteria**

- [ ] Figures saved as `analysis/figures/fig1_capture_rate.pdf` and `fig2_capture_time.pdf`
- [ ] PDF format, 300 DPI minimum, axis labels, legend, error bars visible
- [ ] Colour-blind friendly palette (avoid pure red/green — use blue/orange or add markers)
- [ ] Caption text drafted in `analysis/paper/figure_captions.md`

---

### VIZ-03 · Generate Experiment 2 figures (after EXP-07 complete)
**Priority:** Medium · **Blocks:** Paper §4.2 · **Deadline:** After EXP-07 data arrives

**Figures needed for §4.2 — Protocol Comparison**

1. **Fig 4 — Protocol bar chart at f=0.25 (2/8 Byzantine):** Grouped bar chart, x-axis = protocol
   {none, broadcast, gossip, trimmed_mean}, y-axis = capture rate (%), error bars across 5 seeds.

2. **Fig 5 — Radar/spider chart (optional):** Three axes — capture rate, mean capture time,
   policy entropy convergence — one polygon per protocol. Makes trade-offs visible at a glance.

**Acceptance criteria**

- [ ] Figures at `analysis/figures/fig4_protocol_comparison.pdf`
- [ ] Consistent style with Exp1 figures (same font, same colour scheme)

---

### VIZ-04 · Draft §4 Results and §5 Discussion (after both experiments complete)
**Priority:** High · **Deadline:** Final paper deadline

**§4.1 Results — Byzantine Degradation (~300 words)**

Structure:
1. Report iPPO f=0.0 baseline: "Without Byzantine agents, iPPO achieves X% capture rate
   (mean Y steps) across seeds {42,43,44}."
2. Report MAPPO f=0.0 baseline: "MAPPO with broadcast protocol achieves X% (mean Y steps),
   a Z pp improvement attributable to shared hider position beliefs."
3. Degradation curve: "Capture rate declines monotonically with f: [table or inline numbers].
   At f=0.5, performance degrades by X pp (iPPO) and Y pp (MAPPO)."
4. Cross-reference §3.3 BYZ-04 validation criterion — confirm the 15% threshold was met.

**§4.2 Results — Protocol Comparison (~200 words)**

Structure:
1. At f=0.33, broadcast protocol achieves X% vs Y% for NoneProtocol — message sharing
   partially compensates for Byzantine noise.
2. TrimmedMean recovers Z pp compared to broadcast — outlier trimming effective against
   RandomNoiseByzantine at this fraction.

**§5 Discussion (~400 words)**

Key points to cover:
1. **Why ~60% iPPO floor** — coordination ceiling of independent agents, motivates centralised critic
2. **Honest-movement assumption limitations** — Byzantine movement would compound degradation; future work
3. **Omniscient attacker caveat** — MisdirectionByzantine assumes true hider position is known; weaker
   adversary model would show less degradation
4. **Generalisability** — 20×20 grid, partial obs (obs_radius=5) is more realistic than 10×10 full obs;
   discuss how results might scale to larger grids
5. **Future work:** gossip/reputation protocols, Byzantine-resilient actor gradient aggregation (BFTA),
   partial Byzantine movement

**Files to create:**
- `analysis/paper/sec4_results.md`
- `analysis/paper/sec5_discussion.md`
