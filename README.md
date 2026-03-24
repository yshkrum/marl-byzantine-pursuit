# MARL Byzantine-Resilient Pursuit

**Research question:** Does Byzantine communication corruption degrade cooperative pursuit performance predictably, and can learned or engineered resilience protocols recover it — and at what team size does coordination itself become the bottleneck?

## Project structure

```
marl-byzantine-pursuit/
├── env/                        # PettingZoo custom environment
├── agents/
│   ├── ppo/                    # Independent PPO
│   ├── mappo/                  # MAPPO (CTDE)
│   ├── greedy/                 # Greedy heuristic baseline
│   └── byzantine/              # Byzantine agent subtypes
├── comms/                      # Communication protocols
├── experiments/
│   ├── configs/                # YAML experiment configs
│   └── results/                # Output CSVs (gitignored, see below)
├── analysis/
│   ├── plots/                  # Generated figures
│   └── notebooks/              # Exploration notebooks
├── tests/                      # Unit tests
├── docs/                       # Paper drafts and notes
├── scripts/                    # Sweep runner, logging utils
└── requirements.txt
```

## Setup

```bash
git clone https://github.com/<org>/marl-byzantine-pursuit.git
cd marl-byzantine-pursuit
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running experiments

```bash
python scripts/run_sweep.py --config experiments/configs/exp1_byzantine_degradation.yaml
```

## Branch convention

| Branch | Purpose |
|--------|---------|
| `main` | Stable, reviewed code only |
| `dev` | Integration branch — PRs target here |
| `env/...` | Environment work |
| `agents/...` | RL and agent work |
| `comms/...` | Byzantine and protocol work |
| `exp/...` | Experiment scripts and configs |
| `analysis/...` | Plots and analysis |

## Team

| Role | Branch prefix |
|------|--------------|
| Environment Engineer | `env/` |
| RL Training Lead | `agents/` |
| Byzantine & Comms | `comms/` |
| Experiment Runner | `exp/` |
| Analysis & Viz | `analysis/` |
