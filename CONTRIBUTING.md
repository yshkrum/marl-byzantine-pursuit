# Contributing

## Branch strategy

```
main
└── dev  ← all PRs target here
    ├── env/ENV-01-base-environment
    ├── env/ENV-02-obstacle-maze
    ├── agents/RL-01-greedy-baseline
    ├── agents/RL-02-independent-ppo
    ├── comms/BYZ-01-message-interface
    ├── comms/BYZ-03-byzantine-subtypes
    ├── exp/EXP-01-logging
    └── analysis/VIZ-01-plot-pipeline
```

**Never push directly to `main` or `dev`.**

## Creating a branch

```bash
git checkout dev
git pull origin dev
git checkout -b env/ENV-01-base-environment
```

Branch name format: `<prefix>/<TICKET-ID>-short-description`

| Role | Prefix |
|------|--------|
| Environment Engineer | `env/` |
| RL Training Lead | `agents/` |
| Byzantine & Comms | `comms/` |
| Experiment Runner | `exp/` |
| Analysis & Viz | `analysis/` |

## Commit messages

```
ENV-01: add PettingZoo env scaffold with reset/step stubs
RL-02:  implement independent PPO training loop
BYZ-03: add random noise and misdirection Byzantine subtypes
```

Format: `TICKET-ID: lowercase description of what changed`

## Pull requests

- PR title must include ticket ID: `[ENV-02] Obstacle maze generation`
- Target branch: `dev` (never `main`)
- Require at least 1 reviewer approval before merge
- All tests must pass: `pytest tests/`
- Keep PRs small — one ticket per PR

## Merging to main

Only at end-of-week milestones. One person (rotate weekly) does a single squash merge from `dev` → `main` after team review.

## What NOT to commit

See `.gitignore`. Never commit:
- Model weights (`*.pt`, `*.pth`, `*.ckpt`)
- Raw result CSVs (`experiments/results/`)
- W&B run folders (`wandb/`)
- Jupyter notebook outputs (clear outputs before committing)

Configs go in version control. Data does not.

## Running tests

```bash
pytest tests/ -v
```

Add a test for every non-trivial function. Byzantine subtypes especially must have unit tests — a silent bug in misdirection logic will corrupt all of Experiment 1.
