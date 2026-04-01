"""
BYZ-04 (visual): Matplotlib visualisation of Byzantine degradation results.

Imports run_condition() from validate_byzantine.py, runs both conditions, and
produces a 4-panel figure saved to:
    experiments/results/validation/byzantine_validation_plots.png

Usage
-----
    python scripts/plot_byzantine_results.py
    python scripts/plot_byzantine_results.py --n_episodes 30 --seed 42 --show

Panels
------
  [A] Box plot  — step-count distribution  f=0.0 vs f=0.5
  [B] Per-episode bar chart — steps for every episode, both conditions
  [C] Message quality — mean L2 error per episode (f=0.5 highlighted)
  [D] Behavioural breakdown — fallback%, blind%, capture-rate

Owner : Role C (Byzantine & Comms)
Ticket: BYZ-04
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.validate_byzantine import (   # reuse the same runner
    run_condition,
    N_SEEKERS,
    GRID_SIZE,
    OBS_RADIUS,
    N_EPISODES,
    F_BYZANTINE,
    SEED_BASE,
)

OUT_PATH = Path("experiments/results/validation/byzantine_validation_plots.png")

# Colour palette
COL_HONEST = "#4C9BE8"      # blue — f=0.0
COL_BYZ    = "#E8624C"      # red  — f=0.5
COL_LIGHT  = "#AED4F5"      # light blue
COL_LRED   = "#F5AEA0"      # light red


def make_figure(diags_f0, diags_f5, n_episodes: int) -> plt.Figure:
    steps_f0 = [d.steps    for d in diags_f0]
    steps_f5 = [d.steps    for d in diags_f5]
    err_f0   = [d.mean_msg_err for d in diags_f0]
    err_f5   = [d.mean_msg_err for d in diags_f5]
    fb_f0    = [d.fallback_pct for d in diags_f0]
    fb_f5    = [d.fallback_pct for d in diags_f5]
    bl_f0    = [d.blind_pct    for d in diags_f0]
    bl_f5    = [d.blind_pct    for d in diags_f5]
    rate_f0  = float(np.mean([d.captured for d in diags_f0])) * 100
    rate_f5  = float(np.mean([d.captured for d in diags_f5])) * 100

    mean_f0 = float(np.mean(steps_f0))
    mean_f5 = float(np.mean(steps_f5))
    degradation = (mean_f5 - mean_f0) / mean_f0 * 100.0

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        f"BYZ-04: Byzantine Degradation  |  N={N_SEEKERS}, grid={GRID_SIZE}x{GRID_SIZE}, "
        f"obs_radius={OBS_RADIUS}, {n_episodes} episodes\n"
        f"f=0.5 is {degradation:+.1f}% slower on average  (RandomNoiseByzantine)",
        fontsize=12, fontweight="bold",
    )

    eps = list(range(1, n_episodes + 1))

    # ------------------------------------------------------------------
    # Panel A — box plot: step-count distribution
    # ------------------------------------------------------------------
    ax = axes[0, 0]
    bp = ax.boxplot(
        [steps_f0, steps_f5],
        labels=["f=0.0\n(all honest)", f"f={F_BYZANTINE}\n(RandomNoise)"],
        patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
    )
    bp["boxes"][0].set_facecolor(COL_LIGHT)
    bp["boxes"][1].set_facecolor(COL_LRED)
    ax.set_ylabel("Episode steps to capture")
    ax.set_title("A  Step-count distribution")
    ax.axhline(mean_f0, color=COL_HONEST, linestyle="--", linewidth=1,
               label=f"mean f=0.0 = {mean_f0:.0f}")
    ax.axhline(mean_f5, color=COL_BYZ,    linestyle="--", linewidth=1,
               label=f"mean f=0.5 = {mean_f5:.0f}")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_ylim(bottom=0)

    # ------------------------------------------------------------------
    # Panel B — per-episode bar chart
    # ------------------------------------------------------------------
    ax = axes[0, 1]
    x = np.array(eps)
    w = 0.35
    ax.bar(x - w / 2, steps_f0, w, color=COL_HONEST, label="f=0.0", alpha=0.85)
    ax.bar(x + w / 2, steps_f5, w, color=COL_BYZ,    label=f"f={F_BYZANTINE}", alpha=0.85)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Steps")
    ax.set_title("B  Per-episode steps (both conditions)")
    ax.legend(fontsize=8)
    ax.set_xlim(0.3, n_episodes + 0.7)
    ax.set_ylim(bottom=0)
    if n_episodes <= 30:
        ax.set_xticks(eps)
        ax.tick_params(axis="x", labelsize=7)

    # ------------------------------------------------------------------
    # Panel C — message error per episode (f=0.5)
    # ------------------------------------------------------------------
    ax = axes[1, 0]
    ax.plot(eps, err_f0, color=COL_HONEST, linewidth=1.5, marker="o",
            markersize=3, label="f=0.0 (honest)")
    ax.plot(eps, err_f5, color=COL_BYZ,    linewidth=1.5, marker="s",
            markersize=3, label=f"f={F_BYZANTINE} (Byzantine)")
    ax.axhline(float(np.mean(err_f0)), color=COL_HONEST, linestyle="--",
               linewidth=1, alpha=0.7, label=f"mean={np.mean(err_f0):.3f}")
    ax.axhline(float(np.mean(err_f5)), color=COL_BYZ,    linestyle="--",
               linewidth=1, alpha=0.7, label=f"mean={np.mean(err_f5):.3f}")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Mean L2 message error (normalised)")
    ax.set_title("C  Message quality — L2 error vs true hider position")
    ax.legend(fontsize=8)
    ax.set_xlim(0.3, n_episodes + 0.7)
    ax.set_ylim(bottom=0)

    # ------------------------------------------------------------------
    # Panel D — behavioural breakdown (bar chart)
    # ------------------------------------------------------------------
    ax = axes[1, 1]
    categories = ["Fallback %\n(using msgs)", "Blind %\n(no msgs)", "Capture\nrate %"]
    vals_f0 = [float(np.mean(fb_f0)), float(np.mean(bl_f0)), rate_f0]
    vals_f5 = [float(np.mean(fb_f5)), float(np.mean(bl_f5)), rate_f5]

    x = np.arange(len(categories))
    w = 0.32
    bars0 = ax.bar(x - w / 2, vals_f0, w, color=COL_HONEST, label="f=0.0", alpha=0.85)
    bars5 = ax.bar(x + w / 2, vals_f5, w, color=COL_BYZ,    label=f"f={F_BYZANTINE}", alpha=0.85)

    # Value labels on bars
    for bar in list(bars0) + list(bars5):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                f"{h:.1f}%", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel("%")
    ax.set_title("D  Behavioural breakdown")
    ax.legend(fontsize=8)
    ax.set_ylim(0, max(max(vals_f0), max(vals_f5)) * 1.2 + 5)

    # ------------------------------------------------------------------
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


def main(n_episodes: int, seed: int, show: bool) -> None:
    print("Running condition f=0.0 (all honest) ...")
    diags_f0 = run_condition(0.0, n_episodes, seed)

    print(f"Running condition f={F_BYZANTINE} (RandomNoiseByzantine) ...")
    diags_f5 = run_condition(F_BYZANTINE, n_episodes, seed)

    print("Building figure ...")
    matplotlib.use("Agg" if not show else "TkAgg")
    fig = make_figure(diags_f0, diags_f5, n_episodes)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Plot saved -> {OUT_PATH}")

    if show:
        plt.show()

    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BYZ-04 visualisation.")
    parser.add_argument("--n_episodes", type=int, default=N_EPISODES)
    parser.add_argument("--seed",       type=int, default=SEED_BASE)
    parser.add_argument("--show",       action="store_true",
                        help="Open interactive matplotlib window after saving")
    args = parser.parse_args()
    main(n_episodes=args.n_episodes, seed=args.seed, show=args.show)
