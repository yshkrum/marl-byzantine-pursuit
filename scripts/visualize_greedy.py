"""
Greedy agent visualisation — manual simulation on the real maze.

Drives GreedyAgent(s) against a random hider using env.reset() and
env.observe() (both implemented). Movement is resolved here because
env.step() is not yet available (waiting for RL-04).

Usage
-----
    python scripts/visualize_greedy.py
    python scripts/visualize_greedy.py --n_seekers 2 --grid_size 15 --max_steps 200
    python scripts/visualize_greedy.py --seed 7 --delay 0.15
    python scripts/visualize_greedy.py --save_mp4 outputs/chase.gif

Arguments
---------
--n_seekers   int   number of seeker agents (default 1)
--grid_size   int   side length of the square grid (default 12)
--max_steps   int   episode step limit (default 150)
--seed        int   RNG seed (default 42)
--delay       float seconds between frames / GIF frame duration (default 0.12)
--obs_radius  int   FoV half-side; omit for full observability
--save_mp4    str   if given, save episode as an MP4 to this path (no window shown)
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4

# Ensure project root is on the path when run from scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from env.pursuit_env import ByzantinePursuitEnv
from agents.greedy.greedy_agent import GreedyAgent

# Action deltas: (row_delta, col_delta) — matches schema.py ACTION_MAP
_ACTION_DELTAS = {0: (0, 0), 1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}


# ---------------------------------------------------------------------------
# Manual step helper (replaces env.step() until RL-04 is done)
# ---------------------------------------------------------------------------

def _apply_action(env: ByzantinePursuitEnv, agent: str, action: int) -> None:
    """Move *agent* by *action* in *env*, respecting walls."""
    row, col = env.positions[agent]
    dr, dc = _ACTION_DELTAS[action]
    nr, nc = row + dr, col + dc
    gs = env.grid_size
    if 0 <= nr < gs and 0 <= nc < gs and not env.grid[nr, nc]:
        env.positions[agent] = (nr, nc)


def _is_captured(env: ByzantinePursuitEnv) -> str | None:
    """Return the seeker id that caught the hider, or None."""
    hider_pos = env.positions["hider"]
    for agent_id, pos in env.positions.items():
        if agent_id.startswith("seeker_") and pos == hider_pos:
            return agent_id
    return None


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

# Colour palette
_C_WALL    = np.array([0.25, 0.25, 0.25])   # dark grey
_C_OPEN    = np.array([0.95, 0.95, 0.90])   # off-white
_C_SEEKER  = np.array([0.20, 0.50, 0.90])   # blue
_C_HIDER   = np.array([0.90, 0.25, 0.25])   # red
_C_CAPTURE = np.array([0.20, 0.80, 0.20])   # green flash


def _build_rgb(env: ByzantinePursuitEnv, flash: bool = False) -> np.ndarray:
    """Return H×W×3 float32 image for the current env state."""
    gs = env.grid_size
    img = np.where(env.grid[:, :, np.newaxis], _C_WALL, _C_OPEN).astype(np.float32)

    hider_pos = env.positions.get("hider")
    for agent_id, (r, c) in env.positions.items():
        if agent_id == "hider":
            colour = _C_CAPTURE if flash else _C_HIDER
        else:
            colour = _C_CAPTURE if flash else _C_SEEKER
        img[r, c] = colour

    return img


# ---------------------------------------------------------------------------
# Main episode loop
# ---------------------------------------------------------------------------

def run(
    n_seekers: int,
    grid_size: int,
    max_steps: int,
    seed: int,
    delay: float,
    obs_radius: int | None,
    save_mp4: str | None = None,
) -> None:
    env = ByzantinePursuitEnv(
        n_seekers=n_seekers,
        grid_size=grid_size,
        obs_radius=obs_radius,
        obstacle_density=0.15,
        byzantine_fraction=0.0,
        max_steps=max_steps,
        seed=seed,
    )

    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:6]}"
    print(f"Run ID: {run_id}")

    obs_dict, _ = env.reset()

    seekers = [GreedyAgent(f"seeker_{i}", grid_size, seed=seed + i)
               for i in range(n_seekers)]
    rng = np.random.default_rng(seed)

    # ----------------------------------------------------------------
    # Matplotlib setup — always show the interactive window
    # ----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xticks([])
    ax.set_yticks([])
    fig.patch.set_facecolor("#1a1a1a")
    ax.set_facecolor("#1a1a1a")

    im = ax.imshow(_build_rgb(env), interpolation="nearest", vmin=0, vmax=1)

    legend_patches = [
        mpatches.Patch(color=_C_SEEKER, label="Seeker (greedy BFS)"),
        mpatches.Patch(color=_C_HIDER,  label="Hider (random)"),
        mpatches.Patch(color=_C_WALL,   label="Wall / obstacle"),
    ]
    ax.legend(handles=legend_patches, loc="upper right",
              fontsize=7, framealpha=0.7)

    title = ax.set_title("", color="white", fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.ion()
    plt.show()

    # ----------------------------------------------------------------
    # Episode simulation
    # ----------------------------------------------------------------
    obstacle_map = env.grid.astype(bool)
    frames = []   # populated only when save_mp4 is set
    catcher = None

    for step in range(1, max_steps + 1):
        # --- Seeker actions (greedy BFS) ---
        for i, seeker in enumerate(seekers):
            agent_id = f"seeker_{i}"
            obs = env.observe(agent_id)
            action = seeker.act(obs, obstacle_map)
            _apply_action(env, agent_id, action)

        # --- Hider action (random walk) ---
        hider_action = int(rng.integers(0, 5))
        _apply_action(env, "hider", hider_action)

        # --- Capture check ---
        catcher = _is_captured(env)
        flash = catcher is not None

        # --- Update figure ---
        im.set_data(_build_rgb(env, flash=flash))
        obs_label = f"obs_radius={obs_radius}" if obs_radius else "full obs"
        title.set_text(
            f"Step {step}/{max_steps}  |  {n_seekers} seeker(s)  |  "
            f"seed={seed}  |  {obs_label}"
            + (f"\n  CAPTURED by {catcher}!" if flash else "")
        )
        fig.canvas.draw()
        fig.canvas.flush_events()

        # --- Capture frame for MP4 (must .copy() — buffer_rgba is a memoryview
        #     that gets overwritten on the next draw call) ---
        if save_mp4:
            frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
            frames.append(frame)

        plt.pause(delay)

        if flash:
            if save_mp4:
                # Hold the capture frame for ~1 s in the video
                for _ in range(max(1, int(1.0 / delay))):
                    frames.append(frames[-1])
            else:
                plt.pause(1.0)
            break
    else:
        title.set_text(f"Timeout after {max_steps} steps — hider escaped")
        fig.canvas.draw()
        if save_mp4:
            frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
            frames.append(frame)
        else:
            plt.pause(2.0)

    # ----------------------------------------------------------------
    # Save MP4 after episode ends
    # ----------------------------------------------------------------
    if save_mp4:
        import cv2
        p = Path(save_mp4)
        out_path = p.with_name(f"{p.stem}_{run_id}{p.suffix}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fps = max(1, int(1.0 / delay))
        h, w = frames[0].shape[:2]
        writer = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (w, h),
        )
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
        print(f"MP4 saved: {out_path}  ({len(frames)} frames, {fps} fps)")

    plt.ioff()
    plt.show()

    print(f"Episode finished: {'CAPTURED' if catcher else 'TIMEOUT'} at step {step}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualise greedy pursuer on real maze")
    parser.add_argument("--n_seekers",  type=int,   default=1)
    parser.add_argument("--grid_size",  type=int,   default=12)
    parser.add_argument("--max_steps",  type=int,   default=150)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--delay",      type=float, default=0.12)
    parser.add_argument("--obs_radius", type=int,   default=None)
    parser.add_argument("--save_mp4",   type=str,   default=None,
                        metavar="PATH", help="save episode as GIF to this path")
    args = parser.parse_args()

    run(
        n_seekers=args.n_seekers,
        grid_size=args.grid_size,
        max_steps=args.max_steps,
        seed=args.seed,
        delay=args.delay,
        obs_radius=args.obs_radius,
        save_mp4=args.save_mp4,
    )
