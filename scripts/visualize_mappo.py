"""
MAPPO policy visualisation — runs a live episode using a trained shared actor.

Loads the saved MAPPO actor from checkpoints/mappo_seed{seed}/ep{checkpoint}/actor.pt
and drives all seekers (parameter-sharing, decentralised execution) against a random
hider via the real env.step() (AEC API). The centralised critic is not used at
execution time.

Usage
-----
    python scripts/visualize_mappo.py
    python scripts/visualize_mappo.py --policy_seed 43 --checkpoint ep500
    python scripts/visualize_mappo.py --policy_seed 44 --n_seekers 4 --grid_size 10
    python scripts/visualize_mappo.py --save_mp4 outputs/mappo_chase.mp4

Arguments
---------
--policy_seed  int   training seed whose checkpoint to load (default 42)
--checkpoint   str   which checkpoint to load, e.g. ep500 (default ep500)
--n_seekers    int   number of seekers (default 4)
--grid_size    int   side length of the square grid (default 10)
--max_steps    int   episode step limit (default 150)
--delay        float seconds between frames (default 0.12)
--obs_radius   int   FoV half-side; omit for full observability
--env_seed     int   episode/grid seed (default: random each run)
--save_mp4     str   save episode as MP4 to this path
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch

from agents.mappo.mappo import load_mappo, _SharedActor
from env.pursuit_env import ByzantinePursuitEnv
from env.schema import OBS_DIM, SENTINEL

# ---------------------------------------------------------------------------
# Colour palette (matches visualize_greedy.py and visualize_ippo.py)
# ---------------------------------------------------------------------------
_C_WALL    = np.array([0.25, 0.25, 0.25])
_C_OPEN    = np.array([0.95, 0.95, 0.90])
_C_SEEKER  = np.array([0.20, 0.50, 0.90])   # blue
_C_HIDER   = np.array([0.90, 0.25, 0.25])   # red
_C_CAPTURE = np.array([0.20, 0.80, 0.20])   # green flash


def _build_rgb(env: ByzantinePursuitEnv, flash: bool = False) -> np.ndarray:
    gs = env.grid_size
    img = np.where(env.grid[:, :, np.newaxis], _C_WALL, _C_OPEN).astype(np.float32)
    for agent_id, (r, c) in env.positions.items():
        if agent_id == "hider":
            img[r, c] = _C_CAPTURE if flash else _C_HIDER
        else:
            img[r, c] = _C_CAPTURE if flash else _C_SEEKER
    return img


# ---------------------------------------------------------------------------
# Observation preprocessing (matches mappo.py)
# ---------------------------------------------------------------------------

def _zero_message_slots(obs: np.ndarray, n_seekers: int) -> np.ndarray:
    out = obs.copy()
    msg_start = len(obs) - 2 * (n_seekers - 1)
    out[msg_start:] = 0.0
    return out


# ---------------------------------------------------------------------------
# Policy loading — shared actor only (critic not used at execution time)
# ---------------------------------------------------------------------------

def _load_actor(policy_seed: int, checkpoint: str, obs_dim: int, n_seekers: int) -> _SharedActor:
    root = Path(f"checkpoints/mappo_seed{policy_seed}/{checkpoint}")
    if not root.exists():
        available = sorted(Path(f"checkpoints/mappo_seed{policy_seed}").iterdir()) \
            if Path(f"checkpoints/mappo_seed{policy_seed}").exists() else []
        raise FileNotFoundError(
            f"Checkpoint not found: {root}\n"
            f"Available: {available}"
        )
    actor, _ = load_mappo(str(root), obs_dim, n_seekers)
    print(f"  Loaded shared actor from {root / 'actor.pt'}")
    return actor


# ---------------------------------------------------------------------------
# Main episode loop
# ---------------------------------------------------------------------------

def run(
    policy_seed: int,
    env_seed: int,
    checkpoint: str,
    n_seekers: int,
    grid_size: int,
    max_steps: int,
    delay: float,
    obs_radius: int | None,
    save_mp4: str | None,
) -> None:
    env = ByzantinePursuitEnv(
        n_seekers=n_seekers,
        grid_size=grid_size,
        obs_radius=obs_radius,
        obstacle_density=0.15,
        byzantine_fraction=0.0,
        max_steps=max_steps,
        seed=env_seed,
    )

    seeker_ids = sorted([a for a in env.possible_agents if a.startswith("seeker_")])
    n_seekers_actual = len(seeker_ids)
    obs_dim = OBS_DIM(n_seekers_actual, grid_size, obs_radius)

    print(f"Loading MAPPO actor  policy_seed={policy_seed}  checkpoint={checkpoint}")
    actor = _load_actor(policy_seed, checkpoint, obs_dim, n_seekers_actual)

    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:6]}"
    print(f"Run ID: {run_id}  env_seed={env_seed}")

    rng = np.random.default_rng(env_seed + 9999)

    # Reset before any rendering so env.grid is populated
    from pettingzoo.utils.conversions import aec_to_parallel
    parallel_env = aec_to_parallel(env)
    obs_dict, _ = parallel_env.reset(seed=env_seed)

    # ----------------------------------------------------------------
    # Matplotlib setup
    # ----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xticks([])
    ax.set_yticks([])
    fig.patch.set_facecolor("#1a1a1a")
    ax.set_facecolor("#1a1a1a")

    im = ax.imshow(_build_rgb(env), interpolation="nearest", vmin=0, vmax=1)

    obs_label = f"obs_radius={obs_radius}" if obs_radius else "full obs"
    legend_patches = [
        mpatches.Patch(color=_C_SEEKER, label=f"Seeker (MAPPO seed={policy_seed})"),
        mpatches.Patch(color=_C_HIDER,  label="Hider (random)"),
        mpatches.Patch(color=_C_WALL,   label="Wall / obstacle"),
    ]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=7, framealpha=0.7)
    title = ax.set_title("", color="white", fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.ion()
    plt.show()

    # ----------------------------------------------------------------
    # Episode simulation via parallel API (same as training)
    # One parallel step = all agents act simultaneously → clean render loop
    # All seekers share the same actor weights; each receives its own local obs
    # ----------------------------------------------------------------

    frames = []
    captured = False
    step = 0

    while parallel_env.agents:
        actions: dict = {}

        for sid in seeker_ids:
            if sid not in parallel_env.agents:
                continue
            obs_clean = _zero_message_slots(obs_dict[sid], n_seekers_actual)
            obs_t = torch.tensor(obs_clean[np.newaxis], dtype=torch.float32)
            with torch.no_grad():
                action_t, _, _ = actor.get_action_and_logprob(obs_t)
            actions[sid] = int(action_t.cpu().item())

        if "hider" in parallel_env.agents:
            actions["hider"] = int(rng.integers(0, 5))

        obs_dict, _, term_dict, trunc_dict, _ = parallel_env.step(actions)
        step += 1

        # Capture: any seeker terminates (not truncates)
        flash = any(term_dict.get(s, False) for s in seeker_ids)
        if flash:
            captured = True

        # Render
        im.set_data(_build_rgb(env, flash=flash))
        title.set_text(
            f"MAPPO  policy={policy_seed}  env={env_seed}  {checkpoint}  |  Step {step}/{max_steps}  |  {obs_label}"
            + ("\n  CAPTURED!" if flash else "")
        )
        fig.canvas.draw()
        fig.canvas.flush_events()

        if save_mp4:
            frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
            frames.append(frame)

        plt.pause(delay)

        if captured:
            if save_mp4:
                for _ in range(max(1, int(1.0 / delay))):
                    frames.append(frames[-1])
            else:
                plt.pause(1.0)
            break

    if not captured:
        title.set_text(f"MAPPO  policy={policy_seed}  env={env_seed}  {checkpoint}  |  Timeout after {max_steps} steps")
        fig.canvas.draw()
        if save_mp4:
            frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
            frames.append(frame)
        else:
            plt.pause(2.0)

    # ----------------------------------------------------------------
    # Save MP4
    # ----------------------------------------------------------------
    if save_mp4 and frames:
        import cv2
        p = Path(save_mp4)
        out_dir = p.parent / "mappo"
        out_path = out_dir / f"mappo_{p.stem}_{run_id}{p.suffix}"
        out_dir.mkdir(parents=True, exist_ok=True)
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
    print(f"Episode finished: {'CAPTURED' if captured else 'TIMEOUT'} at step {step}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time as _time
    parser = argparse.ArgumentParser(description="Visualise trained MAPPO shared actor")
    parser.add_argument("--policy_seed", type=int,   default=42,
                        help="which checkpoint to load (42, 43, or 44)")
    parser.add_argument("--env_seed",    type=int,   default=None,
                        help="episode/grid seed (default: random each run)")
    parser.add_argument("--checkpoint",  type=str,   default="ep500",
                        help="checkpoint epoch folder, e.g. ep500 (saved every 50 eps)")
    parser.add_argument("--n_seekers",   type=int,   default=4)
    parser.add_argument("--grid_size",   type=int,   default=10)
    parser.add_argument("--max_steps",   type=int,   default=150)
    parser.add_argument("--delay",       type=float, default=0.12)
    parser.add_argument("--obs_radius",  type=int,   default=None)
    parser.add_argument("--save_mp4",    type=str,   default=None, metavar="PATH")
    args = parser.parse_args()

    env_seed = args.env_seed if args.env_seed is not None else int(_time.time()) % 100000

    run(
        policy_seed=args.policy_seed,
        env_seed=env_seed,
        checkpoint=args.checkpoint,
        n_seekers=args.n_seekers,
        grid_size=args.grid_size,
        max_steps=args.max_steps,
        delay=args.delay,
        obs_radius=args.obs_radius,
        save_mp4=args.save_mp4,
    )
