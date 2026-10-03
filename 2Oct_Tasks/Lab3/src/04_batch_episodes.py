#!/usr/bin/env python3
"""
Stage 4: Batch Episodes and Data Collection
===========================================
Runs N >= 25 randomized pick-and-place episodes automatically using fixed seeds (0..24)
for reproducible statistical evaluation across all three IK solvers (Mink, DLS, QP).

Logs per episode:
- seed, cube initial pose (x, y, yaw)
- success (bool)
- failure_reason (null or string)
- ik_iterations per phase
- time_to_solve (seconds)
- min_clearance to obstacles (meters)
- final cube position and distance to tray center

Saves results to: logs/episode_logs_mink.json
"""

import os
import sys
import time
import json
import argparse
import importlib
from collections import Counter
import numpy as np
import mujoco

# Ensure project root is in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

scene_mod = importlib.import_module("src.02_scene_setup")
build_scene = scene_mod.build_scene
TABLE_H = scene_mod.TABLE_H
CUBE_Z = scene_mod.CUBE_Z
TRAY_X = scene_mod.TRAY_X
TRAY_Y = scene_mod.TRAY_Y
TRAY_Z = scene_mod.TRAY_Z

mink_mod = importlib.import_module("src.03_pick_place_mink")
mink_ik_solver = mink_mod.mink_ik_solver

# ---------------------------------------------------------------------------
# Evaluation Protocol Constants (Shared across Mink, DLS, QP)
# ---------------------------------------------------------------------------
DEFAULT_NUM_EPISODES = 25
EVAL_SEEDS = list(range(DEFAULT_NUM_EPISODES))  # seeds 0..24

# Feasible cube spawn bounds on table surface (derived in Stage 1)
CUBE_X_RANGE = (0.33, 0.44)
CUBE_Y_RANGE = (-0.14, 0.14)


def sample_cube_pose(seed, x_range=CUBE_X_RANGE, y_range=CUBE_Y_RANGE):
    """
    Deterministically samples a cube pose on the table surface given an integer seed.
    """
    rng = np.random.default_rng(seed)
    x = float(rng.uniform(*x_range))
    y = float(rng.uniform(*y_range))
    yaw = float(rng.uniform(-np.pi, np.pi))
    return x, y, yaw


def run_batch_evaluation(solver_fn, solver_name="mink", seeds=EVAL_SEEDS, log_filename=None):
    """
    Runs a batch of randomized episodes using the specified IK solver function.
    
    Args:
        solver_fn: callable with signature (model, data, cube_pos, tray_pos) -> log_dict
        solver_name: identifier string for console/logging ("mink", "dls", "qp")
        seeds: list of integer seeds
        log_filename: path to output JSON file
        
    Returns:
        all_logs: list of per-episode log dicts
    """
    if log_filename is None:
        log_filename = os.path.join(PROJECT_ROOT, "logs", f"episode_logs_{solver_name}.json")
        
    os.makedirs(os.path.dirname(log_filename), exist_ok=True)
    
    N = len(seeds)
    print("=" * 65)
    print(f"BATCH EVALUATION: {solver_name.upper()} IK ({N} Episodes)")
    print("=" * 65)
    print(f"Spawn Bounds: X ∈ [{CUBE_X_RANGE[0]:.2f}, {CUBE_X_RANGE[1]:.2f}] m, "
          f"Y ∈ [{CUBE_Y_RANGE[0]:.2f}, {CUBE_Y_RANGE[1]:.2f}] m, Yaw ∈ [-π, +π]")
    print(f"Seeds: {seeds[0]}..{seeds[-1]}")
    print("-" * 65)

    all_logs = []
    t_batch_start = time.time()

    for idx, seed in enumerate(seeds):
        x, y, yaw = sample_cube_pose(seed)
        model, data = build_scene(cube_x=x, cube_y=y, cube_yaw=yaw)
        cube_pos = [x, y, CUBE_Z]
        tray_pos = [TRAY_X, TRAY_Y, TRAY_Z]

        # Execute episode with solver
        t_ep_start = time.time()
        ep_log = solver_fn(model, data, cube_pos, tray_pos, viewer=None)
        ep_time = time.time() - t_ep_start

        # Record metadata
        ep_log["episode_index"] = idx
        ep_log["seed"] = seed
        ep_log["cube_x"] = x
        ep_log["cube_y"] = y
        ep_log["cube_yaw"] = yaw
        ep_log["method"] = solver_name
        ep_log["time_total"] = ep_time
        all_logs.append(ep_log)

        # Print per-episode progress
        status_str = "✓ SUCCESS" if ep_log["success"] else f"✗ FAIL ({ep_log['failure_reason']})"
        iter_total = sum(ep_log["ik_iterations"]) if ep_log.get("ik_iterations") else 0
        print(f"[{idx+1:02d}/{N:02d}] Seed {seed:02d} | Cube: ({x:.3f}, {y:+.3f}) | {status_str:<26} | "
              f"Iters: {iter_total:3d} | Time: {ep_log['time_to_solve']:.3f}s")

    # Save to JSON
    with open(log_filename, "w") as f:
        json.dump(all_logs, f, indent=2)

    total_duration = time.time() - t_batch_start
    success_count = sum(1 for l in all_logs if l["success"])
    success_rate = 100.0 * success_count / N
    mean_solve_time = float(np.mean([l["time_to_solve"] for l in all_logs]))
    mean_iters = float(np.mean([sum(l["ik_iterations"]) for l in all_logs if l.get("ik_iterations")]))
    mean_clearance = float(np.mean([l["min_clearance"] for l in all_logs]))
    failures = [l["failure_reason"] for l in all_logs if not l["success"]]
    fail_summary = Counter(failures)

    print("-" * 65)
    print(f"=== BATCH SUMMARY: {solver_name.upper()} ===")
    print(f"Total Episodes:       {N}")
    print(f"Success Rate:         {success_count}/{N} ({success_rate:.1f}%)")
    print(f"Mean Time to Solve:   {mean_solve_time:.4f} s")
    print(f"Mean IK Iterations:   {mean_iters:.1f}")
    print(f"Mean Min Clearance:   {mean_clearance:.4f} m")
    print(f"Total Batch Runtime:  {total_duration:.2f} s")
    if failures:
        print(f"Failure Breakdown:    {dict(fail_summary)}")
    else:
        print("Failure Breakdown:    Zero failures recorded!")
    print(f"Saved logs to:        {log_filename}")
    print("=" * 65 + "\n")

    return all_logs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 4: Batch Episode Evaluation")
    parser.add_argument("--episodes", type=int, default=DEFAULT_NUM_EPISODES, help="Number of episodes")
    args = parser.parse_args()

    seeds = list(range(args.episodes))
    run_batch_evaluation(solver_fn=mink_ik_solver, solver_name="mink", seeds=seeds)
