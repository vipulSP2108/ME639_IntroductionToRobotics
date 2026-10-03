#!/usr/bin/env python3
"""
scripts/record_demo.py
=============================================================================
Interactive Demonstration and Recording Runner for Video & Screenshots
=============================================================================
Provides a unified, automated recording suite for:
  - Mink IK (Off-the-shelf baseline)
  - DLS-IK (Custom Closed-Loop Damped Least Squares)
  - QP-IK (Constrained Quadratic Programming with joint limits)

Usage:
  # 1. Record 1-minute video segment for Mink IK (3 consecutive trials):
  python3 scripts/record_demo.py --method mink --trials 3

  # 2. Record 1-minute video segment for DLS-IK (3 consecutive trials):
  python3 scripts/record_demo.py --method dls --trials 3

  # 3. Record 1-minute video segment for QP-IK (3 consecutive trials):
  python3 scripts/record_demo.py --method qp --trials 3

  # 4. Interactive Screenshot Mode: Pauses at specific phase so you can
  #    rotate/zoom the camera and take a clean screenshot:
  python3 scripts/record_demo.py --method qp --pause_phase grip
  python3 scripts/record_demo.py --method qp --pause_phase transit

  # 5. Headless test:
  python3 scripts/record_demo.py --headless --method all --trials 1
"""

import os
import sys
import shutil
import time
import argparse
import importlib
import numpy as np

import mujoco
import mujoco.viewer



# Path resolution
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import importlib

scene_mod = importlib.import_module("src.02_scene_setup")
build_scene = scene_mod.build_scene
TABLE_H = scene_mod.TABLE_H
CUBE_Z = scene_mod.CUBE_Z
TRAY_X = scene_mod.TRAY_X
TRAY_Y = scene_mod.TRAY_Y
TRAY_Z = scene_mod.TRAY_Z

mink_mod = importlib.import_module("src.03_pick_place_mink")
mink_ik_solver = mink_mod.mink_ik_solver

dls_mod = importlib.import_module("src.05_dls_ik")
dls_ik_solver = dls_mod.dls_ik_solver

qp_mod = importlib.import_module("src.06_qp_ik")
qp_ik_solver = qp_mod.qp_ik_solver


def run_trials_for_method(
    method_name,
    trials=3,
    start_seed=100,
    headless=False,
    delay_between_trials=2.0,
    pause_phase=None,
    step_delay=0.02,
):
    print("\n" + "=" * 70)
    print(f"🎬 PREPARING DEMO RECORDING: {method_name.upper()} IK ({trials} Trials)")
    print(f"   Headless: {headless} | Step pacing: {step_delay * 1000:.1f} ms | Pause phase: {pause_phase}")
    print("=" * 70)

    solver_map = {
        "mink": (mink_ik_solver, "Mink Differential IK (Off-the-shelf baseline)"),
        "dls": (dls_ik_solver, "Closed-Loop DLS-IK (λ = 0.01)"),
        "qp": (qp_ik_solver, "Constrained QP-IK (DAQP Solver with joint limits)"),
    }

    if method_name not in solver_map:
        raise ValueError(f"Unknown method '{method_name}'. Choose from: mink, dls, qp")

    solver_fn, desc = solver_map[method_name]
    print(f"Method Description: {desc}")
    print("Camera Tip: Place the camera at an isometric side-view to clearly capture")
    print("            the robot base, table surface, cube grasp, and tray drop-off.\n")

    logs = []

    for t in range(trials):
        seed = start_seed + t
        np.random.seed(seed)
        
        # Sample randomized cube pose within safe table bounds
        cube_x = float(np.random.uniform(0.34, 0.44))
        cube_y = float(np.random.uniform(-0.13, 0.13))
        cube_yaw = float(np.random.uniform(-0.5, 0.5))

        print(f"\n▶ [{method_name.upper()}] Starting Trial {t + 1}/{trials} (Seed {seed})")
        print(f"  Cube Position: X={cube_x:.3f} m, Y={cube_y:.3f} m, Yaw={cube_yaw:.2f} rad")

        model, data = build_scene(cube_x=cube_x, cube_y=cube_y, cube_yaw=cube_yaw)
        cube_pos = [cube_x, cube_y, CUBE_Z]
        tray_pos = [TRAY_X, TRAY_Y, TRAY_Z]

        if not headless:
            try:
                import mujoco.viewer
                with mujoco.viewer.launch_passive(model, data) as viewer:
                    # Let the viewer stabilize and render initial scene
                    for _ in range(30):
                        viewer.sync()
                        time.sleep(0.01)

                    print(f"  Executing {method_name.upper()} pick-and-place trajectory...")
                    log = solver_fn(model, data, cube_pos, tray_pos, viewer=viewer, step_delay=step_delay)

                    # Pause at end of episode so the successful placement is clearly visible on video
                    status = "✓ SUCCESS" if log["success"] else f"✗ FAILED ({log['failure_reason']})"
                    print(f"  Trial {t + 1} Finished: {status} | Duration: {log['time_to_solve']:.2f}s")
                    
                    for _ in range(int(delay_between_trials * 50)):
                        viewer.sync()
                        time.sleep(0.02)
            except Exception as e:
                print(f"  [Viewer Error: {e}] Running headless fallback...")
                log = solver_fn(model, data, cube_pos, tray_pos, viewer=None)
        else:
            log = solver_fn(model, data, cube_pos, tray_pos, viewer=None)
            status = "✓ SUCCESS" if log["success"] else f"✗ FAILED ({log['failure_reason']})"
            print(f"  Trial {t + 1} Finished: {status} | Duration: {log['time_to_solve']:.2f}s")

        logs.append(log)

    success_count = sum(1 for l in logs if l["success"])
    print(f"\n🏁 Finished {trials} trials for {method_name.upper()}: {success_count}/{trials} Successful ({100.0 * success_count / trials:.1f}%)\n")
    return logs


def main():
    parser = argparse.ArgumentParser(description="Demonstration and Screen-Recording Runner")
    parser.add_argument(
        "--method",
        type=str,
        default="all",
        choices=["mink", "dls", "qp", "all"],
        help="Which IK method to run ('mink', 'dls', 'qp', or 'all')",
    )
    parser.add_argument("--trials", type=int, default=3, help="Number of trials per method (default: 3)")
    parser.add_argument("--headless", action="store_true", help="Run without opening GUI viewer window")
    parser.add_argument("--seed", type=int, default=100, help="Starting random seed")
    parser.add_argument("--delay", type=float, default=1.5, help="Pause seconds between trials")
    args = parser.parse_args()

    methods_to_run = ["mink", "dls", "qp"] if args.method == "all" else [args.method]

    print("\n" + "═" * 70)
    print("       ME639 LAB 3: DEMONSTRATION & VIDEO RECORDING RUNNER")
    print("═" * 70)
    print("Recording Tips for the 3-Minute Video:")
    print("  • Minute 1: Run with --method mink (3 trials, shows off-the-shelf baseline)")
    print("  • Minute 2: Run with --method dls  (3 trials, shows closed-loop DLS-IK)")
    print("  • Minute 3: Run with --method qp   (3 trials, shows QP with joint limits)")
    print("Controls in MuJoCo viewer:")
    print("  • Left mouse drag: Rotate camera view")
    print("  • Right mouse drag: Zoom camera")
    print("  • Middle mouse drag: Pan camera")
    print("  • Spacebar: Pause / resume simulation")
    print("═" * 70)

    for m in methods_to_run:
        run_trials_for_method(
            method_name=m,
            trials=args.trials,
            start_seed=args.seed,
            headless=args.headless,
            delay_between_trials=args.delay,
        )


if __name__ == "__main__":
    # On macOS, MuJoCo's GUI viewer requires running under mjpython
    if (
        sys.platform == "darwin"
        and not isinstance(getattr(mujoco.viewer, "_MJPYTHON", None), getattr(mujoco.viewer, "_MjPythonBase", object))
        and "--headless" not in sys.argv
        and "-h" not in sys.argv
        and "--help" not in sys.argv
    ):
        mjpython_bin = shutil.which("mjpython") or "/Library/Frameworks/Python.framework/Versions/3.13/bin/mjpython"
        if os.path.exists(mjpython_bin):
            os.execv(mjpython_bin, [mjpython_bin] + sys.argv)
    main()
