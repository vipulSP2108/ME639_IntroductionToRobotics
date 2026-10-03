#!/usr/bin/env python3
"""
scripts/interactive_play.py
=============================================================================
Interactive MuJoCo 3D Playground for ME639 Lab 3
=============================================================================
Features:
  - Silky-Smooth Continuous Pick-and-Place:
    Uses quintic minimum-jerk spline trajectory interpolation at 50 FPS
    for fluid robotic translation (no discrete jumps, zero lag).
  - Reachable Workspace Sphere & Point Cloud (Key [W]):
    Renders the outer reachable sphere (R = 0.72 m) and inner dead-zone
    sphere (R = 0.18 m) in 3D directly in the MuJoCo viewer window.
  - Multi-Point Workspace Boundary Tour (Key [T]):
    Sweeps the robot through 13 key boundary points across the entire
    reachable envelope with smooth minimum-jerk transitions.
  - Full-Table Domain Randomization (Key [R]):
    Spans the entire table surface (X in [0.29, 0.61], Y in [-0.21, 0.21]),
    rejecting tray and kinematic dead zones.
  - Interactive IK Solver Switching:
    Compare MINK, DLS, and QP in real-time.

Usage:
  mjpython scripts/interactive_play.py
  (or: python3 scripts/interactive_play.py)
"""

import os
import sys
import shutil
import time
import argparse
import importlib
import numpy as np

import mujoco
try:
    import mujoco.viewer
    HAS_VIEWER = True
except (ImportError, AttributeError):
    HAS_VIEWER = False

# Path resolution
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ.setdefault("MPLCONFIGDIR", os.path.join(PROJECT_ROOT, "temp", ".mpl"))
os.makedirs(os.path.join(PROJECT_ROOT, "temp", ".mpl"), exist_ok=True)

from utils.kinematics import get_arm_joint_and_dof_indices
from utils.trajectory import sample_cube_pose_full_table
from scripts.extra_workspace_features import WorkspaceFeatureManager, WORKSPACE_TOUR_POINTS

scene_mod = importlib.import_module("src.02_scene_setup")
build_scene = scene_mod.build_scene
reset_cube_pose = scene_mod.reset_cube_pose
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


def print_banner():
    print("\n" + "═" * 74)
    print("       🤖 ME639 LAB 3: INTERACTIVE MUJOCO 3D PLAYGROUND")
    print("═" * 74)
    print("Welcome! The 3D MuJoCo Viewer is now active on your screen.")
    print("\n🎮 3D VIEWER WINDOW KEYBOARD & MOUSE SHORTCUTS:")
    print("  • Key [W]:                 Toggle Reachable Workspace Sphere & Cloud")
    print("  • Key [T]:                 Run Silky-Smooth Workspace Boundary Tour")
    print("  • Spacebar:                Pause / unpause physics simulation")
    print("  • Backspace:               Reset camera view")
    print("  • Left-click + drag:       Rotate camera view")
    print("  • Right-click + drag:      Zoom in / out")
    print("  • Middle-click + drag:     Pan camera")
    print("  • Ctrl + Right-drag:       Apply physical mouse forces to selected body!")
    print("═" * 74 + "\n")


def reset_simulation(model, data, cube_x=0.38, cube_y=0.0, cube_yaw=0.0):
    """
    Resets the arm configuration and cube pose directly inside the EXISTING data
    object so the MuJoCo viewer remains 100% bound and synchronized.
    """
    arm_joints, _ = get_arm_joint_and_dof_indices(model)

    data.qpos[:] = model.qpos0[:]
    data.qvel[:] = 0.0
    data.qacc[:] = 0.0
    data.ctrl[:] = 0.0

    # Neutral ready posture
    data.qpos[arm_joints] = 0.0

    # Position the cube
    reset_cube_pose(model, data, cube_x=cube_x, cube_y=cube_y, cube_yaw=cube_yaw)

    # Open gripper fully
    gripper_act = 6
    if gripper_act < model.nu:
        data.ctrl[gripper_act] = 0.0

    mujoco.mj_forward(model, data)


def run_interactive_session(default_solver="qp", continuous=False, step_delay=0.018):
    print_banner()

    solvers = {
        "1": ("mink", "Mink Differential IK (Off-the-shelf baseline)"),
        "2": ("dls",  "Custom Closed-Loop DLS-IK (λ = 0.01)"),
        "3": ("qp",   "Constrained QP-IK (Hard joint limits & box constraints)"),
    }

    # Initial cube spawn
    cube_x = 0.38
    cube_y = 0.00
    cube_yaw = 0.00

    print(f"Creating 3D Scene: Spawning red cube on table at X={cube_x:.3f} m, Y={cube_y:.3f} m")
    model, data = build_scene(cube_x=cube_x, cube_y=cube_y, cube_yaw=cube_yaw)
    tray_pos = [TRAY_X, TRAY_Y, TRAY_Z]

    mgr = WorkspaceFeatureManager(model, data, default_solver=default_solver)

    def viewer_key_cb(keycode):
        mgr.key_callback(keycode)

    with mujoco.viewer.launch_passive(model, data, key_callback=viewer_key_cb) as viewer:
        mgr.set_viewer(viewer)
        print("✓ MuJoCo 3D Window is open! Bring the viewer window into view.")

        # Settle scene and let the viewer stabilize
        for _ in range(50):
            viewer.sync()
            time.sleep(0.01)

        episode = 1
        current_solver_key = "3" if default_solver == "qp" else ("2" if default_solver == "dls" else "1")

        if not continuous:
            print("\n" + "─" * 68)
            print("👉 Look at the MuJoCo 3D window on your screen.")
            print("👉 Press [ENTER] in this terminal when ready to watch the robot move!")
            print("─" * 68)
            try:
                input()
            except (EOFError, KeyboardInterrupt):
                return

        while viewer.is_running():
            name, desc = solvers[current_solver_key]
            print(f"\n▶ [Episode {episode}] Executing pick-and-place with {name.upper()}:")
            print(f"  Method:          {desc}")
            print(f"  Cube Pos:        X={cube_x:.3f} m, Y={cube_y:.3f} m, Yaw={cube_yaw:.2f} rad")
            print(f"  Interpolation:   Quintic Minimum-Jerk Spline (~50 FPS smooth translation)")

            cube_pos = [cube_x, cube_y, CUBE_Z]

            # Execute silky-smooth pick-and-place
            log = mgr.run_smooth_pick_and_place(
                cube_pos, tray_pos, solver=name, step_delay=step_delay
            )

            status = "✓ SUCCESS" if log["success"] else f"✗ FAILED ({log.get('failure_reason', 'error')})"
            print(f"  Result:          {status}")
            print(f"  Total Duration:  {log['time_to_solve']:.2f} seconds ({log['total_steps']} smooth frames)")
            print(f"  Cube Final:      In tray at distance {log['dist_to_tray']*1000:.1f} mm")

            if continuous:
                print("\n  [Continuous Mode] Pausing 2.0 seconds to show placed cube, then next trial...")
                for _ in range(100):
                    if not viewer.is_running():
                        break
                    viewer.sync()
                    time.sleep(0.02)

                # Randomize cube pose across entire table
                cube_x, cube_y, cube_yaw = sample_cube_pose_full_table()
                reset_simulation(model, data, cube_x=cube_x, cube_y=cube_y, cube_yaw=cube_yaw)
                viewer.sync()
                episode += 1
                continue

            # Interactive menu
            print("\n" + "─" * 68)
            print("WHAT WOULD YOU LIKE TO TEST NEXT?")
            print("  [ENTER] / [P] Run Silky-Smooth Pick & Place (with current solver)")
            print("  [1]           Select MINK IK (Baseline)")
            print("  [2]           Select DLS-IK (Closed-Loop, λ=0.01)")
            print("  [3]           Select QP-IK (Constrained with Joint Limits)")
            print("  [W]           Toggle Reachable Workspace Sphere & Cloud in 3D Window")
            print("  [T]           Execute Silky-Smooth Workspace Boundary Tour")
            print("  [R]           Randomize Cube Across ENTIRE Table Surface")
            print("  [C]           Switch to Continuous Hands-Free Mode")
            print("  [Q]           Quit Playground")
            print("─" * 68)

            choice = None
            try:
                choice = input(f"Enter choice [P/1/2/3/W/T/R/C/Q] (default: Run {name.upper()}): ").strip().upper()
            except (EOFError, KeyboardInterrupt):
                break

            if choice == "Q":
                print("Exiting playground. Goodbye!")
                break
            elif choice in ["1", "2", "3"]:
                current_solver_key = choice
                print(f"Selected solver: {solvers[current_solver_key][0].upper()}")
            elif choice == "W":
                mgr.toggle_workspace_sphere()
                continue
            elif choice == "T":
                mgr.run_smooth_tour(solver=name, step_delay=step_delay)
                continue
            elif choice == "R":
                cube_x, cube_y, cube_yaw = sample_cube_pose_full_table()
                print(f"\n🎲 New random cube position across table: X={cube_x:.3f} m, Y={cube_y:.3f} m, Yaw={cube_yaw:.2f} rad")
            elif choice == "C":
                continuous = True
            elif choice in ["", "P"]:
                pass

            # Reset arm and cube for next run
            reset_simulation(model, data, cube_x=cube_x, cube_y=cube_y, cube_yaw=cube_yaw)
            for _ in range(25):
                viewer.sync()
                time.sleep(0.01)
            episode += 1


def main():
    parser = argparse.ArgumentParser(description="Interactive MuJoCo 3D Playground")
    parser.add_argument("--solver", type=str, default="qp", choices=["mink", "dls", "qp"], help="Default IK solver")
    parser.add_argument("--continuous", action="store_true", help="Auto-loop episodes continuously without prompt")
    parser.add_argument("--speed", type=float, default=0.018, help="Step delay in seconds (default: 0.018 ~ 55 FPS)")
    args = parser.parse_args()

    run_interactive_session(default_solver=args.solver, continuous=args.continuous, step_delay=args.speed)


if __name__ == "__main__":
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
