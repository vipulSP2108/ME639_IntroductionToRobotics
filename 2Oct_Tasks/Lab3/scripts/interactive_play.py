#!/usr/bin/env python3
"""
scripts/interactive_play.py
=============================================================================
Interactive MuJoCo 3D Playground for ME639 Lab 3
=============================================================================
Allows you to interact with the HEAL arm, table, cube, and gripper in real-time:
  - Run Pick and Place using any IK solver (Mink, DLS, QP)
  - Randomize cube position on the table in real-time (modifies scene in-place)
  - Double-click any body in the viewer and hold Ctrl + Right-Click to apply forces!
  - Watch the arm move at smooth, human-watchable robotic speed (50 FPS)

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
import mujoco.viewer

# Path resolution
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.kinematics import get_arm_joint_and_dof_indices

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


def print_banner():
    print("\n" + "═" * 72)
    print("       🤖 ME639 LAB 3: INTERACTIVE MUJOCO 3D PLAYGROUND")
    print("═" * 72)
    print("Welcome! The 3D MuJoCo Viewer is now active on your screen.")
    print("\n🎮 MOUSE CONTROLS IN THE 3D VIEWER WINDOW:")
    print("  • Left-click + drag:       Rotate camera view")
    print("  • Right-click + drag:      Zoom in / out")
    print("  • Middle-click + drag:     Pan camera")
    print("  • Double-click object:     Select body (e.g. click the red cube)")
    print("  • Ctrl + Right-drag:       Apply physical mouse forces to selected body!")
    print("  • Spacebar:                Pause / unpause physics simulation")
    print("  • Backspace:               Reset camera view")
    print("  • Tab:                     Toggle visual HUD & joint stats")
    print("═" * 72 + "\n")


def reset_simulation(model, data, cube_x=0.38, cube_y=0.0, cube_yaw=0.0):
    """
    Resets the arm posture and cube pose directly inside the EXISTING data
    object so the MuJoCo viewer remains 100% bound and synchronized!
    """
    arm_joints, _ = get_arm_joint_and_dof_indices(model)
    
    # 1. Reset all state to defaults
    data.qpos[:] = model.qpos0[:]
    data.qvel[:] = 0.0
    data.qacc[:] = 0.0
    data.ctrl[:] = 0.0

    # 2. Reset arm joints to neutral ready configuration
    data.qpos[arm_joints] = 0.0

    # 3. Position the cube at the target coordinates
    cube_jnt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free_joint")
    if cube_jnt >= 0:
        qadr = model.jnt_qposadr[cube_jnt]
        dadr = model.jnt_dofadr[cube_jnt]
        half_yaw = cube_yaw / 2.0
        data.qpos[qadr : qadr + 3] = [cube_x, cube_y, CUBE_Z]
        data.qpos[qadr + 3 : qadr + 7] = [np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw)]
        data.qvel[dadr : dadr + 6] = 0.0

    # 4. Open gripper fully
    gripper_act = 6
    if gripper_act < model.nu:
        data.ctrl[gripper_act] = 0.0

    mujoco.mj_forward(model, data)


def run_interactive_session(default_solver="qp", continuous=False, step_delay=0.02):
    print_banner()

    solvers = {
        "1": ("mink", mink_ik_solver, "Mink Differential IK (Off-the-shelf baseline)"),
        "2": ("dls",  dls_ik_solver,  "Custom Closed-Loop DLS-IK (λ = 0.01)"),
        "3": ("qp",   qp_ik_solver,   "Constrained QP-IK (Hard joint limits)"),
    }

    # Initial cube spawn at center of valid region
    cube_x = 0.38
    cube_y = 0.00
    cube_yaw = 0.00

    print(f"Creating 3D Scene: Spawning red cube on table at X={cube_x:.3f} m, Y={cube_y:.3f} m")
    model, data = build_scene(cube_x=cube_x, cube_y=cube_y, cube_yaw=cube_yaw)
    tray_pos = [TRAY_X, TRAY_Y, TRAY_Z]

    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("✓ MuJoCo 3D Window is open! Bring the viewer window into view.")
        
        # Settle scene and let the viewer stabilize
        for _ in range(60):
            viewer.sync()
            time.sleep(0.01)

        episode = 1
        current_solver_key = "3" if default_solver == "qp" else ("2" if default_solver == "dls" else "1")

        if not continuous:
            print("\n" + "─" * 65)
            print("👉 Look at the MuJoCo 3D window on your screen.")
            print("👉 Press [ENTER] in this terminal when ready to watch the robot move!")
            print("─" * 65)
            try:
                input()
            except (EOFError, KeyboardInterrupt):
                return

        while viewer.is_running():
            name, solver_fn, desc = solvers[current_solver_key]
            print(f"\n▶ [Episode {episode}] Executing pick-and-place with {name.upper()}:")
            print(f"  Method:          {desc}")
            print(f"  Cube Pos:        X={cube_x:.3f} m, Y={cube_y:.3f} m, Yaw={cube_yaw:.2f} rad")
            print(f"  Playback Speed:  ~50 FPS (watch the robot in the 3D window)")

            cube_pos = [cube_x, cube_y, CUBE_Z]
            
            # Execute the solver with realistic ~50 FPS step delay
            log = solver_fn(model, data, cube_pos, tray_pos, viewer=viewer, step_delay=step_delay)

            status = "✓ SUCCESS" if log["success"] else f"✗ FAILED ({log.get('failure_reason', 'error')})"
            total_steps = sum(log['ik_iterations'])
            print(f"  Result:          {status}")
            print(f"  Total Duration:  {log['time_to_solve']:.2f} seconds ({total_steps} visual steps)")
            print(f"  Cube Final:      In tray at distance {log['dist_to_tray']*1000:.1f} mm")

            if continuous:
                print("\n  [Continuous Mode] Pausing 2.5 seconds to show placed cube, then next trial...")
                for _ in range(125):
                    if not viewer.is_running():
                        break
                    viewer.sync()
                    time.sleep(0.02)
                
                # Randomize cube pose in-place
                cube_x = float(np.random.uniform(0.34, 0.44))
                cube_y = float(np.random.uniform(-0.13, 0.13))
                cube_yaw = float(np.random.uniform(-0.5, 0.5))
                reset_simulation(model, data, cube_x=cube_x, cube_y=cube_y, cube_yaw=cube_yaw)
                viewer.sync()
                episode += 1
                continue

            # Interactive menu
            print("\n" + "─" * 65)
            print("WHAT WOULD YOU LIKE TO TEST NEXT?")
            print("  [1] Run with MINK IK (Baseline)")
            print("  [2] Run with DLS-IK (Closed-Loop, λ=0.01)")
            print("  [3] Run with QP-IK (Constrained with Joint Limits)")
            print("  [R] Randomize cube location and watch robot re-plan")
            print("  [C] Switch to Continuous Hands-Free Mode")
            print("  [Q] Quit Playground")
            print("─" * 65)

            choice = None
            try:
                choice = input("Enter choice [1/2/3/R/C/Q] (default: keep current): ").strip().upper()
            except (EOFError, KeyboardInterrupt):
                break

            if choice == "Q":
                print("Exiting playground. Goodbye!")
                break
            elif choice in ["1", "2", "3"]:
                current_solver_key = choice
            elif choice == "R":
                cube_x = float(np.random.uniform(0.34, 0.44))
                cube_y = float(np.random.uniform(-0.13, 0.13))
                cube_yaw = float(np.random.uniform(-0.5, 0.5))
                print(f"\n🎲 New random cube position: X={cube_x:.3f} m, Y={cube_y:.3f} m")
            elif choice == "C":
                continuous = True
            elif choice == "":
                pass

            # Reset the arm and cube IN-PLACE in the same data object!
            reset_simulation(model, data, cube_x=cube_x, cube_y=cube_y, cube_yaw=cube_yaw)
            for _ in range(20):
                viewer.sync()
                time.sleep(0.01)
            episode += 1


def main():
    parser = argparse.ArgumentParser(description="Interactive MuJoCo 3D Playground")
    parser.add_argument("--solver", type=str, default="qp", choices=["mink", "dls", "qp"], help="Default IK solver")
    parser.add_argument("--continuous", action="store_true", help="Auto-loop episodes continuously without prompt")
    parser.add_argument("--speed", type=float, default=0.02, help="Sleep delay per step in seconds (default: 0.02 = 50 FPS)")
    args = parser.parse_args()

    run_interactive_session(default_solver=args.solver, continuous=args.continuous, step_delay=args.speed)


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
