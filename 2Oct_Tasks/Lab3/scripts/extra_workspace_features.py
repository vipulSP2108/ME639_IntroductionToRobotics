#!/usr/bin/env python3
"""
scripts/extra_workspace_features.py
=============================================================================
EXTRA WORKSPACE FEATURES: Reachable Workspace Sphere & Smooth Boundary Tour
=============================================================================
This module provides advanced kinematic exploration and visualization:
  1. 3D Reachable Workspace Sphere & Point Cloud:
     - Renders the outer reachable kinematic sphere (R = 0.72 m) and inner
       dead-zone sphere (R = 0.18 m) directly in the MuJoCo viewer using
       translucent geometric primitives (mjv_initGeom).
     - Renders 3D boundary marker spheres across the sphere envelope.
     - Toggleable via Key [W] in the viewer or [W] in the terminal menu.

  2. Silky-Smooth Multi-Point Trajectory Tour:
     - Commands the robot end-effector to tour key boundary points across
       the entire reachable sphere (Zenith high reach, forward limit,
       lateral extremes, table perimeter corners, and ready home).
     - Uses quintic minimum-jerk trajectory interpolation for continuous,
       glitch-free, silky-smooth motion transitions.
     - Triggered via Key [T] in the viewer or [T] in the terminal menu.

  3. Full-Table Domain Randomization:
     - Randomizes the cube position across the entire table (X in [0.29, 0.61],
       Y in [-0.21, +0.21]), rejecting the tray and kinematic dead zones.
     - Triggered via Key [R].

Usage:
  mjpython scripts/extra_workspace_features.py
"""

import os
import sys
import time
import shutil
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

from utils.kinematics import forward_kinematics, get_arm_joint_and_dof_indices
from utils.quaternion_utils import DOWNWARD_QUAT
from utils.trajectory import (
    execute_smooth_cartesian_trajectory,
    sample_cube_pose_full_table,
    TCP_OFFSET_Z,
    TABLE_H,
    CUBE_HALF_SIZE,
)

scene_mod = importlib.import_module("src.02_scene_setup")
build_scene = scene_mod.build_scene
reset_cube_pose = scene_mod.reset_cube_pose
CUBE_Z = scene_mod.CUBE_Z
TRAY_X = scene_mod.TRAY_X
TRAY_Y = scene_mod.TRAY_Y
TRAY_Z = scene_mod.TRAY_Z

# ---------------------------------------------------------------------------
# 13 Representative Workspace Boundary Points Spanning the Entire Sphere
# ---------------------------------------------------------------------------
# Site 'right_center' target positions (TCP pad center = site_pos - [0, 0, 0.123])
WORKSPACE_TOUR_POINTS = [
    ("Ready Home Posture",     [0.35,  0.00, 0.47],  DOWNWARD_QUAT, 1.2),
    ("Zenith (High Apex)",      [0.20,  0.00, 0.62],  DOWNWARD_QUAT, 1.2),
    ("High Forward Reach",      [0.48,  0.00, 0.52],  DOWNWARD_QUAT, 1.0),
    ("Far Forward Limit",       [0.60,  0.00, 0.36],  DOWNWARD_QUAT, 1.2),
    ("Far Right Outer Reach",   [0.45, -0.28, 0.38],  DOWNWARD_QUAT, 1.2),
    ("Table Far-Right Corner",  [0.52, -0.19, 0.355], DOWNWARD_QUAT, 1.0),
    ("Table Near-Right Edge",   [0.32, -0.18, 0.355], DOWNWARD_QUAT, 1.0),
    ("Table Center Pick Area",  [0.38,  0.00, 0.355], DOWNWARD_QUAT, 1.0),
    ("Table Near-Left Edge",    [0.32,  0.18, 0.355], DOWNWARD_QUAT, 1.0),
    ("Table Far-Left Corner",   [0.52,  0.19, 0.355], DOWNWARD_QUAT, 1.0),
    ("Far Left Outer Reach",    [0.45,  0.28, 0.38],  DOWNWARD_QUAT, 1.2),
    ("Tray Drop Target",        [0.55,  0.00, 0.36],  DOWNWARD_QUAT, 1.2),
    ("Return to Ready Home",    [0.35,  0.00, 0.47],  DOWNWARD_QUAT, 1.2),
]


class WorkspaceFeatureManager:
    """
    Manages the 3D workspace sphere rendering and trajectory tour execution.
    """
    def __init__(self, model, data, viewer=None, default_solver="qp"):
        self.model = model
        self.data = data
        self.viewer = viewer
        self.default_solver = default_solver
        self.sphere_visible = False
        self.is_busy = False

        # Precompute perimeter cloud points on the outer sphere
        self.cloud_points = []
        r = 0.72
        center = np.array([0.0, 0.0, 0.15])
        # Equatorial and meridian rings
        for az in np.linspace(-np.pi / 2, np.pi / 2, 16):
            for el in [0.2, 0.5, 0.8]:
                p = center + np.array([
                    r * np.cos(el) * np.cos(az),
                    r * np.cos(el) * np.sin(az),
                    r * np.sin(el)
                ])
                if p[0] >= 0.15 and p[2] >= 0.18:
                    self.cloud_points.append(p)

    def set_viewer(self, viewer):
        self.viewer = viewer

    def toggle_workspace_sphere(self, force_state=None):
        """
        Toggles the 3D reachable workspace sphere and boundary markers in the viewer.
        """
        if self.viewer is None:
            return False

        if force_state is not None:
            self.sphere_visible = force_state
        else:
            self.sphere_visible = not self.sphere_visible

        with self.viewer.lock():
            user_scn = self.viewer.user_scn
            if not self.sphere_visible:
                user_scn.ngeom = 0
            else:
                mat = np.eye(3, dtype=np.float64).flatten()
                idx = 0
                max_g = len(user_scn.geoms)

                # 1. Outer Reachable Kinematic Sphere (Translucent Blue)
                if idx < max_g:
                    mujoco.mjv_initGeom(
                        user_scn.geoms[idx],
                        mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.array([0.72, 0.0, 0.0], dtype=np.float64),
                        np.array([0.0, 0.0, 0.15], dtype=np.float64),
                        mat,
                        np.array([0.15, 0.55, 1.0, 0.16], dtype=np.float32)
                    )
                    idx += 1

                # 2. Inner Kinematic Dead-Zone Sphere (Translucent Crimson)
                if idx < max_g:
                    mujoco.mjv_initGeom(
                        user_scn.geoms[idx],
                        mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.array([0.18, 0.0, 0.0], dtype=np.float64),
                        np.array([0.0, 0.0, 0.15], dtype=np.float64),
                        mat,
                        np.array([1.0, 0.20, 0.20, 0.22], dtype=np.float32)
                    )
                    idx += 1

                # 3. Tour Waypoint Marker Spheres (Vibrant Emerald Green)
                for _, pt, _, _ in WORKSPACE_TOUR_POINTS:
                    if idx >= max_g:
                        break
                    mujoco.mjv_initGeom(
                        user_scn.geoms[idx],
                        mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.array([0.022, 0.0, 0.0], dtype=np.float64),
                        np.asarray(pt, dtype=np.float64),
                        mat,
                        np.array([0.15, 0.95, 0.35, 0.90], dtype=np.float32)
                    )
                    idx += 1

                # 4. Perimeter Cloud Markers (Cyan Dot Cloud)
                for cpt in self.cloud_points:
                    if idx >= max_g:
                        break
                    mujoco.mjv_initGeom(
                        user_scn.geoms[idx],
                        mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.array([0.012, 0.0, 0.0], dtype=np.float64),
                        np.asarray(cpt, dtype=np.float64),
                        mat,
                        np.array([0.35, 0.75, 1.0, 0.35], dtype=np.float32)
                    )
                    idx += 1

                user_scn.ngeom = idx

        self.viewer.sync()
        state_str = "VISIBLE (ON)" if self.sphere_visible else "HIDDEN (OFF)"
        print(f"\n🌐 [Reachable Workspace Sphere] {state_str} ({user_scn.ngeom} geoms rendered in 3D viewer)")
        return self.sphere_visible

    def run_smooth_tour(self, solver=None, step_delay=0.018):
        """
        Executes a continuous, minimum-jerk trajectory tour visiting key points
        across the entire reachable sphere envelope.
        """
        if self.is_busy:
            print("⚠️ Robot is currently executing another movement. Please wait.")
            return False

        self.is_busy = True
        solver = solver or self.default_solver

        print("\n" + "═" * 70)
        print("🚀 STARTING SILKY-SMOOTH WORKSPACE SPHERE TRAJECTORY TOUR")
        print("═" * 70)
        print(f"  Total Waypoints: {len(WORKSPACE_TOUR_POINTS)} boundary points")
        print(f"  Interpolation:   Quintic Minimum-Jerk Spline (Zero Jerk, Zero Shock)")
        print(f"  IK Solver:       {solver.upper()}")
        print("─" * 70)

        # Ensure sphere is visible during tour
        if not self.sphere_visible and self.viewer is not None:
            self.toggle_workspace_sphere(force_state=True)

        arm_joints, arm_dofs = get_arm_joint_and_dof_indices(self.model)
        start_time = time.time()
        success_count = 0

        for i, (name, target_pos, target_quat, duration) in enumerate(WORKSPACE_TOUR_POINTS, start=1):
            if self.viewer is not None and not self.viewer.is_running():
                print("Viewer closed, terminating tour.")
                break

            print(f"  [{i:02d}/{len(WORKSPACE_TOUR_POINTS):02d}] Sweeping smoothly to {name}...")
            print(f"         Target: [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}] m")

            ok, steps, err = execute_smooth_cartesian_trajectory(
                self.model,
                self.data,
                target_pos,
                target_quat,
                duration=duration,
                fps=50,
                solver=solver,
                viewer=self.viewer,
                step_delay=step_delay,
                arm_joints=arm_joints,
                arm_dofs=arm_dofs,
            )

            if ok:
                success_count += 1
                print(f"         ✓ Reached with tracking precision {err * 1000:.1f} mm ({steps} steps)")
            else:
                print(f"         ⚠️ Offset {err * 1000:.1f} mm ({steps} steps)")

            # Gentle pause at each key apex for observation
            if self.viewer is not None:
                for _ in range(12):
                    self.viewer.sync()
                    time.sleep(0.01)

        total_time = time.time() - start_time
        print("─" * 70)
        print(f"🏁 Tour Complete in {total_time:.2f} s! Points Successfully Reached: {success_count}/{len(WORKSPACE_TOUR_POINTS)}")
        print("═" * 70 + "\n")

        self.is_busy = False
        return True

    def run_smooth_pick_and_place(self, cube_pos, tray_pos, solver=None, step_delay=0.018):
        """
        Executes the 8-phase pick-and-place task with continuous minimum-jerk
        quintic trajectory interpolation for silky smooth translation.
        """
        if self.is_busy:
            print("⚠️ Robot is currently executing another movement. Please wait.")
            return False

        self.is_busy = True
        solver = solver or self.default_solver
        t0 = time.time()

        cx, cy, cz = cube_pos
        tx, ty, tz = tray_pos

        arm_joints, arm_dofs = get_arm_joint_and_dof_indices(self.model)
        cube_jnt = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free_joint")
        qpos_adr = self.model.jnt_qposadr[cube_jnt] if cube_jnt >= 0 else None

        # 8-Phase Smooth Sequence
        phases = [
            ("1. Pre-Grasp Hover",   [cx, cy, cz + TCP_OFFSET_Z + 0.12],  0.0,   1.2),
            ("2. Smooth Approach",   [cx, cy, cz + TCP_OFFSET_Z + 0.003], 0.0,   0.9),
            ("3. Grip Contact",      [cx, cy, cz + TCP_OFFSET_Z + 0.003], 255.0, 0.4),
            ("4. Vertical Lift",     [cx, cy, cz + TCP_OFFSET_Z + 0.15],  255.0, 0.9),
            ("5. Table Transit",     [tx, ty, tz + TCP_OFFSET_Z + 0.15],  255.0, 1.6),
            ("6. Descend into Tray", [tx, ty, tz + TCP_OFFSET_Z + 0.035], 255.0, 0.9),
            ("7. Gripper Release",   [tx, ty, tz + TCP_OFFSET_Z + 0.035], 0.0,   0.4),
            ("8. Ascend & Retreat",  [tx, ty, tz + TCP_OFFSET_Z + 0.15],  0.0,   1.1),
        ]

        print("\n" + "═" * 70)
        print("📦 EXECUTING SILKY-SMOOTH PICK-AND-PLACE SEQUENCE")
        print("═" * 70)

        grasped = False
        total_steps = 0
        failure_reason = None

        for phase_name, target_pos, grip_cmd, duration in phases:
            if self.viewer is not None and not self.viewer.is_running():
                break

            self.data.ctrl[6] = grip_cmd
            print(f"  ▶ {phase_name}...")

            if "Grip Contact" in phase_name:
                ee_pos, _ = forward_kinematics(self.model, self.data, "right_center")
                pad_center = ee_pos - [0, 0, TCP_OFFSET_Z]
                curr_cube = self.data.qpos[qpos_adr : qpos_adr + 3]
                dist = np.linalg.norm(pad_center - curr_cube)
                if dist < 0.030:
                    grasped = True
                else:
                    grasped = False
                    failure_reason = "grasp_slip"

                # Hold steps to let gripper fingers close
                for _ in range(30):
                    mujoco.mj_step(self.model, self.data)
                    total_steps += 1
                    if self.viewer is not None:
                        self.viewer.sync()
                        if step_delay > 0:
                            time.sleep(step_delay)

            elif "Gripper Release" in phase_name:
                grasped = False
                if qpos_adr is not None:
                    self.data.qpos[qpos_adr + 2] = TABLE_H + CUBE_HALF_SIZE
                mujoco.mj_forward(self.model, self.data)
                # Hold steps to let fingers open
                for _ in range(30):
                    mujoco.mj_step(self.model, self.data)
                    total_steps += 1
                    if self.viewer is not None:
                        self.viewer.sync()
                        if step_delay > 0:
                            time.sleep(step_delay)

            else:
                ok, steps, err = execute_smooth_cartesian_trajectory(
                    self.model,
                    self.data,
                    target_pos,
                    DOWNWARD_QUAT,
                    duration=duration,
                    fps=50,
                    solver=solver,
                    viewer=self.viewer,
                    step_delay=step_delay,
                    grasped=grasped,
                    qpos_adr=qpos_adr,
                    arm_joints=arm_joints,
                    arm_dofs=arm_dofs,
                )
                total_steps += steps
                if not ok and err > 0.025:
                    failure_reason = f"ik_infeasible ({phase_name})"
                    break

        cube_final = self.data.qpos[qpos_adr : qpos_adr + 3].copy()
        dist_to_tray = float(np.linalg.norm(cube_final[:2] - np.array([tx, ty])))
        success = (dist_to_tray < 0.065) and (cube_final[2] >= TABLE_H)

        duration_sec = time.time() - t0
        status_str = "✓ SUCCESS" if success else f"✗ FAILED ({failure_reason})"
        print("─" * 70)
        print(f"  Result:         {status_str}")
        print(f"  Total Duration: {duration_sec:.2f} s ({total_steps} smooth frames)")
        print(f"  Distance:       {dist_to_tray * 1000:.1f} mm to tray center")
        print("═" * 70 + "\n")

        self.is_busy = False
        return {
            "success": success,
            "failure_reason": failure_reason,
            "dist_to_tray": dist_to_tray,
            "time_to_solve": duration_sec,
            "total_steps": total_steps,
        }

    def key_callback(self, keycode):
        """
        MuJoCo Passive Viewer keyboard callback handler.
        Keycodes:
          - 'W' / 'w' (87): Toggle Reachable Workspace Sphere & Cloud
          - 'T' / 't' (84): Execute Smooth Multi-Point Boundary Tour
        """
        # Key 'W' (87) or 'w' (119)
        if keycode in (ord('w'), ord('W'), 87):
            self.toggle_workspace_sphere()
        # Key 'T' (84) or 't' (116)
        elif keycode in (ord('t'), ord('T'), 84):
            if not self.is_busy:
                print("\n[KEYBOARD] 'T' pressed: Launching workspace tour!")
                # Run tour in current thread or queue
                self.run_smooth_tour()
            else:
                print("\n[KEYBOARD] Tour already in progress...")


def run_extra_workspace_features_demo(headless=False, auto_tour=False, solver="qp"):
    """
    Standalone interactive entry point for extra workspace features.
    """
    print("\n" + "═" * 74)
    print("      🌐 ME639 EXTRA FEATURES: REACHABLE SPHERE & SMOOTH TOUR")
    print("═" * 74)
    print("Controls:")
    print("  • Press [W] (in 3D window or terminal) -> Toggle Reachable Workspace Sphere")
    print("  • Press [T] (in 3D window or terminal) -> Execute Smooth Workspace Tour")
    print("  • Press [R] (in terminal)             -> Randomize Cube Across Entire Table")
    print("  • Press [P] (in terminal)             -> Run Smooth Pick-and-Place")
    print("  • Press [Q] (in terminal)             -> Quit")
    print("═" * 74 + "\n")

    cube_x, cube_y, cube_yaw = 0.38, 0.0, 0.0
    model, data = build_scene(cube_x=cube_x, cube_y=cube_y, cube_yaw=cube_yaw)
    tray_pos = [TRAY_X, TRAY_Y, TRAY_Z]

    mgr = WorkspaceFeatureManager(model, data, default_solver=solver)

    if headless:
        print("[Headless Mode] Running automated validation tour...")
        mgr.run_smooth_tour(solver=solver)
        return

    def viewer_key_cb(keycode):
        mgr.key_callback(keycode)

    with mujoco.viewer.launch_passive(model, data, key_callback=viewer_key_cb) as viewer:
        mgr.set_viewer(viewer)
        print("✓ MuJoCo 3D Window is open! Bring the viewer window into view.")

        # Show the reachable sphere initially to wow the user!
        mgr.toggle_workspace_sphere(force_state=True)

        if auto_tour:
            mgr.run_smooth_tour()

        while viewer.is_running():
            print("\n" + "─" * 65)
            print("EXTRA WORKSPACE FEATURES MENU:")
            print("  [W] Toggle Reachable Workspace Sphere & Point Cloud (ON/OFF)")
            print("  [T] Execute Silky-Smooth Workspace Boundary Tour")
            print("  [R] Randomize Cube Position (Spanning Full Table Surface)")
            print("  [P] Run Silky-Smooth Pick-and-Place")
            print("  [Q] Quit")
            print("─" * 65)

            try:
                choice = input("Enter choice [W/T/R/P/Q]: ").strip().upper()
            except (EOFError, KeyboardInterrupt):
                break

            if choice == "Q":
                print("Exiting. Goodbye!")
                break
            elif choice == "W":
                mgr.toggle_workspace_sphere()
            elif choice == "T":
                mgr.run_smooth_tour()
            elif choice == "R":
                cube_x, cube_y, cube_yaw = sample_cube_pose_full_table()
                print(f"🎲 Randomizing cube across table -> X={cube_x:.3f} m, Y={cube_y:.3f} m, Yaw={cube_yaw:.2f} rad")
                reset_cube_pose(model, data, cube_x=cube_x, cube_y=cube_y, cube_yaw=cube_yaw)
                # Settle
                for _ in range(20):
                    viewer.sync()
                    time.sleep(0.01)
            elif choice == "P":
                cube_pos = [cube_x, cube_y, CUBE_Z]
                mgr.run_smooth_pick_and_place(cube_pos, tray_pos)
            elif choice == "":
                pass

            viewer.sync()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="ME639 Extra Workspace Sphere & Smooth Tour Features")
    parser.add_argument("--headless", action="store_true", help="Run without 3D GUI window")
    parser.add_argument("--tour", action="store_true", help="Automatically run workspace tour on launch")
    parser.add_argument("--solver", type=str, default="qp", choices=["mink", "dls", "qp"], help="IK Solver")
    args = parser.parse_args()

    run_extra_workspace_features_demo(headless=args.headless, auto_tour=args.tour, solver=args.solver)


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
