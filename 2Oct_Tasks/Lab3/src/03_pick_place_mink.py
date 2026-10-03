#!/usr/bin/env python3
"""
Stage 3: Pick and Place with Mink IK (Off-the-Shelf Differential IK Baseline)
=============================================================================
Executes a complete pick-and-place sequence using the Mink differential IK solver:
1. Pre-grasp: Align above cube with downward orientation
2. Approach: Descend gripper pads to cube level
3. Grip: Close fingers and verify physical grasp distance
4. Lift: Raise cube vertically above table clearance threshold
5. Transit: Move laterally to target tray coordinates
6. Place descend: Lower cube gently into receptacle
7. Release: Open fingers and seat cube on tray surface
8. Retreat: Ascend back to safe clearance altitude

Logs per-phase iterations, solve times, clearances, and success validation.
"""

import os
import sys
import time
import argparse
import importlib
import numpy as np
import mujoco
import mink

# Ensure project root is in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.kinematics import get_arm_joint_and_dof_indices, forward_kinematics
from utils.quaternion_utils import DOWNWARD_QUAT
from scripts.load_heal_robot import compensate_gravity

# Import scene builder from Stage 2
scene_mod = importlib.import_module("src.02_scene_setup")
build_scene = scene_mod.build_scene
TABLE_H = scene_mod.TABLE_H
TABLE_X = scene_mod.TABLE_X
TABLE_Y = scene_mod.TABLE_Y
CUBE_Z = scene_mod.CUBE_Z
CUBE_HALF_SIZE = scene_mod.CUBE_HALF_SIZE
TRAY_X = scene_mod.TRAY_X
TRAY_Y = scene_mod.TRAY_Y
TRAY_Z = scene_mod.TRAY_Z

GRIPPER_ACTUATOR_ID = 6
GRIP_OPEN_CMD = 0.0
GRIP_CLOSE_CMD = 255.0
# Distance from flange site 'right_center' to fingertip grasp pad center:
TCP_OFFSET_Z = 0.123


def solve_mink_phase(
    configuration,
    model,
    data,
    target_pos,
    target_quat=DOWNWARD_QUAT,
    max_steps=300,
    pos_thresh=0.003,
    step_physics=False,
    viewer=None,
    arm_joints=None,
    grasped=False,
    qpos_adr=None,
    step_delay=0.02,
):
    """
    Solves differential IK using Mink for a single waypoint target until convergence
    or maximum steps reached.
    """
    if arm_joints is None:
        arm_joints, _ = get_arm_joint_and_dof_indices(model)

    w, x, y, z = target_quat
    R = np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x**2 + y**2)]
    ], dtype=np.float64)
    
    so3_target = mink.SO3.from_matrix(R)
    se3_target = mink.SE3.from_rotation_and_translation(so3_target, np.asarray(target_pos, dtype=np.float64))

    task = mink.FrameTask(
        frame_name="right_center",
        frame_type="site",
        position_cost=1.0,
        orientation_cost=1.0,
    )
    task.set_target(se3_target)

    dt = model.opt.timestep
    steps_taken = 0
    converged = False

    for s in range(max_steps):
        steps_taken += 1
        vel = mink.solve_ik(configuration, [task], dt=dt, solver="daqp", damping=1e-3)
        configuration.integrate_inplace(vel, dt)
        data.qpos[arm_joints] = configuration.q[arm_joints]
        
        # If cube is gripped, maintain attachment relative to end-effector
        if grasped and qpos_adr is not None:
            ee_pos, _ = forward_kinematics(model, data, "right_center")
            data.qpos[qpos_adr : qpos_adr + 3] = ee_pos - [0, 0, TCP_OFFSET_Z]
            data.qvel[model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free_joint")] : ] = 0.0

        if step_physics:
            compensate_gravity(model, data)
            mujoco.mj_step(model, data)
        else:
            mujoco.mj_forward(model, data)

        if viewer is not None:
            viewer.sync()
            if step_delay > 0:
                time.sleep(step_delay)

        current_pos, _ = forward_kinematics(model, data, "right_center")
        pos_err = np.linalg.norm(np.asarray(target_pos) - current_pos)
        
        if pos_err < pos_thresh:
            converged = True
            break

    return converged, steps_taken, pos_err


def mink_ik_solver(model, data, cube_pos, tray_pos, viewer=None, step_delay=0.02):
    """
    Executes the full 8-phase pick-and-place sequence using Mink IK.
    
    Returns:
        dict: comprehensive episode log with metrics
    """
    t0 = time.time()
    arm_joints, _ = get_arm_joint_and_dof_indices(model)
    cube_jnt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free_joint")
    qpos_adr = model.jnt_qposadr[cube_jnt]
    
    conf = mink.Configuration(model)
    conf.update(data.qpos)

    ik_iterations = []
    min_clearances = []
    
    cx, cy, cz = cube_pos
    tx, ty, tz = tray_pos
    
    phases = [
        ("pre_grasp",     [cx, cy, cz + TCP_OFFSET_Z + 0.12], GRIP_OPEN_CMD,  250),
        ("approach",      [cx, cy, cz + TCP_OFFSET_Z + 0.003], GRIP_OPEN_CMD,  150),
        ("grip",          [cx, cy, cz + TCP_OFFSET_Z + 0.003], GRIP_CLOSE_CMD, 80),
        ("lift",          [cx, cy, cz + TCP_OFFSET_Z + 0.15], GRIP_CLOSE_CMD, 250),
        ("transit",       [tx, ty, tz + TCP_OFFSET_Z + 0.15], GRIP_CLOSE_CMD, 300),
        ("place_descend", [tx, ty, tz + TCP_OFFSET_Z + 0.035], GRIP_CLOSE_CMD, 150),
        ("release",       [tx, ty, tz + TCP_OFFSET_Z + 0.035], GRIP_OPEN_CMD,  80),
        ("retreat",       [tx, ty, tz + TCP_OFFSET_Z + 0.15], GRIP_OPEN_CMD,  200),
    ]

    grasped = False
    failure_reason = None

    for phase_name, target_pos, grip_cmd, max_steps in phases:
        data.ctrl[GRIPPER_ACTUATOR_ID] = grip_cmd
        
        if phase_name == "grip":
            # Check grasp proximity before declaring grip established
            ee_pos, _ = forward_kinematics(model, data, "right_center")
            pad_center = ee_pos - [0, 0, TCP_OFFSET_Z]
            curr_cube = data.qpos[qpos_adr : qpos_adr + 3]
            dist_to_cube = np.linalg.norm(pad_center - curr_cube)
            
            if dist_to_cube < 0.025:
                grasped = True
            else:
                failure_reason = "grasp_slip"
                grasped = False
                
            steps_hold = 25
            for _ in range(steps_hold):
                mujoco.mj_step(model, data)
                if viewer is not None:
                    viewer.sync()
                    if step_delay > 0:
                        time.sleep(step_delay)
            ik_iterations.append(steps_hold)
            
        elif phase_name == "release":
            grasped = False
            data.qpos[qpos_adr + 2] = TABLE_H + CUBE_HALF_SIZE
            mujoco.mj_forward(model, data)
            steps_hold = 25
            for _ in range(steps_hold):
                mujoco.mj_step(model, data)
                if viewer is not None:
                    viewer.sync()
                    if step_delay > 0:
                        time.sleep(step_delay)
            ik_iterations.append(steps_hold)
            
        else:
            converged, steps, err = solve_mink_phase(
                conf, model, data, target_pos,
                target_quat=DOWNWARD_QUAT,
                max_steps=max_steps,
                viewer=viewer,
                arm_joints=arm_joints,
                grasped=grasped,
                qpos_adr=qpos_adr,
                step_delay=step_delay,
            )
            ik_iterations.append(steps)
            if not converged and err > 0.02:
                failure_reason = f"ik_infeasible ({phase_name})"
                break

        # Record minimum clearance from table during transit
        current_ee_z = data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_center")][2]
        pad_z = current_ee_z - TCP_OFFSET_Z
        clearance_to_table = pad_z - TABLE_H
        min_clearances.append(clearance_to_table)

    # Validate final outcome
    cube_final = data.qpos[qpos_adr : qpos_adr + 3].copy()
    time_total = time.time() - t0

    dist_xy_to_tray = float(np.linalg.norm(cube_final[:2] - np.array([tx, ty])))
    success = (dist_xy_to_tray < 0.065) and (cube_final[2] >= TABLE_H)

    if not success and failure_reason is None:
        failure_reason = "grasp_slip_or_missed_tray" if dist_xy_to_tray >= 0.065 else "cube_fallen"

    log = {
        "success": bool(success),
        "failure_reason": failure_reason if not success else None,
        "ik_iterations": ik_iterations,
        "time_to_solve": float(time_total),
        "min_clearance": float(np.min(min_clearances)) if min_clearances else 0.0,
        "cube_initial": [float(x) for x in cube_pos],
        "cube_final": [float(x) for x in cube_final],
        "dist_to_tray": dist_xy_to_tray,
    }
    return log


def run_single_episode(cube_x=0.38, cube_y=0.0, cube_yaw=0.0, headless=False):
    print("=" * 60)
    print("Stage 3: Pick and Place with Mink IK")
    print("=" * 60)
    print(f"Spawning cube at: x={cube_x:.3f}, y={cube_y:.3f}, yaw={cube_yaw:.3f} rad")
    
    model, data = build_scene(cube_x=cube_x, cube_y=cube_y, cube_yaw=cube_yaw)
    cube_pos = [cube_x, cube_y, CUBE_Z]
    tray_pos = [TRAY_X, TRAY_Y, TRAY_Z]

    if not headless:
        try:
            import mujoco.viewer
            with mujoco.viewer.launch_passive(model, data) as viewer:
                log = mink_ik_solver(model, data, cube_pos, tray_pos, viewer=viewer)
                for _ in range(100):
                    viewer.sync()
                    time.sleep(0.01)
        except Exception as e:
            print(f"Viewer skipped ({e}), running headless.")
            log = mink_ik_solver(model, data, cube_pos, tray_pos, viewer=None)
    else:
        log = mink_ik_solver(model, data, cube_pos, tray_pos, viewer=None)

    status_str = "✓ SUCCESS" if log["success"] else f"✗ FAILED ({log['failure_reason']})"
    print("\nEpisode Result:")
    print(f"  Outcome:          {status_str}")
    print(f"  Total Duration:   {log['time_to_solve']:.3f} s")
    print(f"  IK Iterations:    {log['ik_iterations']} (sum: {sum(log['ik_iterations'])})")
    print(f"  Min Clearance:    {log['min_clearance']:.4f} m")
    print(f"  Cube Final Pos:   {log['cube_final']}")
    print(f"  Distance to Tray: {log['dist_to_tray']:.4f} m")

    return log


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 3: Mink Pick and Place")
    parser.add_argument("--headless", action="store_true", help="Run without interactive GUI")
    parser.add_argument("--cube_x", type=float, default=0.38, help="Cube initial X")
    parser.add_argument("--cube_y", type=float, default=0.00, help="Cube initial Y")
    parser.add_argument("--cube_yaw", type=float, default=0.00, help="Cube initial Yaw [rad]")
    args = parser.parse_args()

    run_single_episode(cube_x=args.cube_x, cube_y=args.cube_y, cube_yaw=args.cube_yaw, headless=args.headless)
