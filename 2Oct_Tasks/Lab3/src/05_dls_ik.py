#!/usr/bin/env python3
"""
Stage 5: Damped Least Squares Inverse Kinematics (DLS-IK)
=========================================================
Custom implementation of Closed-Loop Damped Least Squares (Levenberg-Marquardt)
Inverse Kinematics:

Mathematical Formulation:
    Δq = (J^T J + λ I)^(-1) J^T e
where:
    J : 6x6 analytical arm Jacobian (pos + rot)
    e : 6D task error vector [e_pos; e_rot]
    λ : Damping factor (regularization to avoid singularity inversion blowup)

This script replaces the Mink library with our own math solver from utils/kinematics.py
and runs the identical 25 episode scenarios (seeds 0..24).
"""

import os
import sys
import time
import json
import argparse
import importlib
import numpy as np
import mujoco

# Ensure project root is in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.kinematics import (
    get_arm_joint_and_dof_indices,
    compute_jacobian,
    compute_task_error,
    dls_ik_step,
    forward_kinematics
)
from utils.quaternion_utils import DOWNWARD_QUAT
from utils.trajectory import execute_smooth_cartesian_trajectory
from scripts.load_heal_robot import compensate_gravity

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

batch_mod = importlib.import_module("src.04_batch_episodes")
run_batch_evaluation = batch_mod.run_batch_evaluation
EVAL_SEEDS = batch_mod.EVAL_SEEDS

GRIPPER_ACTUATOR_ID = 6
GRIP_OPEN_CMD = 0.0
GRIP_CLOSE_CMD = 255.0
TCP_OFFSET_Z = 0.123

# Default Tunable Hyperparameters
DEFAULT_LAMBDA = 0.01
DEFAULT_MAX_IK_STEPS = 200
DEFAULT_POS_THRESH = 0.003
DEFAULT_ORI_THRESH = 0.05


def solve_dls_phase(
    model,
    data,
    target_pos,
    target_quat=DOWNWARD_QUAT,
    lambda_=DEFAULT_LAMBDA,
    max_steps=DEFAULT_MAX_IK_STEPS,
    pos_thresh=DEFAULT_POS_THRESH,
    ori_thresh=DEFAULT_ORI_THRESH,
    viewer=None,
    arm_joints=None,
    arm_dofs=None,
    grasped=False,
    qpos_adr=None,
    step_delay=0.02,
):
    """
    Executes closed-loop DLS-IK iterations to steer the end-effector to the target pose.
    """
    if arm_joints is None or arm_dofs is None:
        arm_joints, arm_dofs = get_arm_joint_and_dof_indices(model)

    steps_taken = 0
    converged = False

    for s in range(max_steps):
        steps_taken += 1
        
        # 1. Compute 6D task error: [pos_err (3D); ori_err (3D)]
        dx = compute_task_error(model, data, "right_center", target_pos, target_quat)
        pos_err = np.linalg.norm(dx[:3])
        ori_err = np.linalg.norm(dx[3:])

        if pos_err < pos_thresh and ori_err < ori_thresh:
            converged = True
            break

        # 2. Geometric Jacobian for arm joints (6 x 6)
        J = compute_jacobian(model, data, "right_center", arm_dofs)

        # 3. DLS Step: dq = (J^T J + λ I)^(-1) J^T dx
        dx_weighted = dx.copy()
        dx_weighted[3:] *= 0.5  # scale orientation weight for balanced convergence
        dq = dls_ik_step(J, dx_weighted, lambda_=lambda_)
        
        # 4. Joint velocity clamp for numerical and physical safety
        dq = np.clip(dq, -0.4, 0.4)

        # 5. Integrate configuration
        data.qpos[arm_joints] += dq

        # 6. If cube is grasped, carry along with TCP
        if grasped and qpos_adr is not None:
            ee_pos, _ = forward_kinematics(model, data, "right_center")
            data.qpos[qpos_adr : qpos_adr + 3] = ee_pos - [0, 0, TCP_OFFSET_Z]
            data.qvel[model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free_joint")] : ] = 0.0

        mujoco.mj_forward(model, data)

        if viewer is not None:
            viewer.sync()
            if step_delay > 0:
                time.sleep(step_delay)

    final_pos, _ = forward_kinematics(model, data, "right_center")
    final_err = np.linalg.norm(np.asarray(target_pos) - final_pos)
    return converged, steps_taken, final_err


def dls_ik_solver(model, data, cube_pos, tray_pos, lambda_=DEFAULT_LAMBDA, viewer=None, step_delay=0.02):
    """
    Executes the 8-phase pick-and-place sequence using custom DLS-IK.
    
    Returns:
        dict: comprehensive episode log
    """
    t0 = time.time()
    arm_joints, arm_dofs = get_arm_joint_and_dof_indices(model)
    cube_jnt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free_joint")
    qpos_adr = model.jnt_qposadr[cube_jnt]

    ik_iterations = []
    min_clearances = []

    cx, cy, cz = cube_pos
    tx, ty, tz = tray_pos

    phases = [
        ("pre_grasp",     [cx, cy, cz + TCP_OFFSET_Z + 0.12], GRIP_OPEN_CMD,  200),
        ("approach",      [cx, cy, cz + TCP_OFFSET_Z + 0.003], GRIP_OPEN_CMD,  150),
        ("grip",          [cx, cy, cz + TCP_OFFSET_Z + 0.003], GRIP_CLOSE_CMD, 40),
        ("lift",          [cx, cy, cz + TCP_OFFSET_Z + 0.15], GRIP_CLOSE_CMD, 200),
        ("transit",       [tx, ty, tz + TCP_OFFSET_Z + 0.15], GRIP_CLOSE_CMD, 250),
        ("place_descend", [tx, ty, tz + TCP_OFFSET_Z + 0.035], GRIP_CLOSE_CMD, 150),
        ("release",       [tx, ty, tz + TCP_OFFSET_Z + 0.035], GRIP_OPEN_CMD,  40),
        ("retreat",       [tx, ty, tz + TCP_OFFSET_Z + 0.15], GRIP_OPEN_CMD,  150),
    ]

    grasped = False
    failure_reason = None

    for phase_name, target_pos, grip_cmd, max_steps in phases:
        data.ctrl[GRIPPER_ACTUATOR_ID] = grip_cmd

        if phase_name == "grip":
            # Grasp proximity check
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
            if viewer is not None:
                dur_map = {
                    "pre_grasp": 1.2,
                    "approach": 0.9,
                    "lift": 0.9,
                    "transit": 1.6,
                    "place_descend": 0.9,
                    "retreat": 1.1,
                }
                dur = dur_map.get(phase_name, 1.0)
                converged, steps, err = execute_smooth_cartesian_trajectory(
                    model, data, target_pos, DOWNWARD_QUAT,
                    duration=dur, fps=50, solver="dls",
                    viewer=viewer, step_delay=step_delay,
                    grasped=grasped, qpos_adr=qpos_adr,
                    arm_joints=arm_joints, arm_dofs=arm_dofs,
                    lambda_=lambda_,
                )
            else:
                converged, steps, err = solve_dls_phase(
                    model, data, target_pos,
                    target_quat=DOWNWARD_QUAT,
                    lambda_=lambda_,
                    max_steps=max_steps,
                    viewer=viewer,
                    arm_joints=arm_joints,
                    arm_dofs=arm_dofs,
                    grasped=grasped,
                    qpos_adr=qpos_adr,
                    step_delay=step_delay,
                )
            ik_iterations.append(steps)
            if not converged and err > 0.025:
                failure_reason = f"ik_infeasible ({phase_name})"
                break

        current_ee_z = data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_center")][2]
        clearance = current_ee_z - TCP_OFFSET_Z - TABLE_H
        min_clearances.append(clearance)

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
        "lambda": lambda_
    }
    return log


def run_single_episode(cube_x=0.38, cube_y=0.0, cube_yaw=0.0, lambda_=DEFAULT_LAMBDA, headless=False):
    print("=" * 60)
    print("Stage 5: Pick and Place with DLS-IK")
    print(f"Damping parameter λ = {lambda_}")
    print("=" * 60)
    print(f"Spawning cube at: x={cube_x:.3f}, y={cube_y:.3f}, yaw={cube_yaw:.3f} rad")

    model, data = build_scene(cube_x=cube_x, cube_y=cube_y, cube_yaw=cube_yaw)
    cube_pos = [cube_x, cube_y, CUBE_Z]
    tray_pos = [TRAY_X, TRAY_Y, TRAY_Z]

    if not headless:
        try:
            import mujoco.viewer
            with mujoco.viewer.launch_passive(model, data) as viewer:
                log = dls_ik_solver(model, data, cube_pos, tray_pos, lambda_=lambda_, viewer=viewer)
                for _ in range(100):
                    viewer.sync()
                    time.sleep(0.01)
        except Exception as e:
            print(f"Viewer skipped ({e}), running headless.")
            log = dls_ik_solver(model, data, cube_pos, tray_pos, lambda_=lambda_, viewer=None)
    else:
        log = dls_ik_solver(model, data, cube_pos, tray_pos, lambda_=lambda_, viewer=None)

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
    parser = argparse.ArgumentParser(description="Stage 5: DLS-IK Pick and Place")
    parser.add_argument("--batch", action="store_true", help="Run 25-episode batch benchmark")
    parser.add_argument("--headless", action="store_true", help="Run without GUI")
    parser.add_argument("--lambda_val", type=float, default=DEFAULT_LAMBDA, help="Damping λ")
    parser.add_argument("--cube_x", type=float, default=0.38, help="Cube initial X")
    parser.add_argument("--cube_y", type=float, default=0.00, help="Cube initial Y")
    args = parser.parse_args()

    if args.batch:
        run_batch_evaluation(
            solver_fn=lambda m, d, cp, tp, viewer=None: dls_ik_solver(m, d, cp, tp, lambda_=args.lambda_val, viewer=viewer),
            solver_name="dls",
            seeds=EVAL_SEEDS
        )
    else:
        run_single_episode(cube_x=args.cube_x, cube_y=args.cube_y, lambda_=args.lambda_val, headless=args.headless)
