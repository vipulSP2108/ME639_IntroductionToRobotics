#!/usr/bin/env python3
"""
Trajectory Planning and Motion Interpolation Utilities
======================================================
Provides minimum-jerk trajectory generation, full-table domain sampling,
and smooth continuous trajectory execution for ME639 robotics lab.
"""

import time
import numpy as np
import mujoco

from utils.kinematics import (
    forward_kinematics,
    compute_task_error,
    compute_jacobian,
    dls_ik_step,
    get_arm_joint_and_dof_indices
)
from utils.quaternion_utils import DOWNWARD_QUAT, quat_slerp

try:
    import qpsolvers
    HAS_QP = True
except ImportError:
    HAS_QP = False

TCP_OFFSET_Z = 0.123
TABLE_H = 0.20
CUBE_HALF_SIZE = 0.025


def quintic_poly(tau):
    """
    Evaluates the 5th-order minimum-jerk polynomial:
        s(tau) = 10*tau^3 - 15*tau^4 + 6*tau^5
    Boundary conditions: s(0)=0, s(1)=1, s'(0)=s'(1)=0, s''(0)=s''(1)=0.
    """
    tau = float(np.clip(tau, 0.0, 1.0))
    return 10.0 * (tau**3) - 15.0 * (tau**4) + 6.0 * (tau**5)


def sample_cube_pose_full_table():
    """
    Samples a valid cube spawn location spanning the ENTIRE reachable table surface.
    Table bounds: X in [0.25, 0.65], Y in [-0.25, +0.25].
    
    Safe spawn range:
      - X in [0.29, 0.61] m (32 cm span)
      - Y in [-0.21, +0.21] m (42 cm span)
      
    Rejection rules:
      1. Exclude tray box: X >= 0.46 and |Y| <= 0.09
      2. Exclude inner base dead-zone: R = sqrt(X^2 + Y^2) < 0.22 m
      3. Exclude arm over-reach: R > 0.68 m
    """
    for _ in range(1000):
        x = float(np.random.uniform(0.29, 0.61))
        y = float(np.random.uniform(-0.21, 0.21))
        
        # Check tray exclusion
        if x >= 0.46 and abs(y) <= 0.09:
            continue
            
        # Check radial reachability
        r = np.hypot(x, y)
        if r < 0.22 or r > 0.68:
            continue
            
        yaw = float(np.random.uniform(-0.78, 0.78))
        return x, y, yaw
        
    return 0.38, 0.0, 0.0


def execute_smooth_cartesian_trajectory(
    model,
    data,
    target_pos,
    target_quat=DOWNWARD_QUAT,
    duration=1.2,
    fps=50,
    solver="qp",
    viewer=None,
    step_delay=0.02,
    grasped=False,
    qpos_adr=None,
    arm_joints=None,
    arm_dofs=None,
    lambda_=0.01,
    max_dq_step=0.04,
):
    """
    Executes a continuous, buttery-smooth minimum-jerk trajectory from the current
    end-effector pose to target_pos and target_quat over the given duration.
    
    At each step:
      - Evaluates the quintic spline for sub-waypoint p_d(k) and q_d(k)
      - Computes small corrective differential IK displacement
      - Smoothly updates joint state and viewer with real-time pacing
      - Synchronously locks grasped object pose with the gripper TCP
    """
    if arm_joints is None or arm_dofs is None:
        arm_joints, arm_dofs = get_arm_joint_and_dof_indices(model)

    start_pos, start_quat = forward_kinematics(model, data, "right_center")
    target_pos = np.asarray(target_pos, dtype=np.float64)
    target_quat = np.asarray(target_quat, dtype=np.float64)

    total_steps = max(int(duration * fps), 15)
    q_min = model.jnt_range[arm_joints, 0]
    q_max = model.jnt_range[arm_joints, 1]
    n = len(arm_joints)

    cube_jnt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free_joint")
    cube_dof = model.jnt_dofadr[cube_jnt] if cube_jnt >= 0 else -1

    for step in range(1, total_steps + 1):
        tau = step / float(total_steps)
        s = quintic_poly(tau)

        # Desired sub-waypoint along the minimum-jerk trajectory
        p_sub = start_pos + s * (target_pos - start_pos)
        q_sub = quat_slerp(start_quat, target_quat, s)

        # 1-2 small corrective IK iterations per sub-step
        for _ in range(2):
            dx = compute_task_error(model, data, "right_center", p_sub, q_sub)
            if np.linalg.norm(dx[:3]) < 0.001 and np.linalg.norm(dx[3:]) < 0.01:
                break

            J = compute_jacobian(model, data, "right_center", arm_dofs)
            dx_weighted = dx.copy()
            dx_weighted[3:] *= 0.4  # Balance rotational vs translational tracking

            if solver == "qp" and HAS_QP:
                H = J.T @ J + lambda_ * np.eye(n, dtype=np.float64)
                c = -J.T @ dx_weighted
                q_now = data.qpos[arm_joints]
                dq_upper = np.minimum(q_max - q_now, max_dq_step)
                dq_lower = np.maximum(q_min - q_now, -max_dq_step)
                G = np.vstack([np.eye(n, dtype=np.float64), -np.eye(n, dtype=np.float64)])
                h = np.concatenate([dq_upper, -dq_lower])
                try:
                    dq = qpsolvers.solve_qp(P=H, q=c, G=G, h=h, solver="daqp")
                except Exception:
                    dq = None
                if dq is None:
                    dq = np.linalg.solve(H, -c)
                    dq = np.clip(dq, -max_dq_step, max_dq_step)
            else:
                dq = dls_ik_step(J, dx_weighted, lambda_=lambda_)
                dq = np.clip(dq, -max_dq_step, max_dq_step)

            data.qpos[arm_joints] += dq

        # Maintain grasped object rigidly attached to TCP
        if grasped and qpos_adr is not None:
            ee_pos, _ = forward_kinematics(model, data, "right_center")
            data.qpos[qpos_adr : qpos_adr + 3] = ee_pos - [0, 0, TCP_OFFSET_Z]
            if cube_dof >= 0:
                data.qvel[cube_dof : cube_dof + 6] = 0.0

        mujoco.mj_forward(model, data)

        if viewer is not None:
            viewer.sync()
            if step_delay > 0:
                time.sleep(step_delay)

    final_pos, _ = forward_kinematics(model, data, "right_center")
    final_err = float(np.linalg.norm(target_pos - final_pos))
    return final_err < 0.015, total_steps, final_err
