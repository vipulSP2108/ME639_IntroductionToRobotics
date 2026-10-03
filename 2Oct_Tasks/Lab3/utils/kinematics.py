#!/usr/bin/env python3
"""
Kinematics Math Utilities for MuJoCo
====================================
Pure mathematical and analytical kinematics helpers for:
- Jacobian computation (using mj_jacSite)
- Forward kinematics extraction
- Task-space tracking error calculation (pos + ori)
- Damped Least Squares (DLS) IK step
- Quadratic Programming (QP) matrix formulation for constrained IK
"""

import numpy as np
import mujoco

from utils.quaternion_utils import rot_matrix_to_quat, quaternion_error_to_angular_velocity


def get_arm_joint_and_dof_indices(model):
    """
    Identifies the arm joints and velocity DOF indices, excluding gripper finger
    joints and floating base / free joints.
    
    Returns:
        arm_joint_ids: list of int joint indices in the arm
        arm_dof_indices: list of int DOF velocity indices corresponding to the arm
    """
    arm_joint_ids = []
    arm_dof_indices = []
    
    for j_id in range(model.njnt):
        j_name = model.joint(j_id).name
        # Skip free joints (floating base / scene objects) and gripper joints
        if model.jnt_type[j_id] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        if j_name.startswith("gripper/") or "finger" in j_name or "driver" in j_name or "follower" in j_name:
            continue
            
        arm_joint_ids.append(j_id)
        dof_adr = model.jnt_dofadr[j_id]
        arm_dof_indices.append(dof_adr)
        
    return arm_joint_ids, arm_dof_indices


def forward_kinematics(model, data, site_name="right_center"):
    """
    Extracts current end-effector position and orientation (quaternion) in world frame.
    
    Returns:
        pos: np.ndarray (3,) [x, y, z]
        quat: np.ndarray (4,) [w, x, y, z] scalar-first
    """
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id < 0:
        raise ValueError(f"Site '{site_name}' not found in MuJoCo model!")
        
    pos = np.array(data.site_xpos[site_id], dtype=np.float64, copy=True)
    R = np.array(data.site_xmat[site_id], dtype=np.float64).reshape((3, 3))
    quat = rot_matrix_to_quat(R)
    return pos, quat


def compute_jacobian(model, data, site_name="right_center", arm_dofs=None):
    """
    Computes the 6xN geometric Jacobian for the specified site.
    Rows 0:3 = positional velocity Jacobian J_pos
    Rows 3:6 = rotational / angular velocity Jacobian J_rot
    
    If arm_dofs is provided, returns only the columns corresponding to the arm joints (6 x 6).
    Otherwise returns the full (6 x model.nv) Jacobian.
    """
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id < 0:
        raise ValueError(f"Site '{site_name}' not found in MuJoCo model!")
        
    jacp = np.zeros((3, model.nv), dtype=np.float64)
    jacr = np.zeros((3, model.nv), dtype=np.float64)
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    
    J = np.vstack([jacp, jacr])
    if arm_dofs is not None:
        J = J[:, arm_dofs]
    return J


def compute_task_error(model, data, site_name, target_pos, target_quat):
    """
    Computes the 6D task-space error vector between current site pose and desired target:
        dx = [ e_pos (3D) ; e_rot (3D) ]
    where:
        e_pos = target_pos - current_pos
        e_rot = angular velocity error pointing towards target_quat
    """
    current_pos, current_quat = forward_kinematics(model, data, site_name)
    
    pos_err = np.asarray(target_pos, dtype=np.float64) - current_pos
    rot_err = quaternion_error_to_angular_velocity(current_quat, target_quat)
    
    return np.concatenate([pos_err, rot_err])


def dls_ik_step(J, dx, lambda_=0.01):
    """
    Computes one Damped Least Squares (Levenberg-Marquardt) joint velocity step:
        dq = (J^T J + lambda * I)^(-1) J^T dx
        
    Uses np.linalg.solve for speed and numerical stability over explicit matrix inversion.
    """
    n = J.shape[1]
    H = J.T @ J + lambda_ * np.eye(n, dtype=np.float64)
    g = J.T @ dx
    dq = np.linalg.solve(H, g)
    return dq


def build_qp_matrices(J, dx, q, q_min, q_max, lambda_=0.01, dt=0.002, v_max=1.0):
    """
    Builds standard QP matrices for constrained velocity IK:
        minimize   0.5 * dq^T H dq + c^T dq
        subject to G dq <= h
        
    Cost:
        Tracking task error (||J dq - dx/dt||^2) regularized by ||dq||^2:
        H = 2 * (J^T J + lambda * I)
        c = -2 * J^T (dx / dt)   [or scaled to dt]
        
    Constraints:
        1. Hard joint position limits:
           q_min <= q + dq * dt <= q_max
           =>  dq <=  (q_max - q) / dt
           => -dq <= -(q_min - q) / dt
           
        2. Velocity magnitude limits (if v_max is specified):
           -v_max <= dq <= v_max
    """
    n = J.shape[1]
    
    # Scale error to velocity target: v_task = dx / dt
    v_task = dx / max(dt, 1e-4)
    
    H = 2.0 * (J.T @ J + lambda_ * np.eye(n, dtype=np.float64))
    c = -2.0 * (J.T @ v_task)
    
    # Position limit velocity bounds
    dq_max_pos = (q_max - q) / dt
    dq_min_pos = (q_min - q) / dt
    
    # Optional velocity clamping
    if v_max is not None:
        dq_max = np.minimum(dq_max_pos, v_max)
        dq_min = np.maximum(dq_min_pos, -v_max)
    else:
        dq_max = dq_max_pos
        dq_min = dq_min_pos
        
    # G dq <= h form:
    # [ I ] dq <= [  dq_max ]
    # [-I ] dq <= [ -dq_min ]
    G = np.vstack([np.eye(n, dtype=np.float64), -np.eye(n, dtype=np.float64)])
    h = np.concatenate([dq_max, -dq_min])
    
    return H, c, G, h
