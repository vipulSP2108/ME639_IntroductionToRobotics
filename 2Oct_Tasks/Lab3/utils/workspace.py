#!/usr/bin/env python3
"""
Workspace Analysis Utilities
============================
Monte Carlo sampling, kinematic reachability mapping, dexterous workspace filtering,
bounding box calculation, and publication-quality Matplotlib visualizations.
"""

import os
import numpy as np
import mujoco
import matplotlib
# Use non-interactive backend for headless / script execution
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from utils.kinematics import get_arm_joint_and_dof_indices


def sample_random_configs(model, N=50000, seed=42, arm_joint_ids=None):
    """
    Samples N random joint configurations uniformly within hardware limits.
    
    Args:
        model: compiled MjModel
        N: number of random samples
        seed: RNG seed for reproducibility
        arm_joint_ids: joint indices to sample. If None, detected automatically.
        
    Returns:
        configs: np.ndarray of shape (N, len(arm_joint_ids))
    """
    if arm_joint_ids is None:
        arm_joint_ids, _ = get_arm_joint_and_dof_indices(model)
        
    rng = np.random.default_rng(seed)
    num_arm_joints = len(arm_joint_ids)
    
    q_min = model.jnt_range[arm_joint_ids, 0]
    q_max = model.jnt_range[arm_joint_ids, 1]
    
    configs = rng.uniform(q_min, q_max, size=(N, num_arm_joints))
    return configs


def compute_tcp_positions(model, data, configs, site_name="right_center", arm_joint_ids=None):
    """
    Runs forward kinematics for each joint configuration and records TCP Cartesian
    positions and orientation matrices.
    
    Returns:
        dict with:
            "positions": np.ndarray of shape (N, 3) [x, y, z]
            "orientations": np.ndarray of shape (N, 9) (flattened 3x3 rotation matrices)
    """
    if arm_joint_ids is None:
        arm_joint_ids, _ = get_arm_joint_and_dof_indices(model)
        
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id < 0:
        raise ValueError(f"Site '{site_name}' not found in model!")
        
    N = len(configs)
    positions = np.zeros((N, 3), dtype=np.float64)
    orientations = np.zeros((N, 9), dtype=np.float64)
    
    # Save original qpos state
    orig_qpos = data.qpos.copy()
    
    for i in range(N):
        data.qpos[arm_joint_ids] = configs[i]
        mujoco.mj_forward(model, data)
        positions[i] = data.site_xpos[site_id]
        orientations[i] = data.site_xmat[site_id]
        
    # Restore original state
    data.qpos[:] = orig_qpos
    mujoco.mj_forward(model, data)
    
    return {
        "positions": positions,
        "orientations": orientations
    }


def filter_dexterous(positions, orientations, angle_threshold_deg=32.0):
    """
    Filters for the 'dexterous workspace' where the end-effector approach axis
    (local Z-axis, pointing out from the wrist toward the fingers) points downward
    within `angle_threshold_deg` of the negative world Z-axis [0, 0, -1].
    
    The local Z-axis in world frame is the 3rd column of the rotation matrix R:
        z_world = [R[0, 2], R[1, 2], R[2, 2]]
    Downward alignment condition:
        z_world[2] <= -cos(angle_threshold_deg)
        
    Returns:
        mask: boolean np.ndarray of shape (N,)
    """
    cos_thresh = np.cos(np.radians(angle_threshold_deg))
    # In row-major flattened 3x3 R:
    # R[0, 2] is index 2, R[1, 2] is index 5, R[2, 2] is index 8
    z_axis_world_z = orientations[:, 8]
    
    # Approach axis points downwards
    mask = z_axis_world_z <= -cos_thresh
    return mask


def compute_bounding_box(points):
    """
    Computes axis-aligned bounding box (min and max along X, Y, Z).
    """
    if len(points) == 0:
        return {"x": [0.0, 0.0], "y": [0.0, 0.0], "z": [0.0, 0.0]}
        
    return {
        "x": [float(np.min(points[:, 0])), float(np.max(points[:, 0]))],
        "y": [float(np.min(points[:, 1])), float(np.max(points[:, 1]))],
        "z": [float(np.min(points[:, 2])), float(np.max(points[:, 2]))],
    }


def plot_workspace(task_pts, dext_pts, save_dir="analysis/plots"):
    """
    Generates and saves three publication-quality workspace visualization plots:
    1. Top-down XY projection (manipulation footprint on table plane)
    2. Side-view XZ projection (vertical reach profile)
    3. 3D scatter plot (Task vs. Dexterous workspace comparison)
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Subsample points if point cloud is large (for snappy rendering and clean vector-like rasterization)
    max_plot_pts = 15000
    if len(task_pts) > max_plot_pts:
        idx_t = np.random.choice(len(task_pts), max_plot_pts, replace=False)
        plot_task = task_pts[idx_t]
    else:
        plot_task = task_pts
        
    if len(dext_pts) > max_plot_pts:
        idx_d = np.random.choice(len(dext_pts), max_plot_pts, replace=False)
        plot_dext = dext_pts[idx_d]
    else:
        plot_dext = dext_pts

    # -------------------------------------------------------------
    # 1. Top-Down XY Projection
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 7), dpi=150)
    ax.scatter(plot_task[:, 0], plot_task[:, 1], s=2, c='#4A90E2', alpha=0.3, label='Task Reachable (Any Orientation)')
    ax.scatter(plot_dext[:, 0], plot_dext[:, 1], s=4, c='#E65100', alpha=0.6, label='Dexterous (Top-Down Graspable, <32°)')
    ax.scatter([0], [0], s=80, c='black', marker='^', label='Robot Base')
    
    ax.set_title("Robot Arm Workspace: Top-Down (X-Y) Footprint", fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel("X (forward) [m]", fontsize=12)
    ax.set_ylabel("Y (lateral) [m]", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.axis('equal')
    ax.legend(loc='upper right', framealpha=0.9)
    plt.tight_layout()
    xy_path = os.path.join(save_dir, "workspace_xy.png")
    plt.savefig(xy_path)
    plt.close(fig)

    # -------------------------------------------------------------
    # 2. Side-View XZ Projection
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    ax.scatter(plot_task[:, 0], plot_task[:, 2], s=2, c='#4A90E2', alpha=0.3, label='Task Reachable')
    ax.scatter(plot_dext[:, 0], plot_dext[:, 2], s=4, c='#E65100', alpha=0.6, label='Dexterous (Top-Down)')
    ax.scatter([0], [0], s=80, c='black', marker='^', label='Robot Base')
    
    ax.set_title("Robot Arm Workspace: Side Elevation (X-Z Reach)", fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel("X (forward) [m]", fontsize=12)
    ax.set_ylabel("Z (height) [m]", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.axis('equal')
    ax.legend(loc='upper right', framealpha=0.9)
    plt.tight_layout()
    xz_path = os.path.join(save_dir, "workspace_xz.png")
    plt.savefig(xz_path)
    plt.close(fig)

    # -------------------------------------------------------------
    # 3. 3D Isometric View
    # -------------------------------------------------------------
    fig = plt.figure(figsize=(9, 8), dpi=150)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(plot_task[:, 0], plot_task[:, 1], plot_task[:, 2], s=1.5, c='#4A90E2', alpha=0.15, label='Task Reachable')
    ax.scatter(plot_dext[:, 0], plot_dext[:, 1], plot_dext[:, 2], s=3, c='#E65100', alpha=0.45, label='Dexterous (Top-Down)')
    ax.scatter([0], [0], [0], s=100, c='black', marker='^', label='Base')
    
    ax.set_title("3D Workspace Envelope: Task vs. Dexterous", fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel("X [m]", fontsize=11)
    ax.set_ylabel("Y [m]", fontsize=11)
    ax.set_zlabel("Z [m]", fontsize=11)
    ax.legend(loc='upper right', framealpha=0.9)
    plt.tight_layout()
    d3_path = os.path.join(save_dir, "workspace_3d.png")
    plt.savefig(d3_path)
    plt.close(fig)

    print(f"Workspace plots successfully saved to: {save_dir}")
    return xy_path, xz_path, d3_path
