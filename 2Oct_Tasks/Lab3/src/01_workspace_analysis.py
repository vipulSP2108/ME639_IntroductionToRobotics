#!/usr/bin/env python3
"""
Stage 1: Workspace Analysis and Dexterous Reachability Mapping
=============================================================
This script performs Monte Carlo sampling over the 6-DOF HEAL robot arm's
joint space to map:
1. The Task Workspace (all reachable 3D end-effector locations)
2. The Dexterous Workspace (all locations reachable with top-down gripper approach)

The resulting bounds directly inform the table design, cube spawn bounds,
and placement tray positioning.
"""

import os
import sys
import time
import numpy as np
import mujoco

# Ensure local modules can be imported
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.load_heal_robot import construct_heal_with_gripper
from utils.kinematics import get_arm_joint_and_dof_indices
from utils.workspace import (
    sample_random_configs,
    compute_tcp_positions,
    filter_dexterous,
    compute_bounding_box,
    plot_workspace
)


def run_workspace_analysis(num_samples=40000, seed=42):
    print("=" * 60)
    print("Stage 1: Workspace Analysis (Monte Carlo Sampling)")
    print("=" * 60)
    
    # 1. Load Robot Model
    print("\n[1/5] Loading HEAL robot with Robotiq 2F-85 gripper...")
    model = construct_heal_with_gripper()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    
    arm_joints, arm_dofs = get_arm_joint_and_dof_indices(model)
    print(f"  -> Arm Joints ({len(arm_joints)}): {arm_joints}")
    print(f"  -> Arm Velocity DOFs ({len(arm_dofs)}): {arm_dofs}")

    # 2. Sample random configurations
    print(f"\n[2/5] Sampling {num_samples:,} random configurations across joint limits...")
    t0 = time.time()
    configs = sample_random_configs(model, N=num_samples, seed=seed, arm_joint_ids=arm_joints)
    print(f"  -> Generated {len(configs)} configurations in {time.time() - t0:.2f}s")

    # 3. Compute TCP positions and orientations
    print("\n[3/5] Evaluating Forward Kinematics for all samples...")
    t1 = time.time()
    results = compute_tcp_positions(model, data, configs, site_name="right_center", arm_joint_ids=arm_joints)
    pos = results["positions"]
    ori = results["orientations"]
    print(f"  -> Forward kinematics computed in {time.time() - t1:.2f}s")

    # 4. Filter for Dexterous Workspace
    print("\n[4/5] Filtering for dexterous workspace (top-down approach <= 32° from -Z)...")
    dext_mask = filter_dexterous(pos, ori, angle_threshold_deg=32.0)
    dext_pos = pos[dext_mask]
    
    dext_pct = 100.0 * len(dext_pos) / len(pos)
    print(f"  -> Dexterous points: {len(dext_pos):,} / {len(pos):,} ({dext_pct:.1f}%)")

    # 5. Compute Bounding Boxes
    task_bbox = compute_bounding_box(pos)
    dext_bbox = compute_bounding_box(dext_pos)
    
    print("\n" + "=" * 60)
    print("WORKSPACE BOUNDS (in Robot Base World Frame)")
    print("=" * 60)
    print("Task Reachable Workspace:")
    print(f"  X: [{task_bbox['x'][0]:+.3f}, {task_bbox['x'][1]:+.3f}] m  (Span: {task_bbox['x'][1] - task_bbox['x'][0]:.3f} m)")
    print(f"  Y: [{task_bbox['y'][0]:+.3f}, {task_bbox['y'][1]:+.3f}] m  (Span: {task_bbox['y'][1] - task_bbox['y'][0]:.3f} m)")
    print(f"  Z: [{task_bbox['z'][0]:+.3f}, {task_bbox['z'][1]:+.3f}] m  (Span: {task_bbox['z'][1] - task_bbox['z'][0]:.3f} m)")
    
    print("\nDexterous (Top-Down Graspable) Workspace:")
    print(f"  X: [{dext_bbox['x'][0]:+.3f}, {dext_bbox['x'][1]:+.3f}] m  (Span: {dext_bbox['x'][1] - dext_bbox['x'][0]:.3f} m)")
    print(f"  Y: [{dext_bbox['y'][0]:+.3f}, {dext_bbox['y'][1]:+.3f}] m  (Span: {dext_bbox['y'][1] - dext_bbox['y'][0]:.3f} m)")
    print(f"  Z: [{dext_bbox['z'][0]:+.3f}, {dext_bbox['z'][1]:+.3f}] m  (Span: {dext_bbox['z'][1] - dext_bbox['z'][0]:.3f} m)")

    # 6. Save Plots
    plot_dir = os.path.join(PROJECT_ROOT, "analysis", "plots")
    print(f"\n[5/5] Generating publication-quality workspace plots in {plot_dir}...")
    plot_workspace(pos, dext_pos, save_dir=plot_dir)
    
    print("\n✓ Stage 1 Complete!")
    return {
        "task_bbox": task_bbox,
        "dext_bbox": dext_bbox,
        "dext_count": len(dext_pos),
        "total_count": len(pos)
    }


if __name__ == "__main__":
    run_workspace_analysis()
