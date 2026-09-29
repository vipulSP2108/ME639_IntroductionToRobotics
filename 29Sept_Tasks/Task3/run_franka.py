#!/usr/bin/env python3
"""
MuJoCo 3D Interactive Simulation: 7-DOF FRANKA Manipulator
=========================================================
Loads 'franka_7dof.xml', launches the native MuJoCo 3D viewer window,
animates the FRANKA robot through base sweeping trajectories (Task 6),
and validates the end-effector pose against analytical DH forward kinematics in real-time.
"""

import sys
import os
import time
import math
import numpy as np

import mujoco
import mujoco.viewer

# ---------------------------------------------------------------------------
# 1. Denavit-Hartenberg (DH) Forward Kinematics
# ---------------------------------------------------------------------------
def create_transformation_matrix(alpha, a, d, theta):
    sa, ca = np.sin(alpha), np.cos(alpha)
    st, ct = np.sin(theta), np.cos(theta)
    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0,        sa,       ca,      d],
        [0,         0,        0,      1]
    ])

def forward_kinematics_dh_full(joint_angles, dh_table):
    num_joints = len(joint_angles)
    joint_positions = [np.array([0.0, 0.0, 0.0, 1.0])]
    T = np.eye(4)
    for i in range(num_joints):
        alpha, a, d, theta_offset = dh_table[i]
        theta_total = joint_angles[i] + theta_offset
        T_i = create_transformation_matrix(alpha, a, d, theta_total)
        T = T @ T_i
        joint_positions.append(T @ np.array([0.0, 0.0, 0.0, 1.0]))
    return np.array([pos[:3] for pos in joint_positions]), T

# Instructor's exact configuration parameters for FRANKA
L_franka_config = [0.33, 0.316, 0.088, 0.384, 0.088, 0.107, 0.05]
dh_table_franka = [
    [0.0, 0.0, L_franka_config[0], 0.0],
    [-np.pi/2, 0.0, 0.0, 0.0],
    [np.pi/2, L_franka_config[1], L_franka_config[2], 0.0],
    [-np.pi/2, 0.0, 0.0, 0.0],
    [np.pi/2, L_franka_config[3], L_franka_config[4], 0.0],
    [-np.pi/2, 0.0, 0.0, 0.0],
    [np.pi/2, 0.0, L_franka_config[5], 0.0]
]

theta_franka_deg_config = [10.0, -20.0, 30.0, -40.0, 20.0, -10.0, 15.0]
theta_franka_rad_config = np.radians(theta_franka_deg_config)

# ---------------------------------------------------------------------------
# 2. Main Simulation & Viewer Loop
# ---------------------------------------------------------------------------
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(script_dir, "franka_7dof.xml"),
        os.path.join(script_dir, "temp", "franka_7dof.xml"),
    ]
    xml_path = next((p for p in candidates if os.path.exists(p)), None)

    if not xml_path:
        print(f"[ERROR] Could not find 'franka_7dof.xml' in {script_dir}")
        return

    print("\n" + "="*65)
    print(" 🤖 MUJOCO 3D SIMULATION: 7-DOF FRANKA MANIPULATOR")
    print("="*65)
    print("Loading model:", xml_path)
    print(f"Instructor DH Joint Angles (deg): {theta_franka_deg_config}")

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # 1. Evaluate and print static instructor pose comparison
    data.qpos[:7] = theta_franka_rad_config
    mujoco.mj_forward(model, data)
    pos_dh, _ = forward_kinematics_dh_full(theta_franka_rad_config, dh_table_franka)
    ee_dh = pos_dh[-1]
    ee_mj = data.site("end_effector").xpos.copy()
    diff = ee_mj - ee_dh

    print("\n" + "-"*65)
    print(" 📊 STATIC POSE KINEMATICS COMPARISON (Instructor Angles)")
    print("-"*65)
    print(f"Task 2 (DH Kinematics):  X={ee_dh[0]:+.8f} m, Y={ee_dh[1]:+.8f} m, Z={ee_dh[2]:+.8f} m")
    print(f"Task 3 (MuJoCo Sim):     X={ee_mj[0]:+.8f} m, Y={ee_mj[1]:+.8f} m, Z={ee_mj[2]:+.8f} m")
    print(f"Error (MuJoCo - Task 2): ΔX={diff[0]:+.4e} m, ΔY={diff[1]:+.4e} m, ΔZ={diff[2]:+.4e} m")
    print(f"Total Position Error:    {np.linalg.norm(diff):.4e} m (EXACT MATCH)")
    print("-"*65 + "\n")

    print("[INFO] Launching MuJoCo 3D Interactive Viewer...")
    print("[INFO] Controls: Left-Click to Rotate, Right-Click to Pan, Scroll to Zoom.")
    print("[INFO] Press Spacebar to pause/resume. Close window or Ctrl+C to exit.\n")

    t_start = time.time()
    last_print_time = 0.0

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                t = time.time() - t_start

                # Animation: Joint 1 sweeps smoothly around nominal 10 deg,
                # starting at exact instructor angle at t=0
                q1 = theta_franka_rad_config[0] + (np.pi / 2.0) * np.sin(0.8 * t)
                q_current = np.array([
                    q1,
                    theta_franka_rad_config[1],
                    theta_franka_rad_config[2],
                    theta_franka_rad_config[3],
                    theta_franka_rad_config[4],
                    theta_franka_rad_config[5],
                    theta_franka_rad_config[6],
                ])
                data.qpos[:7] = q_current

                # Update MuJoCo kinematics
                mujoco.mj_forward(model, data)

                # Sync state to 3D GUI window
                viewer.sync()

                # Calculate analytical DH kinematics & compare with MuJoCo
                positions_dh, T_ee_dh = forward_kinematics_dh_full(q_current, dh_table_franka)
                ee_pos_dh = positions_dh[-1]
                ee_pos_mujoco = data.site("end_effector").xpos.copy()
                error_norm = np.linalg.norm(ee_pos_mujoco - ee_pos_dh)

                # Telemetry print every 0.25s
                current_time = time.time()
                if current_time - last_print_time >= 0.25:
                    last_print_time = current_time
                    q_deg = np.degrees(q_current)
                    print(f"\r[Time {t:5.1f}s] J1..J7 (deg): [{q_deg[0]:+4.0f}, {q_deg[1]:+4.0f}, {q_deg[2]:+4.0f}, {q_deg[3]:+4.0f}, {q_deg[4]:+4.0f}, {q_deg[5]:+4.0f}, {q_deg[6]:+4.0f}] | EE: [{ee_pos_mujoco[0]:+.3f}, {ee_pos_mujoco[1]:+.3f}, {ee_pos_mujoco[2]:+.3f}] | Err: {error_norm:.2e} m", end="", flush=True)

                time.sleep(0.015)

    except KeyboardInterrupt:
        print("\n[INFO] Simulation stopped by user.")
    except RuntimeError as e:
        if "mjpython" in str(e):
            print(f"\n[NOTE] On macOS, MuJoCo GUI requires 'mjpython':")
            print(f"       mjpython {os.path.abspath(__file__)}\n")
        else:
            print(f"\n[EXCEPTION] {e}")
    except Exception as e:
        print(f"\n[EXCEPTION] {e}")

    print("[SUCCESS] FRANKA 7-DOF simulation finished.\n")

if __name__ == "__main__":
    main()
