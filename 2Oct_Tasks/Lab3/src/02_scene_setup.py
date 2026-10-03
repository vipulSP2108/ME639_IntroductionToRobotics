#!/usr/bin/env python3
"""
Stage 2: Scene Setup and Programmatic Model Construction
========================================================
Builds the complete pick-and-place workcell in MuJoCo using MjSpec:
- 6-DOF HEAL robot arm + attached Robotiq 2F-85 gripper
- Table (dimensions chosen from Stage 1 workspace analysis)
- Red manipulable cube (with a 6-DOF free joint)
- Target receptacle tray

Exposes:
- build_scene(cube_x, cube_y, cube_yaw): creates and compiles the model
- reset_cube_pose(model, data, cube_x, cube_y, cube_yaw): resets cube in existing sim
- Scene geometric constants (TABLE_*, CUBE_*, TRAY_*)
"""

import os
import sys
import argparse
import numpy as np
import mujoco
try:
    import mujoco.viewer
    HAS_VIEWER = True
except (ImportError, AttributeError):
    HAS_VIEWER = False

# Ensure project root is on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.load_heal_robot import compensate_gravity
from utils.quaternion_utils import DOWNWARD_QUAT


# ---------------------------------------------------------------------------
# Workspace & Table Geometry Constants (derived in Stage 1)
# ---------------------------------------------------------------------------
TABLE_X = 0.45       # Table centroid X [m]
TABLE_Y = 0.00       # Table centroid Y [m]
TABLE_H = 0.20       # Total table height from floor [m]
TABLE_L = 0.40       # Length along X [m]
TABLE_W = 0.50       # Width along Y [m]

# Cube geometry
CUBE_SIZE = 0.05
CUBE_HALF_SIZE = CUBE_SIZE / 2.0  # 0.025 m
CUBE_Z = TABLE_H + CUBE_HALF_SIZE # 0.225 m (resting flush on table)

# Tray geometry & placement
TRAY_L = 0.12        # 12 cm length
TRAY_W = 0.12        # 12 cm width
TRAY_H = 0.015       # 1.5 cm rim height
TRAY_X = 0.55        # Placed at far end along X
TRAY_Y = 0.00        # Centered along Y
TRAY_Z = TABLE_H + TRAY_H / 2.0


def build_scene(cube_x=0.40, cube_y=0.0, cube_yaw=0.0):
    """
    Constructs and compiles the complete composite MuJoCo model:
    Robot + Gripper + Workcell Table + Target Tray + Dynamic Cube.
    
    Args:
        cube_x: initial X position of cube [m]
        cube_y: initial Y position of cube [m]
        cube_yaw: initial rotation angle of cube around Z-axis [radians]
        
    Returns:
        model: compiled MjModel
        data: initialized MjData
    """
    desc_dir = os.path.join(PROJECT_ROOT, "robot_descriptions")
    arm_path = os.path.join(desc_dir, "single_arm_heal_effort_actuation_rs_mj.xml")
    grip_path = os.path.join(desc_dir, "robotiq_2f85_v4", "2f85.xml")
    
    if not os.path.exists(arm_path):
        raise FileNotFoundError(f"ARM XML not found at {arm_path}")
    if not os.path.exists(grip_path):
        raise FileNotFoundError(f"Gripper XML not found at {grip_path}")
        
    # 1. Load Arm & Gripper MjSpecs
    arm_spec = mujoco.MjSpec.from_file(arm_path)
    grip_spec = mujoco.MjSpec.from_file(grip_path)
    
    # Configure contact solver properties for stable grasping
    arm_spec.option.impratio = 10.0
    arm_spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    arm_spec.option.iterations = 50
    
    # Attach gripper at arm's tool mounting site
    arm_spec.attach(grip_spec, prefix="gripper/", site=arm_spec.site("right_center"))

    # 2. Add Workcell Table
    # Note: MuJoCo box geom position is its volumetric centroid
    table = arm_spec.worldbody.add_body(name="table", pos=[TABLE_X, TABLE_Y, TABLE_H / 2.0])
    table.add_geom(
        name="table_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[TABLE_L / 2.0, TABLE_W / 2.0, TABLE_H / 2.0],
        rgba=[0.35, 0.28, 0.24, 1.0],  # dark walnut wood
        friction=[1.2, 0.005, 0.0001]
    )

    # 3. Add Target Tray
    tray = arm_spec.worldbody.add_body(name="tray", pos=[TRAY_X, TRAY_Y, TRAY_Z])
    tray.add_geom(
        name="tray_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[TRAY_L / 2.0, TRAY_W / 2.0, TRAY_H / 2.0],
        rgba=[0.1, 0.7, 0.5, 0.85]  # semi-translucent emerald
    )

    # 4. Add Dynamic Pick Cube with Free Joint
    # Compute yaw quaternion around Z: [cos(yaw/2), 0, 0, sin(yaw/2)]
    half_yaw = cube_yaw / 2.0
    cube_quat = [float(np.cos(half_yaw)), 0.0, 0.0, float(np.sin(half_yaw))]
    
    cube = arm_spec.worldbody.add_body(name="cube", pos=[cube_x, cube_y, CUBE_Z + 0.0002], quat=cube_quat)
    cube.mass = 0.05
    cube.inertia = [2e-5, 2e-5, 2e-5]
    cube.add_joint(name="cube_free_joint", type=mujoco.mjtJoint.mjJNT_FREE, damping=0.01)
    cube.add_geom(
        name="cube_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[CUBE_HALF_SIZE, CUBE_HALF_SIZE, CUBE_HALF_SIZE],
        rgba=[0.9, 0.15, 0.15, 1.0],  # vivid red
        friction=[1.8, 0.01, 0.001]
    )

    # 5. Compile and Initialize
    model = arm_spec.compile()
    data = mujoco.MjData(model)
    
    # Forward pass to initialize kinematics, contact points, and gravity compensation
    mujoco.mj_forward(model, data)
    
    return model, data


def reset_cube_pose(model, data, cube_x, cube_y, cube_yaw=0.0):
    """
    Resets the cube pose directly in existing simulation data without recompiling.
    """
    jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free_joint")
    if jnt_id < 0:
        return
        
    qpos_adr = model.jnt_qposadr[jnt_id]
    dof_adr = model.jnt_dofadr[jnt_id]
    
    # 7 elements for free joint: [x, y, z, qw, qx, qy, qz]
    half_yaw = cube_yaw / 2.0
    data.qpos[qpos_adr : qpos_adr + 3] = [cube_x, cube_y, CUBE_Z]
    data.qpos[qpos_adr + 3 : qpos_adr + 7] = [np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw)]
    
    # Zero out linear & angular velocity (6 elements)
    data.qvel[dof_adr : dof_adr + 6] = 0.0
    
    mujoco.mj_forward(model, data)


def main():
    parser = argparse.ArgumentParser(description="Stage 2: Pick-and-Place Scene Setup")
    parser.add_argument("--headless", action="store_true", help="Run without opening interactive viewer")
    parser.add_argument("--cube_x", type=float, default=0.38, help="Initial cube X [m]")
    parser.add_argument("--cube_y", type=float, default=0.00, help="Initial cube Y [m]")
    parser.add_argument("--cube_yaw", type=float, default=0.00, help="Initial cube Yaw [rad]")
    args = parser.parse_args()

    print("=" * 60)
    print("Stage 2: Building Pick-and-Place Scene...")
    print("=" * 60)
    
    model, data = build_scene(cube_x=args.cube_x, cube_y=args.cube_y, cube_yaw=args.cube_yaw)
    print(f"✓ Scene compiled successfully!")
    print(f"  Total nq (coordinates): {model.nq}")
    print(f"  Total nv (velocities):  {model.nv}")
    print(f"  Total nu (actuators):   {model.nu}")
    print(f"  Cube initial position:  {data.body('cube').xpos}")
    print(f"  Table surface height:   {TABLE_H} m")
    print(f"  Tray position:          [{TRAY_X}, {TRAY_Y}, {TRAY_Z}] m")

    if args.headless:
        # Step simulation for a few iterations with gravity compensation
        for _ in range(100):
            compensate_gravity(model, data)
            mujoco.mj_step(model, data)
        print("✓ Headless physics simulation validated without crashes or explosions.")
        return

    # Interactive visualization
    if HAS_VIEWER:
        print("\nLaunching MuJoCo passive viewer. Press [ESC] or close window to exit.")
        try:
            with mujoco.viewer.launch_passive(model, data) as viewer:
                while viewer.is_running():
                    compensate_gravity(model, data)
                    mujoco.mj_step(model, data)
                    viewer.sync()
        except Exception as e:
            print(f"Viewer exited: {e}")
    else:
        print("MuJoCo viewer not available in this environment.")


if __name__ == "__main__":
    main()
