import time
import sys
import os
import math
import numpy as np
import json
import mujoco
import mujoco.viewer

def generate_turtlebot_xml(config):
    stable = config.get("enable_stable_physics", False)
    
    timestep = "0.005" if stable else "0.001"
    armature = "0.01" if stable else "0.001"
    wheel_geom = '<geom type="cylinder" size="0.033 0.01" quat="0.7071 0.7071 0 0" rgba="0.1 0.1 0.1 1" mass="0.1"/>' if stable else '<geom type="sphere" size="0.033" rgba="0.1 0.1 0.1 1" mass="0.1"/>'
    caster_friction = "0.01 0.001 0.0001" if stable else "0 0 0"
    actuator = '<velocity joint="left_wheel_joint" kv="1" ctrlrange="-10 10"/>\n    <velocity joint="right_wheel_joint" kv="1" ctrlrange="-10 10"/>' if stable else '<velocity joint="left_wheel_joint" kv="10" forcerange="-2.0 2.0"/>\n    <velocity joint="right_wheel_joint" kv="10" forcerange="-2.0 2.0"/>'

    return f"""
<mujoco model="turtlebot3_waffle_pi">
  <compiler angle="degree" coordinate="local"/>
  <option timestep="{timestep}" gravity="0 0 -9.81"/>

  <visual>
    <headlight ambient="0.4 0.4 0.4" diffuse="0.8 0.8 0.8"/>
  </visual>

  <asset>
    <texture name="grid" type="2d" builtin="checker" width="512" height="512" rgb1=".1 .2 .3" rgb2=".2 .3 .4"/>
    <material name="grid_mat" texture="grid" texrepeat="10 10" texuniform="true" reflectance="0.2"/>
  </asset>

  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <!-- Ground Plane with Checkerboard for motion visibility -->
    <geom name="floor" type="plane" size="10 10 0.1" material="grid_mat"/>

    <!-- World Frame Axes (Static at Origin) -->
    <geom name="world_axis_x" type="cylinder" size="0.005 0.2" pos="0.2 0 0.0" quat="0.7071 0 0.7071 0" rgba="1 0 0 1" contype="0" conaffinity="0"/>
    <geom name="world_axis_y" type="cylinder" size="0.005 0.2" pos="0 0.2 0.0" quat="0.7071 -0.7071 0 0" rgba="0 1 0 1" contype="0" conaffinity="0"/>
    <geom name="world_axis_z" type="cylinder" size="0.005 0.2" pos="0 0 0.2" rgba="0 0 1 1" contype="0" conaffinity="0"/>

    <!-- Ghost Robot (Static Reference at Origin to show starting point) -->
    <body name="ghost_bot" pos="0 0 0.033">
      <geom type="box" size="0.14 0.15 0.04" pos="-0.03 0 0.04" rgba="0.25 0.25 0.25 0.4" contype="0" conaffinity="0"/>
      <geom type="box" size="0.03 0.03 0.02" pos="0.06 0 0.10" rgba="0.8 0.1 0.1 0.4" contype="0" conaffinity="0"/>
      <geom type="cylinder" size="0.035 0.015" pos="-0.03 0 0.10" rgba="0.1 0.1 0.8 0.4" contype="0" conaffinity="0"/>
      <geom type="cylinder" size="0.033 0.01" pos="0 0.14 0" quat="0.7071 0.7071 0 0" rgba="0.1 0.1 0.1 0.4" contype="0" conaffinity="0"/>
      <geom type="cylinder" size="0.033 0.01" pos="0 -0.14 0" quat="0.7071 0.7071 0 0" rgba="0.1 0.1 0.1 0.4" contype="0" conaffinity="0"/>
    </body>

    <!-- TurtleBot3 Waffle Pi Body -->
    <body name="base_link" pos="0 0 0.033">
      <freejoint name="root"/>
      
      <!-- Main Chassis Box -->
      <geom name="chassis" type="box" size="0.14 0.15 0.04" pos="-0.03 0 0.04" rgba="0.25 0.25 0.25 1" mass="1.5" contype="0" conaffinity="0"/>
      
      <!-- Raspberry Pi / Camera Tower -->
      <geom name="pi_camera" type="box" size="0.03 0.03 0.02" pos="0.06 0 0.10" rgba="0.8 0.1 0.1 1" mass="0.1" contype="0" conaffinity="0"/>
      <!-- LiDAR sensor dome -->
      <geom name="lidar" type="cylinder" size="0.035 0.015" pos="-0.03 0 0.10" rgba="0.1 0.1 0.8 1" mass="0.2" contype="0" conaffinity="0"/>

      <!-- Visual Body Frame Axes (X: Red, Y: Green, Z: Blue) attached directly to Base -->
      <geom name="axis_x" type="cylinder" size="0.005 0.1" pos="0.1 0 0.07" quat="0.7071 0 0.7071 0" rgba="1 0 0 1" contype="0" conaffinity="0"/>
      <geom name="axis_y" type="cylinder" size="0.005 0.1" pos="0 0.1 0.07" quat="0.7071 -0.7071 0 0" rgba="0 1 0 1" contype="0" conaffinity="0"/>
      <geom name="axis_z" type="cylinder" size="0.005 0.1" pos="0 0 0.17" rgba="0 0 1 1" contype="0" conaffinity="0"/>

      <!-- Left Wheel -->
      <body name="wheel_left" pos="0 0.14 0">
        <joint name="left_wheel_joint" type="hinge" axis="0 1 0" damping="0.001" armature="{armature}"/>
        {wheel_geom}
      </body>

      <!-- Right Wheel -->
      <body name="wheel_right" pos="0 -0.14 0">
        <joint name="right_wheel_joint" type="hinge" axis="0 1 0" damping="0.001" armature="{armature}"/>
        {wheel_geom}
      </body>

      <!-- Front Caster Wheel (Radius perfectly matching 0.033 wheel height) -->
      <body name="caster_front" pos="0.08 0 -0.023">
        <geom type="sphere" size="0.010" rgba="0.5 0.5 0.5 1" mass="0.05" contype="1" conaffinity="1" friction="{caster_friction}"/>
      </body>

      <!-- Back Caster Wheel -->
      <body name="caster_back" pos="-0.14 0 -0.023">
        <geom type="sphere" size="0.010" rgba="0.5 0.5 0.5 1" mass="0.05" contype="1" conaffinity="1" friction="{caster_friction}"/>
      </body>
    </body>
  </worldbody>

  <actuator>
    {actuator}
  </actuator>
</mujoco>
"""

def quaternion_to_rotation_matrix(q):
    """
    Converts MuJoCo quaternion [w, x, y, z] to 3x3 Rotation Matrix R.
    """
    w, x, y, z = q[0], q[1], q[2], q[3]

    r00 = 1 - 2 * (y**2 + z**2)
    r01 = 2 * (x * y - w * z)
    r02 = 2 * (x * z + w * y)

    r10 = 2 * (x * y + w * z)
    r11 = 1 - 2 * (x**2 + z**2)
    r12 = 2 * (y * z - w * x)

    r20 = 2 * (x * z - w * y)
    r21 = 2 * (y * z + w * x)
    r22 = 1 - 2 * (x**2 + y**2)

    return np.array([
        [r00, r01, r02],
        [r10, r11, r12],
        [r20, r21, r22]
    ])

def main():
    config_path = os.path.join(os.path.dirname(__file__), 'config.json')
    try:
        with open(config_path, 'r') as f:
            config = json.load(f).get('turtlebot', {})
    except Exception as e:
        print(f"Failed to load config.json, using defaults: {e}")
        config = {}

    xml_string = generate_turtlebot_xml(config)
    model = mujoco.MjModel.from_xml_string(xml_string)
    data = mujoco.MjData(model)

    wheel_radius = config.get("wheel_radius", 0.033)
    wheel_separation = config.get("wheel_separation", 0.28)
    linear_speed_step = config.get("linear_speed_step", 0.05)
    angular_speed_step = config.get("angular_speed_step", 0.2)
    max_linear_speed = config.get("max_linear_speed", 0.4)
    max_angular_speed = config.get("max_angular_speed", 1.5)

    v_lin = 0.0
    v_ang = 0.0

    print("=" * 60)
    print("🤖 CHALLENGE 1: MuJoCo TurtleBot3 Waffle Pi Teleop")
    print("=" * 60)
    print("Controls:")
    print("  W / S : Linear Speed Forward / Backward")
    print("  A / D : Turn Left / Right")
    print("  SPACE : Stop")
    print("  Q     : Quit Simulation")
    print("=" * 60)

    try:
        import termios
        import tty
        import select
        
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        tty.setcbreak(sys.stdin.fileno())
        interactive_terminal = True
    except Exception:
        interactive_terminal = False

    def get_key():
        if interactive_terminal:
            if select.select([sys.stdin], [], [], 0)[0]:
                return sys.stdin.read(1)
        return None

    step_counter = 0
    last_print_time = time.time()

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            # Set camera position to view robot nicely
            viewer.cam.distance = 2.0
            viewer.cam.elevation = -20
            viewer.cam.lookat[:] = [0, 0, 0]

            while viewer.is_running():
                key = get_key()
                key_pressed = False
                if key:
                    key = key.lower()
                    if key == 'w':
                        v_lin += linear_speed_step
                        key_pressed = True
                    elif key == 's':
                        v_lin -= linear_speed_step
                        key_pressed = True
                    elif key == 'a':
                        v_ang += angular_speed_step
                        key_pressed = True
                    elif key == 'd':
                        v_ang -= angular_speed_step
                        key_pressed = True
                    elif key == ' ':
                        v_lin = 0.0
                        v_ang = 0.0
                        key_pressed = True
                    elif key == 'q':
                        break

                v_lin = np.clip(v_lin, -max_linear_speed, max_linear_speed)
                v_ang = np.clip(v_ang, -max_angular_speed, max_angular_speed)

                w_left = (v_lin - (v_ang * wheel_separation / 2.0)) / wheel_radius
                w_right = (v_lin + (v_ang * wheel_separation / 2.0)) / wheel_radius

                data.ctrl[0] = w_left
                data.ctrl[1] = w_right

                mujoco.mj_step(model, data)
                viewer.sync()

                current_time = time.time()
                # Print whenever a key is pressed OR every 0.2s for steady updates
                if key_pressed or (current_time - last_print_time >= 0.2):
                    last_print_time = current_time
                    pos = data.qpos[0:3]
                    quat = data.qpos[3:7]
                    R = quaternion_to_rotation_matrix(quat)

                    # Get actual velocity from physics engine (local frame)
                    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'base_link')
                    actual_vel = np.zeros(6)
                    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, base_id, actual_vel, 1)
                    actual_ang = actual_vel[2] # rot_z
                    actual_lin = actual_vel[3] # lin_x

                    sys.stdout.write("\033[H\033[J")
                    print("=" * 60)
                    print("🤖 TURTLEBOT3 WAFFLE PI TELEMETRY")
                    print("-" * 60)
                    print(f"Commanded | Linear: {v_lin:+.2f} m/s | Angular: {v_ang:+.2f} rad/s")
                    print(f"Actual    | Linear: {actual_lin:+.2f} m/s | Angular: {actual_ang:+.2f} rad/s")
                    print("=" * 60)
                    print(f"World Position (x, y, z): [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
                    print("\n3x3 Body Frame Rotation Matrix (R):")
                    print(f"[{R[0,0]:8.4f}  {R[0,1]:8.4f}  {R[0,2]:8.4f}]")
                    print(f"[{R[1,0]:8.4f}  {R[1,1]:8.4f}  {R[1,2]:8.4f}]")
                    print(f"[{R[2,0]:8.4f}  {R[2,1]:8.4f}  {R[2,2]:8.4f}]")
                    print("=" * 60)
                    print("Controls: W/S (Speed), A/D (Turn), SPACE (Stop), Q (Quit)")
                    sys.stdout.flush()

                time.sleep(0.01)
    except RuntimeError as e:
        if "mjpython" in str(e):
            print("\n[NOTE] On macOS GUI, run using 'mjpython' command:")
            print(f"       mjpython {os.path.abspath(__file__)}")
        else:
            raise e
    finally:
        if interactive_terminal:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

if __name__ == '__main__':
    main()
