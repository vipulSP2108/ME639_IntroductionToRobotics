import time
import sys
import os
import math
import numpy as np
import mujoco
import mujoco.viewer

# Path to Skydio X2 model downloaded from MuJoCo Menagerie
MENAGERIE_X2_PATH = "/Users/aai/Desktop/ROS/mujoco_menagerie/skydio_x2/scene.xml"
FALLBACK_CRAZYFLIE_PATH = "/Users/aai/Desktop/ROS/mujoco_menagerie/bitcraze_crazyflie_2/scene.xml"

import json

def generate_quadrotor_xml(base_xml_path):
    # If base_xml_path is provided, we include it. Otherwise, we define a standalone drone.
    if base_xml_path:
        drone_include = f'<include file="{base_xml_path}"/>'
    else:
        drone_include = """
    <!-- Quadrotor Body (Fallback) -->
    <body name="quadrotor" pos="0 0 1.0">
      <freejoint name="root"/>
      <geom name="chassis" type="box" size="0.1 0.08 0.03" rgba="0.2 0.2 0.2 1"/>
      <geom name="camera_gimbal" type="sphere" size="0.025" pos="0.1 0 -0.01" rgba="0.1 0.7 0.9 1"/>
      <!-- Visual Body Frame Axes -->
      <geom name="axis_x" type="cylinder" size="0.005 0.12" pos="0.12 0 0" quat="0.7071 0 0.7071 0" rgba="1 0 0 1" contype="0" conaffinity="0"/>
      <geom name="axis_y" type="cylinder" size="0.005 0.12" pos="0 0.12 0" quat="0.7071 -0.7071 0 0" rgba="0 1 0 1" contype="0" conaffinity="0"/>
      <geom name="axis_z" type="cylinder" size="0.005 0.12" pos="0 0 0.12" rgba="0 0 1 1" contype="0" conaffinity="0"/>
      <geom type="capsule" fromto="0 0 0 0.15 -0.15 0" size="0.008" rgba="0.4 0.4 0.4 1"/>
      <geom type="cylinder" size="0.07 0.002" pos="0.15 -0.15 0.01" rgba="0.9 0.1 0.1 0.7"/>
      <geom type="capsule" fromto="0 0 0 0.15 0.15 0" size="0.008" rgba="0.4 0.4 0.4 1"/>
      <geom type="cylinder" size="0.07 0.002" pos="0.15 0.15 0.01" rgba="0.9 0.1 0.1 0.7"/>
      <geom type="capsule" fromto="0 0 0 -0.15 0.15 0" size="0.008" rgba="0.4 0.4 0.4 1"/>
      <geom type="cylinder" size="0.07 0.002" pos="-0.15 0.15 0.01" rgba="0.1 0.1 0.9 0.7"/>
      <geom type="capsule" fromto="0 0 0 -0.15 -0.15 0" size="0.008" rgba="0.4 0.4 0.4 1"/>
      <geom type="cylinder" size="0.07 0.002" pos="-0.15 -0.15 0.01" rgba="0.1 0.1 0.9 0.7"/>
    </body>
"""

    return f"""
<mujoco model="quadrotor_wrapper">
  {drone_include}
  <compiler angle="degree" coordinate="local"/>
  
  <worldbody>
    <!-- World Frame Axes (Static at Origin) -->
    <geom name="world_axis_x" type="cylinder" size="0.005 0.2" pos="0.2 0 0.0" quat="0.7071 0 0.7071 0" rgba="1 0 0 1" contype="0" conaffinity="0"/>
    <geom name="world_axis_y" type="cylinder" size="0.005 0.2" pos="0 0.2 0.0" quat="0.7071 -0.7071 0 0" rgba="0 1 0 1" contype="0" conaffinity="0"/>
    <geom name="world_axis_z" type="cylinder" size="0.005 0.2" pos="0 0 0.2" rgba="0 0 1 1" contype="0" conaffinity="0"/>

    <!-- Ghost Robot (Static Reference at Start Pos Z=1.0) -->
    <body name="ghost_bot" pos="0 0 1.0">
      <geom type="box" size="0.1 0.08 0.03" rgba="0.2 0.2 0.2 0.3" contype="0" conaffinity="0"/>
      <geom type="capsule" fromto="0 0 0 0.15 -0.15 0" size="0.008" rgba="0.4 0.4 0.4 0.3" contype="0" conaffinity="0"/>
      <geom type="capsule" fromto="0 0 0 0.15 0.15 0" size="0.008" rgba="0.4 0.4 0.4 0.3" contype="0" conaffinity="0"/>
      <geom type="capsule" fromto="0 0 0 -0.15 0.15 0" size="0.008" rgba="0.4 0.4 0.4 0.3" contype="0" conaffinity="0"/>
      <geom type="capsule" fromto="0 0 0 -0.15 -0.15 0" size="0.008" rgba="0.4 0.4 0.4 0.3" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
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

def euler_to_quaternion(roll, pitch, yaw):
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return np.array([qw, qx, qy, qz])

def main():
    config_path = os.path.join(os.path.dirname(__file__), 'config.json')
    try:
        with open(config_path, 'r') as f:
            config = json.load(f).get('quadrotor', {})
    except Exception as e:
        print(f"Failed to load config.json, using defaults: {e}")
        config = {}

    linear_speed_step = config.get("linear_speed_step", 0.05)
    angular_speed_step = config.get("angular_speed_step", 0.1)
    damping = config.get("damping", 0.95)

    print("=" * 60)
    print("CHALLENGE 2: MuJoCo Skydio X2 Quadrotor Teleop & Body Frame Tracker")
    print("=" * 60)
    
    if os.path.exists(MENAGERIE_X2_PATH):
        print(f"Loading Menagerie Quadrotor Model: {MENAGERIE_X2_PATH}")
        xml_string = generate_quadrotor_xml(MENAGERIE_X2_PATH)
    elif os.path.exists(FALLBACK_CRAZYFLIE_PATH):
        print(f"Loading Menagerie Quadrotor Model: {FALLBACK_CRAZYFLIE_PATH}")
        xml_string = generate_quadrotor_xml(FALLBACK_CRAZYFLIE_PATH)
    else:
        print("Loading Standalone Quadrotor Model...")
        xml_string = generate_quadrotor_xml(None)

    # We must pass the directory of the original XML so the include path resolves properly
    model_dir = os.path.dirname(MENAGERIE_X2_PATH) if os.path.exists(MENAGERIE_X2_PATH) else None
    
    # In python bindings from_xml_string does not accept dir, we temporarily chdir
    if model_dir:
        original_cwd = os.getcwd()
        os.chdir(model_dir)
        model = mujoco.MjModel.from_xml_string(xml_string)
        os.chdir(original_cwd)
    else:
        model = mujoco.MjModel.from_xml_string(xml_string)

    data = mujoco.MjData(model)

    # Initial 6-DOF Drone State
    x, y, z = 0.0, 0.0, 1.0
    roll, pitch, yaw = 0.0, 0.0, 0.0
    vx, vy, vz = 0.0, 0.0, 0.0
    w_yaw = 0.0

    dt = 0.01

    # Set terminal non-blocking keyboard input if on Unix
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
            viewer.cam.distance = 3.0
            viewer.cam.elevation = -20
            viewer.cam.lookat[:] = [0, 0, 1.0]

            while viewer.is_running():
                key = get_key()
                key_pressed = False
                if key:
                    key = key.lower()
                    if key == 'w':
                        vx += linear_speed_step
                        pitch = -0.15
                        key_pressed = True
                    elif key == 's':
                        vx -= linear_speed_step
                        pitch = 0.15
                        key_pressed = True
                    elif key == 'a':
                        vy += linear_speed_step
                        roll = -0.15
                        key_pressed = True
                    elif key == 'd':
                        vy -= linear_speed_step
                        roll = 0.15
                        key_pressed = True
                    elif key == 'i':
                        vz += linear_speed_step
                        key_pressed = True
                    elif key == 'k':
                        vz -= linear_speed_step
                        key_pressed = True
                    elif key == 'j':
                        w_yaw += angular_speed_step
                        key_pressed = True
                    elif key == 'l':
                        w_yaw -= angular_speed_step
                        key_pressed = True
                    elif key == ' ':
                        vx, vy, vz = 0.0, 0.0, 0.0
                        w_yaw = 0.0
                        roll, pitch = 0.0, 0.0
                        key_pressed = True
                    elif key == 'q':
                        break

                # Apply max speed limits
                vx = np.clip(vx, -max_linear_speed, max_linear_speed)
                vy = np.clip(vy, -max_linear_speed, max_linear_speed)
                vz = np.clip(vz, -max_linear_speed, max_linear_speed)
                w_yaw = np.clip(w_yaw, -max_angular_speed, max_angular_speed)

                # Velocity friction damping for smooth motion (only coast if no keys pressed)
                if not key_pressed:
                    vx *= damping
                    vy *= damping
                    vz *= damping
                    roll *= damping
                    pitch *= damping
                    w_yaw *= damping

                yaw += w_yaw * dt
                cos_y, sin_y = math.cos(yaw), math.sin(yaw)
                
                # Transform body-frame velocity to world-frame position update
                x += (cos_y * vx - sin_y * vy) * dt
                y += (sin_y * vx + cos_y * vy) * dt
                z += vz * dt
                if z < 0.1:
                    z = 0.1

                # If "stable physics" is false, we simulate real physics vibrations
                if not stable:
                    j_x = x + np.random.uniform(-0.002, 0.002)
                    j_y = y + np.random.uniform(-0.002, 0.002)
                    j_z = z + np.random.uniform(-0.002, 0.002)
                    j_r = roll + np.random.uniform(-0.01, 0.01)
                    j_p = pitch + np.random.uniform(-0.01, 0.01)
                    j_yw = yaw + np.random.uniform(-0.01, 0.01)
                    quat = euler_to_quaternion(j_r, j_p, j_yw)
                    data.qpos[0:3] = [j_x, j_y, j_z]
                    data.qpos[3:7] = quat
                else:
                    quat = euler_to_quaternion(roll, pitch, yaw)
                    data.qpos[0:3] = [x, y, z]
                    data.qpos[3:7] = quat

                mujoco.mj_step(model, data)
                viewer.sync()

                current_time = time.time()
                if key_pressed or (current_time - last_print_time >= 0.2):
                    last_print_time = current_time
                    R = quaternion_to_rotation_matrix(quat)

                    # Get actual velocity from physics engine (even though we manually update qpos, 
                    # mujoco still computes objectVelocity if mj_forward or mj_step was called)
                    # For skydio, the body is usually the root body in the included xml. Let's just find the first free body.
                    # Or we can just use the manual velocities vx, vy, vz as "actual" since we teleport.
                    
                    sys.stdout.write("\033[H\033[J")
                    print("=" * 60)
                    print("🚁 SKYDIO X2 QUADROTOR TELEMETRY")
                    print("-" * 60)
                    print(f"Commanded | VX: {vx:+.2f} | VY: {vy:+.2f} | VZ: {vz:+.2f} | Yaw Rate: {w_yaw:+.2f}")
                    # Since this drone is purely kinematic teleported, actual = commanded.
                    print(f"Actual    | VX: {vx:+.2f} | VY: {vy:+.2f} | VZ: {vz:+.2f} | Yaw Rate: {w_yaw:+.2f}")
                    print("=" * 60)
                    print(f"World Position (x, y, z): [{x:.3f}, {y:.3f}, {z:.3f}]")
                    print("\n3x3 Body Frame Rotation Matrix (R):")
                    print(f"[{R[0,0]:8.4f}  {R[0,1]:8.4f}  {R[0,2]:8.4f}]")
                    print(f"[{R[1,0]:8.4f}  {R[1,1]:8.4f}  {R[1,2]:8.4f}]")
                    print(f"[{R[2,0]:8.4f}  {R[2,1]:8.4f}  {R[2,2]:8.4f}]")
                    print("=" * 60)
                    print("Controls: W/S (Pitch), A/D (Roll), I/K (Alt), J/L (Yaw), Q (Quit)")
                    sys.stdout.flush()

                time.sleep(dt)
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
