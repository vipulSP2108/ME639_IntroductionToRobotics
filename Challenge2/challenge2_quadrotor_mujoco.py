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

# Embedded standalone Quadrotor MJCF XML in case Menagerie scene needs local fallback
QUADROTOR_FALLBACK_XML = """
<mujoco model="skydio_x2_quadrotor">
  <compiler angle="degree" coordinate="local"/>
  <option timestep="0.01" gravity="0 0 -9.81"/>

  <visual>
    <headlight ambient="0.4 0.4 0.4" diffuse="0.8 0.8 0.8"/>
  </visual>

  <worldbody>
    <light pos="0 0 5" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="10 10 0.1" rgba="0.8 0.8 0.8 1"/>

    <!-- Quadrotor Body -->
    <body name="quadrotor" pos="0 0 1.0">
      <freejoint name="root"/>
      
      <!-- Central Body Frame -->
      <geom name="chassis" type="box" size="0.1 0.08 0.03" rgba="0.2 0.2 0.2 1"/>
      <geom name="camera_gimbal" type="sphere" size="0.025" pos="0.1 0 -0.01" rgba="0.1 0.7 0.9 1"/>

      <!-- Visual Body Frame Axes (X: Red, Y: Green, Z: Blue) -->
      <geom name="axis_x" type="cylinder" size="0.005 0.12" pos="0.12 0 0" quat="0.7071 0 0.7071 0" rgba="1 0 0 1" contype="0" conaffinity="0"/>
      <geom name="axis_y" type="cylinder" size="0.005 0.12" pos="0 0.12 0" quat="0.7071 -0.7071 0 0" rgba="0 1 0 1" contype="0" conaffinity="0"/>
      <geom name="axis_z" type="cylinder" size="0.005 0.12" pos="0 0 0.12" rgba="0 0 1 1" contype="0" conaffinity="0"/>

      <!-- Quadrotor Arms & Rotors -->
      <!-- Front Right -->
      <geom type="capsule" fromto="0 0 0 0.15 -0.15 0" size="0.008" rgba="0.4 0.4 0.4 1"/>
      <geom type="cylinder" size="0.07 0.002" pos="0.15 -0.15 0.01" rgba="0.9 0.1 0.1 0.7"/>

      <!-- Front Left -->
      <geom type="capsule" fromto="0 0 0 0.15 0.15 0" size="0.008" rgba="0.4 0.4 0.4 1"/>
      <geom type="cylinder" size="0.07 0.002" pos="0.15 0.15 0.01" rgba="0.9 0.1 0.1 0.7"/>

      <!-- Rear Left -->
      <geom type="capsule" fromto="0 0 0 -0.15 0.15 0" size="0.008" rgba="0.4 0.4 0.4 1"/>
      <geom type="cylinder" size="0.07 0.002" pos="-0.15 0.15 0.01" rgba="0.1 0.1 0.9 0.7"/>

      <!-- Rear Right -->
      <geom type="capsule" fromto="0 0 0 -0.15 -0.15 0" size="0.008" rgba="0.4 0.4 0.4 1"/>
      <geom type="cylinder" size="0.07 0.002" pos="-0.15 -0.15 0.01" rgba="0.1 0.1 0.9 0.7"/>
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
    print("=" * 60)
    print("CHALLENGE 2: MuJoCo Skydio X2 Quadrotor Teleop & Body Frame Tracker")
    print("=" * 60)
    print("Controls:")
    print("  W / S : Forward / Backward (Pitch Tilt)")
    print("  A / D : Left / Right Roll (Roll Tilt)")
    print("  I / K : Ascend / Descend Altitude")
    print("  J / L : Yaw Left / Yaw Right")
    print("  SPACE : Hover Neutral")
    print("  Q     : Quit Simulation")
    print("=" * 60)

    # Attempt loading from Menagerie Skydio X2 first
    if os.path.exists(MENAGERIE_X2_PATH):
        print(f"Loading Menagerie Quadrotor Model: {MENAGERIE_X2_PATH}")
        model = mujoco.MjModel.from_xml_path(MENAGERIE_X2_PATH)
    elif os.path.exists(FALLBACK_CRAZYFLIE_PATH):
        print(f"Loading Menagerie Quadrotor Model: {FALLBACK_CRAZYFLIE_PATH}")
        model = mujoco.MjModel.from_xml_path(FALLBACK_CRAZYFLIE_PATH)
    else:
        print("Loading Standalone Quadrotor Model...")
        model = mujoco.MjModel.from_xml_string(QUADROTOR_FALLBACK_XML)

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
                        vx += 0.05
                        pitch = -0.15
                        key_pressed = True
                    elif key == 's':
                        vx -= 0.05
                        pitch = 0.15
                        key_pressed = True
                    elif key == 'a':
                        vy += 0.05
                        roll = -0.15
                        key_pressed = True
                    elif key == 'd':
                        vy -= 0.05
                        roll = 0.15
                        key_pressed = True
                    elif key == 'i':
                        vz += 0.05
                        key_pressed = True
                    elif key == 'k':
                        vz -= 0.05
                        key_pressed = True
                    elif key == 'j':
                        w_yaw += 0.1
                        key_pressed = True
                    elif key == 'l':
                        w_yaw -= 0.1
                        key_pressed = True
                    elif key == ' ':
                        vx, vy, vz = 0.0, 0.0, 0.0
                        w_yaw = 0.0
                        roll, pitch = 0.0, 0.0
                        key_pressed = True
                    elif key == 'q':
                        break

                # Velocity friction damping for smooth motion
                vx *= 0.95
                vy *= 0.95
                vz *= 0.95
                roll *= 0.90
                pitch *= 0.90
                w_yaw *= 0.90

                yaw += w_yaw * dt
                cos_y, sin_y = math.cos(yaw), math.sin(yaw)
                
                x += (cos_y * vx - sin_y * vy) * dt
                y += (sin_y * vx + cos_y * vy) * dt
                z += vz * dt
                if z < 0.1:
                    z = 0.1

                quat = euler_to_quaternion(roll, pitch, yaw)

                data.qpos[0:3] = [x, y, z]
                data.qpos[3:7] = quat

                mujoco.mj_step(model, data)
                viewer.sync()

                current_time = time.time()
                if key_pressed or (current_time - last_print_time >= 0.2):
                    last_print_time = current_time
                    R = quaternion_to_rotation_matrix(quat)

                    sys.stdout.write("\033[H\033[J")
                    print("=" * 60)
                    print(f"🚁 SKYDIO X2 QUADROTOR | Z: {z:.3f} m | Yaw: {math.degrees(yaw):+.1f}°")
                    print("=" * 60)
                    print(f"Position (x, y, z): [{x:.3f}, {y:.3f}, {z:.3f}]")
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
