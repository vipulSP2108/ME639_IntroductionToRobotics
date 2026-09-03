"""
tf_broadcaster_node.py -- HW1 Part 2, Task 3 (Problem 9).

Broadcasts 'space_frame' and 'body_frame' as TF transforms for the same
motion as Task 1 (01_rotation_sandbox.py), and animates it in rviz2 with
interactive user controls matching Task 1 exactly.

Controls:
  On startup, the body sits stationary at identity. You trigger the animation
  whenever you are ready:

  Keyboard Controls (click the node's terminal window):
    [C]     : Play sequence using CURRENT (Body) Frame
    [F]     : Play sequence using FIXED (Space) Frame
    [R]     : RESET back to initial orientation (Identity)
    [SPACE] : Play current sequence
    [P]     : Pause / Unpause animation

  ROS 2 Parameter CLI (from any other terminal):
    ros2 param set /hw01_tf_broadcaster compose_frame fixed   # Plays Fixed sequence
    ros2 param set /hw01_tf_broadcaster compose_frame current # Plays Current sequence
    ros2 param set /hw01_tf_broadcaster reset true            # Resets to Identity
"""

import sys
import select
import termios
import tty
import threading
import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
from rcl_interfaces.msg import SetParametersResult, ParameterDescriptor


def Rx(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def Ry(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def Rz(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


ELEMENTARY_ROTATIONS = {"x": Rx, "y": Ry, "z": Rz}

# Exact motion sequence as Task 1 (01_rotation_sandbox.py):
# Step 1: Rotate 90 deg about z
# Step 2: Rotate 90 deg about x
STEP_SEQUENCE = [
    ("z", np.deg2rad(90)),
    ("x", np.deg2rad(90)),
]


def R_to_quat_xyzw(R):
    """Convert 3x3 rotation matrix to ROS-convention XYZW quaternion."""
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S
    q = np.array([x, y, z, w], dtype=float)
    norm = np.linalg.norm(q)
    return q / norm if norm > 1e-12 else np.array([0.0, 0.0, 0.0, 1.0])


class Hw01TfBroadcaster(Node):
    def __init__(self):
        super().__init__("hw01_tf_broadcaster")

        # Declare parameters
        self.declare_parameter(
            "compose_frame",
            "current",
            ParameterDescriptor(description="Frame composition mode: 'current' or 'fixed'")
        )
        self.declare_parameter(
            "rotation_duration",
            1.5,
            ParameterDescriptor(description="Duration in seconds per rotation animation")
        )
        self.declare_parameter(
            "pause_duration",
            0.8,
            ParameterDescriptor(description="Pause duration in seconds between steps")
        )
        self.declare_parameter(
            "body_z_offset",
            1.0,
            ParameterDescriptor(description="Z translation offset for body_frame")
        )
        self.declare_parameter(
            "reset",
            False,
            ParameterDescriptor(description="Set to true to reset orientation to identity")
        )

        self.tf_broadcaster = TransformBroadcaster(self)

        # Cache parameters
        self.compose_frame = self.get_parameter("compose_frame").value.lower()
        self.rotation_duration = float(self.get_parameter("rotation_duration").value)
        self.pause_duration = float(self.get_parameter("pause_duration").value)
        self.body_z_offset = float(self.get_parameter("body_z_offset").value)

        # State machine matching Task 1:
        # Starts in IDLE mode so the user can orient the view or start screen recording
        self.R_current = np.eye(3)
        self.R_step_start = np.eye(3)
        self.step_idx = 0
        self.state = "IDLE"  # "IDLE", "ROTATING", "PAUSE_STEP", "FINISHED"
        self.state_time = 0.0
        self.is_paused = False

        # Register parameter callback
        self.add_on_set_parameters_callback(self._on_parameter_change)

        # 30 Hz timer for smooth TF publishing and animation
        self.dt = 1.0 / 30.0
        self.timer = self.create_timer(self.dt, self.on_timer)

        self._print_startup_banner()

        # Start interactive keyboard listener in background if running in an interactive terminal
        self._shutdown_kb = False
        if sys.stdin.isatty():
            self.kb_thread = threading.Thread(target=self._keyboard_listener, daemon=True)
            self.kb_thread.start()

    def _print_startup_banner(self):
        self.get_logger().info("\n" + "=" * 68)
        self.get_logger().info("  HW01 TF Broadcaster Ready! (Matches Task 1 Interactive Controls)")
        self.get_logger().info("  Status: IDLE at starting orientation (Identity).")
        self.get_logger().info("-" * 68)
        self.get_logger().info("  Keyboard Controls (click this terminal, then press):")
        self.get_logger().info("    [C]     : Play sequence using CURRENT (Body) Frame")
        self.get_logger().info("    [F]     : Play sequence using FIXED (Space) Frame")
        self.get_logger().info("    [SPACE] : Play configured sequence")
        self.get_logger().info("    [R]     : RESET back to starting orientation")
        self.get_logger().info("    [P]     : Pause / Unpause motion")
        self.get_logger().info("-" * 68)
        self.get_logger().info("  ROS 2 Parameter CLI Controls:")
        self.get_logger().info("    ros2 param set /hw01_tf_broadcaster compose_frame fixed")
        self.get_logger().info("    ros2 param set /hw01_tf_broadcaster compose_frame current")
        self.get_logger().info("    ros2 param set /hw01_tf_broadcaster reset true")
        self.get_logger().info("=" * 68 + "\n")

    def _on_parameter_change(self, params):
        for param in params:
            if param.name == "compose_frame":
                val = str(param.value).lower().strip()
                if val not in ("current", "fixed"):
                    return SetParametersResult(successful=False, reason="compose_frame must be 'current' or 'fixed'")
                self.compose_frame = val
                self.trigger_play(mode=self.compose_frame)

            elif param.name == "reset" and param.value is True:
                self.reset_to_identity()

            elif param.name == "body_z_offset":
                self.body_z_offset = float(param.value)

        return SetParametersResult(successful=True)

    def reset_to_identity(self):
        self.R_current = np.eye(3)
        self.R_step_start = np.eye(3)
        self.step_idx = 0
        self.state = "IDLE"
        self.state_time = 0.0
        self.is_paused = False
        self.get_logger().info("\n>>> RESET: Back to starting orientation (Identity) <<<\n")

    def trigger_play(self, mode=None):
        if mode is not None:
            self.compose_frame = mode
        self.R_current = np.eye(3)
        self.R_step_start = np.eye(3)
        self.step_idx = 0
        self.state = "ROTATING"
        self.state_time = 0.0
        self.is_paused = False

        axis, ang = STEP_SEQUENCE[0]
        rule = (
            "POST-multiplication: R_new = R_old @ R_step (about moving body axes)"
            if self.compose_frame == "current"
            else "PRE-multiplication: R_new = R_step @ R_old (about static space axes)"
        )
        self.get_logger().info("\n" + "#" * 68)
        self.get_logger().info(f" >>> PLAYING SEQUENCE in {self.compose_frame.upper()} FRAME <<<")
        self.get_logger().info(f" Rule: {rule}")
        self.get_logger().info(f" Step 1: Rotating {np.rad2deg(ang):.0f} deg about {axis.upper()} axis...")
        self.get_logger().info("#" * 68 + "\n")

    def _keyboard_listener(self):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while rclpy.ok() and not self._shutdown_kb:
                rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
                if rlist:
                    ch = sys.stdin.read(1)
                    if ch in ('c', 'C'):
                        self.trigger_play(mode="current")
                    elif ch in ('f', 'F'):
                        self.trigger_play(mode="fixed")
                    elif ch in ('r', 'R'):
                        self.reset_to_identity()
                    elif ch == ' ' or ch in ('s', 'S'):
                        self.trigger_play(mode=self.compose_frame)
                    elif ch in ('p', 'P'):
                        self.is_paused = not self.is_paused
                        st = "PAUSED" if self.is_paused else "RESUMED"
                        self.get_logger().info(f"\n[KEYBOARD] Animation {st}\n")
        except Exception:
            pass
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def on_timer(self):
        # 1. Broadcast fixed space frame at origin (0, 0, 0)
        self.broadcast_frame("world", "space_frame", np.eye(3), z_offset=0.0)

        # 2. Advance animation if active and not paused
        if not self.is_paused and self.state != "IDLE" and self.state != "FINISHED":
            self._update_animation()

        # 3. Broadcast moving body frame with current orientation
        self.broadcast_frame("world", "body_frame", self.R_current, z_offset=self.body_z_offset)

    def _update_animation(self):
        axis, total_angle = STEP_SEQUENCE[self.step_idx]

        if self.state == "ROTATING":
            self.state_time += self.dt
            frac = min(1.0, self.state_time / max(1e-3, self.rotation_duration))
            smooth_frac = 0.5 * (1.0 - np.cos(np.pi * frac))
            R_step = ELEMENTARY_ROTATIONS[axis](total_angle * smooth_frac)

            # Core Task 1 logic:
            # Current frame: post-multiply (R_start @ R_step)
            # Fixed frame:   pre-multiply  (R_step @ R_start)
            if self.compose_frame == "current":
                self.R_current = self.R_step_start @ R_step
            else:
                self.R_current = R_step @ self.R_step_start

            if frac >= 1.0:
                R_full = ELEMENTARY_ROTATIONS[axis](total_angle)
                if self.compose_frame == "current":
                    self.R_current = self.R_step_start @ R_full
                else:
                    self.R_current = R_full @ self.R_step_start

                self.R_step_start = self.R_current.copy()
                self.state_time = 0.0

                if self.step_idx + 1 < len(STEP_SEQUENCE):
                    self.state = "PAUSE_STEP"
                    self.get_logger().info(
                        f"Completed Step {self.step_idx + 1} ({axis.upper()} {np.rad2deg(total_angle):.0f} deg). Pausing..."
                    )
                else:
                    self.state = "FINISHED"
                    self.get_logger().info("\n" + "=" * 68)
                    self.get_logger().info(
                        f">>> Sequence Complete! Final orientation reached ({self.compose_frame.upper()} Frame). <<<"
                    )
                    self.get_logger().info("Press [C] for Current, [F] for Fixed, or [R] to Reset.")
                    self.get_logger().info("=" * 68 + "\n")

        elif self.state == "PAUSE_STEP":
            self.state_time += self.dt
            if self.state_time >= self.pause_duration:
                self.state_time = 0.0
                self.step_idx += 1
                self.state = "ROTATING"
                next_axis, next_ang = STEP_SEQUENCE[self.step_idx]
                self.get_logger().info(
                    f"Starting Step {self.step_idx + 1}: Rotating {np.rad2deg(next_ang):.0f} deg about {next_axis.upper()} axis ({self.compose_frame} frame)..."
                )

    def broadcast_frame(self, parent, child, R, z_offset=0.0):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = parent
        t.child_frame_id = child
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = float(z_offset)

        qx, qy, qz, qw = R_to_quat_xyzw(R)
        t.transform.rotation.x = qx
        t.transform.rotation.y = qy
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw

        self.tf_broadcaster.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    node = Hw01TfBroadcaster()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._shutdown_kb = True
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
