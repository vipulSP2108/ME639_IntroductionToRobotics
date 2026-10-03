#!/usr/bin/env python3
"""
Quaternion Utilities for Inverse Kinematics and Orientation Control
==================================================================
Convention: MuJoCo uses scalar-first quaternion: [w, x, y, z].
(Note: SciPy and ROS use scalar-last: [x, y, z, w]).

All functions here assume unit quaternions in MuJoCo [w, x, y, z] order.
"""

import numpy as np

# Standard straight-down gripper orientation:
# 180-degree rotation about the X-axis: [w, x, y, z] = [0.0, 1.0, 0.0, 0.0]
# In world coordinates, this points the end-effector z-axis straight down.
DOWNWARD_QUAT = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)


def quat_multiply(q1, q2):
    """
    Hamilton product of two quaternions in MuJoCo scalar-first [w, x, y, z].
    
    Formula:
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    ], dtype=np.float64)


def quat_conjugate(q):
    """
    Quaternion conjugate [w, -x, -y, -z].
    For unit quaternions, conjugate equals inverse.
    """
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def quat_error(q_current, q_target):
    """
    Compute relative error quaternion from current to target orientation:
        q_e = q_target * (q_current)^(-1)
    
    Ensures hemisphere continuity (q_e.w >= 0) to always take the shortest
    geodesic rotation path on SO(3).
    """
    q_e = quat_multiply(q_target, quat_conjugate(q_current))
    if q_e[0] < 0.0:
        q_e = -q_e
    return q_e


def quat_to_angular_velocity(q_e):
    """
    Convert a small rotation error quaternion into a 3D angular error vector (axis * angle).
    
    theta = 2 * arccos(clip(w, -1, 1))
    axis = xyz / sin(theta / 2)  [if theta > 1e-6, else 0]
    omega = theta * axis
    """
    w = float(np.clip(q_e[0], -1.0, 1.0))
    theta = 2.0 * np.arccos(w)
    
    if theta < 1e-6:
        return np.zeros(3, dtype=np.float64)
    
    sin_half = np.sin(theta / 2.0)
    if np.abs(sin_half) < 1e-7:
        return np.zeros(3, dtype=np.float64)
        
    axis = q_e[1:] / sin_half
    return theta * axis


def quaternion_error_to_angular_velocity(q_current, q_target):
    """
    Convenience wrapper: compute relative quaternion error and convert directly
    to a 3D angular velocity vector in world frame.
    """
    q_e = quat_error(q_current, q_target)
    return quat_to_angular_velocity(q_e)


def rot_matrix_to_quat(R):
    """
    Convert a 3x3 orthonormal rotation matrix to a scalar-first quaternion [w, x, y, z].
    Uses Shepperd's algorithm for numerical stability across all matrix traces.
    """
    R = np.asarray(R, dtype=np.float64).reshape((3, 3))
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
        
    q = np.array([w, x, y, z], dtype=np.float64)
    # Normalize for safety
    norm = np.linalg.norm(q)
    if norm > 1e-9:
        q /= norm
    if q[0] < 0.0:
        q = -q
    return q


def quat_slerp(q0, q1, tau):
    """
    Spherical Linear Interpolation (SLERP) between two unit quaternions q0 and q1.
    tau in [0, 1].
    """
    tau = float(np.clip(tau, 0.0, 1.0))
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)

    norm0 = np.linalg.norm(q0)
    norm1 = np.linalg.norm(q1)
    if norm0 > 1e-9:
        q0 = q0 / norm0
    if norm1 > 1e-9:
        q1 = q1 / norm1

    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    if dot > 0.9995:
        # Linear interpolation for very close orientations to avoid division by zero
        res = q0 + tau * (q1 - q0)
        return res / np.linalg.norm(res)

    theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * tau
    sin_theta = np.sin(theta)

    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    res = (s0 * q0) + (s1 * q1)
    return res / np.linalg.norm(res)
