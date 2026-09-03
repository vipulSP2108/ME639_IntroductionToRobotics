"""
01_rotation_sandbox.py -- HW1 Part 2, Task 1: does rotation order matter?

STARTER CODE. The model loading, viewer, and simulation loop below
are complete and working -- run this file as-is and you should see
the asymmetric dart sitting in the viewer. Your job is to fill in
the TODOs so that:

  1. The user can queue up a sequence of elemental rotations
     (about x, y, or z), each one EITHER about the current body
     frame OR about the fixed space frame (their choice).
  2. The dart's orientation updates to reflect that sequence.
  3. You can run the SAME sequence of angles twice -- once
     "current frame" and once "fixed frame" -- and see (and
     screen-record) that the final orientation is visibly
     different, exactly as you proved symbolically in HW1
     Problem 3 (Lynch & Park Ch.3 Ex.3.4-style reasoning) and
     the "Composition of Rotations" lecture derivation.

This is intentionally a plain script, not a GUI app -- editing the
`rotation_sequence` list below and re-running is a perfectly good
"sandbox." A slider UI is a nice-to-have, not a requirement. Use AI
freely here; document what you asked it for in your AI Use Note.
"""

import time
import numpy as np
import mujoco
import mujoco.viewer

from utils import Rx, Ry, Rz, ELEMENTARY_ROTATIONS, set_body_orientation

MODEL_PATH = "../model/asymmetric_body.xml"


# ---------------------------------------------------------------
# TODO(student): edit this sequence to try different experiments.
# Each entry is (axis, angle_radians, frame) where frame is either
# "current" (compose on the RIGHT: R_new = R_old @ R_step) or
# "fixed" (compose on the LEFT: R_new = R_step @ R_old).
# Compare the two frame choices for the SAME list of (axis, angle).
# ---------------------------------------------------------------
rotation_sequence = [
    ("z", np.deg2rad(90), "current"),   # TODO: try "fixed" here instead
    ("x", np.deg2rad(90), "current"),   # TODO: try "fixed" here instead
]


def compose_sequence(sequence):
    """TODO(student): implement this.

    Given a list of (axis, angle, frame) tuples, return the final
    3x3 rotation matrix R obtained by applying them in order,
    starting from R = identity.

    Hint: ELEMENTARY_ROTATIONS["x"](angle) gives you R_x(angle) etc.
    Hint: think about *why* current-frame composition is a right-
    multiply and fixed-frame composition is a left-multiply -- this
    is exactly the "Current Frame" vs. "Fixed Frame" distinction
    from the Composition-of-Rotations lecture. Don't just guess;
    check it against your Problem 3/4 reasoning.
    """
    R = np.eye(3)
    for axis, angle, frame in sequence:
        R_step = ELEMENTARY_ROTATIONS[axis](angle)
        if frame == "current":
            R = R @ R_step
        elif frame == "fixed":
            R = R_step @ R
        else:
            raise ValueError(f"Unknown frame type: {frame}")
    return R


def main():
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    R_current = np.eye(3)
    set_body_orientation(data, R_current)
    mujoco.mj_forward(model, data)

    # State dictionary to communicate keyboard events to the main loop
    state = {"action": None}

    def key_callback(keycode):
        # Keyboard handler for MuJoCo passive viewer
        if keycode in (ord('r'), ord('R')):
            state["action"] = "reset"
        elif keycode == 32 or keycode in (ord('s'), ord('S')):  # Spacebar or 'S'
            state["action"] = "play"
        elif keycode in (ord('c'), ord('C')):
            state["action"] = "play_current"
        elif keycode in (ord('f'), ord('F')):
            state["action"] = "play_fixed"

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        print("\n" + "="*65)
        print("MuJoCo Rotation Sandbox Ready!")
        print("Camera controls:")
        print("  - Left-Click and drag in background: Rotate camera view")
        print("  - Right-Click and drag: Pan camera")
        print("  - Scroll wheel: Zoom in/out")
        print("\nKeyboard controls (click viewer window to focus, then press):")
        print("  [SPACE] : Play rotation sequence")
        print("  [C]     : Play sequence using CURRENT (Body) Frame")
        print("  [F]     : Play sequence using FIXED (Space) Frame")
        print("  [R]     : RESET back to initial orientation")
        print("="*65 + "\n")

        while viewer.is_running():
            if state["action"] == "reset":
                state["action"] = None
                R_current = np.eye(3)
                set_body_orientation(data, R_current)
                mujoco.mj_forward(model, data)
                viewer.sync()
                print(">>> RESET: Back to starting orientation (identity) <<<\n")

            elif state["action"] in ("play", "play_current", "play_fixed"):
                act = state["action"]
                state["action"] = None

                # Determine which sequence to run
                if act == "play":
                    seq_to_run = rotation_sequence
                    print(f"\n>>> Running configured sequence: {seq_to_run} <<<")
                elif act == "play_current":
                    seq_to_run = [(axis, angle, "current") for axis, angle, _ in rotation_sequence]
                    print("\n>>> Running in CURRENT (Body) Frame <<<")
                elif act == "play_fixed":
                    seq_to_run = [(axis, angle, "fixed") for axis, angle, _ in rotation_sequence]
                    print("\n>>> Running in FIXED (Space) Frame <<<")

                # Reset to initial before playing so each run starts clean
                R_current = np.eye(3)
                set_body_orientation(data, R_current)
                mujoco.mj_forward(model, data)
                viewer.sync()
                time.sleep(0.3)

                # Animate the sequence step by step
                steps_per_rotation = 90  # 1.5 seconds per rotation at 60 Hz
                interrupted = False
                for step_idx, (axis, angle, frame) in enumerate(seq_to_run, 1):
                    if not viewer.is_running() or state["action"] == "reset":
                        interrupted = True
                        break
                    print(f"Step {step_idx}: Rotate {np.rad2deg(angle):.1f} deg about {axis}-axis ({frame} frame)...")
                    R_start = R_current.copy()
                    for i in range(1, steps_per_rotation + 1):
                        if not viewer.is_running() or state["action"] == "reset":
                            interrupted = True
                            break
                        frac = i / steps_per_rotation
                        R_step = ELEMENTARY_ROTATIONS[axis](angle * frac)

                        if frame == "current":
                            R_current_interp = R_start @ R_step
                        elif frame == "fixed":
                            R_current_interp = R_step @ R_start

                        set_body_orientation(data, R_current_interp)
                        mujoco.mj_forward(model, data)
                        viewer.sync()
                        time.sleep(1 / 60)

                    if interrupted:
                        break
                    R_current = R_current_interp
                    time.sleep(0.5)

                if not interrupted:
                    print(">>> Sequence Complete! Final orientation displayed. <<<")
                    print("Press [R] to Reset, [C] for Current Frame, [F] for Fixed Frame.\n")

            viewer.sync()
            time.sleep(1 / 60)


if __name__ == "__main__":
    main()
