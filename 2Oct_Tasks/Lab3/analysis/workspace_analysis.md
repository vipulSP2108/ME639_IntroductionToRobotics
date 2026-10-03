# Workspace Analysis — `analysis/workspace_analysis.md`

## 1. What We Computed & Why

We sampled **N = 40,000** random joint configurations uniformly within the HEAL arm's hardware joint limits (`model.jnt_range`). For each configuration, we executed MuJoCo forward kinematics (`mujoco.mj_forward`) to compute the 3D position and orientation of the tool-centre point (TCP at site `right_center`).

**Why Monte Carlo sampling?**  
Analytically deriving the 6-DOF workspace boundary with joint limits is computationally difficult and prone to missing interior voids. Uniform Monte Carlo sampling directly maps reachable end-effector poses into Cartesian space, allowing instant identification of reach boundaries, dead zones, and orientation-constrained regions.

---

## 2. Task Workspace

The *task workspace* represents all 3D coordinates reachable by the TCP regardless of tool orientation.

| Axis | Min [m] | Max [m] | Span [m] |
|:----:|:-------:|:-------:|:--------:|
| **X** | -0.745 | +0.748 | 1.493 |
| **Y** | -0.741 | +0.743 | 1.484 |
| **Z** | -0.425 | +1.070 | 1.495 |

**Interpretation:**
- **Reachable Volume:** Bounding envelope volume $\approx 1.493 \times 1.484 \times 1.495 \approx 3.31\text{ m}^3$.
- **Forward Reach:** The manipulator can reach up to **0.748 m** radially from the base pedestal.
- **Symmetry:** As expected for a revolute base turret (`joint_1`), the horizontal footprint is circular and highly symmetric about the origin.

**Plot:** `analysis/plots/workspace_3d.png`

---

## 3. Dexterous Workspace

The *dexterous workspace* is the subset of the reachable workspace where the gripper can maintain a **top-down orientation** (approach vector within $32^\circ$ of vertical $-Z$). This orientation condition is strictly required for stable top-down grasp execution without collision between gripper knuckles and table surface.

$$\mathbf{z}_{\text{EE}} \cdot \begin{bmatrix} 0 \\ 0 \\ -1 \end{bmatrix} \ge \cos(32^\circ) \approx 0.848$$

| Axis | Min [m] | Max [m] | Span [m] |
|:----:|:-------:|:-------:|:--------:|
| **X** | -0.706 | +0.681 | 1.387 |
| **Y** | -0.724 | +0.700 | 1.423 |
| **Z** | -0.399 | +0.826 | 1.225 |

**Dexterous-to-Task ratio:** **5.4%** (2,173 out of 40,000 configurations) maintain top-down vertical grasp capability. The remaining 94.6% correspond to sideways, inverted, or extreme canted configurations unsuitable for downward table pick-and-place.

**Plot:** `analysis/plots/workspace_xy.png` (Top-down footprint) and `analysis/plots/workspace_xz.png` (Side elevation reach)

---

## 4. Feasible Table Region

From the XY projection of the dexterous workspace in the primary manipulation quadrant ($X > 0$):
- **Maximum forward dexterous reach:** $+0.681\text{ m}$
- **Inner kinematic dead-zone:** $X < 0.22\text{ m}$ (arm cannot bend steeply down without self-collision/limit violation)
- **High-dexterity sweet spot:** $X \in [0.25, 0.65]\text{ m}$, $Y \in [-0.25, +0.25]\text{ m}$
- **Vertical sweet spot:** $Z \in [0.10, 0.40]\text{ m}$

Applying safety margins of $\Delta_{\text{margin}} = 0.05\text{ m}$ from the workspace boundary ensures that inverse kinematics solvers (Mink, DLS, QP) do not encounter extreme singular configurations during pick-and-place trajectories.

---

## 5. Key Observations

1. **Dead Zone:** The robot cannot position its end-effector straight downward within a cylinder of radius $\approx 0.20\text{ m}$ around its base pedestal. The table front edge must be set at $X \ge 0.25\text{ m}$.
2. **Optimal Table Height:** At table surface height $Z = 0.20\text{ m}$, the cube grasp height is $Z \approx 0.225\text{ m}$ and pre-grasp clearance is $Z \approx 0.35\text{ m}$. This sits comfortably in the middle of the dexterous vertical envelope $[-0.399, +0.826]\text{ m}$.
3. **Y-Symmetry:** The dexterous region is symmetric about $Y = 0$, confirming that centering the table along the robot's forward centerline maximizes lateral workspace utilization.

---

## 6. Generated Visualization Plots

| Plot File | View | Engineering Insight |
|:----------|:-----|:--------------------|
| `analysis/plots/workspace_xy.png` | Top-down (X-Y) | Visualizes the circular task footprint and the top-down graspable table surface zone |
| `analysis/plots/workspace_xz.png` | Side elevation (X-Z) | Establishes the safe vertical clearance and verifies table surface height |
| `analysis/plots/workspace_3d.png` | 3D Perspective | Dual-color scatter comparing full task envelope (blue) vs. dexterous envelope (orange) |
