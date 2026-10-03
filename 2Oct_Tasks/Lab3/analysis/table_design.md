# Table Design — `analysis/table_design.md`

## Table Description
- **Table Model:** Standard rectangular rigid workcell table
- **Material:** Matte structural surface with high friction
- **In Simulation:** Represented programmatically via MuJoCo `MjSpec` as a static `BOX` geom in the worldbody

---

## Why a Rectangular Table?

1. **Analytical Bounding:** Cartesian bounds ($[X_{\min}, X_{\max}], [Y_{\min}, Y_{\max}]$) map directly to an axis-aligned box, simplifying collision checking and episode sample boundaries.
2. **Stable Contact Plane:** Provides a uniform flat datum for repeatable cube resting without tipping or sliding.
3. **Workspace Matching:** The rectangular footprint aligns with the high-dexterity forward sector of the arm without encroaching on the robot base pedestal.

---

## Chosen Dimensions & Rationale

Derived directly from the Stage 1 workspace analysis results:

| Parameter | Symbol | Value | Unit | Rationale |
|:----------|:------:|:-----:|:----:|:----------|
| **Table Length (X)** | $L$ | **0.40** | m | Spans $X \in [0.25, 0.65]\text{ m}$; well inside the forward reach limit of $0.681\text{ m}$ with safety margin |
| **Table Width (Y)** | $W$ | **0.50** | m | Spans $Y \in [-0.25, +0.25]\text{ m}$; well inside lateral dexterous limits of $\pm 0.700\text{ m}$ |
| **Table Height (Z)** | $H$ | **0.20** | m | Places the tabletop surface at $Z = 0.20\text{ m}$, providing generous vertical stroke for pre-grasp ($Z = 0.35\text{ m}$) |
| **Table Center X** | $X_c$ | **0.45** | m | Midpoint of $X \in [0.25, 0.65]\text{ m}$, safely outside base collision dead-zone |
| **Table Center Y** | $Y_c$ | **0.00** | m | Symmetrically aligned with robot base centerline |

### Geometric Derivation
```
Dexterous Workspace Bounds (Measured):
  X_dex ∈ [-0.706, +0.681] m
  Y_dex ∈ [-0.724, +0.700] m
  Z_dex ∈ [-0.399, +0.826] m

Selected Table Boundary:
  X_min = 0.25 m  (leaves 0.25 m clearance from robot base)
  X_max = 0.65 m  (leaves 0.03 m safety margin before outer limit 0.681 m)
  => L = X_max - X_min = 0.40 m
  => X_center = (X_min + X_max) / 2 = 0.45 m

  Y_min = -0.25 m
  Y_max = +0.25 m
  => W = Y_max - Y_min = 0.50 m
  => Y_center = 0.00 m

  Table Top Surface Z = 0.20 m
  In MuJoCo, box origin is its centroid:
  => Z_center = H / 2 = 0.10 m (with box half-size = [L/2, W/2, H/2] = [0.20, 0.25, 0.10])
```

---

## Scene Placement Constants

These constants are standardized across all stages (`02_scene_setup.py`, `03_pick_place_mink.py`, `04_batch_episodes.py`, `05_dls_ik.py`, `06_qp_ik.py`):

```python
TABLE_X = 0.45       # Table centroid X [m]
TABLE_Y = 0.00       # Table centroid Y [m]
TABLE_H = 0.20       # Total table height from floor [m]
TABLE_L = 0.40       # Length along X [m]
TABLE_W = 0.50       # Width along Y [m]

# Cube dimensions (5 cm side length)
CUBE_SIZE = 0.05
CUBE_HALF_SIZE = 0.025
CUBE_Z = TABLE_H + CUBE_HALF_SIZE  # 0.225 m (resting flush on table)

# Tray dimensions and target placement
TRAY_L = 0.12        # 12 cm length
TRAY_W = 0.12        # 12 cm width
TRAY_H = 0.02        # 2 cm rim height
TRAY_X = 0.55        # Far end along X
TRAY_Y = 0.00        # Centered along Y
TRAY_Z = TABLE_H + TRAY_H / 2  # 0.21 m
```

---

## Cube Spawn Range on Table Surface

For randomized batch evaluation, the cube is sampled uniformly within the safe pick zone:

| Variable | Range | Description |
|:--------:|:-----:|:------------|
| **$x$** | `[0.32, 0.46]` m | Front/mid section of table, leaving clearance to tray |
| **$y$** | `[-0.15, 0.15]` m | Within central ±15 cm lateral strip |
| **$\text{yaw}$** | `[-π, π]` rad | Arbitrary orientation around vertical Z axis |

---

## Design Justification Summary for Viva

1. **Why is the table height 0.20 m?**  
   If the table were too high (e.g., $Z = 0.70\text{ m}$), the arm would hit its top vertical limit during pre-grasp and transit. If the table were at ground level ($Z = 0.0\text{ m}$), the wrist joints would reach near-singular configurations trying to point downward. $Z = 0.20\text{ m}$ places pick and place operations at the center of the arm's highest-manipulability region.

2. **Why is the table set forward at $X = 0.45\text{ m}$?**  
   The workspace analysis confirmed a $0.20\text{ m}$ inner dead-zone around the base. Setting the table front edge at $X = 0.25\text{ m}$ completely eliminates base collisions while keeping all targets within reach of the $0.68\text{ m}$ forward envelope.
