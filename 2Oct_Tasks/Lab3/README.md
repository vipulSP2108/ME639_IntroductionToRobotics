# 🤖 ME639 — Lab 3: Pick and Place with Inverse Kinematics (MuJoCo)

> **Assignment:** Tutorial 4 — Pick and Place of Objects  
> **Robot:** HEAL Arm + Robotiq 2F-85 Gripper  
> **Simulator:** MuJoCo  
> **IK Library (Phase 1):** [Mink](https://github.com/kevinzakka/mink)

---

## 📁 Folder Structure

```
Lab3/
├── README.md                    ← You are here
│
├── robot_descriptions/          ← MuJoCo XML models (arm, gripper, world)
│   ├── arm_gripper.xml
│   ├── ur5.xml
│   ├── world.xml
│   ├── robotiq_2f85_v4/
│   └── heal_meshes/
│
├── utils/                       ← Reusable helper modules
│   ├── mj_velocity_control/
│   │   └── mj_velocity_ctrl.py  ← JointVelocityController & JointPositionController
│   ├── kinematics.py            ← FK, Jacobian, DLS-IK, QP-IK
│   ├── quaternion_utils.py      ← Quaternion error, SLERP, log map, angular velocity
│   ├── trajectory.py            ← Quintic minimum-jerk splines & full-table sampler
│   └── workspace.py             ← Workspace sampling & visualization helpers
│
├── src/                         ← Main runnable pipeline scripts
│   ├── 01_workspace_analysis.py ← Stage 1: Compute & visualize workspace
│   ├── 02_scene_setup.py        ← Stage 2: Spawn table + cube in MuJoCo
│   ├── 03_pick_place_mink.py    ← Stage 3: Phase 1 IK — off-the-shelf Mink
│   ├── 04_batch_episodes.py     ← Stage 4: Run N=25+ episodes, log metrics
│   ├── 05_dls_ik.py             ← Stage 5: Phase 2 — DLS-IK implementation
│   └── 06_qp_ik.py              ← Stage 6: Phase 2 — QP-IK implementation
│
├── scripts/                     ← Standalone interactive & recording tools
│   ├── interactive_play.py      ← 3D viewer playground with mouse forces & shortcuts
│   ├── extra_workspace_features.py ← 3D reachable sphere & silky-smooth tour
│   ├── record_demo.py           ← Automated demo recorder for 3-minute video
│   ├── example_controlling_gripper.py
│   ├── example_controlling_velocity.py
│   ├── load_any_xml.py
│   └── load_heal_robot.py
│
├── analysis/                    ← Analysis documents & output plots
│   ├── workspace_analysis.md    ← How workspace was computed, results
│   ├── table_design.md          ← Table dimension rationale
│   └── plots/                   ← Generated PNG/SVG plots go here
│
├── logs/                        ← Episode logs (auto-generated, gitignored)
│   └── episode_logs.json
│
├── report/                      ← Final report (PDF/LaTeX/Markdown)
│
└── temp/                        ← Private / dev files (GITIGNORED, NOT for public GitHub)
    ├── VIVA_GUIDE.md            ← Private viva prep guide (failure modes, tips, Q&A)
    ├── assgnemnt.md             ← Original assignment prompt
    ├── inspect.md               ← Codebase inspection notes
    └── scaffold_guides/         ← Dev prompt specs (src/*.md, utils/*.md)
```

---

## ⚙️ Setup & Installation

```bash
# 1. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2. Install MuJoCo Python bindings
pip install mujoco

# 3. Install IK library (Phase 1)
pip install mink

# 4. Install QP solver (Phase 2)
pip install qpsolvers[osqp]

# 5. Install plotting & utility dependencies
pip install numpy matplotlib scipy

# 6. Verify MuJoCo works
python -c "import mujoco; print(mujoco.__version__)"
```

---

## 🚀 Running the Pipeline (in order)

| Step | Script | What it does |
|------|--------|-------------|
| 1 | `python src/01_workspace_analysis.py` | Samples reachable positions, saves point-cloud plots |
| 2 | `python src/02_scene_setup.py` | Opens MuJoCo viewer with table + robot |
| 3 | `python src/03_pick_place_mink.py` | Single pick-and-place episode with Mink IK |
| 4 | `python src/04_batch_episodes.py` | 25+ randomized episodes, saves `logs/episode_logs_mink.json` |
| 5 | `python src/05_dls_ik.py` | Same episodes with custom DLS-IK, saves `logs/episode_logs_dls.json` |
| 6 | `python src/06_qp_ik.py` | Same episodes with QP-IK, generates 4 comparison plots |
| 7 | `mjpython scripts/interactive_play.py` | **Interactive 3D Playground**: Live visual test with mouse forces, [W] sphere toggle, [T] tour |
| 8 | `mjpython scripts/extra_workspace_features.py` | **Workspace Sphere & Tour**: 3D reachable sphere & 13-point silky-smooth boundary tour |
| 9 | `mjpython scripts/record_demo.py` | **Automated Video Recorder**: 3-min video runner across Mink, DLS, and QP |

---

## 📊 Key Metrics Logged

| Metric | Description |
|--------|-------------|
| `success` | Boolean — did pick+place complete? |
| `failure_reason` | `"ik_infeasible"` / `"collision"` / `"grasp_slip"` / `"joint_limit"` |
| `ik_iterations` | Steps until IK converged |
| `time_to_solve` | Wall-clock planning time (s) |
| `min_clearance` | Minimum distance to any obstacle (m) |
| `cube_pose` | Initial `(x, y, yaw)` of cube |

---

## 🔧 Tunable Parameters (Quick Reference)

See `VIVA_GUIDE.md` for full discussion.

| Parameter | Location | Effect |
|-----------|----------|--------|
| `lambda` (DLS damping) | `utils/kinematics.py` | Singularity robustness |
| `kd` (velocity gain) | `utils/mj_velocity_control/mj_velocity_ctrl.py` | Joint tracking speed |
| `pre_grasp_offset` | `src/03_pick_place_mink.py` | Approach clearance |
| Cube spawn range `[xmin,xmax]` | `src/04_batch_episodes.py` | Task difficulty |
| `N_episodes` | `src/04_batch_episodes.py` | Batch size |
| `dt` | scene XML | Simulation timestep |

---

## 📚 References

- [Mink IK library](https://github.com/kevinzakka/mink)
- [MuJoCo Documentation](https://mujoco.readthedocs.io)
- [IK Video Series](https://www.youtube.com/watch?v=nin2TbMuhR0&list=PLggLP4f-rq01fi62ek1BoV1yiPHLL3Vt-)
- [Jacobian Series](https://www.youtube.com/watch?v=6tj8QLF69Ok&list=PLggLP4f-rq00oA7lrv6dS5C15RYjqj4Zi)
