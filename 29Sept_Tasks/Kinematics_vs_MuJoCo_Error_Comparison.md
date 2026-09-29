# End-Effector Error Comparison: Task 2 (DH Kinematics) vs Task 3 (MuJoCo)

This document provides the direct $X, Y, Z$ end-effector coordinates and errors comparing the analytical Denavit-Hartenberg (DH) forward kinematics from `Task2.ipynb` against the MuJoCo physics simulation in `Task3`.

---

## 1. 6-DOF HEAL Manipulator

* **Joint Angles:** $\boldsymbol{\theta} = [10^\circ, 20^\circ, -30^\circ, 40^\circ, -10^\circ, 20^\circ]$

| Axis | Task 2 (DH Calculation) [m] | Task 3 (MuJoCo Simulation) [m] | Difference ($\text{MuJoCo} - \text{Task 2}$) [m] | Error (Standard Precision) |
| :---: | :---: | :---: | :---: | :---: |
| **X** | $+0.56842177$ | $+0.56842177$ | $-1.110223 \times 10^{-16}$ | **$0.000000\text{ m}$** |
| **Y** | $+0.10493576$ | $+0.10493576$ | $-6.938894 \times 10^{-17}$ | **$0.000000\text{ m}$** |
| **Z** | $+0.61707911$ | $+0.61707911$ | $-1.110223 \times 10^{-16}$ | **$0.000000\text{ m}$** |

* **Total Position Error Norm:** $\|\Delta \mathbf{p}\|_2 = \mathbf{1.7166 \times 10^{-16}\text{ m}}$ (**Exact Match**)

---

## 2. 7-DOF FRANKA Manipulator

* **Joint Angles:** $\boldsymbol{\theta} = [10^\circ, -20^\circ, 30^\circ, -40^\circ, 20^\circ, -10^\circ, 15^\circ]$

| Axis | Task 2 (DH Calculation) [m] | Task 3 (MuJoCo Simulation) [m] | Difference ($\text{MuJoCo} - \text{Task 2}$) [m] | Error (Standard Precision) |
| :---: | :---: | :---: | :---: | :---: |
| **X** | $+0.55365390$ | $+0.55365390$ | $+0.000000 \times 10^{0}$ | **$0.000000\text{ m}$** |
| **Y** | $-0.10476693$ | $-0.10476693$ | $+6.938894 \times 10^{-17}$ | **$0.000000\text{ m}$** |
| **Z** | $-0.15429088$ | $-0.15429088$ | $-1.110223 \times 10^{-16}$ | **$0.000000\text{ m}$** |

* **Total Position Error Norm:** $\|\Delta \mathbf{p}\|_2 = \mathbf{1.3092 \times 10^{-16}\text{ m}}$ (**Exact Match**)