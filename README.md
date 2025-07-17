# 6‑Axis UR5e Robot Control with **Mink** & **MuJoCo**

This repository contains my solution for *“Robot 6‑Axis – Cartesian Motion”* coding challenge.
It extends the open‑source **[Mink](https://github.com/kevinzakka/mink)** robotics library with a Robotiq 2F‑85 gripper, collision‑aware Cartesian motion, and an autonomous pick‑and‑place routine for a Universal Robots **UR5e** arm simulated in **MuJoCo**.

---

## ✨ Key Features

| Area                    | What was implemented                                                                     |
| ----------------------- | ---------------------------------------------------------------------------------------- |
| **Robot model**         | UR5e + Robotiq 2F‑85 gripper added to the original Mink XML assets.                      |
| **Cartesian Motion**    | Quintic polynomial interpolator for smooth end‑effector paths.                           |
| **Tele‑operation**      | Re‑worked `TeleopMocap` with gripper open/close and active tracking shortcuts.           |
| **Collision avoidance** | Explicit geom pairs (gripper ↔ wall block) enforced via `CollisionAvoidanceLimit`.       |
| **Pick & place**        | Scripted waypoint planner that grasps a cube, transports it and releases it at a target. |

---

## 📁 Runnable Demos & Clips

Below are ready‑to‑run demo scripts **and** short screen‑capture clips so you can preview what each example does before running it locally.

| Script                                    | Description                                             | Preview                                                                      |
| ----------------------------------------- | ------------------------------------------------------- | ---------------------------------------------------------------------------- |
| `examples/arm_ur5e_actuators.py`          | Tele‑op of the bare UR5e arm (no gripper).              | <video src="media/s_arm.mp4" controls loop muted width="320"></video>        |
| `examples/arm_ur5e_gripper.py`            | Tele‑op of UR5e + Robotiq gripper.                      | <video src="media/s_gripper.mp4" controls loop muted width="320"></video>    |
| `examples/arm_ur5e_gripper_collision.py`  | Tele‑op with a wall obstacle and collision constraints. | <video src="media/s_collision.mp4" controls loop muted width="320"></video>  |
| `examples/arm_ur5e_gripper_pick_place.py` | Fully autonomous pick‑and‑place routine.                | <video src="media/s_pick_place.mp4" controls loop muted width="320"></video> |

> **Tip 🔍** If the inline video preview does not play on GitHub, simply click it to download or open it externally.

The MuJoCo assets reside in `examples/universal_robots_ur5e/`.

---

## 🚀 Quick Start

```bash
$ cd mink
$ pip install -e .

# 4 Run a demo (a window will pop‑up)
$ python3 examples/arm_ur5e_actuators.py            # without gripper
$ python3 examples/arm_ur5e_gripper.py            # tele‑op
$ python3 examples/arm_ur5e_gripper_collision.py  # tele‑op + wall
$ python3 examples/arm_ur5e_gripper_pick_place.py # autonomous
```

---

## Key Mappings

| Key            | Action                                                             |
| -------------- | ------------------------------------------------------------------ |
| `9`            | Toggle teleoperation On/Off.                                       |
| `n`            | Toggle between manual and non-manual mode.                         |
| `.`            | Toggle between rotation and translation mode.                      |
| `8`            | Cycle through different mocap data.                                |
| `+`            | Increase movement step size or movement speed.                     |
| `-`            | Decrease movement step size or movement speed.                     |
| **Arrow Keys** | **Move (rotation / translation) along the X, Y, and Z axes**       |
| `Up`           | Move forward (+X) or rotates around X-axis in positive direction.  |
| `Down`         | Move backward (-X) or rotates around X-axis in negative direction. |
| `Right`        | Move right (+Y) or rotates around Y-axis in positive direction.    |
| `Left`         | Move left (-Y) or rotates around Y-axis in negative direction.     |
| `7`            | Move up (+Z) or rotates around Z-axis in positive direction.       |
| `6`            | Move down (-Z) or rotates around Z-axis in negative direction.     |

### Modes

#### **Manual vs. Non-Manual Mode:**

* **Manual Mode**: Iterative movement using arrow keys.
* **Non-Manual Mode**: Continuous movement using arrow keys (to stop continuous movement, re-click the arrow key).

#### **Rotation vs. Translation Mode:**

* **Rotation Mode**: Rotation around an axis.
* **Translation Mode**: Movement along an axis.

---

## 📐 Mathematical Formulation

### Inverse Kinematics as a Quadratic Program

The differential inverse‑kinematics (IK) step that drives the UR5e is solved at **200 Hz** as a strictly convex Quadratic Program (QP):

```math
\begin{aligned}
\min_{\Delta \mathbf{q}} \; & \tfrac{1}{2}\,\Delta\mathbf{q}^\top H\,\Delta\mathbf{q} + c^\top \Delta\mathbf{q}\\
\text{s.t.}\; & G\,\Delta\mathbf{q} \le h
\end{aligned}
```

with

* $\Delta\mathbf{q} = \mathbf{v}\,\Delta t$ (joint displacements over the control period $\Delta t$)
* $H = \lambda I + \sum_i w_i J_i^\top J_i$ (task Hessian with Levenberg–Marquardt damping $\lambda$ and task weights $w_i$)
* $c = -\sum_i w_i J_i^\top e_i$ (linear term built from task errors $e_i$)
* $G, h$ stacked linear inequality constraints enforcing joint, velocity & collision limits

The QP is assembled in `build_ik()` (see `mink.inverse_kinematics`). Any backend supported by **qpsolvers** — OSQP, QPOASES, etc. — can be selected via the `--solver` CLI flag.

### Polynomial Time‑Scaling of Cartesian Way‑points

A short sequence of **SE(3) way‑points** is lifted to a smooth, fully‑parameterised trajectory using a scalar blend function $s(\tau)$ with normalised time $\tau\in[0,1]$. Three closed‑form splines are provided by `PolynomialInterpolator`:

| Order | Continuity | Blend function $s(\tau)$                  |
| :---: | :--------: | :---------------------------------------- |
|   1   |   $C^0$  | $s(\tau)=\tau$                            |
|   3   |   $C^1$  | $s(\tau)=3\tau^{2}-2\tau^{3}$             |
|   5   |   $C^2$  | $s(\tau)=6\tau^{5}-15\tau^{4}+10\tau^{3}$ |

These polynomials guarantee **zero** velocity (and acceleration for the quintic case) at both the start and the goal, greatly reducing jerk at grasp‑ and release‑events.

Given two poses $T_0, T_1\in\mathrm{SE}(3)$ the time‑parametrised pose is

```math
T(\tau) = T_0\;\exp\!\bigl\{\,s(\tau)\,\log\bigl(T_0^{-1}T_1\bigr)\bigr\}, \qquad \tau\in[0,1].
```

Sampling $n$ equidistant points along $s(\tau)$ produces a **C²‑continuous** reference path that can be tracked by the IK controller.

---

## 🗺️ Collision Avoidance Details

*The wall block* in `scene_gripper.xml` is assigned the `wall` geom name.
Identified and named all collision geoms in .xml for referencing in execution code.
Collision pairs are created between every gripper+arm contact geom and this wall via:

```python
collision_pairs = [
    (arm_collision_geoms + gripper_collision_geoms, ["wall"]),
]
limits = [
    mink.CollisionAvoidanceLimit(model, geom_pairs=collision_pairs),
]
```

The UR5e arm links are also listed under `_arm_collision_geom_names` for future extension (e.g. self–collision).

<video src="media/s_collision_viz.mp4" controls loop muted width="480"></video>

> **Yellow debug lines?**  In the clip below, the **yellow lines** are real‑time debug visuals drawn by Mink's collision monitor.  Each line starts at the centre of a monitored gripper/arm geom and ends at the closest point on the `wall` geom.  Its length equals the current signed‑distance, so it shrinks to zero exactly at contact.  This provides an immediate, intuitive view of which constraints are active and how close every geom is to collision.

---

## 🤖 Pick & Place Pipeline

```
           ┌──────────────┐    1  quintic interpolation (200 steps)
start pose │ current EE T │ ─────────────────────────────────────► goal queue
           └──────────────┘
                      ▲                │
                      │                ▼
             IK (QP‑LM‑damped)    Mujoco step + gripper cmd
```

1. **Way‑points:** hard‑coded positions/orientations (approach, descend, lift, place).
2. **Interpolator:** `PolynomialInterpolator(order=5)` yields a C²‑continuous Cartesian path.
3. **Solver:** At 200 Hz the IK QP returns joint velocities that respect

   * joint limits (`ConfigurationLimit`)
   * velocity limits (`VelocityLimit`)
   * and the collision constraint above if scene demands.

<video src="media/s_interpolation.mp4" controls loop muted width="480"></video>

The pick‑and‑place demo is fully scripted—no input needed.


### Goal specification

The pick-and-place planner expresses each **motion segment** as a 4-tuple:

```python
(pos, quat, gripper_cmd, wait_steps)
```

| Element       | Type               | Meaning                                                                                                    |
| ------------- | ------------------ | ---------------------------------------------------------------------------------------------------------- |
| `pos`         | 3-vector **XYZ**   | Desired end-effector translation in **world** coordinates. Use `None` to keep the current position.        |
| `quat`        | 4-vector **w xyz** | Desired end-effector orientation (unit quaternion). Use `None` to retain the current orientation.          |
| `gripper_cmd` | float or `None`    | Command sent to joint 7 — the Robotiq finger spread in **radians** (positive = close). `None` = unchanged. |
| `wait_steps`  | int ≥ 0            | How many control steps (at 200 Hz) to dwell **after** the segment finishes before starting the next one.   |

### Planning process

1. **Plan list** → A Python list of goal tuples (`plan = [...]`) encodes the full task.
2. **Deque scheduler** → Each tuple is converted to a 200-sample quintic path and queued.
3. **Execution loop** (200 Hz):

   * Pop next pose from the active path, update IK task target.
   * Call `solve_ik()` up to 20 Newton iterations to meet `pos_threshold` & `ori_threshold`.
   * Drive joints with the resulting velocity and step MuJoCo.
   * After the path is exhausted, dwell for `wait_steps`, then fetch the next segment.

The state machine therefore has only three explicit states:

* **`ACTIVE_PATH`** – still following poses inside the interpolated list.
* **`WAIT`** – the pose has been reached; maintain it for `wait_steps_remaining` ticks.
* **`IDLE`** – ready to pop the next goal (or terminate if the plan is finished).

This minimalistic approach keeps the control loop fully deterministic and avoids threading or callback hell.

---

## 🧪 Running the Tests

```bash
pytest -q
```

All suites should pass (`✓ 30 passed in XXs`).

---

## 📝 Design Decisions & Rationale

* **Mink** was chosen for its task‑space QP framework and Lie‑group utilities, giving full 6‑DOF IK in a few lines.
* **Robotiq 2F‑85 integration:** The STL/OBJ meshes were positioned using the official URDF, then exported to MuJoCo XML. A single position‑controlled joint (`data.ctrl[6]`) actuates both fingers for simplicity.
* **Polynomial time scaling:** Quintic blend guarantees zero velocity & acceleration at endpoints—important for a physical robot.
* **Collision modelling:** Rather than dense signed‑distance fields, discrete geom pairs are fast and easy to tweak.
* **Explicit way‑point planner:** Adequate for the assignment spec; future work could replace this with sampling‑based planning.

---

## 🔧 Troubleshooting

| Problem                                                        | Fix                                                                     |
| -------------------------------------------------------------- | ----------------------------------------------------------------------- |
| MuJoCo fails with `EGL` / `libstdc++.so` error on CPU          | `export MUJOCO_GL=egl` then `conda install -c conda-forge libstdcxx-ng` |
| `GLFW Error: X11: The DISPLAY environment variable is missing` | Run `export MUJOCO_GL=osmesa` for CPU rendering.                        |
| Solver diverges / arm jitters                                  | Lower `velocity_limit` values or increase `lm_damping`.                 |
| Gripper doesn’t move                                           | Ensure index 6 of `data.ctrl` is assigned in the XML & script.          |

---

## 🙏 Acknowledgements

* [Mink](https://github.com/PetteriAimonen/mink) for the task‑space QP solver.
* [MuJoCo](https://mujoco.org/) for the physics engine & viewer.
* Universal Robots & Robotiq for STL/OBJ asset files.

---

Enjoy exploring the workspace! Feel free to contact me at [fmdazhar@gmail.com](mailto:fmdazhar@gmail.com) with any questions.
