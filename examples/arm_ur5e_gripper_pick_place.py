from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter
from collections import deque
import time

import mink


_HERE = Path(__file__).parent
_XML = _HERE / "universal_robots_ur5e" / "scene_pick_place.xml"


if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())
    data = mujoco.MjData(model)

    ## =================== ##
    ## Setup IK.
    ## =================== ##

    configuration = mink.Configuration(model)

    tasks = [
        end_effector_task := mink.FrameTask(
            frame_name="pinch",
            frame_type="site",
            position_cost=1.0,
            orientation_cost=1.0,
            lm_damping=1e-6,
        ),
        posture_task := mink.PostureTask(model, cost=1e-3),
    ]

    collision_pairs = []

    limits = [
        mink.ConfigurationLimit(model=configuration.model),
        mink.CollisionAvoidanceLimit(
            model=configuration.model,
            geom_pairs=collision_pairs,
        ),
    ]

    max_velocities = {
        "shoulder_pan": 0.1,
        "shoulder_lift": 0.1,
        "elbow": 0.1,
        "wrist_1": 0.1,
        "wrist_2": 0.1,
        "wrist_3": 0.1,
    }
    velocity_limit = mink.VelocityLimit(model, max_velocities)
    limits.append(velocity_limit)

    ## =================== ##

    # IK settings.
    solver = "daqp"
    pos_threshold = 1e-4
    ori_threshold = 1e-4
    max_iters = 20

    path_queue: deque[list[mink.SE3]] = deque()
    active_path: list[mink.SE3] | None = None
    
    path_idx: int = 0
    next_goal_idx: int = 0
    wait_steps_remaining: int = 0 
    wait_steps_current = 0 
    grip_cmd_current = None 
    last_goal_T = None            # mink.SE3 of the EE

    # Polynomial trajectory generator
    interpolator = mink.PolynomialInterpolator(order=5)

    with mujoco.viewer.launch_passive(
        model=model,
        data=data,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        configuration.update(data.qpos)
        mujoco.mj_forward(model, data)

        posture_task.set_target_from_configuration(configuration)
        
        rate = RateLimiter(frequency=200.0, warn=False)

        TABLE_Z      = 0.02      # plane z + half-cube
        APPROACH_Z   = 0.05
        LIFT_Z       = TABLE_Z + 0.15
        PLACE_XY     = (0.30, -0.20)    # where we drop the cube
        OPEN  = -255.0     # joints [rad]; adjust to match your XML
        CLOSE = 255.0
        FACE_DOWN = [0.0, 1.0, 0.0, 0.0]

        plan = [
            # ---- approach above cube ----
            ([0.51, 0.00, APPROACH_Z], FACE_DOWN, OPEN, 200),
            # ---- descend to grasp pose ----
            ([0.51, 0.00, TABLE_Z+0.01], FACE_DOWN, OPEN, 200),
            # ---- close gripper (attach happens automatically) ----
            ([0.51, 0.00, TABLE_Z+0.01], FACE_DOWN, CLOSE, 200),

            # ---- lift straight up ----
            ([0.51, 0.00, LIFT_Z], FACE_DOWN, CLOSE, 200),
            # ---- transfer above place pose ----
            ([*PLACE_XY, LIFT_Z],       FACE_DOWN, CLOSE, 200),
            # ---- descend & open ----
            ([*PLACE_XY, TABLE_Z+0.012], FACE_DOWN, CLOSE, 200),
            ([*PLACE_XY, TABLE_Z+0.012], FACE_DOWN, OPEN, 200),
            # ---- retreat ----
            ([*PLACE_XY, LIFT_Z],   FACE_DOWN, OPEN, 0),
        ]

        while viewer.is_running():
            if wait_steps_remaining:               # holding position
                wait_steps_remaining -= 1
            else:
                # Finished a path segment?
                segment_complete = active_path is not None and path_idx >= len(active_path)
                if segment_complete:
                    print(f"[INFO] Goal {next_goal_idx} command over.") 
                    active_path = None      # mark path as consumed
                    wait_steps_remaining = wait_steps_current

                # Ready for a new segment once wait is over
                if not wait_steps_remaining and active_path is None and next_goal_idx < len(plan):
                    start_T = mink.SE3.from_frame_name(model, data, "pinch", frame_type="site")
                    spec = plan[next_goal_idx]

                    goal = mink.SE3.resolve_goal(spec, start_T)
                    goal_T              = goal.pose
                    grip_cmd_current    = goal.gripper_cmd
                    wait_steps_current  = goal.wait_steps
                    
                    active_path = interpolator(start_T, goal_T, n_steps=200)
                    path_idx = 0
                    next_goal_idx += 1

            # ---------- advance along active path ----------
            if active_path and path_idx < len(active_path):
                last_goal_T = active_path[path_idx]
                path_idx += 1

            # ---------- always keep the EE pose target ----------
            if last_goal_T is not None:
                end_effector_task.set_target(last_goal_T)

            # ---------- gripper command ----------
            if grip_cmd_current is not None:
                data.ctrl[6] = grip_cmd_current

            # Compute velocity and integrate into the next configuration.
            for i in range(max_iters):
                vel = mink.solve_ik(
                    configuration, tasks, rate.dt, solver, limits=limits
                )
                configuration.integrate_inplace(vel, rate.dt)
                err = end_effector_task.compute_error(configuration)
                pos_achieved = np.linalg.norm(err[:3]) <= pos_threshold
                ori_achieved = np.linalg.norm(err[3:]) <= ori_threshold

                if pos_achieved and ori_achieved:
                    break

            data.ctrl[:6] = configuration.q[:6]

            mujoco.mj_step(model, data)

            # Visualize at fixed FPS.
            viewer.sync()
            rate.sleep()
        
