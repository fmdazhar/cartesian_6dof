from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter
import os

import cartesian_6dof
from cartesian_6dof.contrib import TeleopMocap

from cartesian_6dof.planning.prm import PRMPlanner, PRMPlannerOptions # import the PRM planner
from cartesian_6dof.planning.path_shortcutting import shortcut_path
from cartesian_6dof.limits.utils import discretize_joint_space_path, construct_geom_id_pairs
_HERE = Path(__file__).parent
_XML = _HERE / "universal_robots_ur5e" / "scene_gripper.xml"


if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())
    data = mujoco.MjData(model)

    # =================== #
    # Setup IK.
    # =================== #
    
    configuration = cartesian_6dof.Configuration(model)

    tasks = [
        end_effector_task := cartesian_6dof.FrameTask(
            frame_name="pinch",
            frame_type="site",
            position_cost=1.0,
            orientation_cost=1.0,
            lm_damping=1e-6,
        ),
        posture_task := cartesian_6dof.PostureTask(model, cost=1e-3),
    ]

    # Explicit list of every collision geom on the arm
    _arm_collision_geom_names = [
        "shoulder_col",
        "upper_arm_col0",
        "upper_arm_col1",
        "forearm_col0",
        "forearm_col1",
        "wrist_1_col",
        "wrist_2_col",
        "wrist_3_col",
    ]

    arm_collision_geoms = [model.geom(name).id for name in _arm_collision_geom_names]

    gripper_proxy_geom_names = ["gripper_dist_box", "right_pad_ell", "left_pad_ell"]
    gripper_proxy_geoms = [model.geom(name).id for name in gripper_proxy_geom_names]

    collision_pairs = [
        (arm_collision_geoms + gripper_proxy_geoms, ["wall"]),
    ]
    geom_id_pairs = construct_geom_id_pairs(
            model, collision_pairs, enforce_contype_conaffinity=False
        )

    limits = [
        cartesian_6dof.ConfigurationLimit(model=configuration.model),
    ]

    max_velocities = {
        "shoulder_pan": np.pi,
        "shoulder_lift": np.pi,
        "elbow": np.pi,
        "wrist_1": np.pi,
        "wrist_2": np.pi,
        "wrist_3": np.pi,
    }
    velocity_limit = cartesian_6dof.VelocityLimit(model, max_velocities)
    limits.append(velocity_limit)

    mid = model.body("target").mocapid[0]

    # IK settings.
    solver = "daqp"
    pos_threshold = 1e-4
    ori_threshold = 1e-4
    max_iters = 500

    active_path: list[np.ndarray] | None = None
    path_idx: int = 0

    # PRM planner
    prm_file = "my_prm3.graph"
    load_prm = False
    if os.path.isfile(prm_file):
        print(f"Loading pre-existing PRM from: {prm_file}")
        load_prm = True

    options = PRMPlannerOptions(
            max_step_size=0.02,
            max_neighbor_radius=3.14,
            max_neighbor_connections=20,
            max_construction_nodes=10000,
            construction_timeout=1000.0,
            rng_seed=None,
            prm_star=True,
            prm_file=prm_file if load_prm else None,
            minimum_distance_from_collisions=0.05,
            collision_detection_distance=0.3,
        )
    planner = PRMPlanner(
        model=model,
        collision_pairs=collision_pairs,
        options=options
    )

    if not load_prm:
        print(
            f"Initializing the PRM, this will take up to {options.construction_timeout} seconds..."
        )
        planner.construct_roadmap()

        # We can save the graph to file for use in future PRMs.
        if prm_file:
            print(f"Saving generated PRM to {prm_file}")
            planner.graph.save_to_file(prm_file)

    key_callback = TeleopMocap(data)

    with mujoco.viewer.launch_passive(
        model=model,
        data=data,
        show_left_ui=False,
        show_right_ui=False,
        key_callback=key_callback,
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        configuration.update(data.qpos)
        mujoco.mj_forward(model, data)

        posture_task.set_target_from_configuration(configuration)
        
        cartesian_6dof.move_mocap_to_frame(model, data, "target", "pinch", "site")

        rate = RateLimiter(frequency=200.0, warn=False)

        while viewer.is_running():
            if key_callback.tracking:
                T_wt = cartesian_6dof.SE3.from_mocap_name(model, data, "target")
                end_effector_task.set_target(T_wt)
            elif key_callback.new_goal:
                # Compute start and goal
                start_q = data.qpos[:].copy()
                goal_T = cartesian_6dof.SE3.from_mocap_name(model, data, "target")

                # Solve IK once to find a feasible goal configuration
                ik_cfg = cartesian_6dof.Configuration(model)
                ik_cfg.update(configuration.q.copy())
                end_effector_task.set_target(goal_T)
                for _ in range(max_iters):
                    v = cartesian_6dof.solve_ik(ik_cfg, tasks, rate.dt, solver, limits=limits)
                    ik_cfg.integrate_inplace(v, rate.dt)
                    err = end_effector_task.compute_error(ik_cfg)
                    if np.linalg.norm(err[:3]) <= pos_threshold and np.linalg.norm(err[3:]) <= ori_threshold:
                        break
                goal_q = ik_cfg.q[:].copy()
                print(f"[PRM] Planning from {start_q} to {goal_q}")

                # Plan in joint space with RRT
                path = planner.plan(start_q, goal_q)
                if path is not None and len(path) > 1:
                    print(f"[PRM] Found path with {len(path)} waypoints")
                    # short_path = shortcut_path(model, path, geom_id_pairs, max_iters=200,
                    #                      max_step_size=0.05)
                    # print(f"[PRM] Shortcutted path to {len(short_path)} waypoints")
                    active_path = path
                    active_path = discretize_joint_space_path(active_path, 2e-3)
                    print(f"[PRM] Discretized path to {len(active_path)} waypoints")
                    path_idx = 1
                else:
                    print("[PRM] No path found")
                    active_path = None
                key_callback.new_goal = False
            
            if active_path is not None:
                if path_idx < len(active_path):
                    q_target = active_path[path_idx]
                    data.ctrl[:6] = q_target[:6]
                    path_idx += 1
                else:
                    active_path = None

            key_callback.auto_key_move()

            if active_path is None and key_callback.tracking:
                for i in range(max_iters):
                    vel = cartesian_6dof.solve_ik(configuration, tasks, rate.dt, solver, limits=limits)
                    configuration.integrate_inplace(vel, rate.dt)
                    err = end_effector_task.compute_error(configuration)
                    if np.linalg.norm(err[:3]) <= pos_threshold and np.linalg.norm(err[3:]) <= ori_threshold:
                        break
                data.ctrl[:6] = configuration.q[:6]

            mujoco.mj_step(model, data)

            viewer.sync()
            rate.sleep()
