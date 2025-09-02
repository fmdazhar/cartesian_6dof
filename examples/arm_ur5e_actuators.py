from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter

import cartesian_6dof
from cartesian_6dof.contrib import TeleopMocap

_HERE = Path(__file__).parent
_XML = _HERE / "universal_robots_ur5e" / "scene_plain.xml"


if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())
    data = mujoco.MjData(model)


    # print("Links:", [model.body(i).name for i in range(model.nbody)],
    #     "Joints:", [model.joint(i).name for i in range(model.njnt)])

    ## =================== ##
    ## Setup IK.
    ## =================== ##

    configuration = cartesian_6dof.Configuration(model)

    tasks = [
        end_effector_task := cartesian_6dof.FrameTask(
            frame_name="attachment_site",
            frame_type="site",
            position_cost=1.0,
            orientation_cost=1.0,
            lm_damping=1e-6,
        ),
        posture_task := cartesian_6dof.PostureTask(model, cost=1e-3),
    ]

    collision_pairs = []

    limits = [
        cartesian_6dof.ConfigurationLimit(model=configuration.model),
        cartesian_6dof.CollisionAvoidanceLimit(
            model=configuration.model,
            geom_pairs=collision_pairs,
        ),
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

    ## =================== ##

    mid = model.body("target").mocapid[0]

    # IK settings.
    solver = "daqp"
    pos_threshold = 1e-4
    ori_threshold = 1e-4
    max_iters = 20

    # Initialize key_callback function.
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

        # Initialize the mocap target at the end-effector site.
        cartesian_6dof.move_mocap_to_frame(model, data, "target", "attachment_site", "site")

        rate = RateLimiter(frequency=200.0, warn=False)
        while viewer.is_running():
            # Update task target.
            T_wt = cartesian_6dof.SE3.from_mocap_name(model, data, "target")
            end_effector_task.set_target(T_wt)

            # Continuously check for autonomous key movement.
            key_callback.auto_key_move()

            # Compute velocity and integrate into the next configuration.
            for i in range(max_iters):
                vel = cartesian_6dof.solve_ik(
                    configuration, tasks, rate.dt, solver, limits=limits
                )
                configuration.integrate_inplace(vel, rate.dt)
                err = end_effector_task.compute_error(configuration)
                pos_achieved = np.linalg.norm(err[:3]) <= pos_threshold
                ori_achieved = np.linalg.norm(err[3:]) <= ori_threshold
                if pos_achieved and ori_achieved:
                    break

            data.ctrl = configuration.q
            mujoco.mj_step(model, data)

            # Visualize at fixed FPS.
            viewer.sync()
            rate.sleep()
