"""Utility helpers for collision-avoidance limits and MuJoCo geom relations."""

from dataclasses import dataclass
from typing import List, Sequence, Tuple, Union
import itertools
from itertools import product

import mujoco
import numpy as np

# Type aliases.
Geom = Union[int, str]
GeomSequence = Sequence[Geom]
CollisionPair = Tuple[GeomSequence, GeomSequence]
CollisionPairs = Sequence[CollisionPair]


"""Utilities for motion planning."""
def extend_robot_state(q_parent, q_sample, max_connection_distance):
    """
    Determines an incremental robot configuration between the parent and sample states, if one exists.

    Parameters
    ----------
        q_parent : array-like
            The starting robot configuration.
        q_sample : array-like
            The candidate sample configuration to extend towards.
        max_connection_distance : float
            Maximum angular distance, in radians, for connecting nodes.

    Returns
    -------
        array-like
            The resulting robot configuration, or None if it is not feasible.
    """
    q_diff = q_sample - q_parent
    distance = np.linalg.norm(q_diff)
    if distance == 0.0:
        return q_sample
    q_increment = max_connection_distance * q_diff / distance

    q_cur = q_parent
    # Clip the distance between nearest and sampled nodes to max connection distance.
    # If we have reached the sampled node, then we just check that.
    if configuration_distance(q_cur, q_sample) > max_connection_distance:
        q_extend = q_cur + q_increment
    else:
        q_extend = q_sample

    # Then there are no collisions so the extension is valid
    return q_extend


def has_collision_free_path(
    q1,
    q2,
    max_step_size,
    model,
    data,
    geom_id_pairs,
    min_dist,
    detect_dist,
):
    """
    Determines if there is a collision free path between the provided nodes and models.

    Parameters
    ----------
        q1 : array-like
            The starting robot configuration.
        q2 : array-like
            The destination robot configuration.
        max_step_size : float
            Maximum joint configuration step size for collision checking along path segments.
        model : `pinocchio.Model`
            The model for the robot configuration.
        collision_model : `pinocchio.Model`
            The model to use for collision checking.
        data : `pinocchio.Data`, optional
            The model data to use for this solver. If None, data is created automatically.
        collision_data : `pinocchio.GeometryData`, optional
            The collision_model data to use for this solver. If None, data is created automatically.
        distance_padding : float, optional
            The padding, in meters, to use for distance to nearest collision.

    Returns
    -------
        bool
            True if the configurations can be connected, False otherwise.
    """
    # Ensure the destination is collision free.
    if check_collisions_at_state(
        model,
        data,
        q2,
        geom_id_pairs,
        min_dist,
        detect_dist,
    ):
        return False

    # Ensure the discretized path is collision free.
    path_to_q_extend = discretize_joint_space_path([q1, q2], max_step_size)
    if check_collisions_along_path(
        model,
        data,
        path_to_q_extend,
        geom_id_pairs,
        min_dist,
        detect_dist,
    ):
        return False

    return True


def discretize_joint_space_path(q_path, max_angle_distance):
    """
    Discretize a joint-space path so that no single joint moves more than
    max_angle_distance per step (∞-norm bound), across *all* segments.
    """
    if len(q_path) == 0:
        return []
    # Normalize input to arrays
    q_path = [np.asarray(q, dtype=float) for q in q_path]
    out = [q_path[0].copy()]  # include the start once
    for i in range(1, len(q_path)):
        qs = q_path[i - 1]
        qe = q_path[i]
        diff = qe - qs
        # Per-joint bound (∞-norm): conservative and uniform
        step_mag = np.linalg.norm(diff, ord=np.inf)
        num_steps = max(1, int(np.ceil(step_mag / max_angle_distance)))
        for k in range(1, num_steps + 1):  # add intermediates + the segment end
            alpha = k / num_steps
            out.append(qs + alpha * diff)
    return out


def retrace_path(goal_node):
    """
    Retraces a path to the specified `goal_node` from a root node (a node with no parent).

    The resulting path will be returned in order form the start at index `0` to the `goal_node`
    at the index `-1`.

    Parameters
    ----------
        goal_node : `pyroboplan.planning.graph.Node`
            The starting joint configuration.

    Returns
    -------
        list[`pyroboplan.planning.graph.Node`]
            A list a nodes from the root to the specified `goal_node`.

    """
    path = []
    current = goal_node
    while current:
        path.append(current)
        current = current.parent
    path.reverse()
    return path


def discretized_joint_space_generator(model, step_size, generate_random=True):
    """
    Discretizes the entire joint space of the model at step_size increments.
    Once the entire space has been returned, the generator can optionally continue
    returning random samples from the configuration space - in which case this
    generator will never terminate.

    This is an extraordinarily expensive operation for high DOF manipulators
    and small step sizes!

    Parameters
    ----------
        model : `pinocchio.Model`
            The robot model containing lower and upper position limits.
        step_size : float
            The step size for sampling.
        generate_random : bool
            If True, continue randomly sampling the configuration space.
            Otherwise this generator will terminate.

    Yields
    ------
        np.ndarray
            The next point in the configuration space.
    """
    lower = model.lowerPositionLimit
    upper = model.upperPositionLimit

    # Ensure the range is inclusive of endpoints
    ranges = [np.arange(l, u + step_size, step_size) for l, u in zip(lower, upper)]
    for point in product(*ranges):
        yield np.array(point)

    # Once we have iterated through all available points we return random samples.
    while generate_random:
        yield get_random_state(model)



@dataclass(frozen=True)
class Contact:
    """Struct to store contact information between two geoms.

    Attributes:
        dist: Smallest signed distance between geom1 and geom2. If no collision of
            distance smaller than distmax is found, this value is equal to distmax [1].
        fromto: Segment connecting the closest points on geom1 and geom2. The first
            three elements are the coordinates of the closest point on geom1, and the
            last three elements are the coordinates of the closest point on geom2.
        geom1: ID of geom1.
        geom2: ID of geom2.
        distmax: Maximum distance between geom1 and geom2.

    References:
        [1] MuJoCo API documentation. `mj_geomDistance` function.
            https://mujoco.readthedocs.io/en/latest/APIreference/APIfunctions.html
    """

    dist: float
    fromto: np.ndarray
    geom1: int
    geom2: int
    distmax: float

    @property
    def normal(self) -> np.ndarray:
        """Contact normal pointing from geom1 to geom2."""
        normal = self.fromto[3:] - self.fromto[:3]
        mujoco.mju_normalize3(normal)
        return normal

    @property
    def inactive(self) -> bool:
        """Returns True if no distance smaller than distmax is detected between geom1
        and geom2."""
        return self.dist == self.distmax


def compute_contact_normal_jacobian(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    contact: Contact,
) -> np.ndarray:
    """Row vector J_n such that nᵀ(v₂ - v₁) = J_n q̇, projected along contact normal."""
    geom1_body = model.geom_bodyid[contact.geom1]
    geom2_body = model.geom_bodyid[contact.geom2]
    geom1_contact_pos = contact.fromto[:3]
    geom2_contact_pos = contact.fromto[3:]
    jac2 = np.empty((3, model.nv))
    mujoco.mj_jac(model, data, jac2, None, geom2_contact_pos, geom2_body)
    jac1 = np.empty((3, model.nv))
    mujoco.mj_jac(model, data, jac1, None, geom1_contact_pos, geom1_body)
    return contact.normal @ (jac2 - jac1)


def compute_contact_with_minimum_distance(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    geom1_id: int,
    geom2_id: int,
    distmax: float,
) -> Contact:
    """Return the smallest signed distance between a geom pair."""
    fromto = np.empty(6)
    dist = mujoco.mj_geomDistance(
        model,
        data,
        geom1_id,
        geom2_id,
        distmax,
        fromto,
    )
    return Contact(dist, fromto, geom1_id, geom2_id, distmax)

def contact_inside_threshold(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    g1: int,
    g2: int,
    min_dist: float,
    detect_dist: float,
) -> bool:
    """Check if two geoms are colliding or closer than `min_dist`.

    Args:
        model: MuJoCo model.
        data: MuJoCo data.
        g1: ID of first geom.
        g2: ID of second geom.
        min_dist: Minimum allowed distance.
        detect_dist: Maximum distance to consider when querying geom distance.

    Returns:
        True if in collision or too close, False otherwise.
    """
    contact = compute_contact_with_minimum_distance(
        model, data, g1, g2, detect_dist
    )
    return contact.dist <= min_dist

# ----- Geom pairing utilities -------------------------------------------------

def _homogenize_geom_id_list(model: mujoco.MjModel, geom_list: GeomSequence) -> List[int]:
    """Take a heterogeneous list of geoms (specified via ID or name) and return a
    homogenous list of IDs (int)."""
    list_of_int: List[int] = []
    for g in geom_list:
        if isinstance(g, int):
            list_of_int.append(g)
        else:
            assert isinstance(g, str)
            list_of_int.append(model.geom(g).id)
    return list_of_int


def _collision_pairs_to_geom_id_pairs(
    model: mujoco.MjModel,
    collision_pairs: CollisionPairs,
):
    geom_id_pairs = []
    for collision_pair in collision_pairs:
        id_pair_A = _homogenize_geom_id_list(model, collision_pair[0])
        id_pair_B = _homogenize_geom_id_list(model, collision_pair[1])
        id_pair_A = list(set(id_pair_A))
        id_pair_B = list(set(id_pair_B))
        geom_id_pairs.append((id_pair_A, id_pair_B))
    return geom_id_pairs


def _is_welded_together(model: mujoco.MjModel, geom_id1: int, geom_id2: int) -> bool:
    """Returns true if the geoms are part of the same body, or if their bodies are welded together."""
    body1 = model.geom_bodyid[geom_id1]
    body2 = model.geom_bodyid[geom_id2]
    weld1 = model.body_weldid[body1]
    weld2 = model.body_weldid[body2]
    return weld1 == weld2


def _are_geom_bodies_parent_child(
    model: mujoco.MjModel, geom_id1: int, geom_id2: int
) -> bool:
    """Returns true if the geom bodies have a parent-child relationship."""
    body_id1 = model.geom_bodyid[geom_id1]
    body_id2 = model.geom_bodyid[geom_id2]

    # body_weldid is the ID of the body's weld.
    body_weldid1 = model.body_weldid[body_id1]
    body_weldid2 = model.body_weldid[body_id2]

    # weld_parent_id is the ID of the parent of the body's weld.
    weld_parent_id1 = model.body_parentid[body_weldid1]
    weld_parent_id2 = model.body_parentid[body_weldid2]

    # weld_parent_weldid is the weld ID of the parent of the body's weld.
    weld_parent_weldid1 = model.body_weldid[weld_parent_id1]
    weld_parent_weldid2 = model.body_weldid[weld_parent_id2]

    cond1 = body_weldid1 == weld_parent_weldid2
    cond2 = body_weldid2 == weld_parent_weldid1
    return cond1 or cond2


def _is_pass_contype_conaffinity_check(
    model: mujoco.MjModel, geom_id1: int, geom_id2: int
) -> bool:
    """Returns true if the geoms pass the contype/conaffinity check."""
    cond1 = bool(model.geom_contype[geom_id1] & model.geom_conaffinity[geom_id2])
    cond2 = bool(model.geom_contype[geom_id2] & model.geom_conaffinity[geom_id1])
    return cond1 or cond2


def construct_geom_id_pairs(
    model: mujoco.MjModel,
    geom_pairs: CollisionPairs,
    *,
    enforce_contype_conaffinity: bool = False,
):
    """Return a set of geom ID pairs for all possible geom-geom collisions.

    The contacts are added based on the following heuristics:
        1) Geoms that are part of the same body or weld are not included.
        2) Geoms where the body of one geom is a parent of the body of the other
           geom are not included.
        3) Geoms that fail the contype-conaffinity check are ignored (optional).

    Note:
        1) If two bodies are kinematically welded together (no joints between them)
           they are considered to be the same body within this function.
    """
    geom_id_pairs = []
    for id_pair in _collision_pairs_to_geom_id_pairs(model, geom_pairs):
        for geom_a, geom_b in itertools.product(*id_pair):
            weld_body_cond = not _is_welded_together(model, geom_a, geom_b)
            parent_child_cond = not _are_geom_bodies_parent_child(model, geom_a, geom_b)
            if enforce_contype_conaffinity:
                contype_conaffinity_cond = _is_pass_contype_conaffinity_check(
                    model, geom_a, geom_b
                )
            else:
                # Match previous behavior: skip the check.
                contype_conaffinity_cond = True

            if weld_body_cond and parent_child_cond and contype_conaffinity_cond:
                geom_id_pairs.append((min(geom_a, geom_b), max(geom_a, geom_b)))
    return geom_id_pairs



def check_collisions_at_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    q: np.ndarray,
    geom_id_pairs: list[tuple[int, int]],
    min_dist: float,
    detect_dist: float,
) -> bool:
    """
    Check whether a given joint configuration q is in collision.

    Args:
        model: MuJoCo model
        data: MuJoCo data (will be modified as scratch space)
        q: full joint configuration (size model.nq)
        geom_id_pairs: list of (geom1, geom2) pairs to check
        min_dist: minimum allowed distance before treating as collision
        detect_dist: distance threshold to query contacts

    Returns:
        True if in collision, False otherwise.
    """
    # Set full qpos
    data.qpos[:] = q
    mujoco.mj_forward(model, data)

    # Check collisions
    for g1, g2 in geom_id_pairs:
        if contact_inside_threshold(model, data, g1, g2, min_dist, detect_dist):
            return True
    return False


def check_collisions_along_path(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    q_path: list[np.ndarray],
    geom_id_pairs: list[tuple[int, int]],
    min_dist: float = 0.07,
    detect_dist: float = 0.2,
    return_first: bool = True,
) -> bool | list[int]:
    """
    Check whether any configuration along a path collides.

    Args:
        model: MuJoCo model
        data: MuJoCo data (scratch space, will be modified)
        q_path: list of full joint configurations (each size model.nq)
        geom_id_pairs: list of (geom1, geom2) pairs to check
        min_dist: minimum allowed distance before treating as collision
        detect_dist: distance threshold to query contacts
        return_first: 
            - If True → return True/False (collision anywhere?).
            - If False → return list of indices in q_path that are colliding.

    Returns:
        bool if return_first=True, else list of indices of colliding states.
    """
    collisions = []
    for i, q in enumerate(q_path):
        if check_collisions_at_state(
            model, data, q, geom_id_pairs, min_dist, detect_dist
        ):
            if return_first:
                return True
            collisions.append(i)

    return bool(collisions) if return_first else collisions


def configuration_distance(q_start, q_end):
    """
    Returns the distance between two joint configurations.

    Parameters
    ----------
        q_start : array-like
            The start joint configuration.
        q_end : array-like
            The end joint configuration.

    Returns
    -------
        float
            The distance between the two joint configurations.
    """
    return np.linalg.norm(q_end - q_start)


def get_path_length(q_path):
    """
    Returns the configuration distance of a path.

    Parameters
    ----------
        q_path : list[array-like]
            A list of joint configurations describing a path.

    Returns
    -------
        float
            The total configuration distance of the entire path.
    """
    total_distance = 0.0
    for idx in range(1, len(q_path)):
        total_distance += configuration_distance(q_path[idx - 1], q_path[idx])
    return total_distance


def get_random_state(model, padding=0.0):
    """
    Returns a random state that is within the model's joint limits.

    Parameters
    ----------
        joint_limits : tuple[array-like, array-like]
            The lower and upper joint limits.
        padding : float or array-like, optional
            The padding to use around the sampled joint limits.

    Returns
    -------
        array-like
            A set of randomly generated joint variables.
    """
    for jnt in range(model.njnt):
        jnt_type = model.jnt_type[jnt]
        if (
            jnt_type == mujoco.mjtJoint.mjJNT_FREE
            or not model.jnt_limited[jnt]
        ):
            continue
        lows = model.jnt_range[jnt, 0]
        highs = model.jnt_range[jnt, 1]
    return np.random.uniform(lows + padding, highs - padding)


def get_random_state(model, padding=0.0):
    lows = []
    highs = []
    for j_idx in range(model.njnt):
        # MuJoCo stores joint ranges in model.jnt_range for hinge/slide joints,
        # but some models may use +/- inf to indicate no limit. Clamp to +/- pi as fallback.
        lo, hi = model.jnt_range[j_idx]
        if not np.isfinite(lo) or not np.isfinite(hi):
            lo, hi = -np.pi, np.pi
        lows.append(lo)
        highs.append(hi)
    lows, highs = np.asarray(lows), np.asarray(highs)
    return np.random.uniform(lows + padding, highs - padding)
