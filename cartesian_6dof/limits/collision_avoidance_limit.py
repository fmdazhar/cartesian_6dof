"""Collision avoidance limit."""

import numpy as np
import mujoco

from dataclasses import dataclass
from typing import Sequence, Tuple, Union

from ..configuration import Configuration
from .limit import Constraint, Limit

# Type aliases.
Geom = Union[int, str]
GeomSequence = Sequence[Geom]
CollisionPair = Tuple[GeomSequence, GeomSequence]
CollisionPairs = Sequence[CollisionPair]

# Utils
from .utils import (
    Contact,
    compute_contact_normal_jacobian,
    compute_contact_with_minimum_distance,
    construct_geom_id_pairs,
)


class CollisionAvoidanceLimit(Limit):
    """Normal velocity limit between geom pairs.

    Attributes:
        model: MuJoCo model.
        geom_pairs: Set of collision pairs in which to perform active collision
            avoidance. A collision pair is defined as a pair of geom groups. A geom
            group is a set of geom names. For each geom pair, the solver will
            attempt to compute joint velocities that avoid collisions between every
            geom in the first geom group with every geom in the second geom group.
            Self collision is achieved by adding a collision pair with the same
            geom group in both pair fields.
        gain: Gain factor in (0, 1] that determines how fast the geoms are
            allowed to move towards each other at each iteration. Smaller values
            are safer but may make the geoms move slower towards each other.
        minimum_distance_from_collisions: The minimum distance to leave between
            any two geoms. A negative distance allows the geoms to penetrate by
            the specified amount.
        collision_detection_distance: The distance between two geoms at which the
            active collision avoidance limit will be active. A large value will
            cause collisions to be detected early, but may incur high computational
            cost. A negative value will cause the geoms to be detected only after
            they penetrate by the specified amount.
        bound_relaxation: An offset on the upper bound of each collision avoidance
            constraint.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        geom_pairs: CollisionPairs,
        gain: float = 0.85,
        minimum_distance_from_collisions: float = 0.1,
        collision_detection_distance: float = 0.5,
        bound_relaxation: float = 0.0,
    ):
        """Initialize collision avoidance limit."""
        self.model = model
        self.gain = gain
        self.minimum_distance_from_collisions = minimum_distance_from_collisions
        self.collision_detection_distance = collision_detection_distance
        self.bound_relaxation = bound_relaxation

        # Build all eligible geom-geom pairs once.
        self.geom_id_pairs = construct_geom_id_pairs(self.model, geom_pairs)
        self.max_num_contacts = len(self.geom_id_pairs)

    def compute_qp_inequalities(
        self,
        configuration: Configuration,
        dt: float,
    ) -> Constraint:
        upper_bound = np.full((self.max_num_contacts,), np.inf)
        coefficient_matrix = np.zeros((self.max_num_contacts, self.model.nv))

        for idx, (geom1_id, geom2_id) in enumerate(self.geom_id_pairs):
            contact = compute_contact_with_minimum_distance(
                self.model,
                configuration.data,
                geom1_id,
                geom2_id,
                self.collision_detection_distance,
            )

            if contact.inactive:
                continue

            hi_bound_dist = contact.dist
            if hi_bound_dist > self.minimum_distance_from_collisions:
                dist = hi_bound_dist - self.minimum_distance_from_collisions
                upper_bound[idx] = (self.gain * dist / dt) + self.bound_relaxation
            else:
                upper_bound[idx] = self.bound_relaxation

            jac = compute_contact_normal_jacobian(
                self.model, configuration.data, contact
            )
            coefficient_matrix[idx] = -jac

        return Constraint(G=coefficient_matrix, h=upper_bound)
