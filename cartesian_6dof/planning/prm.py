"""Utilities for manipulation-specific Probabilistic Roadmaps (PRMs)."""

import numpy as np
import time
import mujoco

from cartesian_6dof.planning.graph_search import astar
from typing import Iterable, List, Optional, Sequence, Tuple, Union

from cartesian_6dof.limits.utils import (
    check_collisions_at_state,
    get_random_state,
    construct_geom_id_pairs,
    has_collision_free_path
)

from .graph import Node, Graph


def random_model_state_generator(model):
    while True:
        yield get_random_state(model)

Geom = Union[int, str]
GeomGroup = Sequence[Geom]
CollisionPair = Tuple[GeomGroup, GeomGroup]

class PRMPlannerOptions:
    """Options for Probabilistic Roadmap (PRM) planning."""

    def __init__(
        self,
        max_step_size=0.05,
        max_neighbor_radius=0.5,
        max_neighbor_connections=15,
        max_construction_nodes=5000,
        construction_timeout=10.0,
        rng_seed=None,
        prm_star=False,
        prm_file=None,
        minimum_distance_from_collisions=0.05,
        collision_detection_distance=0.3,
    ):
        """
        Initializes a set of PRM planner options.

        Parameters
        ----------
            max_step_size : float
                Maximum joint configuration step size for collision checking along path segments.
            max_neighbor_radius : float
                The maximum allowable connectable distance between nodes.
            max_neighbor_connections : float
                The maximum number of neighbors to check when adding a node to the roadmap.
            max_construction_nodes : int
                The maximum number of samples to generate in the configuration space when growing the graph.
            construction_timeout : float
                Maximum time allotted to sample the configuration space per call.
            rng_seed : int, optional
                Sets the seed for random number generation. Use to generate deterministic results.
            prm_star : str
                If True, use the PRM* approach to dynamically select the radius and max number of neighbors
                during construction of the roadmap.
            prm_file : str, optional
                Full file path of a persisted PRM graph to use in the planner.
                If this is not specified, the PRM will be constructed from scratch.
        """
        self.max_step_size = max_step_size
        self.max_neighbor_radius = max_neighbor_radius
        self.max_neighbor_connections = max_neighbor_connections
        self.rng_seed = rng_seed
        self.construction_timeout = construction_timeout
        self.max_construction_nodes = max_construction_nodes
        self.prm_star = prm_star
        self.prm_file = prm_file
        self.minimum_distance_from_collisions = minimum_distance_from_collisions
        self.collision_detection_distance = collision_detection_distance



class PRMPlanner:
    """Probabilistic Roadmap (PRM) planner.

    This is a sampling-based motion planner that constructs a graph in the free configuration
    space and then searches for a path using standard graph search functions.

    Graphs can be persisted to disk for use in future applications.

    Some helpful resources:
        * The original publication:
          https://www.kavrakilab.org/publications/kavraki-svestka1996probabilistic-roadmaps-for.pdf
        * Modifications of PRM including PRM*:
          https://people.eecs.berkeley.edu/~pabbeel/cs287-fa19/optreadings/rrtstar.pdf
        * A nice guide to higher dimensional motion planning:
          https://motion.cs.illinois.edu/RoboticSystems/MotionPlanningHigherDimensions.html

    """

    def __init__(self, model, collision_pairs: Sequence[CollisionPair], options=PRMPlannerOptions()):
        """
        Creates an instance of a PRM planner.

        Parameters
        ----------
            model : `pinocchio.Model`
                The model to use for this solver.
            collision_model : `pinocchio.Model`
                The model to use for collision checking.
            options : `PRMPlannerOptions`, optional
                The options to use for planning. If not specified, default options are used.
        """
        self.model = model
        self.data = mujoco.MjData(model)
        self.geom_id_pairs = construct_geom_id_pairs(
                    self.model, collision_pairs, enforce_contype_conaffinity=False
                )
        self.options = options
        self.latest_path = None

        if not self.options.prm_file:
            self.graph = Graph()
        else:
            self.graph = Graph.load_from_file(self.options.prm_file)

    def construct_roadmap(self, sample_generator=None):
        """
        Grows the graph by sampling nodes using the provided generator, then connecting
        them to the Graph. The caller can optionally override the default generator, if desired.

        Parameters
        ----------
            sample_generator : Generator[array-like, None, None]
                The sample function to use in construction of the roadmap.
                Defaults to randomly sampling the robot's configuration space.
        """
        # Default to randomly sampling the model if no sample function is provided.
        np.random.seed(self.options.rng_seed)
        if not sample_generator:
            sample_generator = random_model_state_generator(self.model)

        t_start = time.time()
        added_nodes = 0
        while added_nodes < self.options.max_construction_nodes:
            if time.time() - t_start > self.options.construction_timeout:
                print(
                    f"Roadmap construction timed out after {self.options.construction_timeout} seconds."
                )
                break

            # At each iteration we naively sample a valid random state and attempt to connect it to the roadmap.
            q_sample = next(sample_generator)
            if check_collisions_at_state(
                self.model,
                self.data,
                q_sample,
                self.geom_id_pairs,
                self.options.minimum_distance_from_collisions,
                self.options.collision_detection_distance,
            ):
                continue

            radius = self.options.max_neighbor_radius
            max_neighbors = self.options.max_neighbor_connections

            # If using PRM* we dynamically scale the radius and max number of connections
            # each iteration. The scaling is a function of log(num_nodes). For more info refer to section
            # 3.3 of https://people.eecs.berkeley.edu/~pabbeel/cs287-fa19/optreadings/rrtstar.pdf.
            if self.options.prm_star:
                num_nodes = len(self.graph.nodes)
                dimension = len(q_sample)
                if num_nodes > 0:
                    radius = radius * (np.log(num_nodes) / num_nodes) ** (1 / dimension)
                    max_neighbors = int(max_neighbors * np.log(num_nodes))

            # It's a valid configuration so add it to the roadmap
            new_node = Node(q_sample)
            self.graph.add_node(new_node)
            self.connect_node(new_node, radius, max_neighbors)
            added_nodes += 1

    def connect_node(self, new_node, radius, k):
        """
        Identifies all neighbors and makes connections to the added node.

        Parameters
        ----------
            parent_node : `pyroboplan.planning.graph.Node`
                The node to add.
            radius : float
                Only consider connections within the provided radius.
            k : int
                Only consider up to a maximum of k neighbors to make connections.

        Returns
        -------
            bool :
                True if the node was connected to the graph. False otherwise.
        """
        # Add the node and find all neighbors.
        neighbors = self.graph.get_nearest_neighbors(new_node.q, radius)

        # Attempt to connect at most `max_neighbor_connections` neighbors.
        success = False
        for node, _ in neighbors[0:k]:
            # If the nodes are connectable then add an edge.
            if has_collision_free_path(
                node.q,
                new_node.q,
                self.options.max_step_size,
                self.model,
                self.data,
                self.geom_id_pairs,
                self.options.minimum_distance_from_collisions,
                self.options.collision_detection_distance,
            ):
                self.graph.add_edge(node, new_node)
                success |= True

        return success

    def reset(self):
        """
        Resets the PRM's transient data between queries.
        """
        self.latest_path = None
        for node in self.graph.nodes:
            node.parent = None
            node.cost = None

    def connect_planning_nodes(self, start_node, goal_node):
        """
        Ensures the start and goal configurations can be connected to the PRM.

        Parameters
        ----------
            start_node : `pyroboplan.planning.graph.Node`
                The start node to connect.
            goal_node : `pyroboplan.planning.graph.Node`
                The goal node to connect.

        Returns
        -------
            bool :
                True if the nodes were able to be connected, False otherwise.
        """
        success = True

        # If we cannot connect the start and goal nodes then there is no recourse.
        if not self.connect_node(
            start_node,
            self.options.max_neighbor_radius,
            self.options.max_neighbor_connections,
        ):
            print("Failed to connect the start configuration to the PRM.")
            success = False
        if not self.connect_node(
            goal_node,
            self.options.max_neighbor_radius,
            self.options.max_neighbor_connections,
        ):
            print("Failed to connect the goal configuration to the PRM.")
            success = False

        return success

    def plan(self, q_start, q_goal):
        """
        Plans a path from a start to a goal configuration using the constructed graph.

        Parameters
        ----------
            q_start : array-like
                The starting robot configuration.
            q_goal : array-like
                The goal robot configuration.

        Returns
        -------
            list[array-like] :
                A path from the start to the goal state, if one exists. Otherwise None.
        """

        # Check start and end pose collisions.
        if check_collisions_at_state(
            self.model, self.data, q_start, self.geom_id_pairs, self.options.minimum_distance_from_collisions, self.options.collision_detection_distance
        ):
            print("Start configuration in collision.")
            return None
        if check_collisions_at_state(
            self.model, self.data, q_goal, self.geom_id_pairs, self.options.minimum_distance_from_collisions, self.options.collision_detection_distance 
        ):
            print("Goal configuration in collision.")
            return None

        # Ensure the start and goal nodes are in the graph.
        start_node = Node(q_start)
        self.graph.add_node(start_node)
        goal_node = Node(q_goal)
        self.graph.add_node(goal_node)

        # Ensure the start and goal nodes are connected before attempting to plan
        path = None
        if self.connect_planning_nodes(start_node, goal_node):

            # Use a graph search to determine if there is a path between the start and goal poses.
            node_path = astar(self.graph, start_node, goal_node)

            # Reconstruct the path if it exists
            path = [node.q for node in node_path] if node_path else None
            self.latest_path = path

        # Always remove the start and end nodes from the PRM.
        self.graph.remove_node(start_node)
        self.graph.remove_node(goal_node)

        return path

    