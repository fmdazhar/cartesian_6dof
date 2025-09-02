"""Utilities for manipulation-specific Rapidly-Exploring Random Trees (RRTs)."""

import numpy as np
import mujoco
import time
from typing import Iterable, List, Optional, Sequence, Tuple, Union

from cartesian_6dof.limits.utils import (
    check_collisions_at_state,
    check_collisions_along_path,
    configuration_distance,
    get_random_state,
    construct_geom_id_pairs,
    discretize_joint_space_path,
    extend_robot_state,
    has_collision_free_path,
    retrace_path,
)

from .graph import Node, Graph

Geom = Union[int, str]
GeomGroup = Sequence[Geom]
CollisionPair = Tuple[GeomGroup, GeomGroup]


class RRTPlannerOptions:
    """Options for Rapidly-exploring Random Tree (RRT) planning."""

    def __init__(
        self,
        max_step_size=0.05,
        max_connection_dist=np.inf,
        rrt_connect=False,
        bidirectional_rrt=False,
        rrt_star=False,
        max_rewire_dist=np.inf,
        max_planning_time=10.0,
        rng_seed=None,
        fast_return=True,
        goal_biasing_probability=0.0,
        minimum_distance_from_collisions=0.05,
        collision_detection_distance=0.3,
    ):
        """
        Initializes a set of RRT planner options.

        Parameters
        ----------
            max_step_size : float
                Maximum joint configuration step size for collision checking along path segments.
            max_connection_dist : float
                Maximum angular distance, in radians, for connecting nodes.
            rrt_connect : bool
                If true, enables the RRTConnect algorithm, which incrementally extends the most
                recently sampled node in the tree until an invalid state is reached.
            bidirectional_rrt : bool
                If true, uses bidirectional RRTs from both start and goal nodes.
                Otherwise, only grows a tree from the start node.
            rrt_star : bool
                If true, enables the RRT* algorithm to shortcut node connections during planning.
                This in turn will use the `max_rewire_dist` parameter.
            max_rewire_dist : float
                Maximum angular distance, in radians, to consider rewiring nodes for RRT*.
                If set to `np.inf`, all nodes in the trees will be considered for rewiring.
            max_planning_time : float
                Maximum planning time, in seconds.
            rng_seed : int, optional
                Sets the seed for random number generation. Use to generate deterministic results.
            fast_return : bool
                If True, return as soon as a solution is found. Otherwise continuing building the tree
                until max_planning_time is reached.
            goal_biasing_probability : float
                Probability of sampling the goal configuration itself, which can help planning converge.
            minimum_distance_from_collisions : float
                Minimum distance to maintain from obstacles during planning.
            collision_detection_distance : float
                Distance to use for collision detection checks.
        """
        self.max_step_size = max_step_size
        self.max_connection_dist = max_connection_dist
        self.rrt_connect = rrt_connect
        self.bidirectional_rrt = bidirectional_rrt
        self.rrt_star = rrt_star
        self.max_rewire_dist = max_rewire_dist
        self.max_planning_time = max_planning_time
        self.rng_seed = rng_seed
        self.fast_return = fast_return
        self.goal_biasing_probability = goal_biasing_probability
        self.minimum_distance_from_collisions = minimum_distance_from_collisions
        self.collision_detection_distance = collision_detection_distance


class RRTPlanner:
    """Rapidly-expanding Random Tree (RRT) planner.

    This is a sampling-based motion planner that finds collision-free paths from a start to a goal configuration.

    Some good resources:
      * Original RRT paper: https://msl.cs.illinois.edu/~lavalle/papers/Lav98c.pdf
      * RRTConnect paper: https://www.cs.cmu.edu/afs/cs/academic/class/15494-s14/readings/kuffner_icra2000.pdf
      * RRT* and PRM* paper: https://arxiv.org/abs/1105.1186
    """

    def __init__(self, model: mujoco.MjModel, collision_pairs: Sequence[CollisionPair], options=RRTPlannerOptions()):
        """
        Creates an instance of an RRT planner.

        Parameters
        ----------
            model : `pinocchio.Model`
                The model to use for this solver.
            collision_model : `pinocchio.Model`
                The model to use for collision checking.
            options : `RRTPlannerOptions`, optional
                The options to use for planning. If not specified, default options are used.
        """
        self.model = model
        self.data = mujoco.MjData(model)
        self.geom_id_pairs = construct_geom_id_pairs(
            self.model, collision_pairs, enforce_contype_conaffinity=False
        )
        self.options = options
        self.reset()

    def reset(self):
        """Resets all the planning data structures."""
        self.latest_path = None
        self.start_tree = Graph()
        self.goal_tree = Graph()
        np.random.seed(self.options.rng_seed)

    def plan(self, q_start, q_goal):
        """
        Plans a path from a start to a goal configuration.

        Parameters
        ----------
            q_start : array-like
                The starting robot configuration.
            q_goal : array-like
                The goal robot configuration.
        """
        self.reset()
        t_start = time.time()

        start_node = Node(q_start, parent=None, cost=0.0)
        self.start_tree.add_node(start_node)
        goal_node = Node(q_goal, parent=None, cost=0.0)
        self.goal_tree.add_node(goal_node)

        goal_found = False
        latest_start_tree_node = start_node
        latest_goal_tree_node = goal_node

        # Check start and end pose collisions.
        if check_collisions_at_state(
            self.model,
            self.data,
            q_start,
            self.geom_id_pairs,
            self.options.minimum_distance_from_collisions,
            self.options.collision_detection_distance,
        ):
            print("Start configuration in collision.")
            return None
        if check_collisions_at_state(
            self.model,
            self.data,
            q_goal,
            self.geom_id_pairs,
            self.options.minimum_distance_from_collisions,
            self.options.collision_detection_distance,
        ):
            print("Goal configuration in collision.")
            return None

        # Check direct connection to goal.
        if configuration_distance(q_start, q_goal) <= self.options.max_connection_dist:
            path_to_goal = discretize_joint_space_path(
                [q_start, q_goal], self.options.max_step_size
            )
            if not check_collisions_along_path(
                self.model,
                self.data,
                path_to_goal,
                self.geom_id_pairs,
                self.options.minimum_distance_from_collisions,
                self.options.collision_detection_distance,
            ):
                latest_start_tree_node = self.add_node_to_tree(
                    self.start_tree, q_goal, start_node
                )
                print("Start and goal can be directly connected!")
                goal_found = True

        start_tree_phase = True
        while True:
            # Only return on success if specified in the options.
            if goal_found and self.options.fast_return:
                break

            # Check for timeouts.
            if time.time() - t_start > self.options.max_planning_time:
                message = "succeeded" if goal_found else "timed out"
                print(
                    f"Planning {message} after {self.options.max_planning_time} seconds."
                )
                break

            # Choose variables based on whether we're growing the start or goal tree.
            tree = self.start_tree if start_tree_phase else self.goal_tree
            other_tree = self.goal_tree if start_tree_phase else self.start_tree

            # Sample a new configuration.
            if np.random.random() < self.options.goal_biasing_probability:
                q_sample = q_goal if start_tree_phase else q_start
            else:
                q_sample = get_random_state(self.model)

            # Run the extend or connect operation to connect the tree to the new node.
            nearest_node = tree.get_nearest_node(q_sample)
            new_node = self.extend_or_connect(tree, nearest_node, q_sample)

            # Only if extend/connect succeeded, add the new node to the tree.
            if new_node is not None:
                if start_tree_phase:
                    latest_start_tree_node = new_node
                else:
                    latest_goal_tree_node = new_node

                # Check if latest node connects directly to the other tree.
                # If so, add it to the tree and mark planning as complete.
                nearest_node_in_other_tree = other_tree.get_nearest_node(new_node.q)
                distance_to_other_tree = configuration_distance(
                    new_node.q, nearest_node_in_other_tree.q
                )
                if distance_to_other_tree <= self.options.max_connection_dist:
                    path_to_other_tree = discretize_joint_space_path(
                        [new_node.q, nearest_node_in_other_tree.q],
                        self.options.max_step_size,
                    )
                    if not check_collisions_along_path(
                        self.model,
                        self.data,
                        path_to_other_tree,
                        self.geom_id_pairs,
                        self.options.minimum_distance_from_collisions,
                        self.options.collision_detection_distance,
                    ):
                        if distance_to_other_tree > 0:
                            new_node = self.add_node_to_tree(
                                tree, nearest_node_in_other_tree.q, new_node
                            )
                        if start_tree_phase:
                            latest_start_tree_node = new_node
                            latest_goal_tree_node = nearest_node_in_other_tree
                        else:
                            latest_start_tree_node = nearest_node_in_other_tree
                            latest_goal_tree_node = new_node
                        goal_found = True

                # Switch to the other tree next iteration, if bidirectional mode is enabled.
                if self.options.bidirectional_rrt:
                    start_tree_phase = not start_tree_phase

        # Back out the path by traversing the parents from the goal.
        self.latest_path = []
        if goal_found:
            self.latest_path = self.extract_path_from_trees(
                latest_start_tree_node, latest_goal_tree_node
            )
        return self.latest_path

    def extend_or_connect(self, tree, parent_node, q_sample):
        """
        Extends a tree towards a sampled node with steps no larger than the maximum connection distance.

        Parameters
        ----------
            tree : `pyroboplan.planning.graph.Graph`
                The tree to use when performing this operation.
            parent_node : `pyroboplan.planning.graph.Node`
                The node from which to start extending or connecting towards the sample.
            q_sample : array-like
                The robot configuration sample to extend or connect towards.

        Return
        ------
            `pyroboplan.planning.graph.Node`, optional
                The latest node that was added to the tree, or `None` if no node was found.
        """
        # If they are the same node there's nothing to do.
        if np.array_equal(parent_node.q, q_sample):
            return None

        cur_parent_node = parent_node
        cur_node = None
        while True:
            # Compute the next incremental robot configuration.
            q_extend = extend_robot_state(
                cur_parent_node.q,
                q_sample,
                self.options.max_connection_dist,
            )

            # If we can connect then it is a valid state
            if not has_collision_free_path(
                cur_parent_node.q,
                q_extend,
                self.options.max_step_size,
                self.model,
                self.data,
                self.geom_id_pairs,
                min_dist=self.options.minimum_distance_from_collisions,
                detect_dist=self.options.collision_detection_distance,
            ):
                break

            cur_node = self.add_node_to_tree(tree, q_extend, cur_parent_node)

            # If RRT-Connect is disabled, only one iteration is needed.
            if not self.options.rrt_connect:
                break

            # If we have reached the final configuration, we are done.
            if np.array_equal(cur_node.q, q_sample):
                break

            cur_parent_node = cur_node

        return cur_node

    def extract_path_from_trees(self, start_tree_final_node, goal_tree_final_node):
        """
        Extracts the final path from the RRT trees

        from the start tree root to the goal tree root passing through both final nodes.

        Parameters
        ----------
            start_tree_final_node : `pyroboplan.planning.graph.Node`
                The last node of the start tree.
            goal_tree_final_node : `pyroboplan.planning.graph.Node`, optional
                The last node of the goal tree.
                If None, this means the goal tree is ignored.

        Return
        ------
            list[array-like]
                A list of robot configurations describing the path waypoints in order.
        """
        path = retrace_path(start_tree_final_node)

        # extract and reverse the goal tree path to append to the start tree path
        if goal_tree_final_node:
            # the final node itself is already in the start path
            goal_tree_path = retrace_path(goal_tree_final_node.parent)
            goal_tree_path.reverse()
            path += goal_tree_path

        # Convert to robot configuration states
        return [n.q for n in path]

    def add_node_to_tree(self, tree, q_new, parent_node):
        """
        Add a new node to the tree. If the RRT* algorithm is enabled, will also rewire.

        Parameters
        ----------
            tree : `pyroboplan.planning.graph.Graph`
                The tree to which to add the new node.
            q_new : array-like
                The robot configuration from which to create a new tree node.
            parent_node : `pyroboplan.planning.graph.Node`
                The parent node to connect the new node to.

        Returns
        -------
            `pyroboplan.planning.graph.Node`
                The new node that was added to the tree.
        """
        # Add the new node to the tree
        new_node = Node(q_new, parent=parent_node)
        tree.add_node(new_node)
        edge = tree.add_edge(parent_node, new_node)
        new_node.cost = parent_node.cost + edge.cost

        # If RRT* is enable it, rewire that node in the tree.
        if self.options.rrt_star:
            min_cost = new_node.cost
            for other_node in tree.nodes:
                # Do not consider trivial nodes.
                if other_node == new_node or other_node == parent_node:
                    continue
                # Do not consider nodes farther than the configured rewire distance,
                new_distance = configuration_distance(other_node.q, q_new)
                if new_distance > self.options.max_rewire_dist:
                    continue
                # Rewire if this new connections would be of lower cost and is collision free.
                new_cost = other_node.cost + new_distance
                if new_cost < min_cost:
                    new_path = discretize_joint_space_path(
                        [q_new, other_node.q], self.options.max_step_size
                    )
                    if not check_collisions_along_path(
                        self.model,
                        self.data,
                        new_path,
                        self.geom_id_pairs,
                        self.options.minimum_distance_from_collisions,
                        self.options.collision_detection_distance,
                    ):
                        new_node.parent = other_node
                        new_node.cost = new_cost
                        tree.remove_edge(parent_node, new_node)
                        edge = tree.add_edge(other_node, new_node)
                        min_cost = new_cost

        return new_node
