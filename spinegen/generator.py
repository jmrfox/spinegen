"""Stochastic generation of dendritic spine morphologies.

This module provides the SpineGenerator class which implements a growth-based
algorithm for generating synthetic spine morphologies with branching and fusion.
"""

import logging
import numpy as np
import networkx as nx
from typing import Tuple
from .cable_graph import CableGraph
from .prior import SpinePrior
from .utils import euclidean_distance, split_edge

logger = logging.getLogger(__name__)


class SpineGenerator:
    """Generator for synthetic dendritic spine morphologies.

    This class implements a stochastic growth-based algorithm that:
    1. Starts from a root node at the origin
    2. Iteratively extends active tips with sampled segment lengths
    3. Applies curvature perturbations to growth directions
    4. Probabilistically creates branches
    5. Detects and executes fusion events to create cycles

    Attributes:
        prior: SpinePrior instance providing statistical distributions
        graph: CableGraph being constructed (None before generation)
        next_node_id: Counter for assigning unique node IDs
    """

    def __init__(self, prior: SpinePrior):
        """Initialize generator with a statistical prior.

        Args:
            prior: SpinePrior instance with fitted distributions
        """
        self.prior = prior
        self.graph = None
        self.next_node_id = 1
        logger.info("Initialized SpineGenerator")

    def generate(self, max_steps: int = 1000, max_nodes: int = 100) -> CableGraph:
        """Generate a synthetic spine morphology.

        This method implements the main growth loop:
        - Initialize root at origin
        - Maintain list of active growing tips
        - For each tip: extend, branch (probabilistically), attempt fusion
        - Terminate when max_steps or max_nodes reached

        Args:
            max_steps: Maximum number of growth iterations (default: 1000)
            max_nodes: Maximum number of nodes to generate (default: 100)

        Returns:
            CableGraph with generated morphology (may contain cycles)

        Examples:
            >>> generator = SpineGenerator(prior)
            >>> graph = generator.generate(max_steps=500, max_nodes=50)
            >>> print(f"Generated {graph.number_of_nodes()} nodes")
        """
        logger.info(
            f"Starting generation: max_steps={max_steps}, max_nodes={max_nodes}"
        )
        self.graph = CableGraph()
        self.next_node_id = 1

        root_id = self.next_node_id
        self.next_node_id += 1

        self.graph.add_node(
            root_id,
            x=0.0,
            y=0.0,
            z=0.0,
            r=self.prior.sample_radius(0.0),
            t=3,
            parent=-1,
        )
        logger.debug(f"Created root node {root_id} at origin")

        active_tips = [(root_id, None)]

        for step in range(max_steps):
            if len(active_tips) == 0:
                logger.info(f"Terminating: no active tips at step {step}")
                break
            if self.graph.number_of_nodes() >= max_nodes:
                logger.info(
                    f"Terminating: reached max_nodes={max_nodes} at step {step}"
                )
                break

            logger.debug(
                f"Step {step}: {len(active_tips)} active tips, {self.graph.number_of_nodes()} nodes"
            )
            new_tips = []

            for tip_id, prev_direction in active_tips:
                step_length = self.prior.sample_thread_length()

                if prev_direction is None:
                    direction = self._random_unit_vector()
                else:
                    direction = self._perturb_direction(prev_direction)

                tip_pos = self.graph.get_position(tip_id)
                new_pos = tuple(
                    tip_pos[i] + step_length * direction[i] for i in range(3)
                )

                new_node_id = self.next_node_id
                self.next_node_id += 1

                distance_from_root = self._compute_distance_from_root(tip_id)

                self.graph.add_node(
                    new_node_id,
                    x=new_pos[0],
                    y=new_pos[1],
                    z=new_pos[2],
                    r=self.prior.sample_radius(distance_from_root),
                    t=3,
                    parent=tip_id,
                )

                self.graph.add_edge(tip_id, new_node_id)

                self._attempt_fusion(new_node_id)

                if np.random.random() < 0.2:
                    branch_count = self.prior.sample_branch_count()
                    logger.debug(
                        f"Branching at node {new_node_id} with {branch_count} branches"
                    )

                    for _ in range(branch_count):
                        branch_direction = self._generate_branch_direction(direction)
                        new_tips.append((new_node_id, branch_direction))
                else:
                    new_tips.append((new_node_id, direction))

            active_tips = new_tips

        logger.info(
            f"Generation complete: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges"
        )
        cycles = list(nx.simple_cycles(self.graph))
        logger.info(f"Generated graph contains {len(cycles)} cycles")
        return self.graph

    def _random_unit_vector(self) -> np.ndarray:
        """Generate a random unit vector in 3D.

        Returns:
            Random unit vector sampled from uniform distribution on sphere
        """
        vec = np.random.randn(3)
        norm = np.linalg.norm(vec)
        if norm < 1e-9:
            return np.array([0.0, 0.0, 1.0])
        return vec / norm

    def _perturb_direction(self, prev_direction: np.ndarray) -> np.ndarray:
        """Perturb growth direction by sampled curvature angle.

        The perturbation rotates the previous direction by a curvature angle
        around a random axis perpendicular to the previous direction.

        Args:
            prev_direction: Previous growth direction (unit vector)

        Returns:
            New growth direction (unit vector)
        """
        curvature_angle = self.prior.sample_curvature()

        perp1 = np.array([-prev_direction[1], prev_direction[0], 0.0])
        if np.linalg.norm(perp1) < 1e-9:
            perp1 = np.array([1.0, 0.0, 0.0])
        perp1 = perp1 / np.linalg.norm(perp1)

        perp2 = np.cross(prev_direction, perp1)
        perp2 = perp2 / np.linalg.norm(perp2)

        random_angle = np.random.uniform(0, 2 * np.pi)
        rotation_axis = np.cos(random_angle) * perp1 + np.sin(random_angle) * perp2
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)

        new_direction = (
            np.cos(curvature_angle) * prev_direction
            + np.sin(curvature_angle) * rotation_axis
        )

        norm = np.linalg.norm(new_direction)
        if norm < 1e-9:
            return prev_direction
        return new_direction / norm

    def _generate_branch_direction(self, parent_direction: np.ndarray) -> np.ndarray:
        """Generate a branch direction at a sampled angle from parent.

        Args:
            parent_direction: Direction of parent branch (unit vector)

        Returns:
            Branch direction (unit vector) at sampled angle from parent
        """
        branch_angle = self.prior.sample_branch_angle()

        perp1 = np.array([-parent_direction[1], parent_direction[0], 0.0])
        if np.linalg.norm(perp1) < 1e-9:
            perp1 = np.array([1.0, 0.0, 0.0])
        perp1 = perp1 / np.linalg.norm(perp1)

        perp2 = np.cross(parent_direction, perp1)
        perp2 = perp2 / np.linalg.norm(perp2)

        random_angle = np.random.uniform(0, 2 * np.pi)
        rotation_axis = np.cos(random_angle) * perp1 + np.sin(random_angle) * perp2
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)

        new_direction = (
            np.cos(branch_angle) * parent_direction
            + np.sin(branch_angle) * rotation_axis
        )

        norm = np.linalg.norm(new_direction)
        if norm < 1e-9:
            return self._random_unit_vector()
        return new_direction / norm

    def _compute_distance_from_root(self, node_id: int) -> float:
        """Compute path distance from root to a node.

        Args:
            node_id: ID of target node

        Returns:
            Euclidean path length from root to node (0.0 if no path)
        """
        root = self.graph.get_root()
        if root is None or root == node_id:
            return 0.0

        try:
            path = nx.shortest_path(self.graph, root, node_id)
            distance = 0.0
            for i in range(len(path) - 1):
                p1 = self.graph.get_position(path[i])
                p2 = self.graph.get_position(path[i + 1])
                distance += euclidean_distance(p1, p2)
            return distance
        except nx.NetworkXNoPath:
            return 0.0

    def _attempt_fusion(self, new_node_id: int) -> None:
        """Attempt to fuse new node with nearby edges to create cycles.

        This method:
        1. Searches for edges within fusion distance
        2. Computes closest points on candidate edges
        3. Evaluates fusion probability based on spatial and graph distances
        4. Executes fusion by either splitting edge or connecting to endpoint

        Args:
            new_node_id: ID of newly created node to attempt fusion with

        Note:
            Modifies graph in-place if fusion occurs. Only one fusion per call.
        """
        new_pos = self.graph.get_position(new_node_id)
        search_radius = self.prior.sample_fusion_distance()
        logger.debug(
            f"Attempting fusion for node {new_node_id}, search_radius={search_radius:.2f}"
        )

        candidates = []
        for u, v in self.graph.edges():
            if u == new_node_id or v == new_node_id:
                continue

            pos_u = self.graph.get_position(u)
            pos_v = self.graph.get_position(v)

            dist_to_u = euclidean_distance(new_pos, pos_u)
            dist_to_v = euclidean_distance(new_pos, pos_v)

            if dist_to_u < search_radius or dist_to_v < search_radius:
                closest_point, t = self._closest_point_on_segment(new_pos, pos_u, pos_v)
                spatial_dist = euclidean_distance(new_pos, closest_point)

                if spatial_dist < search_radius:
                    candidates.append((u, v, spatial_dist, t, closest_point))

        if not candidates:
            logger.debug(f"No fusion candidates found for node {new_node_id}")
            return

        logger.debug(f"Found {len(candidates)} fusion candidates")
        candidates.sort(key=lambda x: x[2])

        for u, v, spatial_dist, t, closest_point in candidates:
            try:
                graph_dist = nx.shortest_path_length(self.graph, new_node_id, u)
            except nx.NetworkXNoPath:
                graph_dist = float("inf")

            fusion_prob = self.prior.compute_fusion_probability(
                spatial_dist, graph_dist
            )

            if np.random.random() < fusion_prob:
                if 0.01 < t < 0.99:
                    split_node_id = self.next_node_id
                    self.next_node_id += 1

                    avg_radius = (
                        self.graph.get_radius(u) + self.graph.get_radius(v)
                    ) / 2

                    split_edge(
                        self.graph, u, v, split_node_id, closest_point, avg_radius
                    )

                    # Clone new_node at split point for CYCLE_BREAK compliance
                    clone_id = self.next_node_id
                    self.next_node_id += 1
                    self.graph.add_node(
                        clone_id,
                        x=closest_point[0],
                        y=closest_point[1],
                        z=closest_point[2],
                        r=avg_radius,  # Use split node's radius for matching
                        t=3,
                        parent=new_node_id,
                    )
                    self.graph.add_edge(new_node_id, clone_id)
                    self.graph.add_edge(clone_id, split_node_id)
                    logger.info(
                        f"Fusion: split edge ({u},{v}), created split node {split_node_id}, "
                        f"cloned {new_node_id} to {clone_id}, CYCLE_BREAK edge ({clone_id},{split_node_id})"
                    )
                else:
                    target = u if t <= 0.01 else v
                    target_pos = self.graph.get_position(target)
                    target_radius = self.graph.get_radius(target)

                    # Clone new_node at target position for CYCLE_BREAK compliance
                    clone_id = self.next_node_id
                    self.next_node_id += 1
                    self.graph.add_node(
                        clone_id,
                        x=target_pos[0],
                        y=target_pos[1],
                        z=target_pos[2],
                        r=target_radius,  # Use target's radius for matching
                        t=3,
                        parent=new_node_id,
                    )
                    self.graph.add_edge(new_node_id, clone_id)
                    if not self.graph.has_edge(clone_id, target):
                        self.graph.add_edge(clone_id, target)
                        logger.info(
                            f"Fusion: cloned {new_node_id} to {clone_id}, "
                            f"CYCLE_BREAK edge ({clone_id},{target})"
                        )

                break

    def _closest_point_on_segment(
        self,
        point: Tuple[float, float, float],
        seg_start: Tuple[float, float, float],
        seg_end: Tuple[float, float, float],
    ) -> Tuple[Tuple[float, float, float], float]:
        """Find closest point on a line segment to a query point.

        Args:
            point: Query point
            seg_start: Start of line segment
            seg_end: End of line segment

        Returns:
            Tuple of (closest_point, t) where:
                - closest_point: Point on segment closest to query
                - t: Parameter in [0, 1] indicating position along segment
        """
        p = np.array(point)
        a = np.array(seg_start)
        b = np.array(seg_end)

        ab = b - a
        ap = p - a

        ab_squared = np.dot(ab, ab)

        if ab_squared < 1e-9:
            return seg_start, 0.0

        t = np.dot(ap, ab) / ab_squared
        t = np.clip(t, 0.0, 1.0)

        closest = tuple(a + t * ab)

        return closest, t
