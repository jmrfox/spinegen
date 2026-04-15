"""Morphological analysis of dendritic spine cable graphs.

This module provides the SpineAnalyzer class for extracting statistical
features from spine morphologies, including thread lengths, curvatures,
branching patterns, and fusion (cycle) characteristics.

Note: 'Thread' refers to continuous paths through degree-2 vertices,
distinct from 'segment' which refers to frustum geometry of individual edges.
"""

import logging
import numpy as np
import networkx as nx
from typing import List, Dict, Any
from .cable_graph import CableGraph
from .utils import euclidean_distance

logger = logging.getLogger(__name__)


class SpineAnalyzer:
    """Analyzer for extracting morphological statistics from spine graphs.

    This class processes collections of CableGraph instances to compute
    statistical distributions of morphological features that can be used
    to parameterize generative models.

    Attributes:
        graphs: List of CableGraph instances to analyze
        stats: Dictionary of computed statistics (populated after analyze())

    Statistics Computed:
        - thread_lengths: Path lengths of threads (continuous paths)
        - curvatures: Angular deviations along threads
        - branch_counts: Number of branches at branch points
        - branch_angles: Angles between branches
        - radii: Node radii throughout the structure
        - fusion_distances: Euclidean distances of cycle-closing edges
        - fusion_graph_distances: Graph distances of fusion points
        - fusion_angles: Angles at fusion points
    """

    def __init__(self, graphs: List[CableGraph]):
        """Initialize analyzer with a collection of graphs.

        Args:
            graphs: List of CableGraph instances to analyze
        """
        self.graphs = graphs
        self.stats = None
        logger.info(f"Initialized SpineAnalyzer with {len(graphs)} graphs")

    def analyze(self) -> Dict[str, Any]:
        """Analyze all graphs and compute morphological statistics.

        This method processes each graph to extract threads, compute
        geometric features, and analyze branching and fusion patterns.

        Returns:
            Dictionary containing numpy arrays of statistics:
                - thread_lengths: Array of thread path lengths
                - curvatures: Array of curvature angles (radians)
                - branch_counts: Array of branch counts at branch nodes
                - branch_angles: Array of angles between branches (radians)
                - radii: Array of all node radii
                - fusion_distances: Array of fusion Euclidean distances
                - fusion_graph_distances: Array of fusion graph distances
                - fusion_angles: Array of angles at fusion points (radians)

        Note:
            Results are stored in self.stats and also returned.
        """
        logger.info("Starting morphological analysis")
        all_thread_lengths = []
        all_curvatures = []
        all_branch_counts = []
        all_branch_angles = []
        all_radii = []
        all_fusion_distances = []
        all_fusion_graph_distances = []
        all_fusion_angles = []

        for idx, graph in enumerate(self.graphs):
            logger.debug(f"Analyzing graph {idx + 1}/{len(self.graphs)}")
            root = graph.get_root()
            if root is None:
                logger.warning(f"Graph {idx} has no root, skipping")
                continue

            degrees = dict(graph.degree())
            leaves = [n for n in graph.nodes() if degrees[n] == 1 and n != root]
            branch_nodes = [n for n in graph.nodes() if degrees[n] >= 3]
            logger.debug(
                f"Found {len(leaves)} leaves, {len(branch_nodes)} branch nodes"
            )

            threads = self._extract_threads(graph, root, branch_nodes, leaves)
            logger.debug(f"Extracted {len(threads)} threads")

            for thread in threads:
                if len(thread) < 2:
                    continue

                length = self._compute_thread_length(graph, thread)
                all_thread_lengths.append(length)

                curvatures = self._compute_curvatures(graph, thread)
                all_curvatures.extend(curvatures)

                for node_id in thread:
                    all_radii.append(graph.get_radius(node_id))

            for branch_node in branch_nodes:
                neighbors = list(graph.neighbors(branch_node))
                branch_count = len(neighbors)
                all_branch_counts.append(branch_count)

                angles = self._compute_branch_angles(graph, branch_node, neighbors)
                all_branch_angles.extend(angles)

            cycles = list(nx.simple_cycles(graph))
            logger.debug(f"Detected {len(cycles)} cycles")
            for cycle in cycles:
                if len(cycle) < 2:
                    continue

                fusion_data = self._analyze_cycle(graph, cycle)
                if fusion_data:
                    all_fusion_distances.append(fusion_data["euclidean_distance"])
                    all_fusion_graph_distances.append(fusion_data["graph_distance"])
                    if fusion_data["angle"] is not None:
                        all_fusion_angles.append(fusion_data["angle"])

        self.stats = {
            "n_graphs": len(self.graphs),
            "thread_lengths": (
                np.array(all_thread_lengths) if all_thread_lengths else np.array([])
            ),
            "curvatures": np.array(all_curvatures) if all_curvatures else np.array([]),
            "branch_counts": (
                np.array(all_branch_counts) if all_branch_counts else np.array([])
            ),
            "branch_angles": (
                np.array(all_branch_angles) if all_branch_angles else np.array([])
            ),
            "radii": np.array(all_radii) if all_radii else np.array([]),
            "fusion_distances": (
                np.array(all_fusion_distances) if all_fusion_distances else np.array([])
            ),
            "fusion_graph_distances": (
                np.array(all_fusion_graph_distances)
                if all_fusion_graph_distances
                else np.array([])
            ),
            "fusion_angles": (
                np.array(all_fusion_angles) if all_fusion_angles else np.array([])
            ),
        }

        logger.info(
            f"Analysis complete: {len(all_thread_lengths)} threads, "
            f"{len(all_curvatures)} curvatures, {len(all_fusion_distances)} fusions"
        )
        return self.stats

    def _extract_threads(
        self, graph: CableGraph, root: int, branch_nodes: List[int], leaves: List[int]
    ) -> List[List[int]]:
        """Extract threads (continuous paths) between special nodes.

        A thread is a continuous path between two special nodes (root, branch,
        or leaf) that passes only through degree-2 nodes.

        Args:
            graph: CableGraph to analyze
            root: Root node ID
            branch_nodes: List of branch point node IDs (degree >= 3)
            leaves: List of leaf node IDs (degree == 1)

        Returns:
            List of threads, where each thread is a list of node IDs
            from one special node to another
        """
        threads = []
        special_nodes = set([root] + branch_nodes + leaves)

        for start_node in special_nodes:
            for neighbor in graph.neighbors(start_node):
                thread = [start_node]
                current = neighbor
                prev = start_node

                while current not in special_nodes:
                    thread.append(current)
                    neighbors = list(graph.neighbors(current))
                    next_nodes = [n for n in neighbors if n != prev]

                    if not next_nodes:
                        break

                    prev = current
                    current = next_nodes[0]

                thread.append(current)

                if len(thread) > 1:
                    threads.append(thread)

        unique_threads = []
        seen = set()
        for thr in threads:
            key = tuple(sorted([thr[0], thr[-1]]))
            if key not in seen:
                seen.add(key)
                unique_threads.append(thr)

        return unique_threads

    def _compute_thread_length(self, graph: CableGraph, thread: List[int]) -> float:
        """Compute total path length of a thread.

        Args:
            graph: CableGraph containing the thread
            thread: List of node IDs forming the thread

        Returns:
            Total Euclidean path length along the thread
        """
        length = 0.0
        for i in range(len(thread) - 1):
            p1 = graph.get_position(thread[i])
            p2 = graph.get_position(thread[i + 1])
            length += euclidean_distance(p1, p2)
        return length

    def _compute_curvatures(self, graph: CableGraph, thread: List[int]) -> List[float]:
        """Compute curvature angles along a thread.

        Curvature is measured as the angle between consecutive edge vectors
        at each interior node of the thread.

        Args:
            graph: CableGraph containing the thread
            thread: List of node IDs forming the thread

        Returns:
            List of curvature angles in radians (0 = straight, π = reversal)

        Note:
            Requires at least 3 nodes. Returns empty list for shorter threads.
        """
        curvatures = []

        for i in range(len(thread) - 2):
            p1 = np.array(graph.get_position(thread[i]))
            p2 = np.array(graph.get_position(thread[i + 1]))
            p3 = np.array(graph.get_position(thread[i + 2]))

            v1 = p2 - p1
            v2 = p3 - p2

            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)

            if norm1 > 1e-9 and norm2 > 1e-9:
                cos_angle = np.dot(v1, v2) / (norm1 * norm2)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle = np.arccos(cos_angle)
                curvatures.append(angle)

        return curvatures

    def _compute_branch_angles(
        self, graph: CableGraph, branch_node: int, neighbors: List[int]
    ) -> List[float]:
        """Compute angles between all pairs of branches at a branch node.

        Args:
            graph: CableGraph containing the branch
            branch_node: ID of the branch point node
            neighbors: List of neighbor node IDs

        Returns:
            List of angles (in radians) between all pairs of branches

        Note:
            For n neighbors, returns n*(n-1)/2 pairwise angles.
        """
        angles = []

        branch_pos = np.array(graph.get_position(branch_node))

        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                pos_i = np.array(graph.get_position(neighbors[i]))
                pos_j = np.array(graph.get_position(neighbors[j]))

                v1 = pos_i - branch_pos
                v2 = pos_j - branch_pos

                norm1 = np.linalg.norm(v1)
                norm2 = np.linalg.norm(v2)

                if norm1 > 1e-9 and norm2 > 1e-9:
                    cos_angle = np.dot(v1, v2) / (norm1 * norm2)
                    cos_angle = np.clip(cos_angle, -1.0, 1.0)
                    angle = np.arccos(cos_angle)
                    angles.append(angle)

        return angles

    def _analyze_cycle(self, graph: CableGraph, cycle: List[int]) -> Dict[str, Any]:
        """Analyze geometric properties of a cycle (fusion event).

        Args:
            graph: CableGraph containing the cycle
            cycle: List of node IDs forming the cycle

        Returns:
            Dictionary with keys:
                - euclidean_distance: Spatial distance of fusion edge
                - graph_distance: Path length in tree (with fusion edge removed)
                - angle: Angle between branches at fusion point (or None)

            Returns None if cycle has fewer than 2 nodes.

        Note:
            The fusion edge is taken as the edge between the first two
            nodes in the cycle list.
        """
        if len(cycle) < 2:
            return None

        u, v = cycle[0], cycle[1]

        pos_u = np.array(graph.get_position(u))
        pos_v = np.array(graph.get_position(v))

        euclidean_dist = euclidean_distance(pos_u, pos_v)

        temp_graph = graph.copy()
        if temp_graph.has_edge(u, v):
            temp_graph.remove_edge(u, v)

        try:
            graph_dist = nx.shortest_path_length(temp_graph, u, v)
        except nx.NetworkXNoPath:
            graph_dist = float("inf")

        angle = None
        u_neighbors = [n for n in graph.neighbors(u) if n != v]
        v_neighbors = [n for n in graph.neighbors(v) if n != u]

        if u_neighbors and v_neighbors:
            u_neighbor_pos = np.array(graph.get_position(u_neighbors[0]))
            v_neighbor_pos = np.array(graph.get_position(v_neighbors[0]))

            vec_u = u_neighbor_pos - pos_u
            vec_v = v_neighbor_pos - pos_v

            norm_u = np.linalg.norm(vec_u)
            norm_v = np.linalg.norm(vec_v)

            if norm_u > 1e-9 and norm_v > 1e-9:
                cos_angle = np.dot(vec_u, vec_v) / (norm_u * norm_v)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle = np.arccos(cos_angle)

        result = {
            "euclidean_distance": euclidean_dist,
            "graph_distance": graph_dist,
            "angle": angle,
        }
        logger.debug(
            f"Analyzed cycle: euclidean_dist={euclidean_dist:.2f}, "
            f"graph_dist={graph_dist}, angle={angle}"
        )
        return result
