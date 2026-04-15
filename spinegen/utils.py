"""Utility functions for geometric calculations and graph manipulations.

This module provides helper functions for:
- Euclidean distance calculations
- Graph distance computations
- Spatial neighbor searches
- Point interpolation
- Edge splitting operations
"""

import logging
import numpy as np
import networkx as nx
from typing import Tuple, Dict

logger = logging.getLogger(__name__)


def euclidean_distance(
    p1: Tuple[float, float, float], p2: Tuple[float, float, float]
) -> float:
    """Calculate Euclidean distance between two 3D points.

    Args:
        p1: First point as (x, y, z) tuple
        p2: Second point as (x, y, z) tuple

    Returns:
        Euclidean distance between p1 and p2

    Examples:
        >>> euclidean_distance((0, 0, 0), (1, 1, 1))
        1.7320508075688772
    """
    distance = np.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))
    logger.debug(f"Computed distance {distance:.4f} between {p1} and {p2}")
    return distance


def compute_graph_distances(graph: nx.Graph) -> Dict[Tuple[int, int], float]:
    """Compute all-pairs shortest path distances in a graph.

    Args:
        graph: NetworkX graph with optional 'length' edge weights

    Returns:
        Dictionary mapping (node_i, node_j) pairs to shortest path distances

    Note:
        Uses Dijkstra's algorithm for weighted graphs. If 'length' edge
        attribute is not present, treats all edges as having unit weight.
    """
    logger.info(
        f"Computing all-pairs distances for graph with {graph.number_of_nodes()} nodes"
    )
    distances = dict(nx.all_pairs_dijkstra_path_length(graph, weight="length"))
    logger.debug(f"Computed {len(distances)} distance pairs")
    return distances


def find_neighbors_within_radius(
    graph: nx.Graph, point: Tuple[float, float, float], radius: float
) -> list[int]:
    """Find all graph nodes within a given radius of a point.

    Args:
        graph: NetworkX graph with nodes having 'x', 'y', 'z' attributes
        point: Query point as (x, y, z) tuple
        radius: Search radius

    Returns:
        List of node IDs within the specified radius

    Note:
        This performs a brute-force search over all nodes. For large graphs,
        consider using spatial indexing structures (e.g., KD-tree).
    """
    logger.debug(f"Searching for neighbors within radius {radius} of {point}")
    neighbors = []
    for node_id, data in graph.nodes(data=True):
        node_pos = (data["x"], data["y"], data["z"])
        if euclidean_distance(point, node_pos) <= radius:
            neighbors.append(node_id)
    logger.debug(f"Found {len(neighbors)} neighbors within radius")
    return neighbors


def interpolate_point_on_segment(
    p1: Tuple[float, float, float], p2: Tuple[float, float, float], t: float
) -> Tuple[float, float, float]:
    """Linearly interpolate a point on a line segment.

    Args:
        p1: Start point of segment as (x, y, z) tuple
        p2: End point of segment as (x, y, z) tuple
        t: Interpolation parameter in [0, 1], where 0 returns p1 and 1 returns p2

    Returns:
        Interpolated point as (x, y, z) tuple

    Examples:
        >>> interpolate_point_on_segment((0, 0, 0), (10, 10, 10), 0.5)
        (5.0, 5.0, 5.0)
    """
    result = tuple(a + t * (b - a) for a, b in zip(p1, p2))
    logger.debug(f"Interpolated point at t={t}: {result}")
    return result


def split_edge(
    graph: nx.Graph,
    u: int,
    v: int,
    new_node_id: int,
    new_pos: Tuple[float, float, float],
    new_radius: float,
) -> None:
    """Split an edge by inserting a new node at a specified position.

    This operation removes the edge (u, v) and replaces it with two edges:
    (u, new_node) and (new_node, v), where new_node is positioned at new_pos.

    Args:
        graph: NetworkX graph to modify in-place
        u: ID of first endpoint of edge to split
        v: ID of second endpoint of edge to split
        new_node_id: ID for the newly inserted node
        new_pos: Position of new node as (x, y, z) tuple
        new_radius: Radius attribute for new node

    Raises:
        ValueError: If edge (u, v) does not exist in the graph

    Note:
        The new node is assigned type 3 (dendrite) and parent -1.
        This function modifies the graph in-place.
    """
    logger.debug(f"Splitting edge ({u}, {v}) with new node {new_node_id}")

    if not graph.has_edge(u, v):
        logger.error(f"Cannot split edge ({u}, {v}): edge does not exist")
        raise ValueError(f"Edge ({u}, {v}) does not exist in graph")

    graph.remove_edge(u, v)
    logger.debug(f"Removed edge ({u}, {v})")

    graph.add_node(
        new_node_id,
        x=new_pos[0],
        y=new_pos[1],
        z=new_pos[2],
        r=new_radius,
        t=3,
        parent=-1,
    )
    logger.debug(f"Added node {new_node_id} at {new_pos} with radius {new_radius}")

    graph.add_edge(u, new_node_id)
    graph.add_edge(new_node_id, v)
    logger.debug(f"Created edges ({u}, {new_node_id}) and ({new_node_id}, {v})")
