import numpy as np
import networkx as nx
from typing import Tuple, Dict, Any


def euclidean_distance(p1: Tuple[float, float, float], p2: Tuple[float, float, float]) -> float:
    return np.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))


def compute_graph_distances(graph: nx.Graph) -> Dict[Tuple[int, int], float]:
    return dict(nx.all_pairs_dijkstra_path_length(graph, weight='length'))


def find_neighbors_within_radius(
    graph: nx.Graph, 
    point: Tuple[float, float, float], 
    radius: float
) -> list[int]:
    neighbors = []
    for node_id, data in graph.nodes(data=True):
        node_pos = (data['x'], data['y'], data['z'])
        if euclidean_distance(point, node_pos) <= radius:
            neighbors.append(node_id)
    return neighbors


def interpolate_point_on_segment(
    p1: Tuple[float, float, float], 
    p2: Tuple[float, float, float], 
    t: float
) -> Tuple[float, float, float]:
    return tuple(a + t * (b - a) for a, b in zip(p1, p2))


def split_edge(
    graph: nx.Graph,
    u: int,
    v: int,
    new_node_id: int,
    new_pos: Tuple[float, float, float],
    new_radius: float
) -> None:
    if not graph.has_edge(u, v):
        raise ValueError(f"Edge ({u}, {v}) does not exist in graph")
    
    graph.remove_edge(u, v)
    
    graph.add_node(
        new_node_id,
        x=new_pos[0],
        y=new_pos[1],
        z=new_pos[2],
        r=new_radius,
        t=3,
        parent=-1
    )
    
    graph.add_edge(u, new_node_id)
    graph.add_edge(new_node_id, v)
