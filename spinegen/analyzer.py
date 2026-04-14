import numpy as np
import networkx as nx
from typing import List, Dict, Any
from .cable_graph import CableGraph
from .utils import euclidean_distance


class SpineAnalyzer:
    
    def __init__(self, graphs: List[CableGraph]):
        self.graphs = graphs
        self.stats = None
    
    def analyze(self) -> Dict[str, Any]:
        all_segment_lengths = []
        all_curvatures = []
        all_branch_counts = []
        all_branch_angles = []
        all_radii = []
        all_fusion_distances = []
        all_fusion_graph_distances = []
        all_fusion_angles = []
        
        for graph in self.graphs:
            root = graph.get_root()
            if root is None:
                continue
            
            degrees = dict(graph.degree())
            leaves = [n for n in graph.nodes() if degrees[n] == 1 and n != root]
            branch_nodes = [n for n in graph.nodes() if degrees[n] >= 3]
            
            segments = self._extract_segments(graph, root, branch_nodes, leaves)
            
            for segment in segments:
                if len(segment) < 2:
                    continue
                
                length = self._compute_segment_length(graph, segment)
                all_segment_lengths.append(length)
                
                curvatures = self._compute_curvatures(graph, segment)
                all_curvatures.extend(curvatures)
                
                for node_id in segment:
                    all_radii.append(graph.get_radius(node_id))
            
            for branch_node in branch_nodes:
                neighbors = list(graph.neighbors(branch_node))
                branch_count = len(neighbors)
                all_branch_counts.append(branch_count)
                
                angles = self._compute_branch_angles(graph, branch_node, neighbors)
                all_branch_angles.extend(angles)
            
            cycles = list(nx.simple_cycles(graph))
            for cycle in cycles:
                if len(cycle) < 2:
                    continue
                
                fusion_data = self._analyze_cycle(graph, cycle)
                if fusion_data:
                    all_fusion_distances.append(fusion_data['euclidean_distance'])
                    all_fusion_graph_distances.append(fusion_data['graph_distance'])
                    if fusion_data['angle'] is not None:
                        all_fusion_angles.append(fusion_data['angle'])
        
        self.stats = {
            'segment_lengths': np.array(all_segment_lengths) if all_segment_lengths else np.array([]),
            'curvatures': np.array(all_curvatures) if all_curvatures else np.array([]),
            'branch_counts': np.array(all_branch_counts) if all_branch_counts else np.array([]),
            'branch_angles': np.array(all_branch_angles) if all_branch_angles else np.array([]),
            'radii': np.array(all_radii) if all_radii else np.array([]),
            'fusion_distances': np.array(all_fusion_distances) if all_fusion_distances else np.array([]),
            'fusion_graph_distances': np.array(all_fusion_graph_distances) if all_fusion_graph_distances else np.array([]),
            'fusion_angles': np.array(all_fusion_angles) if all_fusion_angles else np.array([]),
        }
        
        return self.stats
    
    def _extract_segments(self, graph: CableGraph, root: int, branch_nodes: List[int], leaves: List[int]) -> List[List[int]]:
        segments = []
        special_nodes = set([root] + branch_nodes + leaves)
        
        for start_node in special_nodes:
            for neighbor in graph.neighbors(start_node):
                segment = [start_node]
                current = neighbor
                prev = start_node
                
                while current not in special_nodes:
                    segment.append(current)
                    neighbors = list(graph.neighbors(current))
                    next_nodes = [n for n in neighbors if n != prev]
                    
                    if not next_nodes:
                        break
                    
                    prev = current
                    current = next_nodes[0]
                
                segment.append(current)
                
                if len(segment) > 1:
                    segments.append(segment)
        
        unique_segments = []
        seen = set()
        for seg in segments:
            key = tuple(sorted([seg[0], seg[-1]]))
            if key not in seen:
                seen.add(key)
                unique_segments.append(seg)
        
        return unique_segments
    
    def _compute_segment_length(self, graph: CableGraph, segment: List[int]) -> float:
        length = 0.0
        for i in range(len(segment) - 1):
            p1 = graph.get_position(segment[i])
            p2 = graph.get_position(segment[i + 1])
            length += euclidean_distance(p1, p2)
        return length
    
    def _compute_curvatures(self, graph: CableGraph, segment: List[int]) -> List[float]:
        curvatures = []
        
        for i in range(len(segment) - 2):
            p1 = np.array(graph.get_position(segment[i]))
            p2 = np.array(graph.get_position(segment[i + 1]))
            p3 = np.array(graph.get_position(segment[i + 2]))
            
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
    
    def _compute_branch_angles(self, graph: CableGraph, branch_node: int, neighbors: List[int]) -> List[float]:
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
            graph_dist = float('inf')
        
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
        
        return {
            'euclidean_distance': euclidean_dist,
            'graph_distance': graph_dist,
            'angle': angle
        }
