import numpy as np
import networkx as nx
from typing import Optional, Tuple, List
from .cable_graph import CableGraph
from .prior import SpinePrior
from .utils import euclidean_distance, split_edge


class SpineGenerator:
    
    def __init__(self, prior: SpinePrior):
        self.prior = prior
        self.graph = None
        self.next_node_id = 1
    
    def generate(self, max_steps: int = 1000, max_nodes: int = 100) -> CableGraph:
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
            parent=-1
        )
        
        active_tips = [(root_id, None)]
        
        for step in range(max_steps):
            if len(active_tips) == 0 or self.graph.number_of_nodes() >= max_nodes:
                break
            
            new_tips = []
            
            for tip_id, prev_direction in active_tips:
                step_length = self.prior.sample_segment_length()
                
                if prev_direction is None:
                    direction = self._random_unit_vector()
                else:
                    direction = self._perturb_direction(prev_direction)
                
                tip_pos = self.graph.get_position(tip_id)
                new_pos = tuple(tip_pos[i] + step_length * direction[i] for i in range(3))
                
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
                    parent=tip_id
                )
                
                self.graph.add_edge(tip_id, new_node_id)
                
                self._attempt_fusion(new_node_id)
                
                if np.random.random() < 0.2:
                    branch_count = self.prior.sample_branch_count()
                    
                    for _ in range(branch_count):
                        branch_direction = self._generate_branch_direction(direction)
                        new_tips.append((new_node_id, branch_direction))
                else:
                    new_tips.append((new_node_id, direction))
            
            active_tips = new_tips
        
        return self.graph
    
    def _random_unit_vector(self) -> np.ndarray:
        vec = np.random.randn(3)
        norm = np.linalg.norm(vec)
        if norm < 1e-9:
            return np.array([0.0, 0.0, 1.0])
        return vec / norm
    
    def _perturb_direction(self, prev_direction: np.ndarray) -> np.ndarray:
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
            np.cos(curvature_angle) * prev_direction +
            np.sin(curvature_angle) * rotation_axis
        )
        
        norm = np.linalg.norm(new_direction)
        if norm < 1e-9:
            return prev_direction
        return new_direction / norm
    
    def _generate_branch_direction(self, parent_direction: np.ndarray) -> np.ndarray:
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
            np.cos(branch_angle) * parent_direction +
            np.sin(branch_angle) * rotation_axis
        )
        
        norm = np.linalg.norm(new_direction)
        if norm < 1e-9:
            return self._random_unit_vector()
        return new_direction / norm
    
    def _compute_distance_from_root(self, node_id: int) -> float:
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
        new_pos = self.graph.get_position(new_node_id)
        search_radius = self.prior.sample_fusion_distance()
        
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
            return
        
        candidates.sort(key=lambda x: x[2])
        
        for u, v, spatial_dist, t, closest_point in candidates:
            try:
                graph_dist = nx.shortest_path_length(self.graph, new_node_id, u)
            except nx.NetworkXNoPath:
                graph_dist = float('inf')
            
            fusion_prob = self.prior.compute_fusion_probability(spatial_dist, graph_dist)
            
            if np.random.random() < fusion_prob:
                if 0.01 < t < 0.99:
                    split_node_id = self.next_node_id
                    self.next_node_id += 1
                    
                    avg_radius = (self.graph.get_radius(u) + self.graph.get_radius(v)) / 2
                    
                    split_edge(self.graph, u, v, split_node_id, closest_point, avg_radius)
                    self.graph.add_edge(new_node_id, split_node_id)
                else:
                    target = u if t <= 0.01 else v
                    if not self.graph.has_edge(new_node_id, target):
                        self.graph.add_edge(new_node_id, target)
                
                break
    
    def _closest_point_on_segment(
        self,
        point: Tuple[float, float, float],
        seg_start: Tuple[float, float, float],
        seg_end: Tuple[float, float, float]
    ) -> Tuple[Tuple[float, float, float], float]:
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
