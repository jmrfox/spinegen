import networkx as nx
import swctools
from pathlib import Path
from typing import Union, Optional


class CableGraph(nx.Graph):
    
    def __init__(self):
        super().__init__()
        self.graph['reconnections'] = []
    
    @classmethod
    def from_swc_file(cls, filepath: Union[str, Path]) -> 'CableGraph':
        swc_model = swctools.SWCModel.from_swc_file(filepath)
        
        cycle_graph = swc_model.make_cycle_connections()
        
        cable_graph = cls()
        
        for node_id, data in cycle_graph.nodes(data=True):
            cable_graph.add_node(node_id, **data)
        
        for u, v in cycle_graph.edges():
            cable_graph.add_edge(u, v)
        
        if 'reconnections' in swc_model.graph:
            cable_graph.graph['reconnections'] = swc_model.graph['reconnections']
        
        return cable_graph
    
    def to_swc_file(self, filepath: Union[str, Path], precision: int = 6) -> None:
        cycles = list(nx.simple_cycles(self))
        
        reconnections = []
        tree_graph = self.copy()
        
        if cycles:
            for cycle in cycles:
                if len(cycle) < 2:
                    continue
                
                u, v = cycle[0], cycle[1]
                if tree_graph.has_edge(u, v):
                    tree_graph.remove_edge(u, v)
                    reconnections.append((u, v))
                    break
        
        roots = [n for n in tree_graph.nodes() if tree_graph.nodes[n].get('parent', -1) == -1]
        if not roots:
            roots = [min(tree_graph.nodes())]
        
        root = roots[0]
        
        records = {}
        node_id_map = {}
        swc_id = 1
        
        visited = set()
        stack = [(root, -1)]
        
        while stack:
            node_id, parent_swc_id = stack.pop()
            
            if node_id in visited:
                continue
            visited.add(node_id)
            
            data = self.nodes[node_id]
            
            records[swc_id] = swctools.SWCRecord(
                n=swc_id,
                t=data.get('t', 3),
                x=data['x'],
                y=data['y'],
                z=data['z'],
                r=data['r'],
                parent=parent_swc_id,
                line=swc_id
            )
            
            node_id_map[node_id] = swc_id
            current_swc_id = swc_id
            swc_id += 1
            
            for neighbor in tree_graph.neighbors(node_id):
                if neighbor not in visited:
                    stack.append((neighbor, current_swc_id))
        
        for u, v in reconnections:
            if u in node_id_map and v in node_id_map:
                u_swc = node_id_map[u]
                v_swc = node_id_map[v]
                
                data_u = self.nodes[u]
                duplicate_record = swctools.SWCRecord(
                    n=swc_id,
                    t=data_u.get('t', 3),
                    x=data_u['x'],
                    y=data_u['y'],
                    z=data_u['z'],
                    r=data_u['r'],
                    parent=v_swc,
                    line=swc_id
                )
                records[swc_id] = duplicate_record
                reconnections.append((swc_id, u_swc))
                swc_id += 1
        
        header_lines = []
        for u_swc, v_swc in reconnections:
            header_lines.append(f"CYCLE_BREAK reconnect {u_swc} {v_swc}")
        
        swc_model = swctools.SWCModel.from_records(records)
        swc_model.to_swc_file(filepath, precision=precision, header=header_lines)
    
    def get_root(self) -> Optional[int]:
        for node_id, data in self.nodes(data=True):
            if data.get('parent', -1) == -1:
                return node_id
        if self.number_of_nodes() > 0:
            return min(self.nodes())
        return None
    
    def get_position(self, node_id: int) -> tuple[float, float, float]:
        data = self.nodes[node_id]
        return (data['x'], data['y'], data['z'])
    
    def get_radius(self, node_id: int) -> float:
        return self.nodes[node_id]['r']
