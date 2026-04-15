"""Cable graph model for neural morphologies with cycle support.

This module provides the CableGraph class, which extends NetworkX Graph
to support loading and saving SWC files with cycle-closure directives.
"""

import logging
import networkx as nx
import swctools
from pathlib import Path
from typing import Union, Optional

logger = logging.getLogger(__name__)


class CableGraph(nx.Graph):
    """Graph representation of neural cable morphology with cycle support.

    CableGraph extends networkx.Graph to provide specialized functionality
    for neural morphologies, including:
    - Loading from SWC files with cycle reconnection directives
    - Saving to SWC format with automatic cycle detection
    - Convenient access to node positions and radii

    Attributes:
        graph['reconnections']: List of (node_i, node_j) tuples representing
            cycle-closing edges that were reconnected from SWC directives

    Node Attributes:
        x (float): X-coordinate position
        y (float): Y-coordinate position
        z (float): Z-coordinate position
        r (float): Radius at this node
        t (int): SWC type tag (typically 3 for dendrite)
        parent (int): Parent node ID (-1 for root)
    """

    def __init__(self):
        """Initialize an empty CableGraph."""
        super().__init__()
        self.graph["reconnections"] = []
        logger.debug("Initialized empty CableGraph")

    @classmethod
    def from_swc_file(cls, filepath: Union[str, Path]) -> "CableGraph":
        """Load a CableGraph from an SWC file with cycle support.

        This method uses swctools to parse SWC files and automatically
        reconnects any cycles specified via CYCLE_BREAK directives in
        the file header.

        Args:
            filepath: Path to SWC file (string or Path object)

        Returns:
            CableGraph instance with nodes, edges, and cycles restored

        Examples:
            >>> graph = CableGraph.from_swc_file("spine.swc")
            >>> print(f"Loaded {graph.number_of_nodes()} nodes")
        """
        logger.info(f"Loading CableGraph from {filepath}")
        swc_model = swctools.SWCModel.from_swc_file(filepath)
        logger.debug(f"Parsed SWC model with {swc_model.number_of_nodes()} nodes")

        cycle_graph = swc_model.make_cycle_connections()
        logger.debug(
            f"Applied cycle connections, graph has {cycle_graph.number_of_edges()} edges"
        )

        cable_graph = cls()

        for node_id, data in cycle_graph.nodes(data=True):
            cable_graph.add_node(node_id, **data)

        for u, v in cycle_graph.edges():
            cable_graph.add_edge(u, v)

        if "reconnections" in swc_model.graph:
            cable_graph.graph["reconnections"] = swc_model.graph["reconnections"]
            logger.info(
                f"Loaded {len(cable_graph.graph['reconnections'])} reconnections"
            )

        logger.info(
            f"Successfully loaded CableGraph: {cable_graph.number_of_nodes()} nodes, {cable_graph.number_of_edges()} edges"
        )
        return cable_graph

    def to_swc_file(self, filepath: Union[str, Path], precision: int = 6) -> None:
        """Save CableGraph to SWC file with cycle-closure directives.

        This method detects cycles in the graph, removes edges to create a
        spanning tree, and adds CYCLE_BREAK reconnect directives to the
        SWC header to preserve cycle information.

        Args:
            filepath: Output path for SWC file (string or Path object)
            precision: Number of decimal places for coordinates (default: 6)

        Note:
            The output SWC file will contain:
            - Header lines with CYCLE_BREAK reconnect directives for each cycle
            - Tree structure with duplicated nodes for reconnections
            - Standard 7-column SWC format (id, type, x, y, z, r, parent)

        Examples:
            >>> graph.to_swc_file("output.swc", precision=4)
        """
        logger.info(f"Saving CableGraph to {filepath}")
        cycles = list(nx.simple_cycles(self))
        logger.debug(f"Detected {len(cycles)} cycles in graph")

        reconnections = []
        tree_graph = self.copy()

        if cycles:
            logger.debug("Breaking cycles to create spanning tree")
            for cycle in cycles:
                if len(cycle) < 2:
                    continue

                u, v = cycle[0], cycle[1]
                if tree_graph.has_edge(u, v):
                    tree_graph.remove_edge(u, v)
                    reconnections.append((u, v))
                    logger.debug(f"Broke cycle edge ({u}, {v})")
                    break

        roots = [
            n for n in tree_graph.nodes() if tree_graph.nodes[n].get("parent", -1) == -1
        ]
        if not roots:
            roots = [min(tree_graph.nodes())]

        root = roots[0]
        logger.debug(f"Using root node {root} for tree traversal")

        records = {}
        node_id_map = {}
        swc_id = 1

        visited = set()
        stack = [(root, -1)]

        logger.debug("Traversing tree to create SWC records")
        while stack:
            node_id, parent_swc_id = stack.pop()

            if node_id in visited:
                continue
            visited.add(node_id)

            data = self.nodes[node_id]

            records[swc_id] = swctools.SWCRecord(
                n=swc_id,
                t=data.get("t", 3),
                x=data["x"],
                y=data["y"],
                z=data["z"],
                r=data["r"],
                parent=parent_swc_id,
                line=swc_id,
            )

            node_id_map[node_id] = swc_id
            current_swc_id = swc_id
            swc_id += 1

            for neighbor in tree_graph.neighbors(node_id):
                if neighbor not in visited:
                    stack.append((neighbor, current_swc_id))

        logger.debug(f"Created {len(records)} SWC records")

        logger.debug("Processing reconnections for CYCLE_BREAK directives")
        cycle_break_pairs = []
        for u, v in reconnections:
            if u in node_id_map and v in node_id_map:
                u_swc = node_id_map[u]
                v_swc = node_id_map[v]

                data_u = self.nodes[u]
                data_v = self.nodes[v]

                # Check if u and v already have matching coordinates and radius
                coords_match = (
                    abs(data_u["x"] - data_v["x"]) < 1e-9
                    and abs(data_u["y"] - data_v["y"]) < 1e-9
                    and abs(data_u["z"] - data_v["z"]) < 1e-9
                    and abs(data_u["r"] - data_v["r"]) < 1e-9
                )

                if coords_match:
                    # Nodes already have matching coordinates and radius, use directly
                    cycle_break_pairs.append((u_swc, v_swc))
                    logger.debug(
                        f"Nodes {u} and {v} already have matching coordinates and radius, "
                        f"using directly for CYCLE_BREAK ({u_swc}, {v_swc})"
                    )
                else:
                    # Create duplicate at v's position with v's radius for CYCLE_BREAK compliance
                    duplicate_record = swctools.SWCRecord(
                        n=swc_id,
                        t=data_u.get("t", 3),
                        x=data_v["x"],
                        y=data_v["y"],
                        z=data_v["z"],
                        r=data_v["r"],  # Use v's radius
                        parent=v_swc,
                        line=swc_id,
                    )
                    records[swc_id] = duplicate_record
                    cycle_break_pairs.append((swc_id, v_swc))
                    logger.debug(
                        f"Created duplicate node {swc_id} at v's position with v's radius "
                        f"for CYCLE_BREAK ({swc_id}, {v_swc})"
                    )
                    swc_id += 1

        header_lines = []
        for u_swc, v_swc in cycle_break_pairs:
            header_lines.append(f"CYCLE_BREAK reconnect {u_swc} {v_swc}")
        logger.debug(f"Generated {len(header_lines)} CYCLE_BREAK directives")

        swc_model = swctools.SWCModel.from_records(records)
        swc_model.to_swc_file(filepath, precision=precision, header=header_lines)
        logger.info(f"Successfully saved CableGraph to {filepath}")

    def get_root(self) -> Optional[int]:
        """Get the root node ID of the graph.

        Returns:
            Node ID of the root (node with parent=-1), or the node with
            minimum ID if no explicit root exists, or None for empty graphs

        Examples:
            >>> root = graph.get_root()
            >>> if root is not None:
            ...     print(f"Root position: {graph.get_position(root)}")
        """
        for node_id, data in self.nodes(data=True):
            if data.get("parent", -1) == -1:
                logger.debug(f"Found root node: {node_id}")
                return node_id
        if self.number_of_nodes() > 0:
            root = min(self.nodes())
            logger.debug(f"No explicit root, using minimum node ID: {root}")
            return root
        logger.debug("Graph is empty, no root node")
        return None

    def get_position(self, node_id: int) -> tuple[float, float, float]:
        """Get the 3D position of a node.

        Args:
            node_id: ID of the node to query

        Returns:
            Tuple of (x, y, z) coordinates

        Raises:
            KeyError: If node_id does not exist in the graph
        """
        data = self.nodes[node_id]
        return (data["x"], data["y"], data["z"])

    def get_radius(self, node_id: int) -> float:
        """Get the radius of a node.

        Args:
            node_id: ID of the node to query

        Returns:
            Radius value at this node

        Raises:
            KeyError: If node_id does not exist in the graph
        """
        return self.nodes[node_id]["r"]
