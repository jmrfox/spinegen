# 📦 Project: Cyclic Spine Morphology Analysis & Generation

## 🧱 Core Objective
Implement a Python package for algorithmic generation of neural spines in cable-graph model format with cycle support.

### Architecture Overview

1. **CableGraph** (inherits from `networkx.Graph`)
   - Internal cable-graph model supporting cycles
   - Load SWC files using `swctools.SWCModel` + `make_cycle_connections()`
   - Write SWC files with cycle-closure directives (CYCLE_BREAK reconnect)

2. **SpineAnalyzer** (formerly SpineStatisticsAnalyzer)
   - Input: one or more CableGraph models
   - Output: statistical distributions describing morphology and loop formation

3. **SpinePrior**
   - Input: statistics from SpineAnalyzer
   - Output: parameterized distributions for sampling
   - Acts as the probability model for SpineGenerator

4. **SpineGenerator**
   - Input: SpinePrior object
   - Output: new synthetic CableGraph models via growth + fusion
   - Samples from the prior distribution to generate morphologies

### Workflow
Real Spines (SWC) → CableGraph.load() → SpineAnalyzer → SpinePrior → SpineGenerator → Synthetic Spines (CableGraph) → CableGraph.save()

---

# ✅ Package Structure

## Module Organization
- `spinegen/__init__.py` - Package exports
- `spinegen/cable_graph.py` - CableGraph class
- `spinegen/analyzer.py` - SpineAnalyzer class
- `spinegen/prior.py` - SpinePrior class
- `spinegen/generator.py` - SpineGenerator class
- `spinegen/utils.py` - Utility functions

---

# 🔌 Class 0: CableGraph (inherits from networkx.Graph)

## Purpose
Internal cable-graph model that supports cycles and interfaces with SWC format.

## 🔹 Node Attributes
Each node stores:
- `x`: float (x-coordinate)
- `y`: float (y-coordinate)
- `z`: float (z-coordinate)
- `r`: float (radius)
- `t`: int (SWC type tag, default=3 for dendrite)
- `parent`: int (original parent in tree, -1 for root)

## 🔹 Methods

- [ ] `__init__(self)` - Initialize empty graph

- [ ] `from_swc_file(cls, filepath)` - Class method
  - Load SWC file using `swctools.SWCModel.from_swc_file()`
  - Call `make_cycle_connections()` to get nx.Graph with cycles
  - Copy nodes and edges into CableGraph instance
  - Preserve all node attributes (x, y, z, r, t)
  - Store reconnections metadata if present

- [ ] `to_swc_file(self, filepath, precision=6)` - Instance method
  - Detect cycles in current graph
  - Identify reconnection pairs (edges that close cycles)
  - Create SWCModel tree by removing cycle-closing edges
  - Generate header with CYCLE_BREAK reconnect directives
  - Write to file using `swctools` format

- [ ] `get_root(self)` - Return root node ID (parent == -1)

- [ ] `get_position(self, node_id)` - Return (x, y, z) tuple

- [ ] `get_radius(self, node_id)` - Return radius

---

# 🛠️ Utility Functions (utils.py)

- [ ] `euclidean_distance(p1, p2)` - Distance between two 3D points

- [ ] `compute_graph_distances(graph)` - All-pairs shortest path (geodesic)

- [ ] `find_neighbors_within_radius(graph, point, radius)` - Spatial search

- [ ] `interpolate_point_on_segment(p1, p2, t)` - Linear interpolation (t ∈ [0,1])

- [ ] `split_edge(graph, u, v, new_node_id, new_pos, new_radius)` - Insert node on edge

---

# 📊 Class 1: SpineAnalyzer

## Purpose
Extract morphological statistics from a collection of CableGraph models.

## 🔹 Initialization

- [ ] Define class:
    class SpineAnalyzer:
        def __init__(self, graphs: List[CableGraph]):
            self.graphs = graphs
            self.stats = None

---

## 🔹 Step 1: Preprocessing per graph

For each graph:

- [ ] Identify root node (parent == -1)
- [ ] Build adjacency list from edges
- [ ] Compute node degrees
- [ ] Identify:
  - [ ] leaf nodes (degree == 1, excluding root)
  - [ ] branch nodes (degree >= 3)
- [ ] Ensure all edges are bidirectional

---

## 🔹 Step 2: Extract branch segments

- [ ] Traverse graph to decompose into segments:
  - Segment = path between:
    - branch node ↔ branch node
    - branch node ↔ leaf
    - root ↔ first branch

- [ ] Algorithm:
  - For each node with degree != 2:
    - For each neighbor:
      - Walk forward until reaching node with degree != 2
      - Record path

- [ ] For each segment:
  - [ ] Compute length (sum of Euclidean distances)
  - [ ] Store ordered list of node coordinates

---

## 🔹 Step 3: Curvature estimation

For each segment:

- [ ] For each triple of consecutive points (p1, p2, p3):
  - [ ] Compute vectors v1 = p2 - p1, v2 = p3 - p2
  - [ ] Compute angle between v1 and v2
- [ ] Store:
  - [ ] mean curvature
  - [ ] curvature variance

---

## 🔹 Step 4: Branching statistics

For each branch node:

- [ ] Count number of children
- [ ] For each child pair:
  - [ ] Compute branching angle

Store:
- [ ] distribution of branch counts
- [ ] distribution of branch angles

---

## 🔹 Step 5: Radius statistics

- [ ] For each segment:
  - [ ] Collect radii along segment
- [ ] Compute:
  - [ ] mean radius
  - [ ] radius vs distance-from-root
  - [ ] taper rate (linear fit of radius vs path length)

---

## 🔹 Step 6: Path length statistics

- [ ] Compute shortest path from root to all nodes
- [ ] Store:
  - [ ] distribution of root-to-leaf path lengths
  - [ ] distribution of segment lengths

---

## 🔹 Step 7: Cycle detection

For each graph:

- [ ] Use cycle detection (DFS or union-find)
- [ ] Extract list of cycles:
  - Represent each cycle as ordered node list

- [ ] For each cycle:
  - [ ] Compute:
    - cycle length (geodesic)
    - mean spatial radius (average distance to centroid)
    - chord length (max pairwise distance)

---

## 🔹 Step 8: Loop formation geometry

For each cycle:

- [ ] Identify edge that closes the loop
- [ ] Compute:
  - [ ] Euclidean distance between endpoints before connection
  - [ ] Graph distance between endpoints before connection
  - [ ] Angle between connecting segments

Store distributions of:
- [ ] fusion distance
- [ ] fusion angle
- [ ] pre-loop graph distance

---

## 🔹 Step 9: Aggregate statistics

- [ ] Combine across all graphs:
  - branch count distribution
  - segment length distribution
  - curvature distribution
  - radius distribution
  - fusion distance distribution
  - fusion probability vs distance

- [ ] Store results in:
    self.stats = {
        "branch_counts": ...,
        "segment_lengths": ...,
        "curvature": ...,
        "radius": ...,
        "fusion_distance": ...,
        "fusion_graph_distance": ...,
        "fusion_angles": ...
    }

---

# � Class 2: SpinePrior

## Purpose
Parameterize statistical distributions from SpineAnalyzer for sampling in SpineGenerator.

## 🔹 Initialization

- [ ] Define class:
    class SpinePrior:
        def __init__(self, stats: dict):
            self.stats = stats
            self.distributions = {}

---

## 🔹 Parameterization

- [ ] For each statistic, fit parametric distributions:
  - [ ] **Segment lengths**: Gamma or Log-normal distribution
  - [ ] **Curvature angles**: Von Mises or wrapped normal distribution
  - [ ] **Branch counts**: Categorical distribution (e.g., {2: 0.8, 3: 0.15, 4: 0.05})
  - [ ] **Branch angles**: Von Mises or normal distribution
  - [ ] **Radii**: Normal or log-normal distribution with taper model
  - [ ] **Fusion distance**: Exponential or gamma distribution
  - [ ] **Fusion probability**: Function of spatial and graph distance

---

## 🔹 Sampling Methods

- [ ] `sample_segment_length()` - Sample from segment length distribution

- [ ] `sample_curvature()` - Sample angular deviation for growth direction

- [ ] `sample_branch_count()` - Sample number of children at branch point

- [ ] `sample_branch_angle()` - Sample angle between branches

- [ ] `sample_radius(distance_from_root)` - Sample radius with taper

- [ ] `sample_fusion_distance()` - Sample search radius for fusion

- [ ] `compute_fusion_probability(spatial_dist, graph_dist)` - Compute fusion probability

---

# � Class 3: SpineGenerator

## Purpose
Generate synthetic spine models using statistical distributions from SpinePrior.

## 🔹 Initialization

- [ ] Define class:
    class SpineGenerator:
        def __init__(self, prior: SpinePrior):
            self.prior = prior
            self.graph = None

---

## 🔹 Step 1: Initialize graph

- [ ] Create root node at (0,0,0)
- [ ] Initialize:
  - nodes dict
  - edges set
  - active tips list = [root]

---

## 🔹 Step 2: Growth loop

- [ ] For t in range(max_steps):

  For each active tip:

  - [ ] Sample step length from segment length distribution
  - [ ] Sample direction:
    - if first step → random unit vector
    - else → perturb previous direction using curvature distribution

  - [ ] Compute new point:
        new_pos = old_pos + step_length * direction

  - [ ] Create new node
  - [ ] Add edge (tip → new node)

---

## 🔹 Step 3: Branching

For each active tip:

- [ ] Sample branching probability from stats
- [ ] If branching occurs:
  - [ ] Sample number of children (usually 2)
  - [ ] Generate new directions using branching angle distribution
  - [ ] Add new tips

- [ ] Else:
  - [ ] Continue single extension

---

## 🔹 Step 4: Fusion detection (loop formation)

For each new node:

- [ ] Find nearby segments within radius R (from fusion_distance distribution)

- [ ] For each candidate segment:
  - [ ] Compute:
    - Euclidean distance d
    - graph distance g

  - [ ] Compute fusion probability:
        p = exp(-d^2 / sigma^2) * (1 - exp(-g / lambda))

  - [ ] Sample random number:
    - If random < p:
        → perform fusion

---

## 🔹 Step 5: Fusion operation

When fusion occurs:

- [ ] Find closest point on segment
- [ ] Insert new node at that location
- [ ] Split segment into two edges
- [ ] Connect current node to new node
- [ ] This creates a cycle

---

## 🔹 Step 6: Radius assignment

- [ ] For each node:
  - [ ] Assign radius based on:
    - distance from root
    - sampled taper distribution

---

## 🔹 Step 7: Termination

- [ ] Stop growth when:
  - max_steps reached OR
  - number of nodes exceeds threshold

---

## 🔹 Step 8: Output

- [ ] `generate(max_steps, max_nodes)` - Main method
  - Execute steps 1-7
  - Return CableGraph instance with generated morphology

- [ ] Return CableGraph object that can be:
  - Analyzed with SpineAnalyzer
  - Saved to SWC file with `to_swc_file()`
  - Visualized or further processed

---

# 🔬 Validation (OPTIONAL BUT RECOMMENDED)

- [ ] Re-run SpineAnalyzer on generated graphs
- [ ] Compare distributions:
  - segment lengths
  - curvature
  - cycle counts
  - fusion distances

- [ ] Ensure generated graphs fall within empirical ranges

---

# 📝 Example Usage

```python
# Load real spine data
graphs = [CableGraph.from_swc_file(f) for f in swc_files]

# Analyze morphology
analyzer = SpineAnalyzer(graphs)
analyzer.analyze()  # Populates analyzer.stats

# Create prior distribution
prior = SpinePrior(analyzer.stats)

# Generate synthetic spines
generator = SpineGenerator(prior)
synthetic_graph = generator.generate(max_steps=1000, max_nodes=100)

# Save to SWC file
synthetic_graph.to_swc_file("synthetic_spine.swc")
```

---

# 🚀 End of Specification