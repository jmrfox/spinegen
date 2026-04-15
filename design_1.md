# 📦 Project: Cyclic Spine Morphology Analysis & Generation

## 🧱 Core Objective
Implement a Python package for algorithmic generation of neural spines in cable-graph model format with cycle support.

## ✅ Status: CORE IMPLEMENTATION COMPLETE
- All classes implemented and functional
- Comprehensive docstrings and logging added
- CYCLE_BREAK format compliance verified
- Prior visualization and inspection tools added

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

- [x] `__init__(self)` - Initialize empty graph ✅

- [x] `from_swc_file(cls, filepath)` - Class method ✅
  - Load SWC file using `swctools.SWCModel.from_swc_file()`
  - Call `make_cycle_connections()` to get nx.Graph with cycles
  - Copy nodes and edges into CableGraph instance
  - Preserve all node attributes (x, y, z, r, t)
  - Store reconnections metadata if present
  - **Includes comprehensive logging**

- [x] `to_swc_file(self, filepath, precision=6)` - Instance method ✅
  - Detect cycles in current graph
  - Identify reconnection pairs (edges that close cycles)
  - Create SWCModel tree by removing cycle-closing edges
  - Generate header with CYCLE_BREAK reconnect directives
  - Write to file using `swctools` format
  - **CYCLE_BREAK compliance: Ensures reconnected nodes have identical xyz and radius**

- [x] `get_root(self)` - Return root node ID (parent == -1) ✅

- [x] `get_position(self, node_id)` - Return (x, y, z) tuple ✅

- [x] `get_radius(self, node_id)` - Return radius ✅

---

# 🛠️ Utility Functions (utils.py)

- [x] `euclidean_distance(p1, p2)` - Distance between two 3D points ✅

- [x] `angle_between_vectors(v1, v2)` - Compute angle between 3D vectors ✅

- [x] `unit_vector(v)` - Normalize vector to unit length ✅

- [x] `split_edge(graph, u, v, new_node_id, new_pos, new_radius)` - Insert node on edge ✅

**All utility functions include comprehensive docstrings and logging**

---

# 📊 Class 1: SpineAnalyzer

## Purpose
Extract morphological statistics from a collection of CableGraph models.

## 🔹 Initialization

- [x] Define class:
    class SpineAnalyzer:
        def __init__(self, graphs: List[CableGraph]):
            self.graphs = graphs
            self.stats = None

---

## 🔹 Step 1: Preprocessing per graph

For each graph:

- [x] Identify root node (parent == -1)
- [x] Build adjacency list from edges
- [x] Compute node degrees
- [x] Identify:
  - [x] leaf nodes (degree == 1, excluding root)
  - [x] branch nodes (degree >= 3)
- [x] Ensure all edges are bidirectional

---

## 🔹 Step 2: Extract branch segments

- [x] Traverse graph to decompose into segments:
  - Segment = path between:
    - branch node ↔ branch node
    - branch node ↔ leaf
    - root ↔ first branch

- [x] Algorithm:
  - For each node with degree != 2:
    - For each neighbor:
      - Walk forward until reaching node with degree != 2
      - Record path

- [x] For each segment:
  - [x] Compute length (sum of Euclidean distances)
  - [x] Store ordered list of node coordinates

---

## 🔹 Step 3: Curvature estimation

For each segment:

- [x] For each triple of consecutive points (p1, p2, p3):
  - [x] Compute vectors v1 = p2 - p1, v2 = p3 - p2
  - [x] Compute angle between v1 and v2
- [x] Store:
  - [x] mean curvature
  - [x] curvature variance

---

## 🔹 Step 4: Branching statistics

For each branch node:

- [x] Count number of children
- [x] For each child pair:
  - [x] Compute branching angle

Store:
- [x] distribution of branch counts
- [x] distribution of branch angles

---

## 🔹 Step 5: Radius statistics

- [x] For each segment:
  - [x] Collect radii along segment
- [x] Compute:
  - [x] mean radius
  - [x] radius vs distance-from-root
  - [x] taper rate (linear fit of radius vs path length)

---

## 🔹 Step 6: Path length statistics

- [x] Compute shortest path from root to all nodes
- [x] Store:
  - [x] distribution of root-to-leaf path lengths
  - [x] distribution of segment lengths

---

## 🔹 Step 7: Cycle detection

For each graph:

- [x] Use cycle detection (DFS or union-find)
- [x] Extract list of cycles:
  - Represent each cycle as ordered node list

- [x] For each cycle:
  - [x] Compute:
    - cycle length (geodesic)
    - mean spatial radius (average distance to centroid)
    - chord length (max pairwise distance)

---

## 🔹 Step 8: Loop formation geometry

For each cycle:

- [x] Identify edge that closes the loop
- [x] Compute:
  - [x] Euclidean distance between endpoints before connection
  - [x] Graph distance between endpoints before connection
  - [x] Angle between connecting segments

Store distributions of:
- [x] fusion distance
- [x] fusion angle
- [x] pre-loop graph distance

---

## 🔹 Step 9: Aggregate statistics

- [x] Combine across all graphs:
  - branch count distribution
  - segment length distribution
  - curvature distribution
  - radius distribution
  - fusion distance distribution
  - fusion probability vs distance

- [x] Store results in:
    self.stats = {
        "segment_lengths": np.array,
        "curvatures": np.array,
        "branch_counts": np.array,
        "branch_angles": np.array,
        "radii": np.array,
        "fusion_distances": np.array,
        "fusion_graph_distances": np.array,
        "fusion_angles": np.array
    }

**SpineAnalyzer is fully implemented with comprehensive docstrings and logging**

---

# � Class 2: SpinePrior

## Purpose
Parameterize statistical distributions from SpineAnalyzer for sampling in SpineGenerator.

## 🔹 Initialization

- [x] Define class:
    class SpinePrior:
        def __init__(self, stats: dict):
            self.stats = stats
            self.distributions = {}

---

## 🔹 Parameterization

- [x] For each statistic, fit parametric distributions:
  - [x] **Segment lengths**: Gamma or Log-normal distribution
  - [x] **Curvature angles**: Von Mises or wrapped normal distribution
  - [x] **Branch counts**: Categorical distribution (e.g., {2: 0.8, 3: 0.15, 4: 0.05})
  - [x] **Branch angles**: Von Mises or normal distribution
  - [x] **Radii**: Normal or log-normal distribution with taper model
  - [x] **Fusion distance**: Exponential or gamma distribution
  - [x] **Fusion probability**: Function of spatial and graph distance

---

## 🔹 Sampling Methods

- [x] `sample_segment_length()` - Sample from segment length distribution ✅

- [x] `sample_curvature()` - Sample angular deviation for growth direction ✅

- [x] `sample_branch_count()` - Sample number of children at branch point ✅

- [x] `sample_branch_angle()` - Sample angle between branches ✅

- [x] `sample_radius(distance_from_root)` - Sample radius with taper ✅

- [x] `sample_fusion_distance()` - Sample search radius for fusion ✅

- [x] `compute_fusion_probability(spatial_dist, graph_dist)` - Compute fusion probability ✅

## 🔹 Inspection & Visualization Methods (NEW)

- [x] `__str__()` - String representation for `print(prior)` ✅

- [x] `print()` - Generate formatted string with all distribution information ✅
  - Shows distribution types and parameters
  - Displays computed statistics (means, etc.)
  - Lists sample counts for each feature

- [x] `plot_distributions(figsize, save_path)` - Visualize all fitted distributions ✅
  - Creates 3×3 grid of histograms + fitted PDFs
  - Uses seaborn for professional styling
  - Supports saving to file at 300 dpi
  - Returns matplotlib Figure object

**SpinePrior is fully implemented with comprehensive docstrings and logging**

---

# � Class 3: SpineGenerator

## Purpose
Generate synthetic spine models using statistical distributions from SpinePrior.

## 🔹 Initialization

- [x] Define class:
    class SpineGenerator:
        def __init__(self, prior: SpinePrior):
            self.prior = prior
            self.graph = None

---

## 🔹 Step 1: Initialize graph

- [x] Create root node at (0,0,0)
- [x] Initialize:
  - nodes dict
  - edges set
  - active tips list = [root]

---

## 🔹 Step 2: Growth loop

- [x] For t in range(max_steps):

  For each active tip:

  - [x] Sample step length from segment length distribution
  - [x] Sample direction:
    - if first step → random unit vector
    - else → perturb previous direction using curvature distribution

  - [x] Compute new point:
        new_pos = old_pos + step_length * direction

  - [x] Create new node
  - [x] Add edge (tip → new node)

---

## 🔹 Step 3: Branching

For each active tip:

- [x] Sample branching probability from stats
- [x] If branching occurs:
  - [x] Sample number of children (usually 2)
  - [x] Generate new directions using branching angle distribution
  - [x] Add new tips

- [x] Else:
  - [x] Continue single extension

---

## 🔹 Step 4: Fusion detection (loop formation)

For each new node:

- [x] Find nearby segments within radius R (from fusion_distance distribution)

- [x] For each candidate segment:
  - [x] Compute:
    - Euclidean distance d
    - graph distance g

  - [x] Compute fusion probability:
        p = exp(-d^2 / sigma^2) * (1 - exp(-g / lambda))

  - [x] Sample random number:
    - If random < p:
        → perform fusion

---

## 🔹 Step 5: Fusion operation

When fusion occurs:

- [x] Find closest point on segment
- [x] Insert new node at that location
- [x] Split segment into two edges
- [x] Connect current node to new node
- [x] This creates a cycle

---

## 🔹 Step 6: Radius assignment

- [x] For each node:
  - [x] Assign radius based on:
    - distance from root
    - sampled taper distribution

---

## 🔹 Step 7: Termination

- [x] Stop growth when:
  - max_steps reached OR
  - number of nodes exceeds threshold

---

## 🔹 Step 8: Output

- [x] `generate(max_steps, max_nodes)` - Main method ✅
  - Execute steps 1-7
  - Return CableGraph instance with generated morphology
  - **Includes comprehensive logging of generation process**

- [x] Return CableGraph object that can be: ✅
  - Analyzed with SpineAnalyzer
  - Saved to SWC file with `to_swc_file()`
  - Visualized or further processed

**SpineGenerator is fully implemented with comprehensive docstrings and logging**

## 🔹 CYCLE_BREAK Compliance (CRITICAL FIX)

- [x] Generator creates clones with matching xyz and radius for fusion ✅
- [x] CableGraph.to_swc_file() ensures CYCLE_BREAK pairs have identical coordinates ✅
- [x] Verified: Reconnected nodes satisfy SWC format requirements ✅

---

# 🔬 Validation & Testing

- [x] Basic pipeline test implemented in `tests/test_pipeline.py` ✅
- [ ] Re-run SpineAnalyzer on generated graphs
- [ ] Compare distributions:
  - segment lengths
  - curvature
  - cycle counts
  - fusion distances
- [ ] Ensure generated graphs fall within empirical ranges
- [ ] Add unit tests for individual components

# 📊 Documentation & Code Quality

- [x] Comprehensive docstrings added to all modules ✅
  - Module-level docstrings
  - Class docstrings with attribute descriptions
  - Method docstrings with Args, Returns, Examples
  - Google-style formatting

- [x] Logging integrated throughout codebase ✅
  - INFO level: Major operations, file I/O, generation milestones
  - DEBUG level: Detailed step-by-step operations
  - WARNING level: Potential issues
  - ERROR level: Failures

- [x] README.md with comprehensive documentation ✅
  - Architecture overview
  - Installation instructions
  - Usage examples
  - Detailed algorithm explanation

# 🔧 Known Issues & Future Work

## Edge Length Tuning
- Generated edge lengths may be too long
- **Solution**: Scale `prior.distributions["segment_length"]["params"]["scale"]` by desired factor
- Edge length calculation verified as correct

## Self-Avoidance
- Generator does NOT implement self-avoidance during tree formation
- Branches can grow through existing structure
- Only spatial interaction is fusion detection (creates cycles, not avoidance)
- **Future**: Add collision detection and avoidance mechanism

## Data Organization
- Real SWC files: `data/real/`
- Generated outputs: `data/synthetic/`
- Test scripts: `tests/`

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