# spinegen

Algorithmic generation of neural spines in cable-graph model format with cycle support.

## Overview

`spinegen` is a Python package for generating synthetic dendritic spine morphologies based on statistical analysis of real spine data. The package supports cyclic graph structures (beyond traditional tree-based SWC models) using the `swctools` library for SWC file handling with cycle-closure directives.

## Installation

```bash
uv pip install -e .
```

## Quick Start

```python
from pathlib import Path
import spinegen

# Load real spine data
data_dir = Path("data/real")
graphs = [spinegen.CableGraph.from_swc_file(f) for f in data_dir.glob("*.swc")]

# Analyze morphology
analyzer = spinegen.SpineAnalyzer(graphs)
stats = analyzer.analyze()

# Create prior distribution
prior = spinegen.SpinePrior(stats)

# Generate synthetic spine
generator = spinegen.SpineGenerator(prior)
synthetic_graph = generator.generate(max_steps=100, max_nodes=50)

# Save to SWC
synthetic_graph.to_swc_file("data/synthetic/output.swc")
```

## Architecture

### Core Classes

- **`CableGraph`**: Graph model inheriting from `networkx.Graph` with SWC I/O support
- **`SpineAnalyzer`**: Extracts morphological statistics from real spine data
- **`SpinePrior`**: Parameterizes statistics into probability distributions
- **`SpineGenerator`**: Samples from prior to generate synthetic morphologies

## Generation Algorithm

The spine generation algorithm is a growth-based stochastic process that produces realistic dendritic spine morphologies with support for branching and fusion (cycle formation).

### Algorithm Overview

1. **Initialization**: Start with a single root node at the origin (0, 0, 0)
2. **Growth Loop**: Iteratively grow the structure until termination criteria are met
3. **Output**: Return a `CableGraph` with nodes, edges, and potential cycles

### Detailed Steps

#### 1. Root Initialization

```python
root_node = {
    'position': (0, 0, 0),
    'radius': sample_radius(distance_from_root=0),
    'type': 3  # dendrite type in SWC
}
```

All generated spines start at the origin, unlike real spine data which may be positioned arbitrarily in 3D space based on their location on the parent dendrite.

#### 2. Growth Loop

The algorithm maintains a list of **active tips** (growing endpoints). For each iteration:

##### a. Extension

For each active tip:
- **Sample segment length** from Gamma distribution fitted to real data
- **Compute growth direction**:
  - If first segment: random unit vector
  - Otherwise: perturb previous direction based on sampled curvature
- **Create new node** at `tip_position + length * direction`
- **Assign radius** using lognormal distribution with tapering (decreases with distance from root)

##### b. Direction Perturbation

Curvature is modeled as angular deviation from the previous direction:

```
curvature_angle ~ Normal(μ_curv, σ_curv)
rotation_axis = random perpendicular vector to previous_direction
new_direction = rotate(previous_direction, rotation_axis, curvature_angle)
```

This creates realistic smooth curves in spine morphology.

##### c. Branching Decision

With probability `p_branch` (typically 0.2):
- **Sample branch count** from categorical distribution (typically 2-3 branches)
- **Generate branch directions** by rotating parent direction by sampled branch angles
- **Add multiple new tips** to active list

Branch angles are sampled from a normal distribution fitted to observed angles between branches in real data.

##### d. Fusion Detection

For each newly created node, the algorithm searches for potential fusion targets:

1. **Spatial search**: Find all edges within `fusion_distance` threshold
2. **Closest point calculation**: For each candidate edge, compute closest point on segment
3. **Fusion probability**: Compute based on two factors:
   - **Spatial distance** `d_spatial`: exponential decay `exp(-d²/2σ²)`
   - **Graph distance** `d_graph`: path length in tree structure
   - Combined: `P_fusion = P_spatial × P_graph`

4. **Fusion operation**:
   - If fusion point is mid-edge: split edge and insert new node
   - If fusion point is near endpoint: connect directly to endpoint
   - Create edge between new node and fusion target → **forms cycle**

This fusion mechanism is what enables the generation of cyclic structures, which are observed in real spine data where membrane compartments reconnect.

##### e. Radius Assignment

Radii are sampled from a lognormal distribution with tapering:

```
base_radius ~ LogNormal(μ_r, σ_r)
taper_factor = exp(-distance_from_root / λ)
final_radius = base_radius × taper_factor
```

This creates realistic thickness variation where structures are thicker near the root and taper toward tips.

#### 3. Termination Criteria

The growth loop terminates when:
- **Maximum steps** reached (prevents infinite loops)
- **Maximum nodes** reached (controls output size)
- **No active tips** remain (all branches terminated)

### Statistical Distributions

The `SpinePrior` class fits the following distributions from real data:

| Feature | Distribution | Parameters |
|---------|-------------|------------|
| Segment length | Gamma | shape, scale, loc |
| Curvature angle | Normal | μ, σ |
| Branch count | Categorical | values, probabilities |
| Branch angle | Normal | μ, σ |
| Radius | LogNormal | μ, σ |
| Fusion distance | Exponential | scale, loc |
| Fusion graph distance | Empirical | mean, std |

### Cycle Handling

Generated graphs may contain cycles due to fusion events. When saving to SWC format:

1. **Cycle detection**: Use `networkx.simple_cycles()` to find all cycles
2. **Tree extraction**: Remove one edge per cycle to create spanning tree
3. **Reconnection directives**: Add `CYCLE_BREAK reconnect <id1> <id2>` headers
4. **Node duplication**: Duplicate reconnection nodes in SWC to maintain tree structure

Example output header:
```
# CYCLE_BREAK reconnect 34 35
# CYCLE_BREAK reconnect 68 11
```

This format is compatible with `swctools` and can be loaded back into `CableGraph` with cycles restored via `make_cycle_connections()`.

## Testing

Run the test suite:

```bash
uv run pytest
```

Run the pipeline test:

```bash
uv run python tests/test_pipeline.py
```

## Data Organization

- `data/real/`: Real spine SWC files from experimental data
- `data/synthetic/`: Generated synthetic spine outputs

## Dependencies

- `networkx`: Graph data structures
- `numpy`: Numerical operations
- `scipy`: Statistical distributions
- `swctools`: SWC file I/O with cycle support

## References

The algorithm is inspired by growth-based morphology generation techniques and statistical shape modeling approaches used in computational neuroscience.