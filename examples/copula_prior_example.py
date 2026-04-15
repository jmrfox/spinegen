"""Example: Using ConditionalSpinePrior with different copula types.

This script demonstrates how to use the ConditionalSpinePrior class with
various copula types to generate spines with correlated morphological features.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from spinegen import (
    CableGraph,
    SpineAnalyzer,
    ConditionalSpinePrior,
    SpineGenerator,
    SpinePrior,
)
import matplotlib.pyplot as plt

# Load real spine data
data_dir = Path(__file__).parent.parent / "data" / "real"
swc_files = list(data_dir.glob("*.swc"))

print(f"Loading {len(swc_files)} SWC files...")
graphs = [CableGraph.from_swc_file(str(f)) for f in swc_files[:5]]

# Analyze morphology
print("Analyzing morphology...")
analyzer = SpineAnalyzer(graphs)
analyzer.analyze()

print("\n" + "=" * 70)
print("COMPARING DIFFERENT COPULA TYPES")
print("=" * 70)

# Test different copula types
copula_types = ["gaussian", "clayton", "frank"]
results = {}

for copula_type in copula_types:
    print(f"\n--- Testing {copula_type.upper()} copula ---")

    # Create conditional prior
    prior = ConditionalSpinePrior(
        analyzer.stats, copula_type=copula_type, use_multivariate=False
    )

    print(prior)

    # Visualize
    fig = prior.plot_copulas(figsize=(15, 5))
    plot_path = f"copula_{copula_type}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot: {plot_path}")
    plt.close()

    # Generate spine
    generator = SpineGenerator(prior)
    synthetic_graph = generator.generate(max_steps=500, max_nodes=100)

    output_path = (
        Path(__file__).parent.parent / "data" / "synthetic" / f"spine_{copula_type}.swc"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    synthetic_graph.to_swc_file(str(output_path))

    results[copula_type] = {
        "nodes": synthetic_graph.number_of_nodes(),
        "edges": synthetic_graph.number_of_edges(),
        "path": output_path,
    }

# Test multivariate copula
print("\n" + "=" * 70)
print("TESTING MULTIVARIATE COPULA")
print("=" * 70)

for mv_type in ["gaussian", "vine"]:
    print(f"\n--- Testing {mv_type.upper()} multivariate copula ---")

    prior_mv = ConditionalSpinePrior(
        analyzer.stats, use_multivariate=True, multivariate_type=mv_type
    )

    print(prior_mv)

    # Generate spine
    generator_mv = SpineGenerator(prior_mv)
    synthetic_graph_mv = generator_mv.generate(max_steps=500, max_nodes=100)

    output_path_mv = (
        Path(__file__).parent.parent
        / "data"
        / "synthetic"
        / f"spine_multivariate_{mv_type}.swc"
    )
    synthetic_graph_mv.to_swc_file(str(output_path_mv))

    results[f"multivariate_{mv_type}"] = {
        "nodes": synthetic_graph_mv.number_of_nodes(),
        "edges": synthetic_graph_mv.number_of_edges(),
        "path": output_path_mv,
    }

# Baseline: Independent prior
print("\n--- Baseline: Independent prior ---")
independent_prior = SpinePrior(analyzer.stats)
generator_indep = SpineGenerator(independent_prior)
synthetic_graph_indep = generator_indep.generate(max_steps=500, max_nodes=100)

output_path_indep = (
    Path(__file__).parent.parent / "data" / "synthetic" / "spine_independent.swc"
)
synthetic_graph_indep.to_swc_file(str(output_path_indep))

results["independent"] = {
    "nodes": synthetic_graph_indep.number_of_nodes(),
    "edges": synthetic_graph_indep.number_of_edges(),
    "path": output_path_indep,
}

# Summary
print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)
for name, result in results.items():
    print(f"{name:25s}: {result['nodes']:3d} nodes, {result['edges']:3d} edges")

print("\n" + "=" * 70)
print("KEY INSIGHTS")
print("=" * 70)
print("- Gaussian copula: Captures linear correlations")
print("- Clayton copula: Models lower tail dependence")
print("- Frank copula: Symmetric dependence structure")
print("- Multivariate: Captures higher-order dependencies")
print("- Independent: No correlation modeling (baseline)")
print("\nCompare generated SWC files to see structural differences!")
print("=" * 70)
