"""Example: Validating generated spines against the prior.

This script demonstrates how to use the Validator class to assess
the quality of generated spines by comparing them to the prior distribution.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from spinegen import (
    CableGraph,
    SpineAnalyzer,
    SpinePrior,
    SpinePriorOptions,
    SpineGenerator,
    Validator,
)
import matplotlib.pyplot as plt

# Load real spine data
data_dir = Path(__file__).parent.parent / "data" / "real"
swc_files = list(data_dir.glob("*.swc"))

print(f"Loading {len(swc_files)} SWC files...")
real_graphs = [CableGraph.from_swc_file(str(f)) for f in swc_files[:5]]

# Analyze real spines
print("Analyzing real spine morphology...")
analyzer = SpineAnalyzer(real_graphs)
analyzer.analyze()

print("\n" + "=" * 70)
print("TESTING DIFFERENT PRIOR CONFIGURATIONS")
print("=" * 70)

# Test 1: Independent prior (no copulas)
print("\n--- Test 1: Independent Prior ---")
prior_independent = SpinePrior(analyzer.stats)
generator_independent = SpineGenerator(prior_independent)

generated_independent = []
for i in range(10):
    graph = generator_independent.generate(max_steps=500, max_nodes=100)
    generated_independent.append(graph)
    print(f"  Generated spine {i+1}: {graph.number_of_nodes()} nodes")

# Validate independent prior
validator_independent = Validator(prior_independent, generated_independent)
print("\n" + validator_independent.summary())

# Save comparison plots
fig = validator_independent.plot_comparison(figsize=(16, 12))
if fig:
    plt.savefig("validation_independent_distributions.png", dpi=300)
    print("Saved: validation_independent_distributions.png")
    plt.close()

# Test 2: Gaussian copula prior
print("\n" + "=" * 70)
print("\n--- Test 2: Gaussian Copula Prior ---")
options_gaussian = SpinePriorOptions(use_copulas=True, copula_type="gaussian")
prior_gaussian = SpinePrior(analyzer.stats, options=options_gaussian)
generator_gaussian = SpineGenerator(prior_gaussian)

generated_gaussian = []
for i in range(10):
    graph = generator_gaussian.generate(max_steps=500, max_nodes=100)
    generated_gaussian.append(graph)
    print(f"  Generated spine {i+1}: {graph.number_of_nodes()} nodes")

# Validate Gaussian copula prior
validator_gaussian = Validator(prior_gaussian, generated_gaussian)
print("\n" + validator_gaussian.summary())

# Save comparison plots
fig = validator_gaussian.plot_comparison(figsize=(16, 12))
if fig:
    plt.savefig("validation_gaussian_distributions.png", dpi=300)
    print("Saved: validation_gaussian_distributions.png")
    plt.close()

# Save copula comparison
fig = validator_gaussian.plot_copula_comparison(figsize=(15, 5))
if fig:
    plt.savefig("validation_gaussian_copulas.png", dpi=300)
    print("Saved: validation_gaussian_copulas.png")
    plt.close()

# Test 3: Multivariate Gaussian copula
print("\n" + "=" * 70)
print("\n--- Test 3: Multivariate Gaussian Copula ---")
options_multivariate = SpinePriorOptions(
    use_copulas=True, use_multivariate=True, multivariate_type="gaussian"
)
prior_multivariate = SpinePrior(analyzer.stats, options=options_multivariate)
generator_multivariate = SpineGenerator(prior_multivariate)

generated_multivariate = []
for i in range(10):
    graph = generator_multivariate.generate(max_steps=500, max_nodes=100)
    generated_multivariate.append(graph)
    print(f"  Generated spine {i+1}: {graph.number_of_nodes()} nodes")

# Validate multivariate copula prior
validator_multivariate = Validator(prior_multivariate, generated_multivariate)
print("\n" + validator_multivariate.summary())

# Save comparison plots
fig = validator_multivariate.plot_comparison(figsize=(16, 12))
if fig:
    plt.savefig("validation_multivariate_distributions.png", dpi=300)
    print("Saved: validation_multivariate_distributions.png")
    plt.close()

# Comparison summary
print("\n" + "=" * 70)
print("COMPARISON ACROSS PRIORS")
print("=" * 70)

# Get distribution validation results
results_independent = validator_independent.validate_distributions()
results_gaussian = validator_gaussian.validate_distributions()
results_multivariate = validator_multivariate.validate_distributions()

print("\nWasserstein Distance (lower is better):")
print("-" * 70)
for feature in ["thread_lengths", "curvatures", "branch_angles"]:
    if feature in results_independent:
        ind_w = results_independent[feature]["wasserstein"]
        gauss_w = results_gaussian.get(feature, {}).get("wasserstein", float("inf"))
        multi_w = results_multivariate.get(feature, {}).get("wasserstein", float("inf"))

        print(
            f"{feature:20s}: Independent={ind_w:.4f}, "
            f"Gaussian={gauss_w:.4f}, Multivariate={multi_w:.4f}"
        )

print("\nKS Test p-values (higher is better, >0.05 = good match):")
print("-" * 70)
for feature in ["thread_lengths", "curvatures", "branch_angles"]:
    if feature in results_independent:
        ind_p = results_independent[feature]["ks_pvalue"]
        gauss_p = results_gaussian.get(feature, {}).get("ks_pvalue", 0.0)
        multi_p = results_multivariate.get(feature, {}).get("ks_pvalue", 0.0)

        print(
            f"{feature:20s}: Independent={ind_p:.4f}, "
            f"Gaussian={gauss_p:.4f}, Multivariate={multi_p:.4f}"
        )

# Copula correlation preservation
print("\nCopula Correlation Preservation:")
print("-" * 70)
copula_results_gaussian = validator_gaussian.validate_copulas()
for copula_name, metrics in copula_results_gaussian.items():
    print(f"{copula_name}:")
    print(f"  Prior correlation: {metrics['prior_correlation']:.4f}")
    print(f"  Generated correlation: {metrics['gen_correlation']:.4f}")
    print(f"  Difference: {metrics['correlation_diff']:.4f}")

print("\n" + "=" * 70)
print("RECOMMENDATIONS")
print("=" * 70)
print("1. Check Wasserstein distances - lower values indicate better match")
print("2. Check KS test p-values - values > 0.05 indicate good distribution match")
print("3. For copula priors, check correlation preservation")
print("4. Visual inspection of plots is crucial for understanding differences")
print("=" * 70)
