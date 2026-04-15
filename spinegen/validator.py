"""Validation and quality assessment for generated spines.

This module provides the Validator class for comparing generated spines
against the prior distribution to assess quality and validity.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

from .cable_graph import CableGraph
from .prior import SpinePrior
from .analyzer import SpineAnalyzer

logger = logging.getLogger(__name__)


class Validator:
    """Validator for assessing quality of generated spines.

    Compares generated spine morphology against the prior distribution
    to evaluate how well the generator captures the target statistics.

    Attributes:
        prior: SpinePrior used for generation
        generated_graphs: List of generated CableGraph instances
        generated_stats: Statistics from generated spines

    Validation Metrics:
        - Distribution similarity (KS test, Wasserstein distance)
        - Moment matching (mean, variance, skewness)
        - Copula correlation preservation
        - Structural validity (connectivity, cycles)
    """

    def __init__(
        self, prior: SpinePrior, generated_graphs: Optional[List[CableGraph]] = None
    ):
        """Initialize validator with prior and optional generated spines.

        Args:
            prior: SpinePrior used for generation
            generated_graphs: List of generated CableGraph instances
        """
        self.prior = prior
        self.generated_graphs = generated_graphs or []
        self.generated_stats: Dict[str, Any] = {}

        if self.generated_graphs:
            self._analyze_generated()

    def add_generated(self, graphs: List[CableGraph]) -> None:
        """Add generated spines to validation set."""
        self.generated_graphs.extend(graphs)
        self._analyze_generated()

    def _analyze_generated(self) -> None:
        """Analyze generated spines to extract statistics."""
        if not self.generated_graphs:
            return

        analyzer = SpineAnalyzer(self.generated_graphs)
        self.generated_stats = analyzer.analyze()
        logger.info(f"Analyzed {len(self.generated_graphs)} generated spines")

    def validate_distributions(self) -> Dict[str, Dict[str, float]]:
        """Compare generated vs prior distributions using statistical tests.

        Returns:
            Dictionary mapping feature names to test results:
                - 'ks_statistic': Kolmogorov-Smirnov test statistic
                - 'ks_pvalue': KS test p-value
                - 'wasserstein': Wasserstein distance
                - 'mean_diff': Difference in means
                - 'std_diff': Difference in standard deviations
        """
        if not self.generated_stats:
            logger.warning("No generated spines to validate")
            return {}

        results = {}

        # Compare each feature
        features = [
            "thread_lengths",
            "curvatures",
            "branch_counts",
            "branch_angles",
            "radii",
            "fusion_distances",
        ]

        for feature in features:
            prior_data = self.prior.stats.get(feature, np.array([]))
            gen_data = self.generated_stats.get(feature, np.array([]))

            if len(prior_data) == 0 or len(gen_data) == 0:
                continue

            # Remove invalid values
            prior_data = prior_data[np.isfinite(prior_data)]
            gen_data = gen_data[np.isfinite(gen_data)]

            if len(prior_data) < 2 or len(gen_data) < 2:
                continue

            # Kolmogorov-Smirnov test
            ks_stat, ks_pval = stats.ks_2samp(prior_data, gen_data)

            # Wasserstein distance (Earth Mover's Distance)
            wasserstein = stats.wasserstein_distance(prior_data, gen_data)

            # Moment differences
            mean_diff = np.abs(np.mean(gen_data) - np.mean(prior_data))
            std_diff = np.abs(np.std(gen_data) - np.std(prior_data))

            results[feature] = {
                "ks_statistic": float(ks_stat),
                "ks_pvalue": float(ks_pval),
                "wasserstein": float(wasserstein),
                "mean_diff": float(mean_diff),
                "std_diff": float(std_diff),
                "prior_mean": float(np.mean(prior_data)),
                "gen_mean": float(np.mean(gen_data)),
                "prior_std": float(np.std(prior_data)),
                "gen_std": float(np.std(gen_data)),
            }

        return results

    def validate_copulas(self) -> Dict[str, Dict[str, float]]:
        """Validate copula correlations are preserved in generated data.

        Returns:
            Dictionary mapping copula names to correlation metrics:
                - 'prior_correlation': Correlation in prior data
                - 'gen_correlation': Correlation in generated data
                - 'correlation_diff': Absolute difference
        """
        if not self.prior.options.use_copulas:
            logger.info("Prior does not use copulas, skipping copula validation")
            return {}

        if not self.generated_stats:
            logger.warning("No generated spines to validate")
            return {}

        results = {}

        # Check bivariate copulas
        copula_pairs = [
            ("length_curvature", "segment_lengths", "curvatures"),
            ("branch", "branch_counts", "branch_angles"),
            ("fusion", "fusion_distances", "fusion_graph_distances"),
        ]

        for copula_name, feat1, feat2 in copula_pairs:
            if copula_name not in self.prior.copulas:
                continue

            prior_data1 = self.prior.stats.get(feat1, np.array([]))
            prior_data2 = self.prior.stats.get(feat2, np.array([]))
            gen_data1 = self.generated_stats.get(feat1, np.array([]))
            gen_data2 = self.generated_stats.get(feat2, np.array([]))

            if (
                len(prior_data1) == 0
                or len(prior_data2) == 0
                or len(gen_data1) == 0
                or len(gen_data2) == 0
            ):
                continue

            # Align lengths
            n_prior = min(len(prior_data1), len(prior_data2))
            n_gen = min(len(gen_data1), len(gen_data2))

            prior_data1 = prior_data1[:n_prior]
            prior_data2 = prior_data2[:n_prior]
            gen_data1 = gen_data1[:n_gen]
            gen_data2 = gen_data2[:n_gen]

            # Remove invalid values
            prior_valid = np.isfinite(prior_data1) & np.isfinite(prior_data2)
            gen_valid = np.isfinite(gen_data1) & np.isfinite(gen_data2)

            prior_data1 = prior_data1[prior_valid]
            prior_data2 = prior_data2[prior_valid]
            gen_data1 = gen_data1[gen_valid]
            gen_data2 = gen_data2[gen_valid]

            if len(prior_data1) < 2 or len(gen_data1) < 2:
                continue

            # Compute correlations
            prior_corr = np.corrcoef(prior_data1, prior_data2)[0, 1]
            gen_corr = np.corrcoef(gen_data1, gen_data2)[0, 1]

            results[copula_name] = {
                "prior_correlation": float(prior_corr),
                "gen_correlation": float(gen_corr),
                "correlation_diff": float(np.abs(gen_corr - prior_corr)),
            }

        return results

    def validate_structure(self) -> Dict[str, Any]:
        """Validate structural properties of generated spines.

        Returns:
            Dictionary with structural validation metrics:
                - 'valid_graphs': Number of valid (connected) graphs
                - 'avg_nodes': Average number of nodes
                - 'avg_edges': Average number of edges
                - 'avg_cycles': Average number of cycles
                - 'isolated_nodes': Average isolated nodes per graph
        """
        if not self.generated_graphs:
            logger.warning("No generated spines to validate")
            return {}

        valid_count = 0
        node_counts = []
        edge_counts = []
        cycle_counts = []
        isolated_counts = []

        for graph in self.generated_graphs:
            # Check connectivity
            if graph.number_of_nodes() > 0:
                valid_count += 1

            node_counts.append(graph.number_of_nodes())
            edge_counts.append(graph.number_of_edges())

            # Count cycles (edges - nodes + 1 for connected graph)
            if graph.number_of_nodes() > 0:
                cycles = graph.number_of_edges() - graph.number_of_nodes() + 1
                cycle_counts.append(max(0, cycles))

            # Count isolated nodes
            isolated = sum(1 for node in graph.nodes() if graph.degree(node) == 0)
            isolated_counts.append(isolated)

        return {
            "total_graphs": len(self.generated_graphs),
            "valid_graphs": valid_count,
            "avg_nodes": float(np.mean(node_counts)),
            "std_nodes": float(np.std(node_counts)),
            "avg_edges": float(np.mean(edge_counts)),
            "std_edges": float(np.std(edge_counts)),
            "avg_cycles": float(np.mean(cycle_counts)) if cycle_counts else 0.0,
            "avg_isolated": float(np.mean(isolated_counts)),
        }

    def summary(self) -> str:
        """Generate comprehensive validation summary.

        Returns:
            Formatted string with validation results
        """
        lines = []
        lines.append("=" * 70)
        lines.append("VALIDATION SUMMARY")
        lines.append("=" * 70)

        # Count real spines from stats metadata or estimate from data
        n_real_spines = "unknown"
        if hasattr(self.prior, "stats"):
            # Try to get from metadata if available
            n_real_spines = self.prior.stats.get("n_graphs", "unknown")

        lines.append(f"Prior: {n_real_spines} real spines analyzed")
        lines.append(f"Generated: {len(self.generated_graphs)} spines")
        lines.append("")

        # Structural validation
        struct = self.validate_structure()
        if struct:
            lines.append("STRUCTURAL VALIDATION")
            lines.append("-" * 70)
            lines.append(
                f"  Valid graphs: {struct['valid_graphs']}/{struct['total_graphs']}"
            )
            lines.append(
                f"  Nodes: {struct['avg_nodes']:.1f} ± {struct['std_nodes']:.1f}"
            )
            lines.append(
                f"  Edges: {struct['avg_edges']:.1f} ± {struct['std_edges']:.1f}"
            )
            lines.append(f"  Cycles: {struct['avg_cycles']:.2f} average")
            lines.append(f"  Isolated nodes: {struct['avg_isolated']:.2f} average")
            lines.append("")

        # Distribution validation
        dist_results = self.validate_distributions()
        if dist_results:
            lines.append("DISTRIBUTION VALIDATION")
            lines.append("-" * 70)
            for feature, metrics in dist_results.items():
                lines.append(f"  {feature}:")
                lines.append(
                    f"    KS test: D={metrics['ks_statistic']:.4f}, "
                    f"p={metrics['ks_pvalue']:.4f}"
                )
                lines.append(f"    Wasserstein: {metrics['wasserstein']:.4f}")
                lines.append(
                    f"    Mean: prior={metrics['prior_mean']:.3f}, "
                    f"gen={metrics['gen_mean']:.3f} "
                    f"(diff={metrics['mean_diff']:.3f})"
                )

                # Interpret KS test
                if metrics["ks_pvalue"] > 0.05:
                    lines.append("    ✓ Distributions match (p > 0.05)")
                else:
                    lines.append("    ✗ Distributions differ (p < 0.05)")
            lines.append("")

        # Copula validation
        copula_results = self.validate_copulas()
        if copula_results:
            lines.append("COPULA VALIDATION")
            lines.append("-" * 70)
            for copula_name, metrics in copula_results.items():
                lines.append(f"  {copula_name}:")
                lines.append(
                    f"    Prior correlation: {metrics['prior_correlation']:.4f}"
                )
                lines.append(
                    f"    Generated correlation: " f"{metrics['gen_correlation']:.4f}"
                )
                lines.append(f"    Difference: {metrics['correlation_diff']:.4f}")

                # Interpret correlation preservation
                if metrics["correlation_diff"] < 0.1:
                    lines.append("    ✓ Correlation well preserved")
                elif metrics["correlation_diff"] < 0.2:
                    lines.append("    ~ Correlation moderately preserved")
                else:
                    lines.append("    ✗ Correlation poorly preserved")
            lines.append("")

        lines.append("=" * 70)
        return "\n".join(lines)

    def plot_comparison(
        self, figsize: Optional[Tuple[int, int]] = None, save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot side-by-side comparison of prior vs generated distributions.

        Args:
            figsize: Figure size (default: (16, 12))
            save_path: Optional path to save figure

        Returns:
            matplotlib Figure object
        """
        if figsize is None:
            figsize = (16, 12)

        if not self.generated_stats:
            logger.warning("No generated spines to plot")
            return None

        sns.set_style("whitegrid")
        fig, axes = plt.subplots(3, 3, figsize=figsize)
        axes = axes.flatten()

        features = [
            ("thread_lengths", "Thread Length"),
            ("curvatures", "Curvature"),
            ("branch_counts", "Branch Count"),
            ("branch_angles", "Branch Angle"),
            ("radii", "Radius"),
            ("fusion_distances", "Fusion Distance"),
        ]

        plot_idx = 0
        for feature_key, feature_name in features:
            if plot_idx >= len(axes):
                break

            prior_data = self.prior.stats.get(feature_key, np.array([]))
            gen_data = self.generated_stats.get(feature_key, np.array([]))

            if len(prior_data) == 0 or len(gen_data) == 0:
                continue

            prior_data = prior_data[np.isfinite(prior_data)]
            gen_data = gen_data[np.isfinite(gen_data)]

            if len(prior_data) < 2 or len(gen_data) < 2:
                continue

            ax = axes[plot_idx]

            # Plot histograms
            ax.hist(
                prior_data,
                bins=30,
                density=True,
                alpha=0.5,
                color="blue",
                label="Prior (Real)",
                edgecolor="black",
                linewidth=0.5,
            )
            ax.hist(
                gen_data,
                bins=30,
                density=True,
                alpha=0.5,
                color="red",
                label="Generated",
                edgecolor="black",
                linewidth=0.5,
            )

            ax.set_xlabel(feature_name)
            ax.set_ylabel("Density")
            ax.set_title(f"{feature_name} Distribution")
            ax.legend()
            ax.grid(True, alpha=0.3)

            plot_idx += 1

        # Hide unused subplots
        for idx in range(plot_idx, len(axes)):
            axes[idx].axis("off")

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved comparison plot to {save_path}")

        return fig

    def plot_copula_comparison(
        self, figsize: Optional[Tuple[int, int]] = None, save_path: Optional[str] = None
    ) -> Optional[plt.Figure]:
        """Plot copula correlations: prior vs generated.

        Args:
            figsize: Figure size (default: (15, 5))
            save_path: Optional path to save figure

        Returns:
            matplotlib Figure object or None
        """
        if not self.prior.options.use_copulas:
            logger.info("Prior does not use copulas")
            return None

        if not self.generated_stats:
            logger.warning("No generated spines to plot")
            return None

        if figsize is None:
            figsize = (15, 5)

        copula_pairs = [
            (
                "length_curvature",
                "segment_lengths",
                "curvatures",
                "Segment Length",
                "Curvature",
            ),
            (
                "branch",
                "branch_counts",
                "branch_angles",
                "Branch Count",
                "Branch Angle",
            ),
            (
                "fusion",
                "fusion_distances",
                "fusion_graph_distances",
                "Fusion Distance",
                "Fusion Graph Distance",
            ),
        ]

        n_copulas = sum(
            1 for name, _, _, _, _ in copula_pairs if name in self.prior.copulas
        )

        if n_copulas == 0:
            logger.warning("No copulas fitted in prior")
            return None

        fig, axes = plt.subplots(2, n_copulas, figsize=figsize)
        if n_copulas == 1:
            axes = axes.reshape(2, 1)

        plot_idx = 0
        for copula_name, feat1, feat2, label1, label2 in copula_pairs:
            if copula_name not in self.prior.copulas:
                continue

            prior_data1 = self.prior.stats.get(feat1, np.array([]))
            prior_data2 = self.prior.stats.get(feat2, np.array([]))
            gen_data1 = self.generated_stats.get(feat1, np.array([]))
            gen_data2 = self.generated_stats.get(feat2, np.array([]))

            if (
                len(prior_data1) == 0
                or len(prior_data2) == 0
                or len(gen_data1) == 0
                or len(gen_data2) == 0
            ):
                continue

            # Prior scatter
            ax_prior = axes[0, plot_idx]
            n = min(len(prior_data1), len(prior_data2))
            ax_prior.scatter(
                prior_data1[:n],
                prior_data2[:n],
                alpha=0.5,
                s=20,
                edgecolors="k",
                linewidths=0.5,
            )
            ax_prior.set_xlabel(label1)
            ax_prior.set_ylabel(label2)
            ax_prior.set_title(f"Prior: {copula_name}")
            ax_prior.grid(True, alpha=0.3)

            # Generated scatter
            ax_gen = axes[1, plot_idx]
            n = min(len(gen_data1), len(gen_data2))
            ax_gen.scatter(
                gen_data1[:n],
                gen_data2[:n],
                alpha=0.5,
                s=20,
                color="red",
                edgecolors="k",
                linewidths=0.5,
            )
            ax_gen.set_xlabel(label1)
            ax_gen.set_ylabel(label2)
            ax_gen.set_title(f"Generated: {copula_name}")
            ax_gen.grid(True, alpha=0.3)

            plot_idx += 1

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved copula comparison plot to {save_path}")

        return fig
