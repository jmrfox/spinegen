"""Statistical prior distributions for spine morphology generation.

This module provides the SpinePrior class which fits parametric distributions
to morphological statistics and provides sampling methods for generation.
"""

import logging
import numpy as np
from typing import Dict, Any, Optional
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class SpinePrior:
    """Prior distribution for spine morphological parameters.

    This class takes statistics from SpineAnalyzer and fits parametric
    probability distributions that can be sampled during generation.

    Attributes:
        stats: Dictionary of morphological statistics from analyzer
        distributions: Dictionary of fitted distribution parameters

    Distributions Fitted:
        - segment_length: Gamma distribution
        - curvature: Normal distribution (clipped to [0, π])
        - branch_count: Categorical distribution
        - branch_angle: Normal distribution (clipped to [0, π])
        - radius: LogNormal distribution with tapering
        - fusion_distance: Exponential distribution
        - fusion_graph_distance: Empirical (mean, std)
    """

    def __init__(self, analyzer_stats: Dict[str, Any]):
        """Initialize prior and fit distributions to statistics.

        Args:
            analyzer_stats: Dictionary of statistics from SpineAnalyzer.analyze()
        """
        self.stats = analyzer_stats
        self.distributions = {}
        logger.info("Initializing SpinePrior and fitting distributions")
        self._fit_distributions()
        logger.info(f"Fitted {len(self.distributions)} distributions")

    def _fit_distributions(self) -> None:
        """Fit parametric distributions to all available statistics.

        This method fits appropriate distributions to each statistic type:
        - Gamma for positive continuous values (lengths)
        - Normal for angular measurements
        - Categorical for discrete counts
        - LogNormal for radii
        - Exponential for fusion distances
        """
        logger.debug("Fitting distributions to statistics")

        if len(self.stats["segment_lengths"]) > 0:
            lengths = self.stats["segment_lengths"]
            lengths = lengths[lengths > 0]
            if len(lengths) > 0:
                shape, loc, scale = stats.gamma.fit(lengths)
                self.distributions["segment_length"] = {
                    "type": "gamma",
                    "params": {"shape": shape, "loc": loc, "scale": scale},
                }
                logger.debug(
                    f"Fitted Gamma to segment_length: shape={shape:.3f}, scale={scale:.3f}"
                )

        if len(self.stats["curvatures"]) > 0:
            curvatures = self.stats["curvatures"]
            mu = np.mean(curvatures)
            sigma = np.std(curvatures)
            self.distributions["curvature"] = {
                "type": "normal",
                "params": {"mu": mu, "sigma": sigma},
            }
            logger.debug(f"Fitted Normal to curvature: μ={mu:.3f}, σ={sigma:.3f}")

        if len(self.stats["branch_counts"]) > 0:
            counts, freq = np.unique(self.stats["branch_counts"], return_counts=True)
            probs = freq / freq.sum()
            self.distributions["branch_count"] = {
                "type": "categorical",
                "params": {"values": counts.tolist(), "probs": probs.tolist()},
            }
            logger.debug(
                f"Fitted Categorical to branch_count: {dict(zip(counts, probs))}"
            )

        if len(self.stats["branch_angles"]) > 0:
            angles = self.stats["branch_angles"]
            mu = np.mean(angles)
            sigma = np.std(angles)
            self.distributions["branch_angle"] = {
                "type": "normal",
                "params": {"mu": mu, "sigma": sigma},
            }
            logger.debug(f"Fitted Normal to branch_angle: μ={mu:.3f}, σ={sigma:.3f}")

        if len(self.stats["radii"]) > 0:
            radii = self.stats["radii"]
            radii = radii[radii > 0]
            if len(radii) > 0:
                mu = np.mean(np.log(radii))
                sigma = np.std(np.log(radii))
                self.distributions["radius"] = {
                    "type": "lognormal",
                    "params": {"mu": mu, "sigma": sigma},
                }
                logger.debug(f"Fitted LogNormal to radius: μ={mu:.3f}, σ={sigma:.3f}")

        if len(self.stats["fusion_distances"]) > 0:
            distances = self.stats["fusion_distances"]
            distances = distances[distances > 0]
            if len(distances) > 0:
                loc, scale = stats.expon.fit(distances)
                self.distributions["fusion_distance"] = {
                    "type": "exponential",
                    "params": {"loc": loc, "scale": scale},
                }
                logger.debug(
                    f"Fitted Exponential to fusion_distance: scale={scale:.3f}"
                )

        if len(self.stats["fusion_graph_distances"]) > 0:
            graph_dists = self.stats["fusion_graph_distances"]
            graph_dists = graph_dists[np.isfinite(graph_dists)]
            if len(graph_dists) > 0:
                mean_dist = np.mean(graph_dists)
                std_dist = np.std(graph_dists)
                self.distributions["fusion_graph_distance"] = {
                    "type": "empirical",
                    "params": {"mean": mean_dist, "std": std_dist},
                }
                logger.debug(
                    f"Fitted Empirical to fusion_graph_distance: mean={mean_dist:.3f}"
                )

    def sample_segment_length(self) -> float:
        """Sample a segment length from the fitted distribution.

        Returns:
            Segment length value (default 10.0 if no distribution fitted)
        """
        if "segment_length" not in self.distributions:
            logger.debug("No segment_length distribution, using default 10.0")
            return 10.0

        dist = self.distributions["segment_length"]
        if dist["type"] == "gamma":
            params = dist["params"]
            value = stats.gamma.rvs(
                params["shape"], loc=params["loc"], scale=params["scale"]
            )
            logger.debug(f"Sampled segment_length: {value:.3f}")
            return value
        return 10.0

    def sample_curvature(self) -> float:
        """Sample a curvature angle from the fitted distribution.

        Returns:
            Curvature angle in radians, clipped to [0, π] (default 0.1)
        """
        if "curvature" not in self.distributions:
            return 0.1

        dist = self.distributions["curvature"]
        if dist["type"] == "normal":
            params = dist["params"]
            angle = np.random.normal(params["mu"], params["sigma"])
            return np.clip(angle, 0, np.pi)
        return 0.1

    def sample_branch_count(self) -> int:
        """Sample number of branches from the fitted distribution.

        Returns:
            Number of branches (default 2 if no distribution fitted)
        """
        if "branch_count" not in self.distributions:
            return 2

        dist = self.distributions["branch_count"]
        if dist["type"] == "categorical":
            params = dist["params"]
            count = int(np.random.choice(params["values"], p=params["probs"]))
            logger.debug(f"Sampled branch_count: {count}")
            return count
        return 2

    def sample_branch_angle(self) -> float:
        """Sample a branch angle from the fitted distribution.

        Returns:
            Branch angle in radians, clipped to [0, π] (default π/3)
        """
        if "branch_angle" not in self.distributions:
            return np.pi / 3

        dist = self.distributions["branch_angle"]
        if dist["type"] == "normal":
            params = dist["params"]
            angle = np.random.normal(params["mu"], params["sigma"])
            return np.clip(angle, 0, np.pi)
        return np.pi / 3

    def sample_radius(self, distance_from_root: float = 0.0) -> float:
        """Sample a radius with distance-based tapering.

        Args:
            distance_from_root: Path distance from root node (for tapering)

        Returns:
            Radius value with exponential tapering applied (default 1.0)

        Note:
            Tapering factor is exp(-distance/1000), causing radii to decrease
            with distance from the root.
        """
        if "radius" not in self.distributions:
            return 1.0

        dist = self.distributions["radius"]
        if dist["type"] == "lognormal":
            params = dist["params"]
            base_radius = np.random.lognormal(params["mu"], params["sigma"])
            taper_factor = np.exp(-distance_from_root / 1000.0)
            final_radius = base_radius * taper_factor
            logger.debug(
                f"Sampled radius: {final_radius:.3f} (base={base_radius:.3f}, taper={taper_factor:.3f})"
            )
            return final_radius
        return 1.0

    def sample_fusion_distance(self) -> float:
        """Sample a fusion search distance from the fitted distribution.

        Returns:
            Fusion search radius (default 50.0 if no distribution fitted)
        """
        if "fusion_distance" not in self.distributions:
            return 50.0

        dist = self.distributions["fusion_distance"]
        if dist["type"] == "exponential":
            params = dist["params"]
            return stats.expon.rvs(loc=params["loc"], scale=params["scale"])
        return 50.0

    def compute_fusion_probability(
        self, spatial_dist: float, graph_dist: float
    ) -> float:
        """Compute probability of fusion based on spatial and graph distances.

        The fusion probability combines two factors:
        1. Spatial proximity: Gaussian decay with distance
        2. Graph distance: Preference for longer graph paths (more likely to
           create interesting cycles)

        Args:
            spatial_dist: Euclidean distance between fusion candidates
            graph_dist: Graph path length (with fusion edge removed)

        Returns:
            Fusion probability in [0, 1]

        Note:
            Returns 0 if graph_dist is infinite (no path exists).
        """
        if "fusion_distance" not in self.distributions:
            sigma = 50.0
        else:
            dist = self.distributions["fusion_distance"]
            if dist["type"] == "exponential":
                sigma = dist["params"]["scale"]
            else:
                sigma = 50.0

        if "fusion_graph_distance" not in self.distributions:
            lambda_param = 100.0
        else:
            dist = self.distributions["fusion_graph_distance"]
            lambda_param = dist["params"].get("mean", 100.0)

        spatial_prob = np.exp(-(spatial_dist**2) / (2 * sigma**2))
        graph_prob = (
            1.0 - np.exp(-graph_dist / lambda_param)
            if graph_dist < float("inf")
            else 0.0
        )

        fusion_prob = spatial_prob * graph_prob
        logger.debug(
            f"Fusion probability: {fusion_prob:.4f} (spatial={spatial_prob:.4f}, graph={graph_prob:.4f})"
        )
        return fusion_prob

    def __str__(self) -> str:
        """Return string representation of prior distributions.

        Returns:
            Formatted string with all distribution information
        """
        return self.print()

    def print(self) -> str:
        """Generate comprehensive information about all fitted distributions.

        Returns:
            Formatted string with distribution types, parameters, and summary
            statistics for all morphological features
        """
        lines = []
        lines.append("=" * 70)
        lines.append("SpinePrior Distribution Information")
        lines.append("=" * 70)
        lines.append("")

        # Segment Length
        if "segment_length" in self.distributions:
            dist = self.distributions["segment_length"]
            lines.append("SEGMENT LENGTH")
            lines.append(f"  Distribution: {dist['type'].capitalize()}")
            if dist["type"] == "gamma":
                p = dist["params"]
                mean = p["shape"] * p["scale"] + p["loc"]
                lines.append(
                    f"  Parameters: shape={p['shape']:.3f}, loc={p['loc']:.3f}, scale={p['scale']:.3f}"
                )
                lines.append(f"  Mean: {mean:.3f}")
        else:
            lines.append("SEGMENT LENGTH: Not fitted (default: 10.0)")
        lines.append("")

        # Curvature
        if "curvature" in self.distributions:
            dist = self.distributions["curvature"]
            lines.append("CURVATURE")
            lines.append(f"  Distribution: {dist['type'].capitalize()}")
            if dist["type"] == "normal":
                p = dist["params"]
                lines.append(f"  Parameters: μ={p['mu']:.3f}, σ={p['sigma']:.3f}")
                lines.append("  Range: [0, π] (clipped)")
        else:
            lines.append("CURVATURE: Not fitted (default: 0.1)")
        lines.append("")

        # Branch Count
        if "branch_count" in self.distributions:
            dist = self.distributions["branch_count"]
            lines.append("BRANCH COUNT")
            lines.append(f"  Distribution: {dist['type'].capitalize()}")
            if dist["type"] == "categorical":
                p = dist["params"]
                lines.append(f"  Values: {p['values']}")
                lines.append(
                    f"  Probabilities: {[f'{prob:.3f}' for prob in p['probs']]}"
                )
        else:
            lines.append("BRANCH COUNT: Not fitted (default: 2)")
        lines.append("")

        # Branch Angle
        if "branch_angle" in self.distributions:
            dist = self.distributions["branch_angle"]
            lines.append("BRANCH ANGLE")
            lines.append(f"  Distribution: {dist['type'].capitalize()}")
            if dist["type"] == "normal":
                p = dist["params"]
                lines.append(f"  Parameters: μ={p['mu']:.3f}, σ={p['sigma']:.3f}")
                lines.append("  Range: [0, π] (clipped)")
        else:
            lines.append("BRANCH ANGLE: Not fitted (default: π/4)")
        lines.append("")

        # Radius
        if "radius" in self.distributions:
            dist = self.distributions["radius"]
            lines.append("RADIUS")
            lines.append(f"  Distribution: {dist['type'].capitalize()}")
            if dist["type"] == "lognormal":
                p = dist["params"]
                mean = np.exp(p["mu"] + p["sigma"] ** 2 / 2)
                lines.append(f"  Parameters: μ={p['mu']:.3f}, σ={p['sigma']:.3f}")
                lines.append(f"  Mean: {mean:.3f}")
                lines.append("  Note: Tapers with distance from root")
        else:
            lines.append("RADIUS: Not fitted (default: 1.0)")
        lines.append("")

        # Fusion Distance
        if "fusion_distance" in self.distributions:
            dist = self.distributions["fusion_distance"]
            lines.append("FUSION DISTANCE (Spatial)")
            lines.append(f"  Distribution: {dist['type'].capitalize()}")
            if dist["type"] == "exponential":
                p = dist["params"]
                lines.append(
                    f"  Parameters: loc={p['loc']:.3f}, scale={p['scale']:.3f}"
                )
                lines.append(f"  Mean: {p['scale']:.3f}")
        else:
            lines.append("FUSION DISTANCE: Not fitted (default scale: 50.0)")
        lines.append("")

        # Fusion Graph Distance
        if "fusion_graph_distance" in self.distributions:
            dist = self.distributions["fusion_graph_distance"]
            lines.append("FUSION GRAPH DISTANCE")
            lines.append(f"  Distribution: {dist['type'].capitalize()}")
            if dist["type"] == "empirical":
                p = dist["params"]
                lines.append(f"  Mean: {p['mean']:.3f}")
                lines.append(f"  Std: {p['std']:.3f}")
        else:
            lines.append("FUSION GRAPH DISTANCE: Not fitted (default mean: 100.0)")
        lines.append("")

        # Summary Statistics
        lines.append("=" * 70)
        lines.append("Summary Statistics from Analyzed Data")
        lines.append("=" * 70)
        lines.append(
            f"  Segment lengths: {len(self.stats.get('segment_lengths', []))} samples"
        )
        lines.append(f"  Curvatures: {len(self.stats.get('curvatures', []))} samples")
        lines.append(
            f"  Branch counts: {len(self.stats.get('branch_counts', []))} samples"
        )
        lines.append(
            f"  Branch angles: {len(self.stats.get('branch_angles', []))} samples"
        )
        lines.append(f"  Radii: {len(self.stats.get('radii', []))} samples")
        lines.append(
            f"  Fusion distances: {len(self.stats.get('fusion_distances', []))} samples"
        )
        lines.append(
            f"  Fusion graph distances: {len(self.stats.get('fusion_graph_distances', []))} samples"
        )
        lines.append("=" * 70)

        return "\n".join(lines)

    def plot_distributions(
        self, figsize: Optional[tuple] = None, save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot all fitted probability distributions.

        Creates a multi-panel figure showing histograms of the original data
        overlaid with the fitted probability density functions for each
        morphological feature.

        Args:
            figsize: Figure size as (width, height). Default: (16, 12)
            save_path: Optional path to save the figure. If None, displays only.

        Returns:
            matplotlib Figure object

        Examples:
            >>> prior.plot_distributions()
            >>> prior.plot_distributions(figsize=(20, 15), save_path="prior.png")
        """
        if figsize is None:
            figsize = (16, 12)

        sns.set_style("whitegrid")
        fig, axes = plt.subplots(3, 3, figsize=figsize)
        axes = axes.flatten()

        plot_idx = 0

        # 1. Segment Length
        if "segment_length" in self.distributions:
            ax = axes[plot_idx]
            data = self.stats["segment_lengths"]
            data = data[data > 0]

            if len(data) > 0:
                ax.hist(
                    data,
                    bins=30,
                    density=True,
                    alpha=0.6,
                    color="skyblue",
                    edgecolor="black",
                    label="Data",
                )

                dist = self.distributions["segment_length"]
                if dist["type"] == "gamma":
                    p = dist["params"]
                    x = np.linspace(data.min(), data.max(), 200)
                    y = stats.gamma.pdf(x, p["shape"], loc=p["loc"], scale=p["scale"])
                    ax.plot(x, y, "r-", linewidth=2, label="Fitted Gamma")

                ax.set_xlabel("Segment Length")
                ax.set_ylabel("Density")
                ax.set_title("Segment Length Distribution")
                ax.legend()
            plot_idx += 1

        # 2. Curvature
        if "curvature" in self.distributions:
            ax = axes[plot_idx]
            data = self.stats["curvatures"]

            if len(data) > 0:
                ax.hist(
                    data,
                    bins=30,
                    density=True,
                    alpha=0.6,
                    color="lightgreen",
                    edgecolor="black",
                    label="Data",
                )

                dist = self.distributions["curvature"]
                if dist["type"] == "normal":
                    p = dist["params"]
                    x = np.linspace(0, np.pi, 200)
                    y = stats.norm.pdf(x, p["mu"], p["sigma"])
                    ax.plot(x, y, "r-", linewidth=2, label="Fitted Normal")

                ax.set_xlabel("Curvature (radians)")
                ax.set_ylabel("Density")
                ax.set_title("Curvature Distribution")
                ax.legend()
            plot_idx += 1

        # 3. Branch Count
        if "branch_count" in self.distributions:
            ax = axes[plot_idx]
            data = self.stats["branch_counts"]

            if len(data) > 0:
                unique, counts = np.unique(data, return_counts=True)
                probs = counts / counts.sum()

                ax.bar(
                    unique,
                    probs,
                    alpha=0.6,
                    color="coral",
                    edgecolor="black",
                    label="Data",
                )

                dist = self.distributions["branch_count"]
                if dist["type"] == "categorical":
                    p = dist["params"]
                    ax.plot(
                        p["values"],
                        p["probs"],
                        "ro-",
                        linewidth=2,
                        markersize=8,
                        label="Fitted Categorical",
                    )

                ax.set_xlabel("Branch Count")
                ax.set_ylabel("Probability")
                ax.set_title("Branch Count Distribution")
                ax.legend()
            plot_idx += 1

        # 4. Branch Angle
        if "branch_angle" in self.distributions:
            ax = axes[plot_idx]
            data = self.stats["branch_angles"]

            if len(data) > 0:
                ax.hist(
                    data,
                    bins=30,
                    density=True,
                    alpha=0.6,
                    color="plum",
                    edgecolor="black",
                    label="Data",
                )

                dist = self.distributions["branch_angle"]
                if dist["type"] == "normal":
                    p = dist["params"]
                    x = np.linspace(0, np.pi, 200)
                    y = stats.norm.pdf(x, p["mu"], p["sigma"])
                    ax.plot(x, y, "r-", linewidth=2, label="Fitted Normal")

                ax.set_xlabel("Branch Angle (radians)")
                ax.set_ylabel("Density")
                ax.set_title("Branch Angle Distribution")
                ax.legend()
            plot_idx += 1

        # 5. Radius
        if "radius" in self.distributions:
            ax = axes[plot_idx]
            data = self.stats["radii"]
            data = data[data > 0]

            if len(data) > 0:
                ax.hist(
                    data,
                    bins=30,
                    density=True,
                    alpha=0.6,
                    color="gold",
                    edgecolor="black",
                    label="Data",
                )

                dist = self.distributions["radius"]
                if dist["type"] == "lognormal":
                    p = dist["params"]
                    x = np.linspace(data.min(), data.max(), 200)
                    y = stats.lognorm.pdf(x, p["sigma"], scale=np.exp(p["mu"]))
                    ax.plot(x, y, "r-", linewidth=2, label="Fitted LogNormal")

                ax.set_xlabel("Radius")
                ax.set_ylabel("Density")
                ax.set_title("Radius Distribution")
                ax.legend()
            plot_idx += 1

        # 6. Fusion Distance
        if "fusion_distance" in self.distributions:
            ax = axes[plot_idx]
            data = self.stats["fusion_distances"]
            data = data[data > 0]

            if len(data) > 0:
                ax.hist(
                    data,
                    bins=30,
                    density=True,
                    alpha=0.6,
                    color="lightcoral",
                    edgecolor="black",
                    label="Data",
                )

                dist = self.distributions["fusion_distance"]
                if dist["type"] == "exponential":
                    p = dist["params"]
                    x = np.linspace(data.min(), data.max(), 200)
                    y = stats.expon.pdf(x, loc=p["loc"], scale=p["scale"])
                    ax.plot(x, y, "r-", linewidth=2, label="Fitted Exponential")

                ax.set_xlabel("Fusion Distance (Spatial)")
                ax.set_ylabel("Density")
                ax.set_title("Fusion Distance Distribution")
                ax.legend()
            plot_idx += 1

        # 7. Fusion Graph Distance
        if "fusion_graph_distance" in self.distributions:
            ax = axes[plot_idx]
            data = self.stats["fusion_graph_distances"]
            data = data[np.isfinite(data)]

            if len(data) > 0:
                ax.hist(
                    data,
                    bins=30,
                    density=True,
                    alpha=0.6,
                    color="lightskyblue",
                    edgecolor="black",
                    label="Data",
                )

                dist = self.distributions["fusion_graph_distance"]
                if dist["type"] == "empirical":
                    p = dist["params"]
                    x = np.linspace(data.min(), data.max(), 200)
                    y = stats.norm.pdf(x, p["mean"], p["std"])
                    ax.plot(x, y, "r-", linewidth=2, label="Empirical (Normal)")

                ax.set_xlabel("Fusion Graph Distance")
                ax.set_ylabel("Density")
                ax.set_title("Fusion Graph Distance Distribution")
                ax.legend()
            plot_idx += 1

        # Hide unused subplots
        for idx in range(plot_idx, len(axes)):
            axes[idx].axis("off")

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved distribution plot to {save_path}")

        return fig
