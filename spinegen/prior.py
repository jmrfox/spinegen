"""Statistical prior distributions for spine morphology generation.

This module provides the SpinePrior class which fits parametric distributions
to morphological statistics and provides sampling methods for generation.
Supports optional copula-based joint distributions for correlated features.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List, Literal
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from copulas.bivariate import Clayton, Frank, Gumbel
from copulas.multivariate import GaussianMultivariate, VineCopula
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Map copula type names to classes
COPULA_CLASSES = {
    "gaussian": GaussianMultivariate,  # Use Gaussian for bivariate too
    "clayton": Clayton,
    "frank": Frank,
    "gumbel": Gumbel,
}

"""Configuration options for SpinePrior.

This module provides the SpinePriorOptions dataclass for configuring
copula-based joint distributions in the SpinePrior class.
"""


CopulaType = Literal["gaussian", "clayton", "gumbel", "frank"]
MultivariateCopulaType = Literal["gaussian", "vine"]


@dataclass
class SpinePriorOptions:
    """Configuration options for SpinePrior copula modeling.

    Attributes:
        use_copulas: If True, fit copulas for joint distributions
        copula_type: Type of copula for bivariate distributions
        use_multivariate: If True, fit single multivariate copula
        multivariate_type: Type of multivariate copula

    Examples:
        # Independent distributions (no copulas)
        options = SpinePriorOptions()

        # Gaussian copulas for pairwise correlations
        options = SpinePriorOptions(use_copulas=True)

        # Clayton copulas (lower tail dependence)
        options = SpinePriorOptions(use_copulas=True, copula_type="clayton")

        # Multivariate Gaussian copula (all features jointly)
        options = SpinePriorOptions(
            use_copulas=True,
            use_multivariate=True,
            multivariate_type="gaussian"
        )

        # Vine copula (flexible multivariate)
        options = SpinePriorOptions(
            use_copulas=True,
            use_multivariate=True,
            multivariate_type="vine"
        )
    """

    use_copulas: bool = False
    copula_type: CopulaType = "gaussian"
    use_multivariate: bool = False
    multivariate_type: MultivariateCopulaType = "gaussian"

    def __post_init__(self):
        """Validate configuration."""
        if self.use_multivariate and not self.use_copulas:
            raise ValueError("use_multivariate=True requires use_copulas=True")


class SpinePrior:
    """Prior distribution for spine morphological parameters.

    This class takes statistics from SpineAnalyzer and fits parametric
    probability distributions that can be sampled during generation.
    Optionally supports copula-based joint distributions for modeling
    correlations between features.

    Attributes:
        stats: Dictionary of morphological statistics from analyzer
        distributions: Dictionary of fitted marginal distribution parameters
        options: Configuration options for copula modeling
        copulas: Dictionary of fitted copula models (if enabled)

    Distributions Fitted:
        - thread_length: Gamma distribution
        - curvature: Normal distribution (clipped to [0, π])
        - branch_count: Categorical distribution
        - branch_angle: Normal distribution (clipped to [0, π])
        - radius: LogNormal distribution with tapering
        - fusion_distance: Exponential distribution
        - fusion_graph_distance: Empirical (mean, std)

    Joint Distributions (if copulas enabled):
        - (thread_length, curvature): Longer threads tend to be straighter
        - (branch_count, branch_angle): Branch count may correlate with angles
        - (fusion_distance, fusion_graph_distance): Spatial-topological correlation
        - Multivariate: All features jointly (optional)
    """

    def __init__(
        self,
        analyzer_stats: Dict[str, Any],
        options: Optional[SpinePriorOptions] = None,
    ):
        """Initialize prior and fit distributions to statistics.

        Args:
            analyzer_stats: Dictionary of statistics from SpineAnalyzer.analyze()
            options: Configuration options for copula modeling. If None, uses
                    default (no copulas, independent distributions)
        """
        self.stats = analyzer_stats
        self.options = options if options is not None else SpinePriorOptions()
        self.distributions: Dict[str, Any] = {}
        self.copulas: Dict[str, Any] = {}

        logger.info("Initializing SpinePrior and fitting distributions")
        self._fit_distributions()
        logger.info(f"Fitted {len(self.distributions)} marginal distributions")

        if self.options.use_copulas:
            self._fit_copulas()
            logger.info(f"Fitted {len(self.copulas)} copula model(s)")

    def _fit_distributions(self) -> None:
        """Fit parametric distributions to all available statistics."""
        logger.debug("Fitting distributions to statistics")

        if len(self.stats["thread_lengths"]) > 0:
            lengths = self.stats["thread_lengths"]
            lengths = lengths[lengths > 0]
            if len(lengths) > 0:
                shape, loc, scale = stats.gamma.fit(lengths)
                self.distributions["thread_length"] = {
                    "type": "gamma",
                    "params": {"shape": shape, "loc": loc, "scale": scale},
                }
                logger.debug(
                    f"Fitted Gamma to thread_length: "
                    f"shape={shape:.3f}, scale={scale:.3f}"
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
                    f"Fitted Empirical to fusion_graph_distance: "
                    f"mean={mean_dist:.3f}"
                )

    def _fit_copulas(self) -> None:
        """Fit copulas for joint distributions."""
        if self.options.use_multivariate:
            logger.info(f"Fitting {self.options.multivariate_type} multivariate copula")
            self._fit_multivariate_copula()
        else:
            logger.info(f"Fitting {self.options.copula_type} bivariate copulas")
            self._fit_bivariate_copulas()

    def _fit_bivariate_copulas(self) -> None:
        """Fit bivariate copulas for pairwise feature correlations."""
        if len(self.stats["thread_lengths"]) > 0 and len(self.stats["curvatures"]) > 0:
            self._fit_bivariate_copula(
                "length_curvature",
                self.stats["thread_lengths"],
                self.stats["curvatures"],
                ["thread_length", "curvature"],
            )

        if (
            len(self.stats["branch_counts"]) > 0
            and len(self.stats["branch_angles"]) > 0
        ):
            self._fit_bivariate_copula(
                "branch",
                self.stats["branch_counts"],
                self.stats["branch_angles"],
                ["branch_count", "branch_angle"],
            )

        if (
            len(self.stats["fusion_distances"]) > 0
            and len(self.stats["fusion_graph_distances"]) > 0
        ):
            self._fit_bivariate_copula(
                "fusion",
                self.stats["fusion_distances"],
                self.stats["fusion_graph_distances"],
                ["fusion_distance", "fusion_graph_distance"],
            )

    def _fit_bivariate_copula(
        self, name: str, data1: np.ndarray, data2: np.ndarray, feature_names: List[str]
    ) -> None:
        """Fit bivariate copula to two features using copulas package."""
        n_samples = min(len(data1), len(data2))
        data1 = data1[:n_samples]
        data2 = data2[:n_samples]

        # Clean data: remove invalid values
        valid_mask = np.isfinite(data1) & np.isfinite(data2)
        if name == "length_curvature":
            valid_mask &= data1 > 0
        elif name == "fusion":
            valid_mask &= (data1 > 0) & (data2 > 0)

        data1 = data1[valid_mask]
        data2 = data2[valid_mask]

        # Check for sufficient data
        if len(data1) < 30:
            logger.warning(
                f"Insufficient data for {name} copula: {len(data1)} samples "
                f"(minimum 30 recommended)"
            )
            return

        # Remove outliers that can cause fitting issues (beyond 3 std devs)
        def remove_outliers(data):
            if len(data) < 10:
                return data
            mean, std = np.mean(data), np.std(data)
            if std > 0:
                mask = np.abs(data - mean) <= 3 * std
                return data[mask]
            return data

        # Create aligned dataset without outliers
        combined = np.column_stack([data1, data2])
        mask1 = np.abs(data1 - np.mean(data1)) <= 3 * np.std(data1)
        mask2 = np.abs(data2 - np.mean(data2)) <= 3 * np.std(data2)
        combined_mask = mask1 & mask2

        data1_clean = data1[combined_mask]
        data2_clean = data2[combined_mask]

        if len(data1_clean) < 20:
            logger.warning(
                f"Too few samples after outlier removal for {name}: "
                f"{len(data1_clean)} (need at least 20)"
            )
            return

        df = pd.DataFrame(
            {feature_names[0]: data1_clean, feature_names[1]: data2_clean}
        )

        try:
            # For Gaussian, use GaussianMultivariate even for bivariate
            if self.options.copula_type == "gaussian":
                copula = GaussianMultivariate()
            else:
                # Get the specific copula class
                copula_class = COPULA_CLASSES.get(self.options.copula_type)
                if copula_class is None:
                    logger.error(f"Unknown copula type: {self.options.copula_type}")
                    return
                copula = copula_class()

            copula.fit(df)

            self.copulas[name] = {
                "model": copula,
                "type": "bivariate",
                "copula_type": self.options.copula_type,
                "marginals": feature_names,
            }

            logger.info(f"Fitted {self.options.copula_type} copula for {name}")
        except Exception as e:
            logger.error(f"Failed to fit copula for {name}: {e}")

    def _fit_multivariate_copula(self) -> None:
        """Fit multivariate copula to all features simultaneously."""
        feature_data = {}
        feature_names = []

        if len(self.stats["thread_lengths"]) > 0:
            feature_data["thread_length"] = self.stats["thread_lengths"]
            feature_names.append("thread_length")

        if len(self.stats["curvatures"]) > 0:
            feature_data["curvature"] = self.stats["curvatures"]
            feature_names.append("curvature")

        if len(self.stats["branch_counts"]) > 0:
            feature_data["branch_count"] = self.stats["branch_counts"]
            feature_names.append("branch_count")

        if len(self.stats["branch_angles"]) > 0:
            feature_data["branch_angle"] = self.stats["branch_angles"]
            feature_names.append("branch_angle")

        if len(self.stats["radii"]) > 0:
            feature_data["radius"] = self.stats["radii"]
            feature_names.append("radius")

        if len(feature_names) < 2:
            logger.warning("Insufficient features for multivariate copula")
            return

        # Align all features to same length
        n_samples = min(len(data) for data in feature_data.values())
        df_data = {name: feature_data[name][:n_samples] for name in feature_names}
        df = pd.DataFrame(df_data)

        # Clean data: remove infinities and NaNs
        df = df.replace([np.inf, -np.inf], np.nan).dropna()

        if len(df) < 30:
            logger.warning(
                f"Insufficient valid data for multivariate copula: {len(df)} "
                f"samples (minimum 30 recommended)"
            )
            return

        # Remove outliers from each feature (beyond 3 std devs)
        for col in df.columns:
            mean, std = df[col].mean(), df[col].std()
            if std > 0:
                df = df[np.abs(df[col] - mean) <= 3 * std]

        if len(df) < 20:
            logger.warning(
                f"Too few samples after outlier removal: {len(df)} "
                f"(need at least 20)"
            )
            return

        try:
            if self.options.multivariate_type == "gaussian":
                copula = GaussianMultivariate()
            else:
                copula = VineCopula(vine_type="regular")

            copula.fit(df)

            self.copulas["multivariate"] = {
                "model": copula,
                "type": "multivariate",
                "copula_type": self.options.multivariate_type,
                "marginals": feature_names,
            }

            logger.info(
                f"Fitted {self.options.multivariate_type} multivariate copula "
                f"with {len(feature_names)} features"
            )
        except Exception as e:
            logger.error(f"Failed to fit multivariate copula: {e}")

    def sample_thread_length(self) -> float:
        """Sample a thread length from the fitted distribution."""
        if "thread_length" not in self.distributions:
            logger.debug("No thread_length distribution, using default 10.0")
            return 10.0

        dist = self.distributions["thread_length"]
        if dist["type"] == "gamma":
            params = dist["params"]
            value = stats.gamma.rvs(
                params["shape"], loc=params["loc"], scale=params["scale"]
            )
            logger.debug(f"Sampled thread_length: {value:.3f}")
            return value
        return 10.0

    def sample_curvature(self) -> float:
        """Sample a curvature angle from the fitted distribution."""
        if "curvature" not in self.distributions:
            return 0.1

        dist = self.distributions["curvature"]
        if dist["type"] == "normal":
            params = dist["params"]
            angle = np.random.normal(params["mu"], params["sigma"])
            return np.clip(angle, 0, np.pi)
        return 0.1

    def sample_branch_count(self) -> int:
        """Sample number of branches from the fitted distribution."""
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
        """Sample a branch angle from the fitted distribution."""
        if "branch_angle" not in self.distributions:
            return np.pi / 3

        dist = self.distributions["branch_angle"]
        if dist["type"] == "normal":
            params = dist["params"]
            angle = np.random.normal(params["mu"], params["sigma"])
            return np.clip(angle, 0, np.pi)
        return np.pi / 3

    def sample_radius(self, distance_from_root: float = 0.0) -> float:
        """Sample a radius with distance-based tapering."""
        if "radius" not in self.distributions:
            return 1.0

        dist = self.distributions["radius"]
        if dist["type"] == "lognormal":
            params = dist["params"]
            base_radius = np.random.lognormal(params["mu"], params["sigma"])
            taper_factor = np.exp(-distance_from_root / 1000.0)
            final_radius = base_radius * taper_factor
            logger.debug(
                f"Sampled radius: {final_radius:.3f} "
                f"(base={base_radius:.3f}, taper={taper_factor:.3f})"
            )
            return final_radius
        return 1.0

    def sample_fusion_distance(self) -> float:
        """Sample a fusion search distance from the fitted distribution."""
        if "fusion_distance" not in self.distributions:
            return 50.0

        dist = self.distributions["fusion_distance"]
        if dist["type"] == "exponential":
            params = dist["params"]
            return stats.expon.rvs(loc=params["loc"], scale=params["scale"])
        return 50.0

    def sample_length_curvature_joint(self) -> Tuple[float, float]:
        """Sample thread length and curvature jointly from copula."""
        if "length_curvature" not in self.copulas:
            return (self.sample_thread_length(), self.sample_curvature())

        copula_model = self.copulas["length_curvature"]["model"]
        sample = copula_model.sample(1)

        length = float(sample["thread_length"].iloc[0])
        curvature = float(sample["curvature"].iloc[0])

        logger.debug(f"Sampled joint (length={length:.2f}, curv={curvature:.3f})")
        return length, curvature

    def sample_branch_joint(self) -> Tuple[int, float]:
        """Sample branch count and angle jointly from copula."""
        if "branch" not in self.copulas:
            return (self.sample_branch_count(), self.sample_branch_angle())

        copula_model = self.copulas["branch"]["model"]
        sample = copula_model.sample(1)

        count = int(round(sample["branch_count"].iloc[0]))
        angle = float(sample["branch_angle"].iloc[0])

        logger.debug(f"Sampled joint (count={count}, angle={angle:.3f})")
        return count, angle

    def sample_fusion_distances_joint(self) -> Tuple[float, float]:
        """Sample fusion spatial and graph distances jointly."""
        if "fusion" not in self.copulas:
            return (
                self.sample_fusion_distance(),
                self._sample_fusion_graph_distance(),
            )

        copula_model = self.copulas["fusion"]["model"]
        sample = copula_model.sample(1)

        spatial = float(sample["fusion_distance"].iloc[0])
        graph = float(sample["fusion_graph_distance"].iloc[0])

        return spatial, graph

    def sample_multivariate(self) -> Dict[str, float]:
        """Sample all features jointly from multivariate copula."""
        if "multivariate" not in self.copulas:
            logger.warning("No multivariate copula fitted")
            return {}

        copula_model = self.copulas["multivariate"]["model"]
        feature_names = self.copulas["multivariate"]["marginals"]

        sample = copula_model.sample(1)

        result = {}
        for feature in feature_names:
            value = float(sample[feature].iloc[0])
            if feature == "branch_count":
                value = int(round(value))
            result[feature] = value

        logger.debug(f"Sampled multivariate: {result}")
        return result

    def _sample_fusion_graph_distance(self) -> float:
        """Sample fusion graph distance (fallback helper)."""
        if "fusion_graph_distance" in self.distributions:
            dist = self.distributions["fusion_graph_distance"]
            if dist["type"] == "empirical":
                p = dist["params"]
                return np.random.normal(p["mean"], p["std"])
        return 100.0

    def compute_fusion_probability(
        self, spatial_dist: float, graph_dist: float
    ) -> float:
        """Compute probability of fusion based on spatial and graph distances."""
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
            f"Fusion probability: {fusion_prob:.4f} "
            f"(spatial={spatial_prob:.4f}, graph={graph_prob:.4f})"
        )
        return fusion_prob

    def __str__(self) -> str:
        """Return string representation of prior distributions."""
        return self.print()

    def print(self) -> str:
        """Generate comprehensive information about all fitted distributions."""
        lines = []
        lines.append("=" * 70)
        lines.append("SpinePrior Distribution Information")
        lines.append("=" * 70)

        if self.options.use_copulas:
            lines.append(f"Copula Mode: {self.options.copula_type}")
            if self.options.use_multivariate:
                lines.append(f"Multivariate: {self.options.multivariate_type}")
        else:
            lines.append("Copula Mode: Disabled (independent distributions)")
        lines.append("")

        # Thread Length
        if "thread_length" in self.distributions:
            dist = self.distributions["thread_length"]
            lines.append("THREAD LENGTH")
            lines.append(f"  Distribution: {dist['type'].capitalize()}")
            if dist["type"] == "gamma":
                p = dist["params"]
                mean = p["shape"] * p["scale"] + p["loc"]
                lines.append(
                    f"  Parameters: shape={p['shape']:.3f}, "
                    f"loc={p['loc']:.3f}, scale={p['scale']:.3f}"
                )
                lines.append(f"  Mean: {mean:.3f}")
        else:
            lines.append("THREAD LENGTH: Not fitted (default: 10.0)")
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

        # Copula information
        if len(self.copulas) > 0:
            lines.append("=" * 70)
            lines.append("Copula Models")
            lines.append("=" * 70)

            for name, copula in self.copulas.items():
                copula_type = copula["copula_type"]
                marginals = " <-> ".join(copula["marginals"])
                lines.append(f"  {name}: {copula_type} ({marginals})")
            lines.append("")

        # Summary Statistics
        lines.append("=" * 70)
        lines.append("Summary Statistics from Analyzed Data")
        lines.append("=" * 70)
        lines.append(
            f"  Thread lengths: {len(self.stats.get('thread_lengths', []))} samples"
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
            f"  Fusion distances: "
            f"{len(self.stats.get('fusion_distances', []))} samples"
        )
        lines.append(
            f"  Fusion graph distances: "
            f"{len(self.stats.get('fusion_graph_distances', []))} samples"
        )
        lines.append("=" * 70)

        return "\n".join(lines)

    def plot_distributions(
        self, figsize: Optional[tuple] = None, save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot all fitted probability distributions."""
        if figsize is None:
            figsize = (16, 12)

        sns.set_style("whitegrid")
        fig, axes = plt.subplots(3, 3, figsize=figsize)
        axes = axes.flatten()

        plot_idx = 0

        # 1. Thread Length
        if "thread_length" in self.distributions:
            ax = axes[plot_idx]
            data = self.stats["thread_lengths"]
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

                dist = self.distributions["thread_length"]
                if dist["type"] == "gamma":
                    p = dist["params"]
                    x = np.linspace(data.min(), data.max(), 200)
                    y = stats.gamma.pdf(x, p["shape"], loc=p["loc"], scale=p["scale"])
                    ax.plot(x, y, "r-", linewidth=2, label="Fitted Gamma")

                ax.set_xlabel("Thread Length")
                ax.set_ylabel("Density")
                ax.set_title("Thread Length Distribution")
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

    def plot_copulas(
        self, figsize: Optional[Tuple[int, int]] = None, save_path: Optional[str] = None
    ) -> Optional[plt.Figure]:
        """Visualize fitted copulas with scatter plots."""
        if figsize is None:
            figsize = (15, 5)

        n_copulas = len([c for c in self.copulas.values() if c["type"] == "bivariate"])
        if n_copulas == 0:
            logger.warning("No bivariate copulas fitted")
            return None

        fig, axes = plt.subplots(1, n_copulas, figsize=figsize)
        if n_copulas == 1:
            axes = [axes]

        plot_idx = 0

        if "length_curvature" in self.copulas:
            ax = axes[plot_idx]
            lengths = self.stats["thread_lengths"]
            curvatures = self.stats["curvatures"]
            n = min(len(lengths), len(curvatures))

            ax.scatter(
                lengths[:n],
                curvatures[:n],
                alpha=0.5,
                s=20,
                edgecolors="k",
                linewidths=0.5,
            )
            ax.set_xlabel("Thread Length")
            ax.set_ylabel("Curvature")
            ax.set_title(f"Length-Curvature ({self.options.copula_type})")
            ax.grid(True, alpha=0.3)
            plot_idx += 1

        if "branch" in self.copulas:
            ax = axes[plot_idx]
            counts = self.stats["branch_counts"]
            angles = self.stats["branch_angles"]
            n = min(len(counts), len(angles))

            ax.scatter(
                counts[:n], angles[:n], alpha=0.5, s=20, edgecolors="k", linewidths=0.5
            )
            ax.set_xlabel("Branch Count")
            ax.set_ylabel("Branch Angle")
            ax.set_title(f"Branch ({self.options.copula_type})")
            ax.grid(True, alpha=0.3)
            plot_idx += 1

        if "fusion" in self.copulas:
            ax = axes[plot_idx]
            spatial = self.stats["fusion_distances"]
            graph = self.stats["fusion_graph_distances"]
            n = min(len(spatial), len(graph))

            ax.scatter(
                spatial[:n], graph[:n], alpha=0.5, s=20, edgecolors="k", linewidths=0.5
            )
            ax.set_xlabel("Fusion Distance (Spatial)")
            ax.set_ylabel("Fusion Graph Distance")
            ax.set_title(f"Fusion ({self.options.copula_type})")
            ax.grid(True, alpha=0.3)
            plot_idx += 1

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved copula plot to {save_path}")

        return fig
