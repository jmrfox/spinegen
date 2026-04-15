"""Conditional prior distributions using copulas for spine morphology.

This module extends SpinePrior with copula-based joint distributions to capture
correlations between morphological features without assuming structural dependencies.
Uses the copulas package for robust copula fitting and sampling.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List, Literal
from scipy import stats
import matplotlib.pyplot as plt

from copulas.bivariate import Bivariate
from copulas.multivariate import GaussianMultivariate, VineCopula

from .prior import SpinePrior

logger = logging.getLogger(__name__)

CopulaType = Literal["gaussian", "student", "clayton", "gumbel", "frank"]


class ConditionalSpinePrior(SpinePrior):
    """Extended prior with copula-based joint distributions.
    
    Uses the copulas package to model correlations between morphological
    features without assuming structural dependencies like branch order.
    
    Attributes:
        stats: Dictionary of morphological statistics from analyzer
        distributions: Dictionary of fitted marginal distribution parameters
        copulas: Dictionary of fitted copula models
        copula_type: Type of copula for bivariate distributions
        use_multivariate: Whether to use multivariate copula
        
    Joint Distributions:
        - Bivariate: (length, curvature), (branch_count, angle), (fusion distances)
        - Multivariate: All features jointly (optional)
    """
    
    def __init__(
        self,
        analyzer_stats: Dict[str, Any],
        copula_type: CopulaType = "gaussian",
        use_multivariate: bool = False,
        multivariate_type: Literal["gaussian", "vine"] = "gaussian",
    ):
        """Initialize conditional prior and fit copulas.
        
        Args:
            analyzer_stats: Statistics from SpineAnalyzer.analyze()
            copula_type: Copula type ("gaussian", "student", "clayton", etc.)
            use_multivariate: If True, fit multivariate copula for all features
            multivariate_type: "gaussian" or "vine" for multivariate copula
        """
        super().__init__(analyzer_stats)
        
        self.copula_type = copula_type
        self.use_multivariate = use_multivariate
        self.multivariate_type = multivariate_type
        self.copulas: Dict[str, Any] = {}
        
        if use_multivariate:
            logger.info(f"Fitting {multivariate_type} multivariate copula")
            self._fit_multivariate_copula()
        else:
            logger.info(f"Fitting {copula_type} bivariate copulas")
            self._fit_bivariate_copulas()
        logger.info(f"Fitted {len(self.copulas)} copula model(s)")
    
    def _fit_bivariate_copulas(self) -> None:
        """Fit bivariate copulas for pairwise feature correlations."""
        if (len(self.stats["segment_lengths"]) > 0 and
            len(self.stats["curvatures"]) > 0):
            self._fit_bivariate_copula(
                "length_curvature",
                self.stats["segment_lengths"],
                self.stats["curvatures"],
                ["segment_length", "curvature"]
            )
        
        if (len(self.stats["branch_counts"]) > 0 and
            len(self.stats["branch_angles"]) > 0):
            self._fit_bivariate_copula(
                "branch",
                self.stats["branch_counts"],
                self.stats["branch_angles"],
                ["branch_count", "branch_angle"]
            )
        
        if (len(self.stats["fusion_distances"]) > 0 and
            len(self.stats["fusion_graph_distances"]) > 0):
            self._fit_bivariate_copula(
                "fusion",
                self.stats["fusion_distances"],
                self.stats["fusion_graph_distances"],
                ["fusion_distance", "fusion_graph_distance"]
            )
    
    def _fit_bivariate_copula(
        self,
        name: str,
        data1: np.ndarray,
        data2: np.ndarray,
        feature_names: List[str]
    ) -> None:
        """Fit bivariate copula to two features using copulas package."""
        n_samples = min(len(data1), len(data2))
        data1 = data1[:n_samples]
        data2 = data2[:n_samples]
        
        valid_mask = np.isfinite(data1) & np.isfinite(data2)
        if name == "length_curvature":
            valid_mask &= (data1 > 0)
        elif name == "fusion":
            valid_mask &= (data1 > 0) & (data2 > 0)
        
        data1 = data1[valid_mask]
        data2 = data2[valid_mask]
        
        if len(data1) < 10:
            logger.warning(f"Insufficient data for {name} copula")
            return
        
        df = pd.DataFrame({feature_names[0]: data1, feature_names[1]: data2})
        
        try:
            copula = Bivariate(copula_type=self.copula_type)
            copula.fit(df)
            
            self.copulas[name] = {
                "model": copula,
                "type": "bivariate",
                "copula_type": self.copula_type,
                "marginals": feature_names,
            }
            
            logger.info(f"Fitted {self.copula_type} copula for {name}")
        except Exception as e:
            logger.error(f"Failed to fit copula for {name}: {e}")
    
    def _fit_multivariate_copula(self) -> None:
        """Fit multivariate copula to all features simultaneously."""
        feature_data = {}
        feature_names = []
        
        if len(self.stats["segment_lengths"]) > 0:
            feature_data["segment_length"] = self.stats["segment_lengths"]
            feature_names.append("segment_length")
        
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
        
        n_samples = min(len(data) for data in feature_data.values())
        df_data = {name: feature_data[name][:n_samples] for name in feature_names}
        df = pd.DataFrame(df_data)
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(df) < 10:
            logger.warning("Insufficient valid data for multivariate copula")
            return
        
        try:
            if self.multivariate_type == "gaussian":
                copula = GaussianMultivariate()
            else:
                copula = VineCopula(vine_type="regular")
            
            copula.fit(df)
            
            self.copulas["multivariate"] = {
                "model": copula,
                "type": "multivariate",
                "copula_type": self.multivariate_type,
                "marginals": feature_names,
            }
            
            logger.info(
                f"Fitted {self.multivariate_type} multivariate copula "
                f"with {len(feature_names)} features"
            )
        except Exception as e:
            logger.error(f"Failed to fit multivariate copula: {e}")
    
    def sample_length_curvature_joint(self) -> Tuple[float, float]:
        """Sample segment length and curvature jointly from copula."""
        if "length_curvature" not in self.copulas:
            return (self.sample_segment_length(), self.sample_curvature())
        
        copula_model = self.copulas["length_curvature"]["model"]
        sample = copula_model.sample(1)
        
        length = float(sample["segment_length"].iloc[0])
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
    
    def plot_copulas(
        self,
        figsize: Optional[Tuple[int, int]] = None,
        save_path: Optional[str] = None
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
            lengths = self.stats["segment_lengths"]
            curvatures = self.stats["curvatures"]
            n = min(len(lengths), len(curvatures))
            
            ax.scatter(lengths[:n], curvatures[:n], alpha=0.5, s=20,
                      edgecolors="k", linewidths=0.5)
            ax.set_xlabel("Segment Length")
            ax.set_ylabel("Curvature")
            ax.set_title(f"Length-Curvature ({self.copula_type})")
            ax.grid(True, alpha=0.3)
            plot_idx += 1
        
        if "branch" in self.copulas:
            ax = axes[plot_idx]
            counts = self.stats["branch_counts"]
            angles = self.stats["branch_angles"]
            n = min(len(counts), len(angles))
            
            ax.scatter(counts[:n], angles[:n], alpha=0.5, s=20,
                      edgecolors="k", linewidths=0.5)
            ax.set_xlabel("Branch Count")
            ax.set_ylabel("Branch Angle")
            ax.set_title(f"Branch ({self.copula_type})")
            ax.grid(True, alpha=0.3)
            plot_idx += 1
        
        if "fusion" in self.copulas:
            ax = axes[plot_idx]
            spatial = self.stats["fusion_distances"]
            graph = self.stats["fusion_graph_distances"]
            n = min(len(spatial), len(graph))
            
            ax.scatter(spatial[:n], graph[:n], alpha=0.5, s=20,
                      edgecolors="k", linewidths=0.5)
            ax.set_xlabel("Fusion Distance (Spatial)")
            ax.set_ylabel("Fusion Graph Distance")
            ax.set_title(f"Fusion ({self.copula_type})")
            ax.grid(True, alpha=0.3)
            plot_idx += 1
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved copula plot to {save_path}")
        
        return fig
    
    def __str__(self) -> str:
        """String representation with copula information."""
        base_str = super().__str__()
        
        if len(self.copulas) > 0:
            copula_info = ["\n\n" + "=" * 70]
            copula_info.append("Copula Models")
            copula_info.append("=" * 70)
            
            for name, copula in self.copulas.items():
                copula_type = copula["copula_type"]
                marginals = " <-> ".join(copula["marginals"])
                copula_info.append(f"  {name}: {copula_type} ({marginals})")
            
            copula_info.append("=" * 70)
            return base_str + "\n".join(copula_info)
        
        return base_str
