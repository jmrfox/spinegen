import numpy as np
from typing import Dict, Any
from scipy import stats


class SpinePrior:
    
    def __init__(self, analyzer_stats: Dict[str, Any]):
        self.stats = analyzer_stats
        self.distributions = {}
        self._fit_distributions()
    
    def _fit_distributions(self) -> None:
        if len(self.stats['segment_lengths']) > 0:
            lengths = self.stats['segment_lengths']
            lengths = lengths[lengths > 0]
            if len(lengths) > 0:
                shape, loc, scale = stats.gamma.fit(lengths)
                self.distributions['segment_length'] = {
                    'type': 'gamma',
                    'params': {'shape': shape, 'loc': loc, 'scale': scale}
                }
        
        if len(self.stats['curvatures']) > 0:
            curvatures = self.stats['curvatures']
            mu = np.mean(curvatures)
            sigma = np.std(curvatures)
            self.distributions['curvature'] = {
                'type': 'normal',
                'params': {'mu': mu, 'sigma': sigma}
            }
        
        if len(self.stats['branch_counts']) > 0:
            counts, freq = np.unique(self.stats['branch_counts'], return_counts=True)
            probs = freq / freq.sum()
            self.distributions['branch_count'] = {
                'type': 'categorical',
                'params': {'values': counts.tolist(), 'probs': probs.tolist()}
            }
        
        if len(self.stats['branch_angles']) > 0:
            angles = self.stats['branch_angles']
            mu = np.mean(angles)
            sigma = np.std(angles)
            self.distributions['branch_angle'] = {
                'type': 'normal',
                'params': {'mu': mu, 'sigma': sigma}
            }
        
        if len(self.stats['radii']) > 0:
            radii = self.stats['radii']
            radii = radii[radii > 0]
            if len(radii) > 0:
                mu = np.mean(np.log(radii))
                sigma = np.std(np.log(radii))
                self.distributions['radius'] = {
                    'type': 'lognormal',
                    'params': {'mu': mu, 'sigma': sigma}
                }
        
        if len(self.stats['fusion_distances']) > 0:
            distances = self.stats['fusion_distances']
            distances = distances[distances > 0]
            if len(distances) > 0:
                loc, scale = stats.expon.fit(distances)
                self.distributions['fusion_distance'] = {
                    'type': 'exponential',
                    'params': {'loc': loc, 'scale': scale}
                }
        
        if len(self.stats['fusion_graph_distances']) > 0:
            graph_dists = self.stats['fusion_graph_distances']
            graph_dists = graph_dists[np.isfinite(graph_dists)]
            if len(graph_dists) > 0:
                self.distributions['fusion_graph_distance'] = {
                    'type': 'empirical',
                    'params': {'mean': np.mean(graph_dists), 'std': np.std(graph_dists)}
                }
    
    def sample_segment_length(self) -> float:
        if 'segment_length' not in self.distributions:
            return 10.0
        
        dist = self.distributions['segment_length']
        if dist['type'] == 'gamma':
            params = dist['params']
            return stats.gamma.rvs(params['shape'], loc=params['loc'], scale=params['scale'])
        return 10.0
    
    def sample_curvature(self) -> float:
        if 'curvature' not in self.distributions:
            return 0.1
        
        dist = self.distributions['curvature']
        if dist['type'] == 'normal':
            params = dist['params']
            angle = np.random.normal(params['mu'], params['sigma'])
            return np.clip(angle, 0, np.pi)
        return 0.1
    
    def sample_branch_count(self) -> int:
        if 'branch_count' not in self.distributions:
            return 2
        
        dist = self.distributions['branch_count']
        if dist['type'] == 'categorical':
            params = dist['params']
            return int(np.random.choice(params['values'], p=params['probs']))
        return 2
    
    def sample_branch_angle(self) -> float:
        if 'branch_angle' not in self.distributions:
            return np.pi / 3
        
        dist = self.distributions['branch_angle']
        if dist['type'] == 'normal':
            params = dist['params']
            angle = np.random.normal(params['mu'], params['sigma'])
            return np.clip(angle, 0, np.pi)
        return np.pi / 3
    
    def sample_radius(self, distance_from_root: float = 0.0) -> float:
        if 'radius' not in self.distributions:
            return 1.0
        
        dist = self.distributions['radius']
        if dist['type'] == 'lognormal':
            params = dist['params']
            base_radius = np.random.lognormal(params['mu'], params['sigma'])
            taper_factor = np.exp(-distance_from_root / 1000.0)
            return base_radius * taper_factor
        return 1.0
    
    def sample_fusion_distance(self) -> float:
        if 'fusion_distance' not in self.distributions:
            return 50.0
        
        dist = self.distributions['fusion_distance']
        if dist['type'] == 'exponential':
            params = dist['params']
            return stats.expon.rvs(loc=params['loc'], scale=params['scale'])
        return 50.0
    
    def compute_fusion_probability(self, spatial_dist: float, graph_dist: float) -> float:
        if 'fusion_distance' not in self.distributions:
            sigma = 50.0
        else:
            dist = self.distributions['fusion_distance']
            if dist['type'] == 'exponential':
                sigma = dist['params']['scale']
            else:
                sigma = 50.0
        
        if 'fusion_graph_distance' not in self.distributions:
            lambda_param = 100.0
        else:
            dist = self.distributions['fusion_graph_distance']
            lambda_param = dist['params'].get('mean', 100.0)
        
        spatial_prob = np.exp(-spatial_dist**2 / (2 * sigma**2))
        graph_prob = 1.0 - np.exp(-graph_dist / lambda_param) if graph_dist < float('inf') else 0.0
        
        return spatial_prob * graph_prob
