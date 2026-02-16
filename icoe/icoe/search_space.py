import numpy as np
import scipy.stats as stats
from typing import Dict, Any, Optional, List, Union

class SearchSpace:
    """
    Manages the hyperparameter search space using Geometric Heuristics.
    """

    def __init__(self, X: Any, param_distributions: Optional[Dict[str, Any]] = None, random_state: Optional[int] = None):
        self.random_state = np.random.RandomState(random_state)
        
        # Heuristics based on dataset shape
        n_samples = X.shape[0] if hasattr(X, 'shape') else len(X)
        n_features = X.shape[1] if hasattr(X, 'shape') else 1

        # Sensible Defaults tailored to dataset size
        # "Geometric" scaling: Larger data -> deeper trees, less regularization needed?
        # Actually, larger data usually allows for deeper trees (lower bias), 
        # but requires more regularization to prevent overfitting on noise.
        
        default_leaves = int(np.clip(n_samples / 500, 31, 255))
        default_min_child = int(np.clip(n_samples / 1000, 20, 100))

        self.defaults = {
            'learning_rate': stats.uniform(0.01, 0.2), # 0.01 to 0.21
            'num_leaves': stats.randint(31, default_leaves + 1),
            'min_child_samples': stats.randint(20, default_min_child + 1),
            'subsample': stats.uniform(0.5, 0.5),      # 0.5 to 1.0
            'colsample_bytree': stats.uniform(0.5, 0.5),
            'reg_alpha': stats.uniform(0, 1.0),
            'reg_lambda': stats.uniform(0, 5.0)
        }

        # Override defaults if user provided params
        if param_distributions:
            self.defaults.update(param_distributions)
            
        self.distributions = self.defaults.copy()
        
        # Feature Pruning Mask (Starts as all included)
        # We store the *indices* of active features
        self.active_features: List[int] = list(range(n_features))
        
        # Exploration Rate (Phase 5)
        self.exploration_rate = 0.2

    def sample_params(self) -> Dict[str, Any]:
        """Sample a set of hyperparameters."""
        # Exploration Logic: 20% chance to sample from Global Defaults
        if self.random_state.rand() < self.exploration_rate:
            source = self.defaults
        else:
            source = self.distributions
            
        params = {}
        for k, v in source.items():
            if hasattr(v, 'rvs'):
                params[k] = v.rvs(random_state=self.random_state)
            elif isinstance(v, list):
                # Sample from list (Categorical or discrete grid)
                params[k] = list(v)[self.random_state.randint(len(v))]
            else:
                params[k] = v # Fixed value
        return params

    def get_active_features(self) -> List[int]:
        """Return the current list of active feature indices."""
        return self.active_features

    def prune_features(self, surviving_indices: List[int]) -> None:
        """Update the active feature set (Intersection)."""
        # Input is a list of indices valid in the ORIGINAL X space.
        self.active_features = surviving_indices

    def refine(self, history: List[Dict[str, Any]], direction: str = 'minimize') -> None:
        """
        Phase 5: Refines the search space based on top performing trials.
        Shrinks the bounds of self.distributions around the best values.
        """
        if len(history) < 5:
            return

        # 1. Select Top 20%
        values = [t['value'] for t in history]
        if direction == 'minimize':
            # Lower is better
            cutoff = np.percentile(values, 20)
            top_trials = [t for t in history if t['value'] <= cutoff]
        else:
            # Higher is better
            cutoff = np.percentile(values, 80)
            top_trials = [t for t in history if t['value'] >= cutoff]
            
        if not top_trials:
            return

        # 2. Update Distributions
        new_distributions = self.distributions.copy()
        
        for k, v in self.defaults.items():
            # We only refine numeric distributions provided in defaults
            # If user provided fixed values or custom objects we might skip?
            # Assuming standard scipy stats usage for now.
            
            # Extract values from top trials
            param_values = [t['params'].get(k) for t in top_trials if k in t['params']]
            if not param_values:
                continue
                
            min_v = min(param_values)
            max_v = max(param_values)
            
            if isinstance(v, (str, bool)): 
                continue # Categorical not handled yet
                
            if hasattr(v, 'dist'):
                # It's a scipy distribution
                # We identify type by names (hacky but effective for standard usage)
                dist_name = v.dist.name
                
                # Span padding (10%)
                span = max_v - min_v
                if span == 0: span = 1.0 # Avoid collapse if all same
                pad = span * 0.1
                
                new_min = max(0.0001, min_v - pad) # Assuming positive params mostly
                new_max = max_v + pad
                
                # Clamp specific parameters
                if k in ['subsample', 'colsample_bytree']:
                    new_max = min(1.0, new_max)
                    new_min = min(new_max - 0.05, new_min) # Ensure non-zero range
                
                # Integer params
                if 'randint' in dist_name or isinstance(v.dist, stats.rv_discrete):
                     new_distributions[k] = stats.randint(int(new_min), int(new_max) + 2)
                # Float params
                else: 
                     new_distributions[k] = stats.uniform(new_min, (new_max - new_min))
        
        self.distributions = new_distributions
