import abc
import array
from typing import List, Dict, Any, Optional

class BaseStorage(abc.ABC):
    """Abstract Base Class for ICOE trial storage."""

    @abc.abstractmethod
    def log_trial(self, trial: Dict[str, Any]) -> None:
        """Log a completed trial."""
        pass

    @abc.abstractmethod
    def get_trials(self, phase_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """Retrieve trials, optionally filtered by phase."""
        pass

    @abc.abstractmethod
    def get_best_trial(self, metric: str = 'score', mode: str = 'min') -> Optional[Dict[str, Any]]:
        """Retrieve the best trial based on a metric."""
        pass


class DictStorage(BaseStorage):
    """
    In-memory storage using Python lists and optimized arrays.

    Optimizations:
    - 'features' (list of indices) are stored as array.array('I') to save memory.
    """

    def __init__(self):
        self.trials = []
        # We store heavy feature sets separately in compact arrays to avoid
        # overhead of millions of small list objects.
        # Format: list of array.array('I')
        self._feature_store: List[array.array] = []

    def log_trial(self, trial: Dict[str, Any]) -> None:
        """
        Logs a trial.
        Expected keys: 'trial_id', 'phase_id', 'params', 'value', 'features' (list[int]).
        """
        # Deep copy params to avoid mutation issues, but handle features separately
        stored_trial = {
            k: v for k, v in trial.items() if k != 'features'
        }
        
        # Optimize Feature Storage
        features = trial.get('features', [])
        if features:
            # 'I' is unsigned int (at least 2 bytes, usually 4). 
            # Much more compact than List[int] pointers (28 bytes per int + list overhead).
            compact_features = array.array('I', features)
            self._feature_store.append(compact_features)
            # Store index pointer
            stored_trial['_feature_idx'] = len(self._feature_store) - 1
        else:
            stored_trial['_feature_idx'] = -1
            
        self.trials.append(stored_trial)

    def get_trials(self, phase_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """Reconstructs and returns trials."""
        result = []
        for t in self.trials:
            if phase_id is not None and t.get('phase_id') != phase_id:
                continue
            
            # Reconstruct
            reconstructed = t.copy()
            idx = reconstructed.pop('_feature_idx', -1)
            if idx >= 0:
                reconstructed['features'] = list(self._feature_store[idx])
            else:
                reconstructed['features'] = []
            result.append(reconstructed)
        return result

    def get_best_trial(self, metric: str = 'score', mode: str = 'min') -> Optional[Dict[str, Any]]:
        if not self.trials:
            return None
        
        sorted_trials = sorted(
            self.trials, 
            key=lambda x: x.get(metric, float('inf') if mode == 'min' else float('-inf')),
            reverse=(mode == 'max')
        )
        
        best = sorted_trials[0]
        # Dehydrate
        reconstructed = best.copy()
        idx = reconstructed.pop('_feature_idx', -1)
        if idx >= 0:
            reconstructed['features'] = list(self._feature_store[idx])
        else:
            reconstructed['features'] = []
            
        return reconstructed


class RDBStorage(BaseStorage):
    """Placeholder for SQL Storage."""
    def __init__(self, url: str):
        raise NotImplementedError("RDBStorage is not yet implemented in v0.1.")
        
    def log_trial(self, trial: Dict[str, Any]) -> None:
        pass

    def get_trials(self, phase_id: Optional[int] = None) -> List[Dict[str, Any]]:
        return []

    def get_best_trial(self, metric: str = 'score', mode: str = 'min') -> Optional[Dict[str, Any]]:
        return None
