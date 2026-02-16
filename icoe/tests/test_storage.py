import pytest
import array
from icoe.storage import DictStorage

def test_dict_storage_roundtrip():
    storage = DictStorage()
    
    trial_data = {
        'trial_id': 1,
        'phase_id': 0,
        'params': {'learning_rate': 0.1},
        'value': 0.5,
        'features': [1, 2, 100, 500]
    }
    
    storage.log_trial(trial_data)
    
    # Retrieve
    trials = storage.get_trials(phase_id=0)
    assert len(trials) == 1
    assert trials[0]['trial_id'] == 1
    assert trials[0]['features'] == [1, 2, 100, 500]
    assert isinstance(trials[0]['features'], list) # Should be converted back to list

def test_dict_storage_optimization():
    """Verify that we are actually using array.array internally."""
    storage = DictStorage()
    storage.log_trial({'trial_id': 1, 'features': [1, 2, 3]})
    
    assert len(storage._feature_store) == 1
    assert isinstance(storage._feature_store[0], array.array)
    assert storage._feature_store[0].typecode == 'I'

def test_filtering():
    storage = DictStorage()
    storage.log_trial({'trial_id': 1, 'phase_id': 0, 'features': []})
    storage.log_trial({'trial_id': 2, 'phase_id': 1, 'features': []})
    
    assert len(storage.get_trials(phase_id=0)) == 1
    assert len(storage.get_trials(phase_id=1)) == 1
    assert len(storage.get_trials(phase_id=99)) == 0

def test_get_best_trial():
    storage = DictStorage()
    storage.log_trial({'trial_id': 1, 'score': 10})
    storage.log_trial({'trial_id': 2, 'score': 5})
    storage.log_trial({'trial_id': 3, 'score': 20})
    
    best = storage.get_best_trial(metric='score', mode='min')
    assert best['trial_id'] == 2
    
    best_max = storage.get_best_trial(metric='score', mode='max')
    assert best_max['trial_id'] == 3
