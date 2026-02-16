import numpy as np
from icoe.search_space import SearchSpace

def test_search_space_initialization():
    X = np.random.rand(1000, 50)
    space = SearchSpace(X, random_state=42)
    
    # Check default active features
    assert len(space.get_active_features()) == 50
    
    # Check sampling
    params = space.sample_params()
    assert 'learning_rate' in params
    assert 'num_leaves' in params
    assert 31 <= params['num_leaves'] <= 255

def test_search_space_pruning():
    X = np.random.rand(100, 10)
    space = SearchSpace(X)
    
    assert len(space.get_active_features()) == 10
    
    # Prune to first 5
    space.prune_features([0, 1, 2, 3, 4])
    assert len(space.get_active_features()) == 5
    assert space.get_active_features() == [0, 1, 2, 3, 4]
    
    # Parameters should still sample fine
    params = space.sample_params()
    assert isinstance(params, dict)
