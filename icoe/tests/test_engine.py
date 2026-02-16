import numpy as np
from icoe.engine import CausalEngine

def test_engine_pruning():
    engine = CausalEngine(alpha=1.96)
    
    # Fake Coefficients: 
    # 0 = Very Bad (Positive) 
    # 1 = Useless (Zero)
    # 2 = Good (Negative)
    # 3 = Very Good (More Negative)
    coefs = np.array([1.0, 0.0, -0.5, -2.0])
    std_errors = np.array([0.1, 0.1, 0.1, 0.1])
    
    # Upper Bounds:
    # 0: 1.0 - 1.96*0.1 = 0.804 (>0, Prune)
    # 1: 0.0 - 1.96*0.1 = -0.196 (<0, Keep - Optimistic LCB)
    # 2: -0.5 - 1.96*0.1 = -0.7 (<0, Keep)
    # 3: -2.0 - 1.96*0.1 = -2.2 (<0, Keep)
    
    surviving = engine.prune(coefs, std_errors)
    
    assert 0 not in surviving
    assert 1 in surviving
    assert 2 in surviving
    assert 3 in surviving

def test_get_treatment_effect():
    engine = CausalEngine()
    
    # Mock history: 2 features. 
    # Feature 0 used in trial 0 (val=10)
    # Feature 1 used in trial 1 (val=5) -> Feature 1 is better.
    history = [
        {'features': [0], 'value': 10.0, 'params': {'a': 1}},
        {'features': [1], 'value': 5.0, 'params': {'a': 1}},
        {'features': [0, 1], 'value': 7.5, 'params': {'a': 1}}
    ]
    
    coefs, se = engine._get_treatment_effect(history, n_features=2)
    
    # Feature 1 should be more negative (lower value = better) than Feature 0
    assert coefs[1] < coefs[0]

