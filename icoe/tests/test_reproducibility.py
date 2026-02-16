import numpy as np
import pytest
from sklearn.datasets import make_regression
from icoe.estimator import ICOERegressor

def test_determinism():
    X, y = make_regression(n_samples=100, n_features=20, random_state=42)
    
    # Run 1
    est1 = ICOERegressor(n_phases=2, n_jobs=1, random_state=42, verbose=0)
    est1.fit(X, y)
    pred1 = est1.predict(X)
    feat1 = est1.active_features_
    
    # Run 2
    est2 = ICOERegressor(n_phases=2, n_jobs=1, random_state=42, verbose=0)
    est2.fit(X, y)
    pred2 = est2.predict(X)
    feat2 = est2.active_features_
    
    np.testing.assert_array_almost_equal(pred1, pred2)
    assert feat1 == feat2

def test_different_seeds():
    X, y = make_regression(n_samples=100, n_features=20, random_state=42)
    
    est1 = ICOERegressor(n_phases=2, n_jobs=1, random_state=42, verbose=0)
    est1.fit(X, y)
    
    est2 = ICOERegressor(n_phases=2, n_jobs=1, random_state=99, verbose=0)
    est2.fit(X, y)
    
    # Should be different (unlikely to be identical with 20 features and random sampling)
    assert est1.active_features_ != est2.active_features_ or \
           not np.allclose(est1.predict(X), est2.predict(X))
