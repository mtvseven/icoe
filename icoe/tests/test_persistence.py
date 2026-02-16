import numpy as np
import pickle
import pytest
from sklearn.datasets import make_regression
from icoe.estimator import ICOERegressor
import tempfile
import os
import joblib

def test_pickle_persistence():
    X, y = make_regression(n_samples=50, n_features=10, random_state=42)
    
    est = ICOERegressor(n_phases=1, n_jobs=1, random_state=42, verbose=0)
    est.fit(X, y)
    pred_orig = est.predict(X)
    
    # Pickle
    serialized = pickle.dumps(est)
    est_loaded = pickle.loads(serialized)
    
    pred_loaded = est_loaded.predict(X)
    
    np.testing.assert_array_almost_equal(pred_orig, pred_loaded)
    assert est.active_features_ == est_loaded.active_features_

def test_joblib_persistence():
    X, y = make_regression(n_samples=50, n_features=10, random_state=42)
    
    est = ICOERegressor(n_phases=1, n_jobs=1, random_state=42, verbose=0)
    est.fit(X, y)
    pred_orig = est.predict(X)
    
    with tempfile.NamedTemporaryFile(delete=False) as f:
        joblib.dump(est, f.name)
        f.close()
        
        est_loaded = joblib.load(f.name)
        os.unlink(f.name)
        
    pred_loaded = est_loaded.predict(X)
    np.testing.assert_array_almost_equal(pred_orig, pred_loaded)
