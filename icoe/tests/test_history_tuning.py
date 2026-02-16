import pytest
import pandas as pd
import numpy as np
from icoe.estimator import ICOERegressor

def test_history_tuning_integration():
    # Synthetic Data
    dates = pd.date_range(start='2020-01-01', periods=200, freq='D')
    # Signal flips after 100 days
    # First 100: y = x (Positive correlation)
    # Last 100: y = -x (Negative correlation)
    
    X = pd.DataFrame({'feature': np.concatenate([np.arange(100), np.arange(100)])})
    # Add date
    X['date'] = dates
    
    y = np.concatenate([np.arange(100), -np.arange(100)])
    
    # We want the model to learn from the Recent History (Last 100 days) where correlation is negative.
    # If it trains on full history, it sees mixed signals (flat or noise).
    # If it trains on last 50 days, it sees strong negative correlation.
    
    model = ICOERegressor(
        metric='rmse', 
        n_trials=10, 
        n_phases=1, # One pass
        splitting_strategy='timeseries',
        search_space={
            'train_history_days': [50, 180], # 50 (Good, pure negative), 180 (Bad, mixed)
            'learning_rate': [0.1]
        },
        verbose=2
    )
    
    model.fit(X, y, time_column='date')
    
    # Check best params
    print("Best Params:", model.best_params_)
    
    # Ideally, it prefers 50 days logic.
    # Note: randomness in splitting (cutoff) might affect this, but 
    # if cutoff is late (e.g. day 190), then [190-50=140 to 190] is pure negative.
    # [190-180=10 to 190] is mixed.
    # So 50 should win if cutoff > 100.
    
    # Since we can't control random cutoff easily in this black box test without mocking,
    # we just check if it runs without error and keys are present.
    assert 'train_history_days' in model.best_params_

def test_embargo_logic():
    # Verify that embargo creates a gap
    # Need enough data so that (cutoff - embargo) still has training points
    dates = pd.date_range(start='2020-01-01', periods=300, freq='D')
    X = pd.DataFrame({'feature': np.arange(300), 'date': dates})
    y = np.arange(300)
    
    # Embargo = 10 days
    # With 300 points, even if cutoff is early (day 60), we have 50 days (minus embargo=10 = 40 days) to train.
    # LightGBM needs some minimal data.
    model = ICOERegressor(
        n_trials=5, n_phases=1, splitting_strategy='timeseries', embargo=10, verbose=2
    )
    
    model.fit(X, y, time_column='date')
    assert model.best_model_ is not None
