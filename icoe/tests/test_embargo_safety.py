import pytest
import pandas as pd
import numpy as np
from icoe.estimator import ICOERegressor

def test_embargo_validation_days():
    """Test that integer embargo is accepted for datetime IF interpreted as days."""
    dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
    X = pd.DataFrame({'val': np.arange(100)})
    y = np.arange(100)
    
    # 1. Int Embargo + Datetime Time -> Warning + Conversion
    est = ICOERegressor(embargo=5, verbose=1) 
    # This should work but might warn
    with pytest.warns(UserWarning, match="Assuming 'Days'"):
        est.fit(X, y, time_column=dates) 
        
    assert isinstance(est.embargo, int) # Original param untouched
    # But internal execution used Timedelta. 
    # We can't easily check private variables inside the worker, but success implies it worked.

def test_embargo_validation_timedelta():
    """Test explicit Timedelta embargo."""
    dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
    X = pd.DataFrame({'val': np.arange(100)})
    y = np.arange(100)
    
    est = ICOERegressor(embargo=pd.Timedelta(days=5), verbose=0)
    est.fit(X, y, time_column=dates) 
    # Should pass without warning

def test_embargo_error_mismatch():
    """Test that Timedelta embargo fails for Integer index."""
    X = pd.DataFrame({'val': np.arange(100), 'step': np.arange(100)})
    y = np.arange(100)
    
    est = ICOERegressor(embargo=pd.Timedelta(days=5), verbose=0)
    
    with pytest.raises(ValueError, match="cannot be Timedelta"):
        est.fit(X.drop(columns='step'), y, time_column=X['step'])

def test_embargo_int_index():
    """Test integer embargo works for integer index."""
    X = pd.DataFrame({'val': np.arange(100), 'step': np.arange(100)})
    y = np.arange(100)
    
    est = ICOERegressor(embargo=10, splitting_strategy='timeseries', verbose=0)
    est.fit(X.drop(columns='step'), y, time_column=X['step'])
    # Should pass
