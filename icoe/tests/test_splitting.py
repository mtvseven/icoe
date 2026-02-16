import pytest
import pandas as pd
import numpy as np
from icoe.splitting import TimeSeriesCutoffSplit

def test_timeseries_cutoff_split_monotonicity():
    # Setup Data
    dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
    df = pd.DataFrame({'date': dates, 'val': np.arange(100)})
    
    splitter = TimeSeriesCutoffSplit(time_column='date', n_splits=3)
    
    splits = list(splitter.split(df))
    assert len(splits) == 3
    
    for train_idx, val_idx in splits:
        train_dates = df.iloc[train_idx]['date']
        val_dates = df.iloc[val_idx]['date']
        
        # Strict Temporal Separation
        assert train_dates.max() < val_dates.min()
        
        # Coverage
        assert len(train_idx) > 0
        assert len(val_idx) > 0

def test_timeseries_split_integrity():
    # Random shuffle input shouldn't break logic if we sort internally or index correctly?
    # Our impl relies on value comparison, so order of rows shouldn't matter for correctness,
    # only for efficiency if we used slicing.
    
    dates = pd.to_datetime(['2022-01-01', '2022-01-03', '2022-01-02'])
    df = pd.DataFrame({'date': dates, 'val': [1, 3, 2]})
    
    splitter = TimeSeriesCutoffSplit(time_column='date', n_splits=1)
    train_idx, val_idx = next(splitter.split(df))
    
    # With 3 dates, split might be after 1st or 2nd. 
    # Let's check max train < min val
    train_dates = df.iloc[train_idx]['date']
    val_dates = df.iloc[val_idx]['date']
    
    assert train_dates.max() < val_dates.min()
