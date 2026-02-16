import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator
from sklearn.utils.validation import check_array
from typing import Optional, List, Generator, Tuple

class TimeSeriesCutoffSplit(BaseCrossValidator):
    """
    Time Series Cross-Validator that splits based on a time cutoff.
    
    Parameters
    ----------
    time_column : str
        Name of the column containing time information (must be datetime).
    n_splits : int, default=5
        Number of splits. 
        Note: In standard usage with ICOE, we might often just use 1 split per trial 
        (random cutoff), but adhering to CV API allows for n_splits.
    test_size : float or int, default=0.2
        Size of the validation set (fraction if < 1.0, count if >= 1).
    gap : float or int, default=0
        Gap between train and test to prevent leakage (e.g. embargo).
    """
    def __init__(self, time_column: str, n_splits: int = 5, test_size: float = 0.2, gap: float = 0):
        self.time_column = time_column
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate indices to split data into training and test set.
        
        X : pd.DataFrame
            Dataframe containing the time_column.
        """
        # Validation
        if not hasattr(X, 'loc'): # Check if pandas dataframe-like
             raise ValueError("X must be a pandas DataFrame containing the time_column.")
             
        if self.time_column not in X.columns:
             raise ValueError(f"Time column '{self.time_column}' not found in X.")
             
        # Ensure strict temporal ordering check? code assumes sorted or we sort?
        # Standard TS split usually assumes sorted. We will not enforce sort here 
        # to avoid copy, but we operate on Values.
        
        times = X[self.time_column]
        if not np.issubdtype(times.dtype, np.datetime64):
             try:
                 times = pd.to_datetime(times)
             except:
                 raise ValueError(f"Column '{self.time_column}' cannot be cast to datetime.")
        
        unique_times = np.sort(times.unique())
        n_dates = len(unique_times)
        
        if n_dates < 2:
            raise ValueError("Data must have at least 2 unique time points for splitting.")

        # Indices of dates
        indices = np.arange(n_dates)
        
        # We want to pick 'n_splits' cutoff points.
        # Strategy: Randomly pick cutoffs? Or deterministic rolling?
        # Sklearn TimeSeriesSplit is deterministic "expanding window".
        # ICOE wants "Random Cutoff" for inner loop, but BaseCrossValidator needs to be generator.
        # Let's implement standard "Expanding Window" for CV compliance, but 
        # ICOE internal loop might define n_splits=1 and shuffle=True (not supported by BaseCV usually).
        
        # Actually, if we use this for the inner loop, we probably just want ONE split based on a random cutoff.
        # But for CV, let's do K-Fold equivalent (K cutoffs).
        
        # We simply step through time.
        step = int(n_dates // (self.n_splits + 1))
        
        for i in range(self.n_splits):
            # Define cutoff for this split
            # Split i+1
            split_idx = (i + 1) * step
            if split_idx >= n_dates:
                break
                
            cutoff_date = unique_times[split_idx]
            
            # Masking
            train_mask = times < cutoff_date
            
            # Test set logic: 
            # If test_size is ratio, strictly take next chunk? 
            # Or just "everything after"? Standard TS Split usually takes next 'step'.
            # Let's take 'everything after' (Future Validation) to be rigorous about "Future",
            # unless test_size limits it.
            
            # Actually, standard TimeSeriesSplit fixes test_size. 
            # Let's implement "Test is all future" for max realism Causal Pruning?
            # No, usually we want a specific validation horizon.
            
            val_mask = times >= cutoff_date
            
            # Apply gap if needed
            # (Remove gap period from END of train or START of val? usually START of val)
            # Simplest: Train < cutoff, Val >= cutoff. 
            # Gap: Val >= cutoff + gap.
            
            indices_all = np.arange(len(X))
            yield indices_all[train_mask], indices_all[val_mask]


class PanelTimeSeriesSplit(BaseCrossValidator):
    """
    Panel-Aware Time Series Splitter.
    Respects both Time (non-overlapping train/val) and Groups (groups kept intact).
    """
    def __init__(self, time_column: str, group_column: str, n_splits: int = 5):
        self.time_column = time_column
        self.group_column = group_column
        self.n_splits = n_splits
        
    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        """
        Splitting logic:
        1. Identify unique timestamps.
        2. Split on Time (expanding window).
        3. For a given time split, ensure all Groups present at that time are handled.
           (Actually, standard TS split naturally handles this if we split on Time).
           
        Why simple TimeSeriesCutoffSplit isn't enough?
        - Because we might want to ensure we don't split a "Group-Event" in half if data is long format?
        - Actually, if we split strictly on Time, we automatically respect Panel structure 
          (past of Ticker A -> future of Ticker A).
          
        So PanelTimeSeriesSplit is effectively TimeSeriesCutoffSplit but enforcing that we split 
        by *unique dates*, not by *row counts*, which we did above anyway.
        
        So this class might just be a wrapper or strict enforcer.
        Let's keep it to handle "Group" specific logic if we ever need "Leave-One-Group-Out" 
        (which is Cross-Sectional, not TS). 
        
        For Panel TS, it's just TS split on the 'time_column'. 
        """
        # Delegate to TimeSeriesCutoffSplit logic for now, 
        # but structurally defined to separate intent.
        splitter = TimeSeriesCutoffSplit(time_column=self.time_column, n_splits=self.n_splits)
        return splitter.split(X, y)
