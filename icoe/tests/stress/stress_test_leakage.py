import numpy as np
import pandas as pd
from icoe.estimator import ICOERegressor
from sklearn.metrics import mean_squared_error

def run_leakage_test():
    print("\n--- Running Leakage (Embargo) Test ---")
    
    dates = pd.date_range(start='2020-01-01', periods=500, freq='D')
    # Random walk
    y = np.cumsum(np.random.randn(500))
    df = pd.DataFrame({'y': y})
    df['lag1'] = df['y'].shift(1)
    df['date'] = dates
    df = df.dropna()
    
    # Needs 'date' in X so we can pass time_column='date'
    X = df[['lag1', 'date']]
    y = df['y']
    
    # 1. No Embargo
    print("Training with Embargo=0...")
    model_0 = ICOERegressor(n_trials=10, n_phases=1, splitting_strategy='timeseries', embargo=0, verbose=0, random_state=42)
    model_0.fit(X, y, time_column='date')
    score_0 = model_0.best_global_score_
    print(f"Score (Embargo=0): {score_0:.4f}")
    
    # 2. Embargo = 50
    # Must re-create X because fit drops the time_column
    X = df[['lag1', 'date']]
    
    print("Training with Embargo=50...")
    model_50 = ICOERegressor(n_trials=10, n_phases=1, splitting_strategy='timeseries', embargo=50, verbose=0, random_state=42)
    model_50.fit(X, y, time_column='date')
    score_50 = model_50.best_global_score_
    print(f"Score (Embargo=50): {score_50:.4f}")
    
    if score_50 > score_0 * 1.5:
         print("PASS: Embargo successfully degraded performance (Verification of Gap).")
    else:
         print(f"FAIL: Embargo {score_50:.4f} not clearly worse than {score_0:.4f}")
         # It might fail due to randomness or small sample, but let's see.
         # Actually with RW, lag1 is perfect predictor. Lag50 is terrible. Difference should be huge.

if __name__ == "__main__":
    run_leakage_test()
