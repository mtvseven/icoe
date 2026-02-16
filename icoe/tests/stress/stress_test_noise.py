import numpy as np
import pandas as pd
from icoe.estimator import ICOERegressor

def run_noise_test():
    print("\n--- Running Noise Robustness Test ---")
    
    np.random.seed(42)
    n_samples = 2000
    n_features = 100
    
    # 2 Informative features
    X = np.random.randn(n_samples, n_features)
    # y = 2*X0 - 3*X1 + noise
    y = 2*X[:, 0] - 3*X[:, 1] + 0.5 * np.random.randn(n_samples)
    
    # ICOE should pick 0 and 1, and drop most of 2..99
    
    model = ICOERegressor(
        n_trials=20,
        n_phases=3, # Multi-phase to allow pruning
        n_jobs=-1,
        verbose=0
    )
    
    model.fit(X, y)
    
    active = model.active_features_
    print(f"Final Active Features: {len(active)} / {n_features}")
    print(f"Indices: {active}")
    
    # Assertions
    assert 0 in active, "Feature 0 (Signal) was dropped!"
    assert 1 in active, "Feature 1 (Signal) was dropped!"
    
    # Allow some false positives (e.g. 10% = 10 noise features)
    # With 3 phases, it should be quite aggressive.
    n_noise_kept = len([f for f in active if f > 1])
    print(f"Noise Features Kept: {n_noise_kept}")
    
    if n_noise_kept > 20: 
        print("FAIL: Too many noise features kept.")
        exit(1)
    else:
        print("PASS: Noise filtering successful.")

if __name__ == "__main__":
    run_noise_test()
