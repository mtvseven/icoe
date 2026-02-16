import time
import psutil
import pandas as pd
import numpy as np
from sklearn.datasets import make_regression
from icoe.estimator import ICOERegressor
import gc
import os

def measure_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024 # MB

def run_scale_test(n_samples, n_features, tag):
    print(f"\n--- Running Scale Test: {tag} (N={n_samples}, P={n_features}) ---")
    mem_start = measure_memory()
    
    # Generate Data
    t0 = time.time()
    X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=0.1, random_state=42)
    t_gen = time.time() - t0
    print(f"Data Gen Time: {t_gen:.2f}s")
    
    mem_loaded = measure_memory()
    print(f"Memory (Data Loaded): {mem_loaded:.2f} MB (+{mem_loaded - mem_start:.2f} MB)")
    
    # Init Model
    model = ICOERegressor(
        n_trials=20, # Limited trials for speed
        n_phases=2, 
        n_jobs=-1, # Parallel
        verbose=0
    )
    
    t0 = time.time()
    model.fit(X, y)
    t_fit = time.time() - t0
    
    mem_peak = measure_memory()
    print(f"Fit Time: {t_fit:.2f}s")
    print(f"Memory (After Fit): {mem_peak:.2f} MB (+{mem_peak - mem_loaded:.2f} MB)")
    print(f"Score: {model.best_global_score_:.4f}")
    
    del X, y, model
    gc.collect()

if __name__ == "__main__":
    # Regime S (Small)
    run_scale_test(10000, 50, "Small")
    
    # Regime M (Medium)
    run_scale_test(100000, 100, "Medium")
    
    # Regime L (Large - Reduced for CI/Environment constraints)
    # Full 1M rows might be too slow for this interactive session, testing 200k
    run_scale_test(200000, 200, "Large-ISH")
    
    # Regime W (Wide)
    run_scale_test(10000, 1000, "Wide")
