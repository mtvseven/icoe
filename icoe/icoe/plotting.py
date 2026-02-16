import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Optional

def plot_optimization_history(estimator):
    """
    Plots the optimization history of an ICOE estimator.
    """
    if not hasattr(estimator, 'history_'):
        raise ValueError("Estimator is not fitted yet.")
        
    history = estimator.history_.get_trials()
    if not history:
        print("No history to plot.")
        return
        
    df = pd.DataFrame(history)
    
    # Check if we have multiple phases
    if 'phase_id' not in df.columns:
        df['phase_id'] = 0
        
    phases = sorted(df['phase_id'].unique())
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot dots for trials
    scatter = ax.scatter(df.index, df['value'], c=df['phase_id'], cmap='viridis', alpha=0.6, label='Trials')
    
    # Plot best-so-far line
    if hasattr(estimator, 'direction') and estimator.direction == 'maximize':
        best_vals = df['value'].cummax()
    else:
        # Default to minimize
        best_vals = df['value'].cummin()
        
    ax.plot(df.index, best_vals, color='red', linestyle='--', linewidth=2, label='Best so far')
    
    # Add phase boundaries
    # We can't easily know exact trial indices per phase without grouping, 
    # but the scatter color helps.
    
    ax.set_title('ICOE Optimization History')
    ax.set_xlabel('Trial Number')
    ax.set_ylabel(f'Metric ({estimator.metric})')
    ax.legend()
    plt.colorbar(scatter, label='Phase ID')
    plt.grid(True, alpha=0.3)
    
    return fig

def plot_feature_stability(estimator):
    """
    Visualizes which features survived across phases.
    """
    if not hasattr(estimator, 'history_'):
        raise ValueError("Estimator is not fitted yet.")
        
    history = estimator.history_.get_trials()
    df = pd.DataFrame(history)
    
    # Get best trial per phase to see what the "Phase Winner" looked like?
    # Or intersection of all trials?
    # Design says: "X-Axis: Phase ID. Y-Axis: Top 20 Features. Cell Color: Importance."
    
    # Problem: We don't log feature importance for every trial, currently only return 'model' in memory 
    # but the storage doesn't keep the model.
    # In v0.1: We only know WHICH features were active.
    
    # Let's plot "Feature Survival Rate"
    # For each phase, calculate % of trials including feature X?
    # Or just look at the Active Features at the START of each phase (from search space).
    # The Estimator loop updates search_space active features.
    
    # We can reconstruct the "Active Set" per phase from the storage logs.
    # For phase P, look at any trial. The 'features' list is the active set for that phase.
    
    phases = sorted(df['phase_id'].unique())
    active_sets = {}
    
    for p in phases:
        # Get first trial of phase
        trial = df[df['phase_id'] == p].iloc[0]
        active_sets[p] = set(trial['features'])
        
    # Identification of "Top" features
    # Union of all active sets? Or just the final set?
    # Let's visualize the union of Phase 0 and Final Phase.
    
    all_features = sorted(list(active_sets[0]))
    n_features = len(all_features)
    
    if n_features > 50:
        # Too many to plot all. Filter to those present in the final phase?
        final_set = active_sets[phases[-1]]
        # Plus some that were dropped
        dropped_set = active_sets[0] - final_set
        display_feats = sorted(list(final_set) + list(dropped_set)[:20])
    else:
        display_feats = all_features
        
    # Create Heatmap Data
    matrix = np.zeros((len(display_feats), len(phases)))
    
    feat_to_idx = {f: i for i, f in enumerate(display_feats)}
    
    for p_idx, p in enumerate(phases):
        for f in active_sets[p]:
            if f in feat_to_idx:
                matrix[feat_to_idx[f], p_idx] = 1
                
    fig, ax = plt.subplots(figsize=(8, max(6, len(display_feats)*0.3)))
    im = ax.imshow(matrix, cmap='Blues', aspect='auto')
    
    ax.set_xticks(np.arange(len(phases)))
    ax.set_xticklabels([f"Phase {p}" for p in phases])
    
    ax.set_yticks(np.arange(len(display_feats)))
    ax.set_yticklabels([f"Feat {f}" for f in display_feats])
    
    ax.set_title("Feature Survival Monitor")
    
    return fig
