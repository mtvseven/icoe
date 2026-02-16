from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin, check_is_fitted
from sklearn.utils.validation import check_X_y, check_array, check_random_state
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import mean_squared_error, roc_auc_score, accuracy_score
from joblib import Parallel, delayed
import numpy as np
import lightgbm as lgb
from typing import Optional, Union, List, Dict, Any
import sys
import copy
import pandas as pd
import warnings

from .storage import DictStorage, BaseStorage
from .search_space import SearchSpace
from .engine import CausalEngine

def _score_predictions(metric: str, y_true: np.ndarray, y_pred: np.ndarray, objective: str = 'regression') -> float:
    """Helper to score predictions without needing the full estimator instance."""
    if metric == 'rmse':
        return np.sqrt(mean_squared_error(y_true, y_pred))
    elif metric == 'mse':
        return mean_squared_error(y_true, y_pred)
    elif metric == 'auc':
        return roc_auc_score(y_true, y_pred)
    elif metric == 'accuracy':
        if objective == 'binary':
             y_pred_cls = (y_pred > 0.5).astype(int)
             return accuracy_score(y_true, y_pred_cls)
        return accuracy_score(y_true, y_pred > 0.5)
    # Default fallback
    try:
        if objective in ['regression', 'regression_l1', 'huber']:
            return np.sqrt(mean_squared_error(y_true, y_pred))
        return roc_auc_score(y_true, y_pred)
    except:
        return float('inf')

def _execute_trial(
    trial_seed: int,
    X: np.ndarray, 
    y: np.ndarray,
    search_space: SearchSpace,
    initial_active: List[int],
    dropout_rate: float,
    splitting_config: Dict[str, Any],
    objective_params: Dict[str, Any],
    metric: str,
    class_weight_type: Optional[str]
) -> Dict[str, Any]:
    """
    Module-level function to execute a single trial.
    Must be pickleable for joblib.
    """
    # Sample Params
    # We passed search_space OBJECT. Is it pickleable? Yes.
    # But calling sample_params on it might affect its internal random state if not handled carefully?
    # Actually, SearchSpace has its own random_state. If we pass the SAME object to all workers,
    # and they fork, they get copies.
    params = search_space.sample_params()
    
    # We use a trial-specific RNG for dropout/splitting to ensure determinism solely from trial_seed
    trial_rng = np.random.RandomState(trial_seed)

    # Dynamic Dropout
    n_curr = len(initial_active)
    if n_curr > 0:
        keep_mask = trial_rng.choice([True, False], size=n_curr, p=[1 - dropout_rate, dropout_rate])
        if not np.any(keep_mask):
            keep_mask[trial_rng.randint(0, n_curr)] = True
        trial_active_features = [initial_active[i] for i, kept in enumerate(keep_mask) if kept]
    else:
        trial_active_features = []

    # Train LightGBM
    X_subset = X[:, trial_active_features]
    
    # --- Splitting Logic ---
    strategy = splitting_config.get('strategy', 'random')
    time_values = splitting_config.get('time_values')
    group_values = splitting_config.get('group_values')
    embargo = splitting_config.get('embargo', 0)
    
    train_idx = np.array([])
    val_idx = np.array([])
    
    if strategy == 'timeseries':
        if time_values is None:
             raise ValueError("splitting_strategy='timeseries' requires time_values")
        
        # Random Cutoff for Inner Loop
        unique_times = np.unique(time_values)
        n_times = len(unique_times)
        if n_times < 2:
             # Fallback
             cutoff_idx = 0
             cutoff_date = unique_times[0]
        else:
            cutoff_idx = trial_rng.randint(int(n_times * 0.2), int(n_times * 0.9))
            cutoff_date = unique_times[cutoff_idx]
        
        # Apply Embargo
        # Embargo is already validated/converted in fit() to be compatible (Timedelta or Int)
        if isinstance(embargo, (pd.Timedelta, np.timedelta64)):
             train_end_date = cutoff_date - embargo
        else:
             # If embargo is int and time is not TimeDelta compatible (e.g. Int index), simple subtraction
             train_end_date = cutoff_date - embargo

        train_mask = time_values < train_end_date
        val_mask = time_values >= cutoff_date
        
        train_idx = np.where(train_mask)[0]
        val_idx = np.where(val_mask)[0]
        
    elif strategy == 'panel':
         if time_values is not None:
             # Panel Time Series -> Use Time Split with Embargo
             unique_times = np.unique(time_values)
             n_times = len(unique_times)
             cutoff_idx = trial_rng.randint(int(n_times * 0.2), int(n_times * 0.9))
             cutoff_date = unique_times[cutoff_idx]
             
             if isinstance(embargo, (pd.Timedelta, np.timedelta64)):
                 train_end_date = cutoff_date - embargo
             else:
                 train_end_date = cutoff_date - embargo
             
             train_mask = time_values < train_end_date
             val_mask = time_values >= cutoff_date
             
             train_idx = np.where(train_mask)[0]
             val_idx = np.where(val_mask)[0]
             
         elif group_values is not None:
             unique_groups = np.unique(group_values)
             n_groups = len(unique_groups)
             n_val_groups = max(1, int(n_groups * 0.2))
             val_groups = trial_rng.choice(unique_groups, size=n_val_groups, replace=False)
             
             val_mask = np.isin(group_values, val_groups)
             train_mask = ~val_mask
             train_idx = np.where(train_mask)[0]
             val_idx = np.where(val_mask)[0]
         else:
             raise ValueError("splitting_strategy='panel' requires groups or time.")

    else:
        # Random Split
        n_train = int(len(X) * 0.8)
        idxs = trial_rng.permutation(len(X))
        train_idx, val_idx = idxs[:n_train], idxs[n_train:]
    
    if len(train_idx) == 0 or len(val_idx) == 0:
        # Fallback if split failed (e.g. embargo too large)
        return {
            'params': params,
            'value': float('inf') if objective_params.get('metric') not in ['auc', 'auc_mu'] else 0.0,
            'features': trial_active_features
        }

    X_tr, X_val = X_subset[train_idx], X_subset[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]
    
    # --- Training History Logic ---
    if 'train_history_days' in params and time_values is not None:
        history_days = params['train_history_days']
        tr_times = time_values[train_idx]
        if len(tr_times) > 0:
            split_point = tr_times.max() 
            # Convert history_days to Timedelta if time is datetime
            if np.issubdtype(time_values.dtype, np.datetime64):
                 start_date = split_point - pd.Timedelta(days=history_days)
            else:
                 start_date = split_point - history_days # Assume int index
            
            mask_history = tr_times >= start_date
            X_tr = X_tr[mask_history]
            y_tr = y_tr[mask_history]
    
    if len(y_tr) < 2: 
         return {
            'params': params,
            'value': float('inf') if objective_params.get('metric') not in ['auc'] else 0.0,
            'features': trial_active_features
        }

    # Weights
    weights = None
    if class_weight_type == 'balanced':
         try:
            weights = compute_sample_weight(class_weight='balanced', y=y_tr)
         except: pass
    
    lgb_train = lgb.Dataset(X_tr, label=y_tr, weight=weights)
    lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)
    
    run_params = params.copy()
    run_params.update(objective_params)
    run_params['verbosity'] = -1
    run_params['n_jobs'] = 1 
    
    bst = lgb.train(
        run_params,
        lgb_train,
        num_boost_round=1000,
        valid_sets=[lgb_val],
        callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)]
    )
    
    # Manual Scoring
    y_pred_val = bst.predict(X_val)
    score = _score_predictions(metric, y_val, y_pred_val, objective=objective_params.get('objective'))
    
    return {
        'params': params,
        'value': score,
        'features': trial_active_features,
        'model': bst 
    }


class _BaseICOE(BaseEstimator):
    """
    Base class for Iterative Causal Optimization Engine.
    Handles the core logic: Search Space, Causal Pruning Loop, and Adaptive Tuning.
    """
    
    def __init__(self, 
                 objective='regression',
                 metric='rmse',
                 n_phases=3,
                 n_jobs=-1,
                 random_state=None,
                 warm_start=False,
                 feature_dropout=0.2,
                 n_trials=None,
                 early_stopping_tolerance=0.0,
                 direction='minimize',
                 verbose=1,
                 splitting_strategy='random',
                 search_space=None, 
                 embargo=0, 
                 ):
        self.objective = objective
        self.metric = metric
        self.n_phases = n_phases
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.warm_start = warm_start
        self.feature_dropout = feature_dropout
        self.n_trials = n_trials
        self.early_stopping_tolerance = early_stopping_tolerance
        self.direction = direction
        self.verbose = verbose
        self.splitting_strategy = splitting_strategy
        self.search_space = search_space
        self.embargo = embargo

    def _get_objective_params(self) -> Dict[str, Any]:
        """Subclasses must return specific LGBM objective params."""
        raise NotImplementedError

    def _score_model(self, model, X, y) -> float:
        """Deprecated internal method."""
        y_pred = model.predict(X)
        return _score_predictions(self.metric, y, y_pred, self.objective)

    def _validate_embargo(self, time_values):
        """
        Validates and converts embargo based on time_values type.
        Returns the safe embargo value to use.
        """
        if self.embargo == 0:
            return 0
            
        is_datetime = np.issubdtype(time_values.dtype, np.datetime64)
        
        if is_datetime:
            if isinstance(self.embargo, (int, float, np.integer)):
                if self.verbose > 0:
                    warnings.warn(f"Embargo provided as number ({self.embargo}) but time is datetime. Assuming 'Days'.")
                return pd.Timedelta(days=int(self.embargo))
            elif isinstance(self.embargo, (pd.Timedelta, np.timedelta64)):
                return self.embargo
            else:
                raise ValueError("For datetime time_column, embargo must be int (days) or Timedelta.")
        else:
            # Integer/Float index
            if isinstance(self.embargo, (int, float, np.integer)):
                return self.embargo
            elif isinstance(self.embargo, (pd.Timedelta, np.timedelta64)):
                raise ValueError("For numeric time_column, embargo cannot be Timedelta.")
            else:
                # Fallback, maybe user passed something weird
                return self.embargo

    def fit(self, X, y, time_column=None, group_column=None):
        # 1. Input Validation
        if 'polars' in sys.modules and hasattr(X, "to_pandas"):
             X = X.to_pandas()

        # Handle Time Column Extraction
        self._time_values = None
        if time_column:
            if isinstance(X, pd.DataFrame):
                if time_column not in X.columns:
                    raise ValueError(f"Time column '{time_column}' not found in X")
                
                # Check / Convert to datetime
                t_vals = X[time_column]
                if not np.issubdtype(t_vals.dtype, np.datetime64) and not np.issubdtype(t_vals.dtype, np.number):
                    try:
                        t_vals = pd.to_datetime(t_vals)
                    except:
                         pass 
                
                self._time_values = t_vals.values 
                X = X.drop(columns=[time_column])
            else:
                pass
        
        # Handle Group Column Extraction
        self._group_values = None
        if group_column:
            if isinstance(X, pd.DataFrame):
                if group_column not in X.columns:
                    raise ValueError(f"Group column '{group_column}' not found in X")
                
                self._group_values = X[group_column].values
                X = X.drop(columns=[group_column])
            else:
                pass

        # Check splitting strategy requirements
        if self.splitting_strategy == 'timeseries' and self._time_values is None:
             raise ValueError("splitting_strategy='timeseries' requires 'time_column' to be provided and valid.")
        
        if self.splitting_strategy == 'panel':
             if self._group_values is None and self._time_values is None:
                  raise ValueError("splitting_strategy='panel' requires 'group_column' or 'time_column'.")
             if self._time_values is None and self.verbose > 0:
                  warnings.warn("splitting_strategy='panel' used without time_column. Autocorrelation leakage possible.")

        # Check X y
        X, y = check_X_y(X, y, accept_sparse=False, y_numeric=True, multi_output=False)
        self.n_features_in_ = X.shape[1]
        
        # 2. Initialization
        if self.warm_start and hasattr(self, 'history_'):
            pass
        else:
            self.history_ = DictStorage()
            self.search_space_ = SearchSpace(X, param_distributions=self.search_space, random_state=self.random_state)
            self.engine_ = CausalEngine()
            self.best_model_ = None
            self.best_params_ = None
            self.best_global_score_ = float('inf') if self.direction == 'minimize' else float('-inf')
            self.active_features_ = list(range(X.shape[1]))
            
        rng = check_random_state(self.random_state)

        # Validate Embargo once
        safe_embargo = 0
        if self._time_values is not None:
            safe_embargo = self._validate_embargo(self._time_values)
            
        # 3. Phase Loop
        for phase in range(self.n_phases):
            current_active = self.search_space_.get_active_features()
            n_active = len(current_active)
            
            if self.verbose >= 1:
                print(f"[ICOE] Phase {phase+1}/{self.n_phases} Start. Active Features: {n_active}")
                
            if n_active == 0:
                print("[ICOE] Warning: All features pruned. Stopping.")
                break

            # 3.1 Run Simulation Batch
            if self.n_trials is not None:
                n_trials_phase = self.n_trials
            else:
                n_trials_phase = max(10, n_active // 2) 

            # Prepare configuration for worker
            splitting_config = {
                'strategy': self.splitting_strategy,
                'embargo': safe_embargo,
                'time_values': self._time_values,
                'group_values': self._group_values
            }
            obj_params = self._get_objective_params()
            current_dropout = self.feature_dropout * (1 - phase / self.n_phases)

            # Run Parallel
            # We must be careful passing 'self.search_space_' if it's large, but it's usually small config.
            # Passing 'X' and 'y' (large arrays) to every worker is efficient w/ joblib memmapping.
            
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(_execute_trial)(
                    trial_seed=rng.randint(0, 1000000),
                    X=X,
                    y=y,
                    search_space=self.search_space_,
                    initial_active=current_active,
                    dropout_rate=current_dropout,
                    splitting_config=splitting_config,
                    objective_params=obj_params,
                    metric=self.metric,
                    class_weight_type=getattr(self, 'class_weight', None)
                ) for _ in range(n_trials_phase)
            )
            
            # 3.2 Log & Update
            best_phase_score = float('inf') if self.direction == 'minimize' else float('-inf')
            
            valid_results = [r for r in results if r is not None]
            
            for res in valid_results:
                log_entry = {k: v for k, v in res.items() if k != 'model'}
                log_entry['phase_id'] = phase
                self.history_.log_trial(log_entry)
                
                val = res['value']
                if self.direction == 'minimize':
                    if val < best_phase_score: best_phase_score = val
                else:
                    if val > best_phase_score: best_phase_score = val
                     
            # Global Best Logic
            mode = 'max' if self.direction == 'maximize' else 'min'
            best_trial_now = self.history_.get_best_trial(metric='value', mode=mode)
            
            if best_trial_now:
                score_best_now = best_trial_now['value']
                is_new_best = False
                if self.best_params_ is None:
                    is_new_best = True
                elif self.direction == 'maximize' and score_best_now > self.best_global_score_:
                    is_new_best = True
                elif self.direction == 'minimize' and score_best_now < self.best_global_score_:
                    is_new_best = True
                    
                if is_new_best:
                     self.best_params_ = best_trial_now['params']
                     self.active_features_ = best_trial_now['features']
                     self.best_global_score_ = score_best_now
                     
                     if self.verbose >= 2:
                         print(f"   [New Best] Score: {score_best_now:.4f} | Features: {len(self.active_features_)}")
            
            if self.verbose >= 1:
                print(f"[ICOE] Phase {phase+1} Best Score: {best_phase_score:.4f} (Global Best: {self.best_global_score_:.4f})")

            # SAFETY STOP
            stop_condition = False
            if self.direction == 'minimize':
                if best_phase_score > self.best_global_score_ + self.early_stopping_tolerance:
                    # Allow slight degradation? Default tol=0.0 means STRICT.
                    stop_condition = True
            else:
                 if best_phase_score < self.best_global_score_ - self.early_stopping_tolerance:
                    stop_condition = True
                    
            if stop_condition:
                if self.verbose >= 1:
                    print(f"[ICOE] Early Stopping: Performance degraded. Stopping.")
                break

            # 3.3 Causal Pruning & Adaptive Tuning (Mid-Phase)
            if phase < self.n_phases - 1:
                trials = self.history_.get_trials(phase_id=phase)
                
                # Pruning
                coefs, se = self.engine_._get_treatment_effect(trials, n_features=X.shape[1])
                surviving = self.engine_.prune(coefs, se, direction=self.direction)
                self.search_space_.prune_features(surviving)
                
                # Adaptive Tuning
                self.search_space_.refine(trials, direction=self.direction)
                
        # 4. Final Refit
        if self.verbose >= 1:
            print("[ICOE] Optimization Complete. Refitting best model...")
            
        final_features = self.active_features_
        final_params = self.best_params_
        
        X_subset = X[:, final_features]
        
        final_weights = None
        if hasattr(self, 'class_weight') and self.class_weight == 'balanced':
             final_weights = compute_sample_weight(class_weight='balanced', y=y)

        lgb_train = lgb.Dataset(X_subset, label=y, weight=final_weights)
        
        run_params = final_params.copy()
        run_params.update(self._get_objective_params())
        run_params['verbosity'] = -1
        run_params['n_jobs'] = self.n_jobs 
        
        self.bst_ = lgb.train(run_params, lgb_train, num_boost_round=1000)
        self.best_model_ = self.bst_ 
        self.active_features_ = final_features
        
        return self

    def predict(self, X):
        check_is_fitted(self, ['bst_', 'active_features_'])
        if 'polars' in sys.modules and hasattr(X, "to_pandas"):
             X = X.to_pandas()
        X = check_array(X, accept_sparse=False)
        X_subset = X[:, self.active_features_]
        return self.bst_.predict(X_subset)


class ICOERegressor(_BaseICOE, RegressorMixin):
    """
    Iterative Causal Optimization Engine (Regressor).
    """
    def __init__(self, objective='regression', metric='rmse', **kwargs):
        kwargs.setdefault('direction', 'minimize')
        super().__init__(objective=objective, metric=metric, **kwargs)

    def _get_objective_params(self):
        return {'objective': self.objective, 'metric': self.metric}

    def score(self, X, y, sample_weight=None):
        y_pred = self.predict(X)
        return _score_predictions(self.metric, y, y_pred, self.objective)


class ICOEClassifier(_BaseICOE, ClassifierMixin):
    """
    Iterative Causal Optimization Engine (Classifier).
    """
    def __init__(self, objective='binary', metric='auc', class_weight='balanced', **kwargs):
        if metric in ['auc']:
            kwargs.setdefault('direction', 'maximize')
        else:
            kwargs.setdefault('direction', 'minimize')
            
        self.class_weight = class_weight
        super().__init__(objective=objective, metric=metric, **kwargs)

    def _get_objective_params(self):
        return {'objective': self.objective, 'metric': self.metric}

    def predict_proba(self, X):
        conf = self.predict(X)
        return np.vstack([1-conf, conf]).T

    def predict(self, X):
         probs = super().predict(X)
         if hasattr(self, 'objective') and self.objective == 'binary':
             return (probs > 0.5).astype(int)
         return probs # for other objectives?
    
    def score(self, X, y, sample_weight=None):
         if self.metric == 'auc':
             # Use raw probabilities for AUC
             return roc_auc_score(y, super().predict(X))
         return accuracy_score(y, self.predict(X))
