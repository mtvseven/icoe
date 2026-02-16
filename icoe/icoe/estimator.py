from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin, check_is_fitted
from sklearn.utils.validation import check_X_y, check_array, check_random_state
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import mean_squared_error, roc_auc_score
from joblib import Parallel, delayed
import numpy as np
import lightgbm as lgb
from typing import Optional, Union, List, Dict, Any
import sys
import copy

from .storage import DictStorage, BaseStorage
from .search_space import SearchSpace
from .engine import CausalEngine

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
                 verbose=1):
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

    def _get_objective_params(self) -> Dict[str, Any]:
        """Subclasses must return specific LGBM objective params."""
        raise NotImplementedError

    def _score_model(self, model, X, y) -> float:
        """Subclasses must return the metric score."""
        raise NotImplementedError

    def fit(self, X, y):
        # 1. Input Validation
        if 'polars' in sys.modules and hasattr(X, "to_pandas"):
             X = X.to_pandas()

        # Check X y
        # We handle classification vs regression check in subclasses or let LGBM handle it
        X, y = check_X_y(X, y, accept_sparse=False, y_numeric=True, multi_output=False)
        self.n_features_in_ = X.shape[1]
        
        # 2. Initialization
        if self.warm_start and hasattr(self, 'history_'):
            pass
        else:
            self.history_ = DictStorage()
            self.search_space_ = SearchSpace(X, random_state=self.random_state)
            self.engine_ = CausalEngine()
            self.best_model_ = None
            self.best_params_ = None
            self.best_global_score_ = float('inf') if self.direction == 'minimize' else float('-inf')
            self.active_features_ = list(range(X.shape[1]))
            
        rng = check_random_state(self.random_state)
            
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
            
            def run_trial(trial_seed):
                params = self.search_space_.sample_params()
                trial_rng = np.random.RandomState(trial_seed)

                # Dynamic Dropout
                current_dropout = self.feature_dropout * (1 - phase / self.n_phases)
                
                n_curr = len(current_active)
                if n_curr > 0:
                    keep_mask = trial_rng.choice([True, False], size=n_curr, p=[1 - current_dropout, current_dropout])
                    if not np.any(keep_mask):
                        keep_mask[trial_rng.randint(0, n_curr)] = True
                    trial_active_features = [current_active[i] for i, kept in enumerate(keep_mask) if kept]
                else:
                    trial_active_features = []

                # Train LightGBM
                X_subset = X[:, trial_active_features]
                
                # Simple Validation Split
                n_train = int(len(X) * 0.8)
                idxs = rng.permutation(len(X))
                train_idx, val_idx = idxs[:n_train], idxs[n_train:]
                
                X_tr, X_val = X_subset[train_idx], X_subset[val_idx]
                y_tr, y_val = y[train_idx], y[val_idx]
                
                # Weights
                weights = None
                if hasattr(self, 'class_weight') and self.class_weight == 'balanced':
                     weights = compute_sample_weight(class_weight='balanced', y=y_tr)
                
                lgb_train = lgb.Dataset(X_tr, label=y_tr, weight=weights)
                lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)
                
                run_params = params.copy()
                run_params.update(self._get_objective_params()) # Subclass params
                run_params['verbosity'] = -1
                run_params['n_jobs'] = 1 
                
                # We need to manually score if custom metric, but for now rely on LGBM's internal metric?
                # Actually, LGBM valid_sets is convenient but we need to control the metric name.
                # Let's rely on manual scoring to ensure consistency with 'self.metric'.
                
                bst = lgb.train(
                    run_params,
                    lgb_train,
                    num_boost_round=1000,
                    valid_sets=[lgb_val],
                    callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)]
                )
                
                # Manual Scoring for consistency
                score = self._score_model(bst, X_val, y_val)
                
                return {
                    'params': params,
                    'value': score,
                    'features': trial_active_features,
                    'model': bst 
                }

            # Run Parallel
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(run_trial)(rng.randint(0, 10000)) for _ in range(n_trials_phase)
            )
            
            # 3.2 Log & Update
            best_phase_score = float('inf') if self.direction == 'minimize' else float('-inf')
            
            for res in results:
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
                
                # Adaptive Tuning (Phase 5)
                self.search_space_.refine(trials, direction=self.direction)
                
        # 4. Final Refit
        if self.verbose >= 1:
            print("[ICOE] Optimization Complete. Refitting best model...")
            
        final_features = self.active_features_
        final_params = self.best_params_
        
        X_subset = X[:, final_features]
        
        # Weights for Final Fit
        final_weights = None
        if hasattr(self, 'class_weight') and self.class_weight == 'balanced':
             final_weights = compute_sample_weight(class_weight='balanced', y=y)

        lgb_train = lgb.Dataset(X_subset, label=y, weight=final_weights)
        
        run_params = final_params.copy()
        run_params.update(self._get_objective_params())
        run_params['verbosity'] = -1
        run_params['n_jobs'] = self.n_jobs # Use parallel for final fit?
        
        self.bst_ = lgb.train(run_params, lgb_train, num_boost_round=1000)
        self.active_features_ = final_features
        
        return self

    def predict(self, X):
        check_is_fitted(self, ['bst_', 'active_features_'])
        if 'polars' in sys.modules and hasattr(X, "to_pandas"):
             X = X.to_pandas()
        X = check_array(X, accept_sparse=False)
        # Check n_features? (Loose check for now)
        X_subset = X[:, self.active_features_]
        return self.bst_.predict(X_subset)


class ICOERegressor(_BaseICOE, RegressorMixin):
    """
    Iterative Causal Optimization Engine (Regressor).
    """
    def __init__(self, objective='regression', metric='rmse', **kwargs):
        # Default direction 'minimize'
        kwargs.setdefault('direction', 'minimize')
        super().__init__(objective=objective, metric=metric, **kwargs)

    def _get_objective_params(self):
        return {'objective': self.objective, 'metric': self.metric}

    def _score_model(self, model, X, y):
        y_pred = model.predict(X)
        if self.metric == 'rmse':
            return np.sqrt(mean_squared_error(y, y_pred))
        elif self.metric == 'mse':
            return mean_squared_error(y, y_pred)
        else:
            # Fallback
            return np.sqrt(mean_squared_error(y, y_pred))

    def score(self, X, y):
        """Return the optimized metric (RMSE/MSE)."""
        y_pred = self.predict(X)
        if self.metric == 'rmse':
            return np.sqrt(mean_squared_error(y, y_pred))
        elif self.metric == 'mse':
            return mean_squared_error(y, y_pred)
        else:
            return np.sqrt(mean_squared_error(y, y_pred))


class ICOEClassifier(_BaseICOE, ClassifierMixin):
    """
    Iterative Causal Optimization Engine (Classifier).
    """
    def __init__(self, objective='binary', metric='auc', class_weight='balanced', **kwargs):
        # Default direction 'maximize' for AUC
        if metric in ['auc']:
            kwargs.setdefault('direction', 'maximize')
        else:
            kwargs.setdefault('direction', 'minimize') # LogLoss?
            
        self.class_weight = class_weight
        super().__init__(objective=objective, metric=metric, **kwargs)

    def _get_objective_params(self):
        return {'objective': self.objective, 'metric': self.metric}

    def _score_model(self, model, X, y):
        # For AUC we need probabilities
        y_pred = model.predict(X) # LGBM predict returns proba for binary by default
        
        if self.metric == 'auc':
             return roc_auc_score(y, y_pred)
        # Add LogLoss support if needed
        return roc_auc_score(y, y_pred)

    def score(self, X, y):
        """Return the optimized metric (AUC/LogLoss)."""
        if self.metric == 'auc':
             # Need probabilities for AUC
             return self._score_model(self, X, y)
        else:
             from sklearn.metrics import accuracy_score
             return accuracy_score(y, self.predict(X))

    def predict_proba(self, X):
        # LGBM predict returns raw probabilities
        conf = self.predict(X)
        # Return (n_samples, 2)
        return np.vstack([1-conf, conf]).T

    def predict(self, X):
         # Base returns probabilities for LGBM binary
         probs = super().predict(X)
         return (probs > 0.5).astype(int)
