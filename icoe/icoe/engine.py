import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from typing import List, Dict, Any, Tuple, Optional
from scipy import sparse

class CausalEngine:
    """
    The core logic for Causal Feature Pruning using DML principles.
    """
    
    def __init__(self, pruning_strategy: str = 'lcb', alpha: float = 1.96):
        self.pruning_strategy = pruning_strategy
        self.alpha = alpha # 1.96 for 95% CI

    def _get_treatment_effect(self, history: List[Dict[str, Any]], n_features: int, n_bootstraps: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimates the 'treatment effect' using Double Machine Learning (DML) / Partial Linear Model.
        
        Model: Y = T*theta + g(W) + epsilon
        Where:
        - Y: Outcome (metric)
        - T: Treatments (Features mask)
        - W: Confounders (Hyperparameters)
        
        Algorithm:
        1. Fit g(W) -> Y using Random Forest (Nuisance Model).
        2. Calculate Residuals: Y_res = Y - g(W).
        3. Fit Ridge(T, Y_res) to estimate theta (Marginal contribution of features).
        """
        if not history:
            return np.zeros(n_features), np.zeros(n_features)

        # 1. Prepare Data
        rows = []
        cols = []
        data = []
        y = []
        W_list = [] # List of param dicts
        
        for i, trial in enumerate(history):
            feats = trial.get('features', [])
            if feats:
                rows.extend([i] * len(feats))
                cols.extend(feats)
                data.extend([1] * len(feats))
            y.append(trial['value'])
            W_list.append(trial['params'])
            
        T = sparse.csr_matrix((data, (rows, cols)), shape=(len(history), n_features))
        y = np.array(y)
        
        # 2. Vectorize Hyperparameters (W)
        # Simple approach: DictVectorizer or pandas.get_dummies equivalent
        # Since we don't have sklearn DictVectorizer imported, and params are simple...
        # Let's assume params are numeric or handle basic categorical.
        from sklearn.feature_extraction import DictVectorizer
        vec = DictVectorizer(sparse=False)
        W = vec.fit_transform(W_list)
        
        # 3. Nuisance Model (g(W))
        # We use RF to capture non-linear relationships between Params and Metric.
        rf = RandomForestRegressor(n_estimators=50, max_depth=5, n_jobs=1, random_state=42)
        rf.fit(W, y)
        y_pred_nuisance = rf.predict(W)
        
        # 4. Calculate Residuals
        y_res = y - y_pred_nuisance
        
        # 5. Bootstrap Loop on RESIDUALS
        n_samples = T.shape[0]
        bootstrap_coefs = []
        
        # Base model for point estimate
        base_model = Pipeline([
            ('scaler', StandardScaler(with_mean=False)),
            ('ridge', Ridge(alpha=1.0, fit_intercept=True)) # Intercept captures mean residual (should be ~0)
        ])
        base_model.fit(T, y_res)
        point_coefs = base_model.named_steps['ridge'].coef_
        
        rng = np.random.RandomState(42) 
        
        for _ in range(n_bootstraps):
            indices = rng.randint(0, n_samples, n_samples)
            T_boot = T[indices]
            y_boot_res = y_res[indices]
            
            try:
                model = Pipeline([
                    ('scaler', StandardScaler(with_mean=False)),
                    ('ridge', Ridge(alpha=1.0, fit_intercept=True))
                ])
                model.fit(T_boot, y_boot_res)
                bootstrap_coefs.append(model.named_steps['ridge'].coef_)
            except:
                pass
                
        if not bootstrap_coefs:
            return point_coefs, np.zeros_like(point_coefs)
            
        bootstrap_coefs = np.array(bootstrap_coefs)
        std_errors = np.std(bootstrap_coefs, axis=0)
        std_errors = np.maximum(std_errors, 1e-6)
        
        return point_coefs, std_errors

    def prune(self, coefs: np.ndarray, std_errors: np.ndarray, direction: str = 'minimize') -> List[int]:
        """
        Returns indices of features to KEEP.
        
        Args:
            direction: 'minimize' (RMSLE, RMSE) or 'maximize' (AUC, Accuracy).
        """
        
        if direction == 'minimize':
            # GOOD = Negative Coef.
            # BAD = Positive Coef.
            # PRUNE if (Coef - alpha*SE) > 0 (Definitely Positive/Bad).
            lower_bound = coefs - self.alpha * std_errors
            keep_mask = lower_bound <= 0
            
        else: # maximize
            # GOOD = Positive Coef.
            # BAD = Negative Coef.
            # PRUNE if (Coef + alpha*SE) < 0 (Definitely Negative/Bad).
            upper_bound = coefs + self.alpha * std_errors
            keep_mask = upper_bound >= 0
        
        surviving_indices = np.where(keep_mask)[0].tolist()
        
        
        
        # Safety net: Ensure at least top 10% survive (based on point estimate)
        if len(surviving_indices) < len(coefs) * 0.1:
            top_k = int(max(1, len(coefs) * 0.1))
            surviving_indices = np.argsort(coefs)[:top_k].tolist()
            
        return surviving_indices
