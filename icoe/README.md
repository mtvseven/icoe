# ICOE: Iterative Causal Optimization Engine

**Version:** 0.1.0 (Alpha)

ICOE is a robust feature selection and model optimization engine designed for high-stakes data science. It combines **Geometric Heuristics** for search space definition with **Double Machine Learning (DML)** for causal feature pruning, all wrapped in a standard Scikit-Learn estimator.

## Philosophy

1.  **Scikit-Learn Native:** Plug-and-play with `Pipeline`, `GridSearchCV`, and the rest of the ecosystem.
2.  **Sensible Defaults:** No more guessing hyperparameters. We use heuristics based on dataset properties ($N, p$).
3.  **Causal, Not Just Correlational:** We prune features based on their *conditional* treatment effect, not just global importance.

## Installation

```bash
pip install sim-icoe
```

## Quick Start

```python
from icoe import ICOERegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

# 1. Load Data
X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Initialize
model = ICOERegressor(
    objective='regression',
    metric='rmse',
    n_phases=3,
    n_jobs=-1,
    verbose=1
)

# 3. Fit (The Engine runs: Sim -> DML -> Prune -> Repeat)
model.fit(X_train, y_train)


# 4. Inspect Results
print(f"Best RMSE: {-model.score(X_test, y_test):.4f}")
model.plot_optimization_history()
```

## Classification Example

```python
from icoe import ICOEClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# 1. Load Data
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Initialize
model = ICOEClassifier(
    objective='binary',
    metric='auc',
    n_phases=3,
    n_jobs=-1,
    verbose=1
)

# 3. Fit
model.fit(X_train, y_train)

# 4. Inspect Results
print(f"Best AUC: {model.score(X_test, y_test):.4f}")
```

---

## Key Concepts

### Feature Selection Philosophy
ICOE uses an **"Optimistic Lower Confidence Bound"** strategy for feature pruning.
*   **Behavior**: It keeps features unless they are *statistically proven* to be harmful or irrelevant (Negative treatment effect with high confidence).
*   **Implication**: Expect **High Recall** (keeps all good features) but potentially **Low Precision** (keeps some noise/ambiguous features).
*   **Tip**: If you have very noisy data, increase the `n_trials` (e.g., 50+) to tighten the confidence intervals, or accept that the model prefers to be safe than sorry.

### Safety: The Embargo
To prevent **Look-Ahead Bias** in time-series, ICOE implements an `embargo` period.
*   **Definition**: The gap between the end of the Training set and the start of the Validation set.
*   **Usage**:
    ```python
    # For daily data, enforce a 5-day gap
    model = ICOERegressor(
        splitting_strategy='timeseries',
        embargo=5, # Interpreted as 5 Days if time_column is datetime
        ...
    )
    model.fit(X, y, time_column='date')
    ```
*   **Validation**: The system now strictly validates `embargo` units. If you pass an integer for a datetime column, it warns and assumes "Days". Ideally, use `pd.Timedelta(days=5)`.

## Performance & Scale

### Memory Usage
*   **Optimization**: ICOE uses optimized `array.array` storage for feature indices, reducing memory overhead by ~70% compared to standard lists.
*   **Guideline**:
    *   **< 100k rows**: Use `n_jobs=-1` (All cores).
    *   **> 1M rows**: Monitor RAM. Each worker copies the dataset (via `joblib` memmapping, but overhead exists). Consider `n_jobs=4` or `n_jobs=2` if RAM is tight (<16GB).

### Wide Datasets (P > 1000)
For datasets with thousands of features:
*   **Initialization**: ICOE starts with all features active. The first few trials will be slow.
*   **Pruning**: Pruning happens after each Phase. Use `n_phases=3` or `5` to aggressively whittle down the feature set.

## Troubleshooting

| Error | Cause | Fix |
| :--- | :--- | :--- |
| `ValueError: splitting_strategy='timeseries' requires time_column` | You forgot to pass `time_column` to `fit`. | `model.fit(X, y, time_column='date')` |
| `UserWarning: Embargo provided as number...` | Ambiguous units for embargo. | Use `pd.Timedelta(days=X)` or ignore if Days is intended. |
| `PicklingError` / `Joblib` failure | Using closure-based `run_trial`. | **Fixed in v0.2**: Ensure you are using the latest version with module-level `_execute_trial`. |

## Deployment Checklist
- [ ] **Time Column**: Always use explicit datetime objects for `time_column` to ensure correct splitting.
- [ ] **Embargo**: Set `embargo > 0` for any financial/economic time series (autocorrelation risk).
- [ ] **Persistence**: Use `joblib.dump(model, 'model.pkl')` to save. The internal storage is pickle-friendly.
