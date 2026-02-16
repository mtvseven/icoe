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
