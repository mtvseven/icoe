
import numpy as np
from icoe.estimator import ICOEClassifier
from sklearn.metrics import f1_score

def test_f1_metric_maximization():
    print("Testing F1 Metric Initialization...")
    clf = ICOEClassifier(metric='f1', n_phases=1, n_trials=2, n_jobs=1)
    
    # Check Direction
    if clf.direction != 'maximize':
        print(f"FAILURE: Expected direction='maximize' for metric='f1', got '{clf.direction}'")
        return False
    print("SUCCESS: Direction correctly set to 'maximize'")
    
    print("\nTesting F1 Metric Optimization Run...")
    # Create imbalanced dataset where F1 is more interesting than accuracy
    X = np.random.rand(100, 5)
    # y has 10% positives
    y = (np.random.rand(100) > 0.9).astype(int)
    
    try:
        clf.fit(X, y)
        print("SUCCESS: Fit completed without error.")
    except Exception as e:
        print(f"FAILURE: Fit failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    # Check if we can score
    score = clf.score(X, y)
    print(f"Final F1 Score: {score}")
    
    # Manual check
    y_pred = clf.predict(X)
    manual_f1 = f1_score(y, y_pred)
    
    if abs(score - manual_f1) < 1e-9:
        print("SUCCESS: Score method matches manual f1_score.")
    else:
        print(f"FAILURE: Score mismatch. Class: {score}, Manual: {manual_f1}")
        return False
        
    return True

if __name__ == "__main__":
    if test_f1_metric_maximization():
        print("\nALL TESTS PASSED")
        # System exit 0
    else:
        print("\nTEST FAILED")
        import sys
        sys.exit(1)
