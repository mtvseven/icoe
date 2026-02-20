
import sys
from unittest.mock import patch
import numpy as np
from icoe.estimator import ICOERegressor

def test_early_stopping_enabled():
    print("\n--- Test 1: Early Stopping ENABLED (Default) ---")
    X = np.random.rand(100, 5)
    y = np.random.rand(100)
    
    # Phase 1: 0.1 (Good)
    # Phase 2: 0.5 (Bad) -> Should trigger stop
    # Phase 3: Should not run
    side_effect = [0.1, 0.1, 0.5, 0.5, 0.1, 0.1]
    
    with patch('icoe.estimator._score_predictions') as mock_score:
        mock_score.side_effect = side_effect
        
        est = ICOERegressor(
            n_phases=3, 
            n_trials=2, 
            n_jobs=1, 
            verbose=1,
            early_stopping_tolerance=0.0,
            early_stopping=True 
        )
        est.fit(X, y)
        
        phase_0_trials = est.history_.get_trials(phase_id=0)
        phase_1_trials = est.history_.get_trials(phase_id=1)
        phase_2_trials = est.history_.get_trials(phase_id=2)
        
        print(f"Phase 0 trials: {len(phase_0_trials)}")
        print(f"Phase 1 trials: {len(phase_1_trials)}")
        print(f"Phase 2 trials: {len(phase_2_trials)}")
        
        if len(phase_2_trials) == 0 and len(phase_1_trials) > 0:
            print("SUCCESS: Early stopping triggered as expected.")
        else:
            print("FAILURE: Early stopping did NOT trigger.")
            return False
    return True

def test_early_stopping_disabled():
    print("\n--- Test 2: Early Stopping DISABLED ---")
    X = np.random.rand(100, 5)
    y = np.random.rand(100)
    
    # Phase 1: 0.1 (Good)
    # Phase 2: 0.5 (Bad) -> Should NOT trigger stop
    # Phase 3: 0.1 (Good) -> Should run
    side_effect = [0.1, 0.1, 0.5, 0.5, 0.1, 0.1]
    
    with patch('icoe.estimator._score_predictions') as mock_score:
        mock_score.side_effect = side_effect
        
        est = ICOERegressor(
            n_phases=3, 
            n_trials=2, 
            n_jobs=1, 
            verbose=1,
            early_stopping_tolerance=0.0,
            early_stopping=False # DISABLED
        )
        est.fit(X, y)
        
        phase_0_trials = est.history_.get_trials(phase_id=0)
        phase_1_trials = est.history_.get_trials(phase_id=1)
        phase_2_trials = est.history_.get_trials(phase_id=2)
        
        print(f"Phase 0 trials: {len(phase_0_trials)}")
        print(f"Phase 1 trials: {len(phase_1_trials)}")
        print(f"Phase 2 trials: {len(phase_2_trials)}")
        
        if len(phase_2_trials) > 0:
            print("SUCCESS: Early stopping was skipped as expected.")
        else:
            print("FAILURE: Early stopping triggered when it should have been disabled.")
            return False
    return True

if __name__ == "__main__":
    res1 = test_early_stopping_enabled()
    res2 = test_early_stopping_disabled()
    
    if res1 and res2:
        print("\nALL TESTS PASSED")
        sys.exit(0)
    else:
        print("\nSOME TESTS FAILED")
        sys.exit(1)
