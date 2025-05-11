# Adaptive Random Forest (ARF) Drift Detector
# concept_drift/arf.py

import logging
import numpy as np
from river import forest
from river import metrics
from collections import deque

class ARFDriftDetector:
    """
    Drift detector based on the Adaptive Random Forest algorithm.
    
    This detector uses window-based accuracy monitoring to detect concept drift,
    while leveraging ARF's built-in adaptation capabilities.
    """
    
    def __init__(self, 
                 n_models=10, 
                 max_features='sqrt', 
                 lambda_value=6,
                 warning_window_size=50,
                 drift_window_size=30,
                 warning_threshold=0.85,
                 drift_threshold=0.75,
                 seed=None):
        """
        Initialize the ARF-based drift detector.
        
        Parameters:
        -----------
        n_models : int, default=10
            Number of trees in the forest.
        max_features : int or str, default='sqrt'
            Number of features to consider when looking for the best split.
        lambda_value : float, default=6
            The lambda parameter for the ARF algorithm.
        warning_window_size : int, default=50
            Size of the sliding window for warning detection.
        drift_window_size : int, default=30
            Size of the sliding window for drift detection.
        warning_threshold : float, default=0.85
            Accuracy threshold below which a warning is triggered.
        drift_threshold : float, default=0.75
            Accuracy threshold below which drift is detected.
        seed : int, default=None
            Random seed for reproducibility.
        """
        self.n_models = n_models
        self.max_features = max_features
        self.lambda_value = lambda_value
        self.seed = seed
        self.warning_window_size = warning_window_size
        self.drift_window_size = drift_window_size
        self.warning_threshold = warning_threshold
        self.drift_threshold = drift_threshold
        
        # Configure logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize the model and state
        self.reset()

    def reset(self):
        """Reset the detector to its initial state."""
        # Initialize the ARF classifier
        self.model = forest.ARFClassifier(
            n_models=self.n_models, 
            max_features=self.max_features, 
            lambda_value=self.lambda_value,
            seed=self.seed
        )
        
        # Initialize metrics
        self.accuracy = metrics.Accuracy()
        
        # State variables
        self.drift_detected = False
        self.warning_detected = False
        self.num_instances = 0
        self.num_drifts_detected = 0
        self.num_warnings_detected = 0
        
        # Recent predictions and true labels for window-based detection
        self.recent_predictions = deque(maxlen=max(self.warning_window_size, self.drift_window_size))
        self.recent_true_labels = deque(maxlen=max(self.warning_window_size, self.drift_window_size))
        
        # Track the number of trees replaced (as an indicator of drift)
        self.trees_replaced_count = 0
        self.previous_trees_replaced = 0

    def _preprocess_features(self, X):
        """
        Preprocess the input features.
        
        Parameters:
        -----------
        X : dict or array-like
            Input features.
            
        Returns:
        --------
        dict
            Preprocessed features as a dictionary.
        """
        # Convert X to a dictionary if it's not already
        if not isinstance(X, dict):
            if isinstance(X, (list, np.ndarray)):
                X = {f"feature_{i}": val for i, val in enumerate(X)}
            else:
                X = {"feature_0": X}
        
        # Handle None, NaN, and Inf values
        processed_X = {}
        for k, v in X.items():
            if v is None or (isinstance(v, (float, int)) and (np.isnan(v) or np.isinf(v))):
                processed_X[k] = 0.0  # Default value for missing/invalid data
            else:
                processed_X[k] = v
                
        return processed_X

    def _evaluate_drift(self):
        """
        Evaluate whether drift or warning should be detected based on recent performance.
        
        Returns:
        --------
        str
            'drift', 'warning', or 'no_drift' indicating the current status.
        """
        # Reset status flags
        self.drift_detected = False
        self.warning_detected = False
        
        # Not enough data to evaluate
        if len(self.recent_predictions) < self.drift_window_size:
            return 'no_drift'
        
        # Calculate recent accuracy for drift detection
        recent_preds = list(self.recent_predictions)[-self.drift_window_size:]
        recent_true = list(self.recent_true_labels)[-self.drift_window_size:]
        recent_accuracy = sum(p == y for p, y in zip(recent_preds, recent_true)) / len(recent_preds)
        
        # Check for drift
        if recent_accuracy < self.drift_threshold:
            self.drift_detected = True
            self.num_drifts_detected += 1
            self.logger.info(f"Drift detected: accuracy={recent_accuracy:.4f}, threshold={self.drift_threshold}")
            return 'drift'
        
        # Check for warning
        if len(self.recent_predictions) >= self.warning_window_size:
            warning_preds = list(self.recent_predictions)[-self.warning_window_size:]
            warning_true = list(self.recent_true_labels)[-self.warning_window_size:]
            warning_accuracy = sum(p == y for p, y in zip(warning_preds, warning_true)) / len(warning_preds)
            
            if warning_accuracy < self.warning_threshold:
                self.warning_detected = True
                self.num_warnings_detected += 1
                self.logger.debug(f"Warning detected: accuracy={warning_accuracy:.4f}, threshold={self.warning_threshold}")
                return 'warning'
        
        return 'no_drift'

    def _check_arf_internal_drift(self):
        """
        Check if ARF's internal drift adaptation mechanism has been triggered
        by monitoring the number of background trees that became active.
        
        Returns:
        --------
        bool
            True if ARF's internal mechanism indicates drift, False otherwise.
        """
        # Try to access ARF's internal state to check for tree replacements
        try:
            # In ARFClassifier, we can check if trees have been replaced by 
            # examining the switch count if it's available
            if hasattr(self.model, '_n_switches'):
                current_switches = self.model._n_switches
                if current_switches > self.previous_trees_replaced:
                    self.trees_replaced_count += (current_switches - self.previous_trees_replaced)
                    self.previous_trees_replaced = current_switches
                    return True
            return False
        except (AttributeError, Exception) as e:
            self.logger.debug(f"Could not check ARF internal drift state: {e}")
            return False

    def update(self, X, y):
        """
        Update the detector with a new observation.
        
        Parameters:
        -----------
        X : dict or array-like
            Input features.
        y : any
            True label.
            
        Returns:
        --------
        str
            'drift', 'warning', or 'no_drift' indicating the current status.
        """
        self.num_instances += 1
        
        # Preprocess features
        processed_X = self._preprocess_features(X)
        
        # Make a prediction
        try:
            pred = self.model.predict_one(processed_X)
        except Exception as e:
            self.logger.error(f"Prediction error: {e}")
            pred = None
        
        # Update the recent predictions and true labels
        if pred is not None:
            self.recent_predictions.append(pred)
            self.recent_true_labels.append(y)
            
            # Update accuracy metric
            self.accuracy.update(y, pred)
        
        # Store state before learning to compare later
        if hasattr(self.model, '_n_switches'):
            self.previous_trees_replaced = getattr(self.model, '_n_switches', 0)
        
        # Update the model with the new sample
        try:
            self.model.learn_one(processed_X, y)
        except Exception as e:
            self.logger.error(f"Learning error: {e}")
        
        # Check for drift using our window-based mechanism
        drift_status = self._evaluate_drift()
        
        # Also check if ARF's internal mechanism has adapted to drift
        arf_drift_detected = self._check_arf_internal_drift()
        if arf_drift_detected and not self.drift_detected:
            self.drift_detected = True
            self.num_drifts_detected += 1
            drift_status = 'drift'
            self.logger.info("Drift detected by ARF internal mechanism (tree replacement)")
        
        return drift_status

    def predict(self, X):
        """
        Make a prediction for the given input.
        
        Parameters:
        -----------
        X : dict or array-like
            Input features.
            
        Returns:
        --------
        any
            Predicted label.
        """
        processed_X = self._preprocess_features(X)
        return self.model.predict_one(processed_X)
    
    def get_status(self):
        """
        Get the current status of the detector.
        
        Returns:
        --------
        dict
            A dictionary containing the current status.
        """
        return {
            "drift_detected": self.drift_detected,
            "warning_detected": self.warning_detected,
            "num_instances": self.num_instances,
            "num_drifts_detected": self.num_drifts_detected,
            "num_warnings_detected": self.num_warnings_detected,
            "trees_replaced_count": self.trees_replaced_count,
            "current_accuracy": self.accuracy.get(),
            "n_models": self.n_models,
            "max_features": self.max_features,
            "warning_threshold": self.warning_threshold,
            "drift_threshold": self.drift_threshold
        }