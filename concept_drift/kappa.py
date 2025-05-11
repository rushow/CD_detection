import numpy as np
from sklearn.metrics import cohen_kappa_score

class KappaDriftDetector:
    def __init__(self, window_size=100, alpha=0.05):
        self.window_size = window_size
        self.alpha = alpha
        self.pred_history = []
        self.true_history = []
        self._drift_detected = False

    def update(self, y_true, y_pred):
        # Ensure y_true and y_pred are binary or categorical
        y_true = self._validate_data(y_true)
        y_pred = self._validate_data(y_pred)

        # Update the history with the new data
        if len(self.pred_history) >= self.window_size:
            self.pred_history.pop(0)
            self.true_history.pop(0)
        self.pred_history.append(y_pred)
        self.true_history.append(y_true)
        
        # Check for drift detection after updating
        self._drift_detected = self.detect()

    def detect(self):
        if len(self.pred_history) < self.window_size:
            return False
        
        # Calculate Kappa statistic over the current window
        kappa_statistic = cohen_kappa_score(self.true_history, self.pred_history)

        # Define drift detection logic
        drift_threshold = self.alpha
        
        return kappa_statistic < drift_threshold

    def _validate_data(self, y):
        # Convert y to a numpy array if it's not already
        y = np.array(y)
        
        # Ensure binary classification values (0, 1) for the Kappa calculation
        unique_values = np.unique(y)
        if len(unique_values) > 2:
            raise ValueError(f"Expected binary classification, but got values: {unique_values}")
        
        return y

    @property
    def drift_detected(self):
        return self._drift_detected