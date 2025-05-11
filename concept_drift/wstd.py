import numpy as np
from scipy.stats import mannwhitneyu
import logging

class WSTDDriftDetector:
    """
    Window-based Statistical Test Drift Detector using the Wilcoxon rank-sum test.
    
    This detector compares two halves of a sliding window of observations to detect
    distributional shifts in the data stream.
    """
    
    def __init__(self, window_size=100, 
                 alpha=0.05, 
                 warning_threshold=0.10,
                 min_samples=10,
                 max_window_size=10000):
        """
        Initialize the drift detector.
        
        Parameters:
        -----------
        window_size : int, default=100
            The size of the sliding window to maintain.
        alpha : float, default=0.05
            The significance level for the statistical test (drift threshold).
        warning_threshold : float, default=0.10
            The warning level threshold (should be > alpha).
        min_samples : int, default=10
            The minimum number of samples required in each half to perform the test.
        max_window_size : int, default=10000
            Maximum number of instances to store before forcing a window reset.
        """
        if window_size <= 1:
            raise ValueError("window_size must be greater than 1")
        if alpha <= 0 or alpha >= 1:
            raise ValueError("alpha must be between 0 and 1")
        if warning_threshold <= alpha or warning_threshold >= 1:
            raise ValueError("warning_threshold must be between alpha and 1")
        if min_samples < 2:
            raise ValueError("min_samples must be at least 2")
            
        self.window_size = window_size
        self.alpha = alpha
        self.warning_threshold = warning_threshold
        self.min_samples = min_samples
        self.max_window_size = max_window_size
        
        # Configure logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize the detector
        self.reset()

    def reset(self):
        """Reset the detector to its initial state."""
        self.history = []
        self.num_instances = 0
        self.p_value = None
        self.drift_detected = False
        self.warning_detected = False
        self.warning_count = 0
        self.test_statistic = None

    def update(self, y_true, y_pred=None):
        """
        Update the detector with a new observation.
        
        Parameters:
        -----------
        y_true : float or convertible to float
            The true value to add to the window.
        y_pred : any, default=None
            Prediction value (not used in this implementation but included for API consistency).
            
        Returns:
        --------
        str
            'drift', 'warning', or 'no_drift' indicating the current status.
        """
        # Reset status flags at the beginning of the update
        self.drift_detected = False
        self.warning_detected = False
        
        # Handle non-numeric or None values
        if y_true is None:
            self.logger.debug("Ignoring None value")
            return 'no_drift'
            
        try:
            y_true = float(y_true)
        except (ValueError, TypeError):
            # self.logger.warning(f"Ignoring non-numeric value: {y_true}")
            return 'no_drift'
            
        if np.isnan(y_true) or np.isinf(y_true):
            # self.logger.debug(f"Ignoring NaN or Inf value: {y_true}")
            return 'no_drift'

        # Update history with the new data point
        self.history.append(y_true)
        self.num_instances += 1
        
        # Maintain the window size
        if len(self.history) > self.window_size:
            self.history.pop(0)
        
        # Check if we have enough data for detection
        if len(self.history) < self.window_size:
            return 'no_drift'
        
        # Split the history into two halves
        mid_point = len(self.history) // 2
        first_half = np.array(self.history[:mid_point], dtype=float)
        second_half = np.array(self.history[mid_point:], dtype=float)
        
        # Check if there are enough valid samples in each half
        valid_first = first_half[~np.isnan(first_half) & ~np.isinf(first_half)]
        valid_second = second_half[~np.isnan(second_half) & ~np.isinf(second_half)]
        
        if len(valid_first) < self.min_samples or len(valid_second) < self.min_samples:
            self.logger.debug(f"Not enough valid samples for test: {len(valid_first)} and {len(valid_second)}")
            return 'no_drift'
        
        try:
            # Perform the Wilcoxon rank-sum test
            stat, p_value = mannwhitneyu(valid_first, valid_second, alternative='two-sided')
            self.p_value = p_value
            self.test_statistic = stat
            
            # Drift detection logic
            if p_value < self.alpha:
                # Drift detected
                self.drift_detected = True
                self.warning_detected = False
                self.warning_count = 0
                
                # Log the drift detection
                self.logger.info(f"Drift detected: p-value={p_value}, alpha={self.alpha}")
                
                # Optional: reset after drift to start fresh
                # self.reset()
                return 'drift'
                
            elif p_value < self.warning_threshold:
                # Warning zone
                self.warning_detected = True
                self.drift_detected = False
                self.warning_count += 1
                
                self.logger.debug(f"Warning detected: p-value={p_value}, warning_threshold={self.warning_threshold}")
                return 'warning'
                
            else:
                # In-control
                self.warning_detected = False
                self.drift_detected = False
                self.warning_count = 0
                return 'no_drift'
                
        except Exception as e:
            self.logger.error(f"Error performing statistical test: {e}")
            return 'no_drift'
        
        # Check for max window size exceeded
        if self.num_instances >= self.max_window_size:
            self.logger.info(f"Max window size ({self.max_window_size}) exceeded, resetting detector")
            self.reset()
            
        return 'no_drift'
        
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
            "p_value": self.p_value,
            "test_statistic": self.test_statistic,
            "window_size": self.window_size,
            "current_window_size": len(self.history),
            "num_instances": self.num_instances,
            "alpha": self.alpha,
            "warning_threshold": self.warning_threshold,
            "warning_count": self.warning_count
        }