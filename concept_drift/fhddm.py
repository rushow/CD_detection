import math
from collections import deque

class FHDDMDriftDetector:
    def __init__(self, sliding_window_size=50, confidence_level=0.000001, short_window_size=None):

        self.sliding_window_size = sliding_window_size
        self.confidence_level = confidence_level
        self.short_window_size = short_window_size
        self.n_one = 0
        self.reset()

    def reset(self):
        """Reset the FHDDM detector"""
        self.drift_detected = False
        self._sliding_window = deque(maxlen=self.sliding_window_size)
        
        # Calculate epsilon for Hoeffding's inequality
        self._epsilon = math.sqrt(
            (math.log(1 / self.confidence_level)) / (2 * self.sliding_window_size)
        )
        
        self._u_max = 0  # Maximum mean observed in long window
        self.n_one = 0  # Count of 1's (errors) in sliding window
        
        # Initialize short window for Stacking FHDDM if specified
        if self.short_window_size is not None:
            self._short_window = deque(maxlen=self.short_window_size)
            self._u_short_max = 0  # Maximum mean observed in short window
            self._epsilon_s = math.sqrt(
                (math.log(1 / self.confidence_level)) / (2 * self.short_window_size)
            )

    def update(self, prediction, true_label):
        """
        Update the FHDDM with new prediction result and check for drift.
        
        Parameters:
        - prediction: Predicted value
        - true_label: True value
        
        Returns:
        - 'drift' if drift is detected, 'no_drift' otherwise
        """
        if self.drift_detected:
            self.drift_detected = False
            self.reset()
        
        # Convert to error (1 for incorrect, 0 for correct)
        error = 1 if prediction != true_label else 0
        
        # Add to sliding window
        self._sliding_window.append(error)
        self.n_one += error
        
        # Only detect drift when sliding window is full
        if len(self._sliding_window) == self.sliding_window_size:
            # Calculate mean of sliding window
            u = self.n_one / self.sliding_window_size
            
            # Update maximum mean
            self._u_max = u if self._u_max < u else self._u_max
            
            short_win_drift_status = False
            long_win_drift_status = False
            
            # Check short window drift if short_window_size is specified (Stacking FHDDM)
            if self.short_window_size is not None:
                # Calculate mean of the last short_window_size elements
                short_window_start = self.sliding_window_size - self.short_window_size
                u_s = sum(list(self._sliding_window)[short_window_start:]) / self.short_window_size
                
                # Update short window maximum mean
                self._u_short_max = u_s if self._u_short_max < u_s else self._u_short_max
                
                # Check for drift in short window
                short_win_drift_status = (self._u_short_max - u_s) > self._epsilon_s
            
            # Check for drift in long window
            long_win_drift_status = (self._u_max - u) > self._epsilon
            
            # Drift detected if either window detects drift
            self._drift_detected = long_win_drift_status or short_win_drift_status
            
            # Remove oldest element and update count
            oldest = self._sliding_window.popleft()
            self.n_one -= oldest
            
            if self._drift_detected:
                self.drift_detected = True
                return 'drift'
        
        return 'no_drift'
