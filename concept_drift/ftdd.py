# import numpy as np
# from scipy.stats import fisher_exact

# class FTDDDriftDetector:
#     def __init__(self, window_size=100, p_value_threshold=0.05):
#         self.window_size = window_size
#         self.p_value_threshold = p_value_threshold
#         self.reset()

#     def reset(self):
#         self.reference_window = []
#         self.current_window = []
#         self.drift_detected = False

#     def update(self, prediction, true_label):
#         error = 1 if prediction != true_label else 0
#         self.current_window.append(error)

#         # If current window is full, perform the Fisher's exact test
#         if len(self.current_window) == self.window_size:
#             if len(self.reference_window) < self.window_size:
#                 # Populate the reference window if not yet full
#                 self.reference_window.extend(self.current_window)
#             else:
#                 # Perform Fisher's exact test
#                 contingency_table = self._create_contingency_table()
#                 _, p_value = fisher_exact(contingency_table)

#                 # Check if drift is detected
#                 if p_value < self.p_value_threshold:
#                     self.drift_detected = True
#                     self.reset()  # Reset windows on drift detection
#                     return 'drift'
#                 else:
#                     self.drift_detected = False
#                     self.reference_window = self.current_window.copy()

#             # Reset current window
#             self.current_window = []

#         return 'no_drift'

#     def _create_contingency_table(self):
#         # Calculate the number of errors in both windows
#         ref_errors = sum(self.reference_window)
#         curr_errors = sum(self.current_window)

#         # Create a 2x2 contingency table
#         ref_correct = self.window_size - ref_errors
#         curr_correct = self.window_size - curr_errors

#         contingency_table = np.array([[ref_correct, ref_errors], [curr_correct, curr_errors]])
#         return contingency_table




import numpy as np
from scipy.stats import fisher_exact
from typing import Tuple, Optional, List
from dataclasses import dataclass, field

@dataclass
class FTDDStats:
    p_value: float = 1.0
    drift_count: int = 0
    warning_count: int = 0
    error_rate_reference: float = 0.0
    error_rate_current: float = 0.0

class FTDDDriftDetector:
 
    def __init__(
        self, 
        window_size: int = 100,
        p_value_threshold: float = 0.05,
        warning_threshold: float = 0.1,
        min_window_size: int = 30
    ):
        """
        Initialize the FTDD drift detector.
        
        Args:
            window_size: Size of the reference and detection windows
            p_value_threshold: Threshold for drift detection
            warning_threshold: Threshold for warning level
            min_window_size: Minimum samples before testing for drift
        """
        self.window_size = window_size
        self.p_value_threshold = p_value_threshold
        self.warning_threshold = warning_threshold
        self.min_window_size = min_window_size
        self.stats = FTDDStats()
        self.reset()

    def reset(self) -> None:
        """Reset the detector state."""
        self.reference_window: List[int] = []
        self.current_window: List[int] = []
        self.drift_detected = False
        self.warning_zone = False
        
    def get_current_stats(self) -> FTDDStats:
        """Return current detection statistics."""
        return self.stats

    def _calculate_error_rates(self) -> Tuple[float, float]:
        """Calculate error rates for both windows."""
        ref_errors = sum(self.reference_window)
        curr_errors = sum(self.current_window)
        
        ref_rate = ref_errors / len(self.reference_window) if self.reference_window else 0
        curr_rate = curr_errors / len(self.current_window) if self.current_window else 0
        
        return ref_rate, curr_rate

    def _create_contingency_table(self) -> np.ndarray:
        """Create contingency table for Fisher's exact test."""
        ref_errors = sum(self.reference_window)
        curr_errors = sum(self.current_window)
        
        ref_correct = len(self.reference_window) - ref_errors
        curr_correct = len(self.current_window) - curr_errors
        
        return np.array([[ref_correct, ref_errors], 
                        [curr_correct, curr_errors]])

    def _check_for_drift(self) -> Tuple[bool, float]:
        """
        Perform Fisher's exact test and check for drift.
        
        Returns:
            Tuple of (drift_detected, p_value)
        """
        contingency_table = self._create_contingency_table()
        _, p_value = fisher_exact(contingency_table)
        
        self.stats.p_value = p_value
        self.stats.error_rate_reference, self.stats.error_rate_current = self._calculate_error_rates()
        
        drift_detected = p_value < self.p_value_threshold
        warning_zone = self.p_value_threshold <= p_value < self.warning_threshold
        
        if drift_detected:
            self.stats.drift_count += 1
        if warning_zone:
            self.stats.warning_count += 1
            
        return drift_detected, p_value

    def update(self, prediction: any, true_label: any) -> str:
        """
        Update the detector with a new sample.
        
        Args:
            prediction: The predicted value
            true_label: The true value
            
        Returns:
            str: Status ('drift', 'warning', or 'normal')
        """
        error = 1 if prediction != true_label else 0
        self.current_window.append(error)
        
        # Early drift detection with minimum window size
        if (len(self.current_window) >= self.min_window_size and 
            len(self.reference_window) >= self.min_window_size):
            
            drift_detected, p_value = self._check_for_drift()
            
            if drift_detected:
                self.drift_detected = True
                self.warning_zone = False
                self.reset()
                return 'drift'
            
            elif p_value < self.warning_threshold:
                self.warning_zone = True
                return 'warning'
        
        # Handle full window
        if len(self.current_window) == self.window_size:
            if len(self.reference_window) < self.window_size:
                self.reference_window.extend(self.current_window)
            else:
                drift_detected, _ = self._check_for_drift()
                
                if drift_detected:
                    self.drift_detected = True
                    self.reset()
                    return 'drift'
                else:
                    self.reference_window = self.current_window.copy()
            
            self.current_window = []
            
        return 'normal'
