import math
import sys

class RDDMDriftDetector:
    def __init__(self, min_instances=129,
                 warning_threshold=1.773,
                 drift_threshold=2.258,
                 max_concept_size=40000,
                 min_stable_size=7000,
                 warning_limit=1400,
                 window_size=50):
        self.min_instances = min_instances
        self.warning_threshold = warning_threshold
        self.drift_threshold = drift_threshold
        self.max_concept_size = max_concept_size
        self.min_stable_size = min_stable_size
        self.warning_limit = warning_limit
        self.window_size = window_size
        self.reset()

    def reset(self):
        self.num_instances = 0
        self.warning_count = 0
        self.rddm_drift = False
        self.drift_detected = False
        self.warning_detected = False
        # Two sliding windows for comparison
        self.window1 = []
        self.window2 = []
        self.window1_error_sum = 0
        self.window2_error_sum = 0

    def update(self, prediction, true_label):
        # Calculate error: 1 for incorrect prediction, 0 for correct prediction
        error = 1 if prediction != true_label else 0
        
        self.num_instances += 1

        # Add new error to both windows
        self.window1.append(error)
        self.window1_error_sum += error
        
        self.window2.append(error)
        self.window2_error_sum += error
        
        # Remove oldest error from window1 if it exceeds size
        if len(self.window1) > self.window_size:
            removed_error = self.window1.pop(0)
            self.window1_error_sum -= removed_error
        
        # Remove oldest error from window2 if it exceeds size
        if len(self.window2) > self.window_size:
            removed_error = self.window2.pop(0)
            self.window2_error_sum -= removed_error
        
        # Only proceed if both windows are full
        if len(self.window1) < self.window_size or len(self.window2) < self.window_size:
            return 'no_drift'
        
        # Calculate statistics for window1 (reference window)
        window1_mean = self.window1_error_sum / len(self.window1)
        if window1_mean > 0 and window1_mean < 1:
            window1_std = math.sqrt(window1_mean * (1 - window1_mean) / len(self.window1))
        else:
            window1_std = 0
        
        # Calculate statistics for window2 (current window)
        window2_mean = self.window2_error_sum / len(self.window2)
        if window2_mean > 0 and window2_mean < 1:
            window2_std = math.sqrt(window2_mean * (1 - window2_mean) / len(self.window2))
        else:
            window2_std = 0
        
        # Drift detection: compare window2 statistics against window1
        window2_plus_std = window2_mean + window2_std

        # Drift detection: compare window2 statistics against window1
        window2_plus_std = window2_mean + window2_std

        # Drift detection logic
        if self.num_instances >= self.min_instances:
            if window2_plus_std > window1_mean + self.drift_threshold * window1_std:
                # Drift detected
                self.rddm_drift = True
                self.drift_detected = True
                self.warning_detected = False
                self.reset()  # Reset stats after drift is detected
                return 'drift'

            if window2_plus_std > window1_mean + self.warning_threshold * window1_std:
                # Warning zone
                self.warning_detected = True
                self.drift_detected = False
                self.warning_count += 1
                if self.warning_count >= self.warning_limit:
                    # Drift detected after exceeding warning limit
                    self.rddm_drift = True
                    self.drift_detected = True
                    self.warning_detected = False
                    self.reset()  # Reset stats after drift is detected
                    return 'drift'
                return 'warning'
            else:
                # In-control
                self.warning_detected = False
                self.drift_detected = False
                self.warning_count = 0

        # Check for drift based on max concept size
        if self.num_instances >= self.max_concept_size and not self.warning_detected:
            self.rddm_drift = True
            self.drift_detected = True
            self.reset()  # Reset stats after drift is detected
            return 'drift'

        return 'no_drift'


# import numpy as np
# from capymoa.drift.detectors import RDDM

# class RDDMDriftDetector:
#     def __init__(self):
#         self.detector = RDDM()
#         self.drift_detected = False
#         self.error_count = 0
#         self.sample_count = 0

#     def update(self, prediction, true_label):
#         # Calculate the error
#         error = int(prediction != true_label)
        
#         # Manually keep track of errors
#         self.error_count += error
#         self.sample_count += 1

#         # Check if RDDM has detected change
#         if self.detector.detected_change:
#             self.drift_detected = True
#             self.detector.reset()  # Reset the detector after drift is detected
#             # Reset counters after drift
#             self.error_count = 0
#             self.sample_count = 0
#         else:
#             self.drift_detected = False

#     def check_drift(self):
#         return self.drift_detected
