import math
import sys

class RDDMDriftDetector:
    def __init__(self, min_instances=129,
                 warning_threshold=1.773,
                 drift_threshold=2.258,
                 max_concept_size=40000,
                 min_stable_size=7000,
                 warning_limit=1400):
        self.min_instances = min_instances
        self.warning_threshold = warning_threshold
        self.drift_threshold = drift_threshold
        self.max_concept_size = max_concept_size
        self.min_stable_size = min_stable_size
        self.warning_limit = warning_limit
        self.reset()

    def reset(self):
        self.num_instances = 0
        self.error_rate_mean = 0
        self.error_rate_std = 0
        self.min_error_rate = sys.maxsize
        self.min_error_rate_std = sys.maxsize
        self.min_error_rate_plus_std = sys.maxsize
        self.predictions = [0] * self.min_stable_size
        self.prediction_pos = 0
        self.warning_count = 0
        self.rddm_drift = False
        self.drift_detected = False
        self.warning_detected = False

    def update(self, prediction, true_label):
        # Calculate error: 1 for incorrect prediction, 0 for correct prediction
        error = 1 if prediction != true_label else 0
        self.predictions[self.prediction_pos] = error
        self.prediction_pos = (self.prediction_pos + 1) % self.min_stable_size

        self.num_instances += 1

        # Update running mean and standard deviation of error rate
        if self.num_instances > 1:
            self.error_rate_mean += (error - self.error_rate_mean) / self.num_instances
        else:
            self.error_rate_mean = error

        if self.num_instances > 1:
            self.error_rate_std = math.sqrt(self.error_rate_mean * (1 - self.error_rate_mean) / self.num_instances)

        error_rate_plus_std = self.error_rate_mean + self.error_rate_std

        # Check and update minimum error rate plus standard deviation
        if error_rate_plus_std < self.min_error_rate_plus_std:
            self.min_error_rate = self.error_rate_mean
            self.min_error_rate_std = self.error_rate_std
            self.min_error_rate_plus_std = error_rate_plus_std

        # Drift detection logic
        if self.num_instances >= self.min_instances:
            if error_rate_plus_std > self.min_error_rate + self.drift_threshold * self.min_error_rate_std:
                # Drift detected
                self.rddm_drift = True
                self.drift_detected = True
                self.warning_detected = False
                self.reset()  # Reset stats after drift is detected
                return 'drift'

            if error_rate_plus_std > self.min_error_rate + self.warning_threshold * self.min_error_rate_std:
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
