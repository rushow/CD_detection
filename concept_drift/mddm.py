import math

# class MDDMDriftDetector:
#     def __init__(self, n=100, difference=0.01, delta=0.000001):
#         self.win = []
#         self.n = n
#         self.difference = difference
#         self.delta = delta

#         self.e = math.sqrt(0.5 * self.cal_sigma() * (math.log(1 / self.delta, math.e)))
#         self.u_max = 0
#         self.drift_detected = False

#     def update(self, prediction, true_label):
#         error = 1 if prediction != true_label else 0

#         # Add error to the window
#         if len(self.win) == self.n:
#             self.win.pop(0)
#         self.win.append(error)

#         drift_status = False

#         if len(self.win) == self.n:
#             u = self.cal_w_sigma()
#             self.u_max = max(u, self.u_max)
#             drift_status = self.u_max - u > self.e

#         self.drift_detected = drift_status  # Update drift_detected based on drift status
#         return self.drift_detected

#     def reset(self):
#         self.win.clear()
#         self.u_max = 0
#         self.drift_detected = False  # Reset drift_detected

#     def cal_sigma(self):
#         sum_, sigma = 0, 0
#         for i in range(self.n):
#             sum_ += (1 + i * self.difference)
#         for i in range(self.n):
#             sigma += math.pow((1 + i * self.difference) / sum_, 2)
#         return sigma

#     def cal_w_sigma(self):
#         total_sum, win_sum = 0, 0
#         for i in range(self.n):
#             total_sum += 1 + i * self.difference
#             win_sum += self.win[i] * (1 + i * self.difference)
#         return win_sum / total_sum

#     def get_settings(self):
#         settings = [f'{self.n}.{self.delta}',
#                     f'$n$:{self.n}, $d$:{self.difference}, $\\delta$:{self.delta}']
#         return settings



class MDDMDriftDetector:
    def __init__(self, window_size=50, confidence_level=0.05):
        """
        Initialize the MDDM detector with a sliding window size and confidence level.
        
        Parameters:
        - window_size (int): The size of the sliding window.
        - confidence_level (float): Confidence threshold for drift detection.
        """
        self.window_size = window_size
        self.confidence_level = confidence_level
        self.scores = []  # To store scores (1 if error, 0 if correct)
        self.max_mean_score = 0  # Maximum mean score within the window
        self.n_errors = 0  # Counter for errors within the window
        self.drift_detected = False  # Attribute to track drift detection status

    def mcdiarmid_bound(self):
        """
        Calculate the McDiarmid bound based on window size and confidence level.
        
        Returns:
        - bound (float): The calculated McDiarmid bound.
        """
        return math.sqrt((1 / (2 * self.window_size)) * math.log(1 / self.confidence_level))

    def update(self, prediction, true_label):
        """
        Update the detector with the latest instance prediction and check for drift.
        
        Parameters:
        - prediction: Predicted label for the instance.
        - true_label: True label for the instance.
        
        Returns:
        - drift_detected (bool): True if drift is detected, False otherwise.
        """
        is_error = int(prediction != true_label)  # 1 if error, 0 if correct
        self.scores.append(is_error)

        # Maintain the sliding window size
        if len(self.scores) > self.window_size:
            oldest_score = self.scores.pop(0)
            if oldest_score == 1:
                self.n_errors -= 1  # Adjust error count if removing an error

        # Update error count and calculate current mean score
        if is_error == 1:
            self.n_errors += 1
        mean_score = sum(self.scores) / self.window_size

        # Update maximum mean score if the current mean is higher
        if mean_score > self.max_mean_score:
            self.max_mean_score = mean_score

        # Calculate McDiarmid bound and check for drift
        bound = self.mcdiarmid_bound()
        if self.max_mean_score - mean_score > bound:
            # Drift detected, set drift_detected flag and reset detector
            self.drift_detected = True
            self.reset()
        else:
            self.drift_detected = False
        return self.drift_detected

    def reset(self):
        """
        Reset the detector's state after detecting drift.
        """
        self.scores = []
        self.max_mean_score = 0
        self.n_errors = 0
