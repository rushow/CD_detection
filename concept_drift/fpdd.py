# concept_drift/fpdd.py
import numpy as np
from scipy.stats import fisher_exact

class FPDDDriftDetector:
    def __init__(self, window_size=30, alpha=0.05):
        """
        Fisher Proportions Drift Detector (FPDD)

        :param window_size: Size of each sliding window for monitoring errors.
        :param alpha: Significance level for Fisher's Exact Test.
        """
        self.window_size = window_size
        self.alpha = alpha
        self.window1_errors = []
        self.window2_errors = []
        self.drift_detected = False

    def update(self, y_pred, y_true):
        """
        Update the FPDD with new predictions and check for drift.

        :param y_pred: Predicted label (0 or 1)
        :param y_true: True label (0 or 1)
        :return: None
        """
        # Calculate the number of errors (1 for error, 0 for correct prediction)
        error = int(y_pred != y_true)

        # Update window2_errors
        self.window2_errors.append(error)

        # If window2 is full, move the oldest errors to window1
        if len(self.window2_errors) > self.window_size:
            if len(self.window1_errors) < self.window_size:
                self.window1_errors.append(self.window2_errors.pop(0))
            else:
                self.window1_errors.pop(0)
                self.window1_errors.append(self.window2_errors.pop(0))

        # Reset drift_detected flag
        self.drift_detected = False

        # Proceed only if both windows are full
        if len(self.window1_errors) == self.window_size and len(self.window2_errors) == self.window_size:
            # Create contingency table
            table = [
                [sum(self.window1_errors), self.window_size - sum(self.window1_errors)],
                [sum(self.window2_errors), self.window_size - sum(self.window2_errors)]
            ]

            # Perform Fisher's Exact Test
            _, p_value = fisher_exact(table, alternative='two-sided')

            # Determine drift based on p-value
            if p_value < self.alpha:
                self.drift_detected = True

