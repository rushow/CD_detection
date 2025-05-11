import math

class EWMADriftDetector:
    def __init__(self, min_instances=30, lambda_=0.2):
        """
        EWMA Drift Detector
        
        Parameters:
        - min_instances: Minimum number of instances before starting to detect drift.
        - lambda_: Smoothing factor for the moving average. (0 < lambda_ <= 1)
        """
        self.min_instances = min_instances
        self.lambda_ = lambda_
        self.reset()

    def reset(self):
        """
        Reset the EWMA Drift Detector.
        """
        self.num_instances = 1
        self.m_sum = 0.0
        self.m_p = 0.0
        self.m_s = 0.0
        self.z_t = 0.0
        self.drift_detected = False

    def update(self, y_pred, y_true):
        """
        Update the EWMA with new prediction result and check for drift.
        
        Parameters:
        - y_pred: Predicted value (binary).
        - y_true: True value (binary).
        
        Returns:
        - 'drift' if drift is detected, 'no_drift' otherwise.
        """
        # Convert prediction correctness to binary (1 for incorrect, 0 for correct)
        pr = 1 if y_pred != y_true else 0

        # Update the cumulative sum and proportion of positives
        self.m_sum += pr
        self.m_p = self.m_sum / self.num_instances

        # Update standard deviation estimate
        self.m_s = math.sqrt(
            self.m_p * (1.0 - self.m_p) * self.lambda_ * 
            (1.0 - math.pow(1.0 - self.lambda_, 2.0 * self.num_instances)) / (2.0 - self.lambda_)
        )

        # Update the EWMA (z_t)
        self.z_t += self.lambda_ * (pr - self.z_t)
        self.num_instances += 1

        # Calculate the L_t control limit
        L_t = (
            3.97 - 6.56 * self.m_p + 48.73 * math.pow(self.m_p, 3) - 
            330.13 * math.pow(self.m_p, 5) + 848.18 * math.pow(self.m_p, 7)
        )

        # Detect drift or warning
        if self.num_instances < self.min_instances:
            return 'no_drift'

        if self.z_t > self.m_p + L_t * self.m_s:
            self.drift_detected = True
            return 'drift'
        elif self.z_t > self.m_p + 0.5 * L_t * self.m_s:
            return 'warning'
        else:
            self.drift_detected = False
            return 'no_drift'

