class CUSUMDriftDetector:
    def __init__(self, min_instances=30, delta=0.005, threshold=50):
        self.min_instances = min_instances
        self.delta = delta
        self.threshold = threshold
        self.reset()

    def reset(self):
        self.num_instances = 0
        self.mean = 0
        self.cusum = 0
        self.drift_detected = False

    def update(self, y_pred, y_true):
        # Convert predictions to binary if necessary (1 for incorrect, 0 for correct)
        pr = 1 if y_pred != y_true else 0

        # Increment the instance counter
        self.num_instances += 1

        # Update the mean
        self.mean += (pr - self.mean) / self.num_instances
        
        # Update the CUSUM
        self.cusum = max(0, self.cusum + pr - self.mean - self.delta)

        # Check for drift after minimum instances
        if self.num_instances >= self.min_instances:
            if self.cusum > self.threshold:
                self.drift_detected = True
                return 'drift'
            else:
                self.drift_detected = False
                return 'no_drift'
        else:
            return 'no_drift'

    def reset(self):
        self.num_instances = 0
        self.mean = 0
        self.cusum = 0
        self.drift_detected = False
