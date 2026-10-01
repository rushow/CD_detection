import math

class EDDMDriftDetector:
    def __init__(self, warm_start=30, alpha=0.95, beta=0.9, window_size=50):

        if alpha < beta:
            raise ValueError("'alpha' must be greater or equal to 'beta'.")
        
        self.warm_start = warm_start
        self.alpha = alpha
        self.beta = beta
        self.window_size = window_size
        self.reset()

    def reset(self):
        """Reset the EDDM detector"""
        self.drift_detected = False
        self.warning_detected = False
        
        self.n = 0  # Total number of observations
        
        # Two sliding windows for comparison
        self.window1 = []  # Reference window
        self.window2 = []  # Current window
        
        # Track error positions for distance calculation in each window
        self.window1_error_positions = []
        self.window2_error_positions = []

    def _calculate_distances_stats(self, error_positions):
        """Calculate mean and std of distances between errors in a window"""
        if len(error_positions) < 2:
            return 0, 0
        
        # Calculate distances between consecutive errors
        distances = []
        for i in range(1, len(error_positions)):
            distances.append(error_positions[i] - error_positions[i-1])
        
        if len(distances) == 0:
            return 0, 0
        
        # Calculate mean
        mean_distance = sum(distances) / len(distances)
        
        # Calculate standard deviation
        if len(distances) > 1:
            variance = sum((d - mean_distance) ** 2 for d in distances) / len(distances)
            std_distance = math.sqrt(max(0, variance))
        else:
            std_distance = 0
        
        return mean_distance, std_distance

    def update(self, prediction, true_label):

        if self.drift_detected:
            self.reset()
        
        # Update the sample counter
        self.n += 1
        
        # Check if there's an error
        error = 1 if prediction != true_label else 0
        
        # Add to both windows
        self.window1.append(error)
        self.window2.append(error)
        
        # Track error positions (relative to window start)
        if error == 1:
            self.window1_error_positions.append(len(self.window1) - 1)
            self.window2_error_positions.append(len(self.window2) - 1)
        
        # Remove oldest from window1 if it exceeds size
        if len(self.window1) > self.window_size:
            removed_error = self.window1.pop(0)
            # Adjust error positions and remove if necessary
            if removed_error == 1 and len(self.window1_error_positions) > 0:
                self.window1_error_positions.pop(0)
            # Shift all positions down by 1
            self.window1_error_positions = [pos - 1 for pos in self.window1_error_positions]
        
        # Remove oldest from window2 if it exceeds size
        if len(self.window2) > self.window_size:
            removed_error = self.window2.pop(0)
            # Adjust error positions and remove if necessary
            if removed_error == 1 and len(self.window2_error_positions) > 0:
                self.window2_error_positions.pop(0)
            # Shift all positions down by 1
            self.window2_error_positions = [pos - 1 for pos in self.window2_error_positions]
        
        # Only proceed if both windows are full
        if len(self.window1) < self.window_size or len(self.window2) < self.window_size:
            return 'no_drift'
        
        # Calculate statistics for window1 (reference window)
        mean1, std1 = self._calculate_distances_stats(self.window1_error_positions)
        p2s_prime1 = mean1 + 2 * std1
        
        # Calculate statistics for window2 (current window)
        mean2, std2 = self._calculate_distances_stats(self.window2_error_positions)
        p2s_prime2 = mean2 + 2 * std2
        
        # Need enough errors to make comparison
        if len(self.window1_error_positions) < 2 or len(self.window2_error_positions) < 2:
            return 'no_drift'
        
        # Only proceed if we have enough data
        if self.n < self.warm_start:
            return 'no_drift'
        
        # Compare window2 against window1 (reference)
        if p2s_prime1 > 0:
            level = p2s_prime2 / p2s_prime1
            
            if level < self.beta:
                # Drift detected
                self.drift_detected = True
                self.warning_detected = False
                self.reset()
                return 'drift'
            elif level < self.alpha:
                # Warning zone
                self.warning_detected = True
                self.drift_detected = False
                return 'warning'
            else:
                # No drift
                self.warning_detected = False
                self.drift_detected = False
        
        return 'no_drift'
