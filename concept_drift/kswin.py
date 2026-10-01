import random
import math
from collections import deque
from scipy import stats
import warnings

class KSWINDriftDetector:
    def __init__(self, alpha=0.005, window_size=50, stat_size=20, seed=None):

        if alpha < 0 or alpha > 1:
            raise ValueError("Alpha must be between 0 and 1.")
        
        if window_size < 0:
            raise ValueError("window_size must be greater than 0.")
        
        if window_size < stat_size:
            raise ValueError("stat_size must be smaller than window_size.")
        
        if window_size < 2 * stat_size:
            raise ValueError(f"window_size ({window_size}) must be at least 2 * stat_size ({stat_size}). "
                           f"Minimum required window_size: {2 * stat_size}")
        
        self.alpha = alpha
        self.window_size = window_size
        self.stat_size = stat_size
        self.seed = seed
        self.reset()

    def reset(self):
        """Reset the KSWIN detector"""
        self.drift_detected = False
        self.p_value = 0
        self.n = 0
        self.window = deque(maxlen=self.window_size)
        self._rng = random.Random(self.seed)

    def update(self, prediction, true_label):

        if self.drift_detected:
            self.reset()
        
        self.n += 1
        
        # Convert to error (1 for incorrect, 0 for correct)
        error = 1 if prediction != true_label else 0
        
        # Add to sliding window
        self.window.append(error)
        
        # Perform KS-test when window is full
        if len(self.window) >= self.window_size:
            # Sample r elements uniformly from first (n-r) samples
            sample_range = self.window_size - self.stat_size
            rnd_indices = self._rng.sample(range(sample_range), self.stat_size)
            rnd_window = [self.window[i] for i in rnd_indices]
            
            # Get last r samples (most recent concept)
            most_recent = list(self.window)[self.window_size - self.stat_size:]
            
            # Perform Kolmogorov-Smirnov test
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                st, self.p_value = stats.ks_2samp(rnd_window, most_recent, method="auto")
            
            # Check for drift
            if self.p_value <= self.alpha and st > 0.1:
                self._drift_detected = True
                self.drift_detected = True
                # Keep only most recent samples in window
                self.window = deque(most_recent, maxlen=self.window_size)
                return 'drift'
            else:
                self._drift_detected = False
                self.drift_detected = False
        else:
            # Not enough samples for valid test
            self._drift_detected = False
            self.drift_detected = False
        
        return 'no_drift'
