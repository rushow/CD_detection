import math
from collections import deque

class Bucket:
    """Helper class to store bucket information"""
    def __init__(self):
        self.total = 0.0
        self.variance = 0.0
        self.width = 0

class ADWINDriftDetector:
    def __init__(self, delta=0.002, clock=32, max_buckets=5, min_window_length=50, grace_period=10):
  
        self.delta = delta
        self.clock = clock
        self.max_buckets = max_buckets
        self.min_window_length = min_window_length
        self.grace_period = grace_period
        self.reset()

    def reset(self):
        """Reset the ADWIN detector"""
        self.drift_detected = False
        self._drift_detected = False
        self.n_detections = 0
        
        # Window statistics
        self.width = 0
        self.total = 0.0
        self.variance = 0.0
        
        # Buckets to compress window data
        self.buckets = deque()
        
        # Clock for periodic checking
        self._clock_counter = 0
        self._n = 0  # Total number of elements seen

    def _compress(self):
        """Compress buckets by merging when there are too many of the same size"""
        # Group buckets by width
        bucket_sizes = {}
        for i, bucket in enumerate(self.buckets):
            if bucket.width not in bucket_sizes:
                bucket_sizes[bucket.width] = []
            bucket_sizes[bucket.width].append(i)
        
        # Merge buckets if there are more than max_buckets of same size
        for width, indices in bucket_sizes.items():
            while len(indices) > self.max_buckets:
                # Merge first two buckets of this size
                idx1, idx2 = indices[0], indices[1]
                b1, b2 = self.buckets[idx1], self.buckets[idx2]
                
                # Create merged bucket
                new_bucket = Bucket()
                new_bucket.width = b1.width + b2.width
                new_bucket.total = b1.total + b2.total
                new_bucket.variance = b1.variance + b2.variance
                
                # Remove old buckets and add new one
                # Remove in reverse order to maintain indices
                if idx2 > idx1:
                    del self.buckets[idx2]
                    del self.buckets[idx1]
                else:
                    del self.buckets[idx1]
                    del self.buckets[idx2]
                
                self.buckets.appendleft(new_bucket)
                
                # Recalculate indices
                bucket_sizes[width] = [i for i, b in enumerate(self.buckets) if b.width == width]
                indices = bucket_sizes[width]

    def _check_drift(self):
        """Check for concept drift between subwindows"""
        if self.width < 2 * self.min_window_length:
            return False
        
        # Try different split points
        n0 = 0  # Width of W0
        sum0 = 0.0  # Sum of W0
        var0 = 0.0  # Variance of W0
        
        for i in range(len(self.buckets) - 1):
            bucket = self.buckets[i]
            n0 += bucket.width
            sum0 += bucket.total
            var0 += bucket.variance
            
            if n0 < self.min_window_length:
                continue
            
            n1 = self.width - n0  # Width of W1
            if n1 < self.min_window_length:
                continue
            
            sum1 = self.total - sum0  # Sum of W1
            var1 = self.variance - var0  # Variance of W1
            
            # Calculate means
            mean0 = sum0 / n0
            mean1 = sum1 / n1
            
            # Calculate cut threshold
            m = 1.0 / (1.0 / n0 + 1.0 / n1)
            dd = math.log(2 * math.log(self.width) / self.delta)
            epsilon = math.sqrt(2 * m * var0 / n0 * dd) + 2.0 / 3 * m * dd
            
            # Check if means are significantly different
            if abs(mean0 - mean1) > epsilon:
                return True
        
        return False

    def update(self, prediction, true_label):

        if self.drift_detected:
            self.reset()
        
        self._n += 1
        
        # Convert to error (1 for incorrect, 0 for correct)
        error = 1 if prediction != true_label else 0
        
        # Update window statistics
        self.width += 1
        self.total += error
        
        # Update variance incrementally
        if self.width > 1:
            mean_old = (self.total - error) / (self.width - 1)
            mean_new = self.total / self.width
            self.variance += (error - mean_old) * (error - mean_new)
        
        # Add to newest bucket
        new_bucket = Bucket()
        new_bucket.width = 1
        new_bucket.total = error
        new_bucket.variance = 0.0
        self.buckets.append(new_bucket)
        
        # Compress buckets if needed
        self._compress()
        
        # Check for drift periodically
        self._clock_counter += 1
        if self._clock_counter % self.clock == 0 and self._n >= self.grace_period:
            if self._check_drift():
                self._drift_detected = True
                self.drift_detected = True
                self.n_detections += 1
                
                # Remove oldest buckets (W0) and keep W1
                n_remove = 0
                sum_remove = 0.0
                var_remove = 0.0
                
                for i in range(len(self.buckets)):
                    if n_remove >= self.width // 2:
                        break
                    bucket = self.buckets[i]
                    n_remove += bucket.width
                    sum_remove += bucket.total
                    var_remove += bucket.variance
                
                # Remove oldest buckets
                for _ in range(i):
                    if len(self.buckets) > 0:
                        self.buckets.popleft()
                
                # Update window statistics
                self.width -= n_remove
                self.total -= sum_remove
                self.variance -= var_remove
                
                return 'drift'
        
        return 'no_drift'

    @property
    def estimation(self):
        """Estimate of mean value in the window"""
        if self.width == 0:
            return 0.0
        return self.total / self.width
