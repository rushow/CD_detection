# concept_drift/d3.py

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
import numpy as np
from typing import Union, List, Tuple, Optional, Literal

class D3DriftDetector:
    def __init__(
        self, 
        window_size: int = 20, 
        classifier = None, 
        threshold: float = 0.7,
        metric: Literal['accuracy', 'roc_auc'] = 'accuracy'
    ):
        """
        Initialize the D3 drift detector.
        
        Args:
            window_size: Number of samples in each window
            classifier: Classifier to distinguish between distributions (default: RandomForestClassifier)
            threshold: Detection threshold for the metric
            metric: Performance metric to use ('accuracy' or 'roc_auc')
        """
        self.window_size = window_size
        self.classifier = classifier if classifier is not None else RandomForestClassifier()
        self.threshold = threshold
        self.metric = metric
        self.samples = []
        self.drift_detected = False
        # track feature keys when samples are dict-like (e.g., River)
        self.feature_keys = None
        self._validate_params()
        
    def _to_array(self, samples: List) -> np.ndarray:
        """
        Convert a list of samples to a 2D numpy array.
        Supports samples that are:
          - dict-like (feature name -> value)
          - list/tuple/1D numpy arrays (feature vector)
          - scalar numbers (will be cast to shape (n,1))
        """
        if len(samples) == 0:
            return np.empty((0, 0))

        # If any sample is a dict, construct a feature matrix from dict keys
        if any(isinstance(s, dict) for s in samples):
            # compute union of keys across samples to ensure consistent columns
            keys = sorted(set().union(*(s.keys() for s in samples if isinstance(s, dict))))
            self.feature_keys = keys
            rows = []
            for s in samples:
                if isinstance(s, dict):
                    row = []
                    for k in keys:
                        v = s.get(k, 0.0)
                        try:
                            row.append(float(v))
                        except Exception:
                            # fallback to 0.0 for non-convertible entries
                            row.append(0.0)
                    rows.append(row)
                else:
                    # non-dict sample: try to convert to numeric vector
                    try:
                        arr = np.asarray(s, dtype=float).ravel()
                        # if lengths mismatch, pad/truncate to len(keys)
                        if arr.size < len(keys):
                            padded = np.zeros(len(keys), dtype=float)
                            padded[:arr.size] = arr
                            rows.append(padded.tolist())
                        else:
                            rows.append(arr[:len(keys)].tolist())
                    except Exception:
                        rows.append([0.0] * len(keys))
            return np.asarray(rows, dtype=float)

        # If samples are already sequences/numeric, let numpy handle them
        try:
            arr = np.asarray(samples, dtype=float)
            # If arr is 1D (a list of scalars), convert to column vector
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            return arr
        except Exception:
            # fallback: try converting each sample to numeric vector elementwise
            rows = []
            for s in samples:
                try:
                    row = np.asarray(s, dtype=float).ravel()
                    rows.append(row)
                except Exception:
                    rows.append([0.0])
            # pad rows to equal length
            max_len = max(len(r) for r in rows) if rows else 0
            padded = [list(r) + [0.0] * (max_len - len(r)) for r in rows]
            return np.asarray(padded, dtype=float)
    
    def _validate_params(self) -> None:
        """Validate initialization parameters."""
        if not isinstance(self.window_size, int) or self.window_size <= 0:
            raise ValueError(f"window_size must be a positive integer, got {self.window_size}")
        
        if not (0 < self.threshold < 1):
            raise ValueError(f"threshold must be between 0 and 1, got {self.threshold}")
            
        if self.metric not in ['accuracy', 'roc_auc']:
            raise ValueError(f"metric must be 'accuracy' or 'roc_auc', got {self.metric}")
    
    def update(self, X: Union[np.ndarray, List], y = None) -> str:
        """
        Update the detector with new samples and check for drift.
        
        Args:
            X: New data samples (can be single sample or batch)
            y: Labels (not used for drift detection, included for API consistency)
            
        Returns:
            'drift' if drift is detected, 'no_drift' otherwise
        """
        # Handle both single samples and batches
        if not isinstance(X, list) and not isinstance(X, np.ndarray):
            X = [X]
        
        X_array = np.asarray(X)
        
        # Handle both single sample and batch updates
        if X_array.ndim == 1:
            # Single sample with multiple features
            self.samples.append(X_array)
        else:
            # Batch of samples
            for sample in X_array:
                self.samples.append(sample)
        
        # Check if we have enough samples to detect drift
        if len(self.samples) >= 2 * self.window_size:
            return self._detect_drift()
        
        return 'no_drift'
    
    def _detect_drift(self) -> str:
        """Perform drift detection using the classifier."""
        # Convert collected samples to numeric arrays (handles dict-like samples)
        try:
            samples_array = self._to_array(self.samples)
        except Exception as e:
            print(f"Error converting samples to array for drift detection: {e}")
            self.drift_detected = False
            return 'no_drift'

        # Create past and current windows
        X_past = samples_array[:self.window_size]
        X_current = samples_array[-self.window_size:]
        
        # Label past samples as 1, current as 0
        y_past = np.ones(self.window_size)
        y_current = np.zeros(self.window_size)
        
        # Combine windows
        X_combined = np.vstack((X_past, X_current))
        y_combined = np.concatenate((y_past, y_current))
        
        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_combined, y_combined, test_size=0.3, random_state=42, stratify=y_combined
        )
        
        # Train the classifier
        try:
            self.classifier.fit(X_train, y_train)
            
            # Predict probabilities if using ROC AUC
            if self.metric == 'roc_auc':
                y_pred_proba = self.classifier.predict_proba(X_test)[:, 1]
                performance = roc_auc_score(y_test, y_pred_proba)
            else:  # accuracy
                y_pred = self.classifier.predict(X_test)
                performance = accuracy_score(y_test, y_pred)
            
            # Determine if drift has occurred
            self.drift_detected = performance > self.threshold
            
        except Exception as e:
            print(f"Error during drift detection: {e}")
            self.drift_detected = False
        
        # Slide the window (keep only the most recent window_size samples)
        self.samples = self.samples[-self.window_size:]
        
        return 'drift' if self.drift_detected else 'no_drift'
    
    def reset(self) -> None:
        """Reset the detector's state."""
        self.samples = []
        self.drift_detected = False
    
    @property
    def is_drift_detected(self) -> bool:
        """Return whether drift is currently detected."""
        return self.drift_detected
        
    def get_status(self) -> dict:
        """Get the current status of the detector."""
        return {
            'samples_collected': len(self.samples),
            'window_size': self.window_size,
            'threshold': self.threshold,
            'drift_detected': self.drift_detected,
            'ready': len(self.samples) >= 2 * self.window_size
        }