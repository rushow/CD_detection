# # concept_drift/ace.py

# from river import tree
# import numpy as np

# class ACEDriftDetector:
#     def __init__(self, n_classifiers=5, base_learner=tree.HoeffdingTreeClassifier, seed=None):
#         self.classifiers = [base_learner() for _ in range(n_classifiers)]
#         self.weights = np.ones(n_classifiers)
#         self.drift_detected = False
#         self.seed = seed

#     def reset(self):
#         # Reset each classifier and reinitialize the weights
#         self.classifiers = [self.classifiers[i].clone() for i in range(len(self.classifiers))]
#         self.weights = np.ones(len(self.classifiers))
#         self.drift_detected = False

#     def update(self, X, y):
#         # Ensure X is treated as a dictionary of features
#         if not isinstance(X, dict):
#             X = {f"feature_{i}": X for i in range(1)}

#         # Check for None values in X and replace them with a default value (e.g., 0.0 or mean)
#         X = {k: (v if v is not None else 0.0) for k, v in X.items()}

#         correct_predictions = np.zeros(len(self.classifiers))

#         for i, clf in enumerate(self.classifiers):
#             pred = clf.predict_one(X)
#             correct_predictions[i] = (pred == y)
#             clf.learn_one(X, y)

#         # Calculate the accuracy and update weights
#         total_correct = np.sum(correct_predictions)
        
#         if total_correct > 0:
#             self.weights = correct_predictions / total_correct
#         else:
#             # If no classifier made a correct prediction, reset weights to equal distribution
#             self.weights = np.ones(len(self.classifiers)) / len(self.classifiers)

#         # Determine if drift has occurred
#         self.drift_detected = (total_correct / len(self.classifiers)) < 0.5

#         if self.drift_detected:
#             self.reset_worst_classifier()

#         return 'drift' if self.drift_detected else 'no_drift'

#     def reset_worst_classifier(self):
#         # Identify the worst-performing classifier and reset it
#         worst_index = np.argmin(self.weights)
#         self.classifiers[worst_index] = self.classifiers[worst_index].clone()
#         self.weights[worst_index] = 1.0 / len(self.classifiers)


from river import base, metrics
import numpy as np
from scipy.stats import norm
from copy import deepcopy
from collections import deque

class ACEDriftDetector:
    """
    Adaptive Classifiers Ensemble (ACE) drift detector adapted for River framework
    """
    def __init__(
        self,
        short_term_memory_size=50,
        chunk_size=100,
        confidence_level=0.95,
        adjustment_factor=0.5
    ):
        """
        Initialize ACE drift detector
        
        Parameters:
        -----------
        short_term_memory_size : int
            Size of short-term memory (Sa)
        chunk_size : int
            Size of data chunks (Sc)
        confidence_level : float
            Confidence level for drift detection (1 - alpha)
        adjustment_factor : float
            Adjustment factor for ensemble weights (μ)
        """
        self.Sa = short_term_memory_size
        self.Sc = chunk_size
        self.alpha = 1 - confidence_level
        self.mu = adjustment_factor
        
        # Initialize ensemble variables
        self.J = 0  # Number of batch classifiers
        self.N0 = 0  # Starting point of current chunk
        self.ensemble = []  # List of classifiers
        self.buffer = []  # Buffer for storing instances (Bl)
        
        # Performance tracking
        self.CR = {}  # Correct predictions for each classifier
        self.A = {}   # Accuracy for each classifier
        self.N = {}   # Starting point for each classifier
        
        # Initialize performance tracking for online classifier
        self.CR[0] = deque(maxlen=self.Sa)
        self.A[0] = deque(maxlen=self.Sa)
        self.N[0] = 0
        
        # Drift detection flag
        self.drift_detected = False
        
        # Metrics for evaluation
        self.metric = metrics.Accuracy()
        
        # Current model being used for predictions
        self.current_model = None
    
    def _compute_confidence_interval(self, accuracy, n):
        """
        Compute confidence interval for proportion
        """
        z = norm.ppf(1 - self.alpha/2)
        error = z * np.sqrt(accuracy * (1 - accuracy) / n)
        lower = max(0, accuracy - error)
        upper = min(1, accuracy + error)
        return lower, upper
    
    def _get_ensemble_prediction(self, x):
        """
        Get weighted ensemble prediction and probabilities
        """
        if not self.ensemble:
            return 0, {0: 0.5, 1: 0.5}
        
        votes = {}
        proba_sum = {0: 0.0, 1: 0.0}
        total_weight = 0
        
        for j, clf in enumerate(self.ensemble):
            if clf is not None and (j == 0 or (j in self.A and len(self.A[j]) > 0)):
                try:
                    # Get prediction and probabilities
                    pred = clf.predict_one(x)
                    probs = clf.predict_proba_one(x)
                    
                    # Calculate weight based on recent accuracy
                    weight = 1
                    if j in self.A and len(self.A[j]) > 0:
                        acc = np.mean(list(self.A[j]))
                        if acc < 1:
                            weight = (1 / (1 - acc)) ** self.mu
                    
                    # Accumulate weighted votes
                    if pred not in votes:
                        votes[pred] = 0
                    votes[pred] += weight
                    
                    # Accumulate weighted probabilities
                    for class_label, prob in probs.items():
                        proba_sum[class_label] = proba_sum.get(class_label, 0) + prob * weight
                    total_weight += weight
                except Exception as e:
                    print(f"Warning: Error in classifier {j}: {str(e)}")
                    continue
        
        # Normalize probabilities
        if total_weight > 0:
            for class_label in proba_sum:
                proba_sum[class_label] /= total_weight
        
        # Get most voted prediction
        prediction = max(votes.items(), key=lambda x: x[1])[0] if votes else 0
            
        return prediction, proba_sum
    
    def update(self, prediction, true_label):
        """
        Update the detector with a new sample
        
        Parameters:
        -----------
        prediction : object
            The prediction made by the model
        true_label : object
            The true label
        """
        self.drift_detected = False
        
        # Check if current_model is initialized
        if self.current_model is None:
            return
        
        # Update performance metrics
        correct = int(prediction == true_label)
        if 0 in self.CR:
            self.CR[0].append(correct)
            
            # Update accuracy
            acc = np.mean(list(self.CR[0]))
            self.A[0].append(acc)
        
        # Store instance in buffer
        self.buffer.append((prediction, true_label))
        
        # Check for drift
        acquire_new = False
        
        # Check accuracy drop condition
        if len(self.buffer) - self.N0 >= self.Sa:
            for j in range(1, len(self.ensemble)):
                if j in self.A and len(self.A[j]) >= self.Sa:
                    current_acc = self.A[j][-1]
                    old_acc = self.A[j][-self.Sa]
                    lower, upper = self._compute_confidence_interval(old_acc, self.Sa)
                    
                    if current_acc < lower or current_acc > upper:
                        acquire_new = True
                        break
        
        # Check chunk size condition
        if len(self.buffer) - self.N0 >= self.Sc:
            acquire_new = True
        
        # If drift detected, create new classifier
        if acquire_new:
            self.drift_detected = True
            
            # Create new classifier (will be initialized in main.py)
            self.J += 1
            
            # Reset buffer and tracking
            self.N0 = len(self.buffer)
            self.buffer = []
            
            # Initialize tracking for new classifier
            self.CR[self.J] = deque(maxlen=self.Sa)
            self.A[self.J] = deque(maxlen=self.Sa)
            self.N[self.J] = len(self.buffer)
            
            # Copy recent performance from online classifier
            if 0 in self.CR and 0 in self.A:
                for m in range(min(self.Sa, len(self.CR[0]))):
                    self.CR[self.J].append(self.CR[0][m])
                    self.A[self.J].append(self.A[0][m])
    
    def detect_warning_zone(self):
        """
        Not used in ACE but implemented for compatibility
        """
        return False