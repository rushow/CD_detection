# # concept_drift/aue.py

# import numpy as np
# from collections import deque
# from river import tree

# class AUEDriftDetector:
#     def __init__(self, base_model=tree.HoeffdingTreeClassifier, n_models=10, window_size=100, threshold=0.5):
#         self.base_model = base_model
#         self.n_models = n_models
#         self.window_size = window_size
#         self.threshold = threshold
#         self.reset()

#     def reset(self):
#         self.ensemble = [self.base_model() for _ in range(self.n_models)]
#         self.windows = [deque(maxlen=self.window_size) for _ in range(self.n_models)]
#         self.weights = np.ones(self.n_models)
#         self.drift_detected = False

#     def update(self, X, y):
#         # Ensure X is a dictionary with feature names as keys
#         if isinstance(X, (np.ndarray, list)):
#             X = {f'X{i}': X[i] for i in range(len(X))}
#         elif isinstance(X, (int, float)):
#             X = {'X0': X}

#         predictions = np.zeros(self.n_models)

#         for i, model in enumerate(self.ensemble):
#             pred = model.predict_one(X)
#             predictions[i] = 1 if pred == y else 0
            
#             # Update the sliding window and model weight
#             self.windows[i].append(predictions[i])
#             accuracy = np.mean(self.windows[i])
#             self.weights[i] = accuracy

#             # Update the model with the new sample
#             model.learn_one(X, y)

#         # Normalize the weights
#         if np.sum(self.weights) > 0:
#             self.weights /= np.sum(self.weights)

#         # Combine predictions based on weights
#         combined_prediction = np.dot(self.weights, predictions)

#         # Check if drift is detected
#         if combined_prediction < self.threshold:
#             self.drift_detected = True
#             self.reset()  # Reset the ensemble if drift is detected
#             return 'drift'
#         else:
#             self.drift_detected = False
#             return 'no_drift'



from river import base, metrics, naive_bayes
import numpy as np
from copy import deepcopy
from collections import deque

class AUEDriftDetector:
    def __init__(self):
        """
        Initialize AUE drift detector
        
        Parameters:
        -----------
        base_classifier : river.base.Classifier
            Base classifier that implements River's stream learning interface
        ensemble_size : int
            Number of ensemble members (k)
        chunk_size : int
            Size of data chunks for training and evaluation
        """
        base_classifier=naive_bayes.GaussianNB()
        ensemble_size=10
        chunk_size=100


        self.ensemble_size = ensemble_size
        self.chunk_size = chunk_size
        self.base_classifier = base_classifier
        self.ensemble = []  # List of (classifier, weight) tuples
        self.stored_classifiers = []  # All classifiers with their weights
        
        # Buffer for collecting chunks
        self.X_buffer = []
        self.y_buffer = []
        self.chunk_count = 0
        
        # Drift detection
        self.drift_detected = False
        
    def _compute_mse(self, classifier, X_chunk, y_chunk):
        """
        Compute MSE for a classifier on a chunk of data
        """
        errors = []
        for x, y_true in zip(X_chunk, y_chunk):
            y_pred = classifier.predict_one(x)
            error = 1 if y_pred != y_true else 0
            errors.append(error)
        return np.mean(errors)
    
    def _compute_weight(self, mse):
        """
        Compute weight for a classifier based on its MSE
        """
        epsilon = 1e-10
        return 1 / (mse + epsilon)
    
    def _update_ensemble(self, X_chunk, y_chunk):
        """
        Update ensemble with new chunk of data
        """
        # Train new classifier
        new_classifier = deepcopy(self.base_classifier) if self.base_classifier else None
        if new_classifier:
            for x, y in zip(X_chunk, y_chunk):
                new_classifier.learn_one(x, y)
            
            # Compute MSE and weight for new classifier
            new_mse = self._compute_mse(new_classifier, X_chunk, y_chunk)
            new_weight = self._compute_weight(new_mse)
            
            # Update existing classifiers' weights
            for i, (clf, _) in enumerate(self.stored_classifiers):
                mse = self._compute_mse(clf, X_chunk, y_chunk)
                weight = self._compute_weight(mse)
                self.stored_classifiers[i] = (clf, weight)
                
                # Update classifier if condition is met
                if weight > 1 - new_mse:
                    for x, y in zip(X_chunk, y_chunk):
                        clf.learn_one(x, y)
            
            # Add new classifier to stored classifiers
            self.stored_classifiers.append((new_classifier, new_weight))
            
            # Select top k classifiers based on weights
            self.stored_classifiers = sorted(self.stored_classifiers, 
                                           key=lambda x: x[1], 
                                           reverse=True)[:self.ensemble_size * 2]
            self.ensemble = self.stored_classifiers[:self.ensemble_size]
            
            # Signal drift detection
            self.drift_detected = True
        
    def update(self, prediction, true_label):
        """
        Update the detector with a new sample
        """
        self.drift_detected = False
        
        # Store the example in the buffer
        self.X_buffer.append(prediction)
        self.y_buffer.append(true_label)
        self.chunk_count += 1
        
        # When buffer is full, update the ensemble
        if self.chunk_count >= self.chunk_size:
            self._update_ensemble(self.X_buffer, self.y_buffer)
            
            # Reset buffer
            self.X_buffer = []
            self.y_buffer = []
            self.chunk_count = 0
    
    def predict_one(self, x):
        """
        Make prediction using weighted voting
        """
        if not self.ensemble:
            return 0  # Default prediction for empty ensemble
        
        # Collect weighted votes
        class_votes = {}
        total_weight = 0
        
        for clf, weight in self.ensemble:
            pred = clf.predict_one(x)
            if pred not in class_votes:
                class_votes[pred] = 0
            class_votes[pred] += weight
            total_weight += weight
        
        # Normalize votes and select winner
        if total_weight > 0:
            for class_label in class_votes:
                class_votes[class_label] /= total_weight
            
            return max(class_votes.items(), key=lambda x: x[1])[0]
        return 0
    
    def predict_proba_one(self, x):
        """
        Predict class probabilities using weighted voting
        """
        if not self.ensemble:
            return {0: 0.5, 1: 0.5}  # Default probabilities for empty ensemble
        
        # Collect weighted probability predictions
        proba_sum = {0: 0.0, 1: 0.0}
        total_weight = 0
        
        for clf, weight in self.ensemble:
            probs = clf.predict_proba_one(x)
            for class_label, prob in probs.items():
                proba_sum[class_label] = proba_sum.get(class_label, 0) + prob * weight
            total_weight += weight
        
        # Normalize probabilities
        if total_weight > 0:
            for class_label in proba_sum:
                proba_sum[class_label] /= total_weight
                
        return proba_sum