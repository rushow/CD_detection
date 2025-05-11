# import numpy as np
# from river import naive_bayes
# from river import base

# class DWMDriftDetector(base.DriftDetector):
#     def __init__(self, n_estimators=5, base_estimator=None, period=50, beta=0.5, theta=0.01):
#         super().__init__()
#         self.n_estimators = n_estimators
#         self.base_estimator = base_estimator if base_estimator is not None else naive_bayes.GaussianNB()
#         self.period = period
#         self.beta = beta
#         self.theta = theta
#         self.dwm = DynamicWeightedMajorityClassifier(
#             n_estimators=self.n_estimators,
#             base_estimator=self.base_estimator,
#             period=self.period,
#             beta=self.beta,
#             theta=self.theta
#         )
#         self.drift_detected = False
#         self.warning_detected = False
#         self.n_samples = 0

#     def update(self, prediction, label):
#         self.n_samples += 1
#         self.drift_detected = False
#         self.warning_detected = False

#         X = np.array([[0]])  # Dummy feature
#         y = np.array([label])

#         if self.n_samples == 1:
#             self.dwm.partial_fit(X, y, classes=np.unique(y))
#         else:
#             expert_predictions = self.dwm.get_expert_predictions(X)
#             ensemble_prediction = self.dwm.predict(X)

#             # Check for drift
#             if ensemble_prediction != y:
#                 self.drift_detected = True

#             # Check for warning (when half of the experts disagree)
#             if sum([1 for pred in expert_predictions if pred != y]) >= len(expert_predictions) / 2:
#                 self.warning_detected = True

#             self.dwm.partial_fit(X, y)

#     def reset(self):
#         self.dwm.reset()
#         self.drift_detected = False
#         self.warning_detected = False
#         self.n_samples = 0

# class DynamicWeightedMajorityClassifier:
#     class WeightedExpert:
#         def __init__(self, estimator, weight):
#             self.estimator = estimator
#             self.weight = weight

#     def __init__(self, n_estimators=5, base_estimator=naive_bayes.GaussianNB(),
#                  period=50, beta=0.5, theta=0.01):
#         self.n_estimators = n_estimators
#         self.base_estimator = base_estimator
#         self.beta = beta
#         self.theta = theta
#         self.period = period
#         self.epochs = 0
#         self.num_classes = None
#         self.experts = None
#         self.reset()

#     def partial_fit(self, X, y, classes=None, sample_weight=None):
#         for i in range(len(X)):
#             self.fit_single_sample(
#                 X[i:i + 1, :], y[i:i + 1], classes, sample_weight
#             )
#         return self

#     def predict(self, X):
#         preds = np.array([np.array(exp.estimator.predict(X)) * exp.weight
#                           for exp in self.experts])
#         sum_weights = sum(exp.weight for exp in self.experts)
#         aggregate = np.sum(preds / sum_weights, axis=0)
#         return (aggregate + 0.5).astype(int)

#     def fit_single_sample(self, X, y, classes=None, sample_weight=None):
#         self.epochs += 1
#         self.num_classes = max(
#             len(classes) if classes is not None else 0,
#             (int(np.max(y)) + 1), self.num_classes)
#         predictions = np.zeros((self.num_classes,))
#         max_weight = 0
#         weakest_expert_weight = 1
#         weakest_expert_index = None

#         for i, exp in enumerate(self.experts):
#             y_hat = exp.estimator.predict(X)
#             if np.any(y_hat != y) and (self.epochs % self.period == 0):
#                 exp.weight *= self.beta

#             predictions[y_hat] += exp.weight
#             max_weight = max(max_weight, exp.weight)

#             if exp.weight < weakest_expert_weight:
#                 weakest_expert_index = i
#                 weakest_expert_weight = exp.weight

#         y_hat = np.array([np.argmax(predictions)])
#         if self.epochs % self.period == 0:
#             self._scale_weights(max_weight)
#             self._remove_experts()
#             if np.any(y_hat != y):
#                 if len(self.experts) == self.n_estimators:
#                     self.experts.pop(weakest_expert_index)
#                 self.experts.append(self._construct_new_expert())

#         for exp in self.experts:
#             exp.estimator.partial_fit(X, y, classes, sample_weight)

#     def get_expert_predictions(self, X):
#         return [exp.estimator.predict(X) for exp in self.experts]

#     def reset(self):
#         self.epochs = 0
#         self.num_classes = 2
#         self.experts = [
#             self._construct_new_expert()
#         ]

#     def _scale_weights(self, max_weight):
#         scale_factor = 1 / max_weight
#         for exp in self.experts:
#             exp.weight *= scale_factor

#     def _remove_experts(self):
#         self.experts = [ex for ex in self.experts if ex.weight >= self.theta]

#     def _construct_new_expert(self):
#         return self.WeightedExpert(self.base_estimator.clone(), 1)

from river import base
import copy
import numpy as np
from typing import Dict, Any

class DWMDriftDetector:
    def __init__(
        self,
        base_estimator=None,
        n_classes=2,
        beta=0.5,
        theta=0.1,
        period=50
    ):
        """
        Dynamic Weighted Majority (DWM) drift detector and ensemble learner
        
        Parameters
        ----------
        base_estimator : river.base.Estimator
            Base estimator for the ensemble
        n_classes : int
            Number of classes
        beta : float
            Factor for decreasing weights (0 <= beta < 1)
        theta : float
            Threshold for removing experts
        period : int
            Period between expert removal, creation, and weight update
        """
        self.base_estimator = base_estimator
        self.n_classes = n_classes
        self.beta = beta
        self.theta = theta
        self.period = period
        
        # Initialize ensemble
        self.experts = []
        self.weights = []
        self.sample_count = 0
        self.drift_detected = False
        self._current_x = None
        
        # Initialize first expert
        self.reset()
    
    def _normalize_weights(self):
        """Normalize the weights of experts"""
        if self.weights:
            sum_weights = sum(self.weights)
            if sum_weights > 0:
                self.weights = [w/sum_weights for w in self.weights]
    
    def _remove_experts(self):
        """Remove experts whose weight is below threshold"""
        if not self.weights:
            return
        
        keep_indices = [i for i, w in enumerate(self.weights) if w > self.theta]
        self.experts = [self.experts[i] for i in keep_indices]
        self.weights = [self.weights[i] for i in keep_indices]
    
    def predict_one(self, x: Dict) -> int:
        """
        Predict the class of a single instance
        
        Parameters
        ----------
        x : dict
            Instance to predict
            
        Returns
        -------
        int
            Predicted class
        """
        if not self.experts:
            return 0
            
        weighted_predictions = np.zeros(self.n_classes)
        
        for expert, weight in zip(self.experts, self.weights):
            try:
                pred = expert.predict_one(x)
                weighted_predictions[pred] += weight
            except Exception as e:
                print(f"Prediction error: {e}")
                continue
            
        return int(np.argmax(weighted_predictions))
    
    def predict_proba_one(self, x: Dict) -> Dict[int, float]:
        """
        Predict probabilities for a single instance
        
        Parameters
        ----------
        x : dict
            Instance to predict
            
        Returns
        -------
        dict
            Dictionary mapping class labels to probabilities
        """
        if not self.experts:
            return {i: 1/self.n_classes for i in range(self.n_classes)}
            
        weighted_predictions = np.zeros(self.n_classes)
        total_weight = sum(self.weights)
        
        for expert, weight in zip(self.experts, self.weights):
            try:
                proba = expert.predict_proba_one(x)
                for label, prob in proba.items():
                    weighted_predictions[label] += weight * prob
            except Exception as e:
                print(f"Probability prediction error: {e}")
                continue
                
        if total_weight > 0:
            weighted_predictions /= total_weight
            
        return {i: float(p) for i, p in enumerate(weighted_predictions)}
    
    def learn_one(self, x: Dict, y: int):
        """
        Update the model with a single instance
        
        Parameters
        ----------
        x : dict
            Instance to learn from
        y : int
            True label
        """
        self._current_x = x  # Store current instance for weight updates
        
        # Initialize first expert if none exists
        if not self.experts and self.base_estimator is not None:
            self.experts.append(copy.deepcopy(self.base_estimator))
            self.weights.append(1.0)
        
        # Train all experts
        for expert in self.experts:
            try:
                expert.learn_one(x, y)
            except Exception as e:
                print(f"Learning error: {e}")
                continue
    
    def update(self, y_pred: int, y_true: int):
        """
        Update the detector with a new prediction and true label
        
        Parameters
        ----------
        y_pred : int
            Predicted label
        y_true : int
            True label
        """
        if self._current_x is None or not self.experts:
            return
            
        self.drift_detected = False
        self.sample_count += 1
        
        # Update weights and check for drift periodically
        if self.sample_count % self.period == 0:
            # Update weights based on predictions
            predictions = []
            for expert in self.experts:
                try:
                    pred = expert.predict_one(self._current_x)
                    predictions.append(pred)
                except Exception as e:
                    print(f"Update prediction error: {e}")
                    predictions.append(None)
            
            # Update weights for experts that made predictions
            for i, pred in enumerate(predictions):
                if pred is not None and pred != y_true:
                    self.weights[i] *= self.beta
            
            # Normalize weights
            self._normalize_weights()
            
            # Remove experts below threshold
            self._remove_experts()
            
            # Add new expert if global prediction was wrong
            if y_pred != y_true and self.base_estimator is not None:
                self.drift_detected = True
                new_expert = copy.deepcopy(self.base_estimator)
                try:
                    new_expert.learn_one(self._current_x, y_true)
                    self.experts.append(new_expert)
                    self.weights.append(1.0)
                except Exception as e:
                    print(f"New expert initialization error: {e}")
    
    def reset(self):
        """Reset the detector"""
        self.experts = []
        self.weights = []
        self.sample_count = 0
        self.drift_detected = False
        self._current_x = None
        
        if self.base_estimator is not None:
            try:
                self.experts.append(copy.deepcopy(self.base_estimator))
                self.weights.append(1.0)
            except Exception as e:
                print(f"Reset error: {e}")