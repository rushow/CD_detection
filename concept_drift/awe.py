# import numpy as np
# from sklearn.base import BaseEstimator, ClassifierMixin
# from sklearn.model_selection import KFold
# from sklearn.naive_bayes import GaussianNB
# from sklearn.impute import SimpleImputer
# import copy as cp

# class AWEDriftDetector(BaseEstimator, ClassifierMixin):
#     class WeightedClassifier:
#         def __init__(self, estimator, weight, seen_labels):
#             self.estimator = estimator
#             self.weight = weight
#             self.seen_labels = seen_labels

#         def __lt__(self, other):
#             return self.weight < other.weight

#     def __init__(self, window_size=200, n_estimators=10, n_kept_estimators=30, base_estimator=GaussianNB(), n_splits=5):
#         self.window_size = window_size
#         self.n_estimators = n_estimators
#         self.n_kept_estimators = n_kept_estimators
#         self.base_estimator = base_estimator
#         self.n_splits = n_splits
#         self.models_pool = []
#         self.X_chunk = None
#         self.y_chunk = None
#         self.p = 0
#         self.drift_detected = False
#         self.imputer = SimpleImputer(strategy='mean')

#     def update(self, y_pred, y_true):
#         X = np.array([y_pred])  # We use y_pred as a feature
#         y = np.array([y_true])
#         self.partial_fit(X, y)
#         return self.drift_detected

#     def partial_fit(self, X, y, classes=None):
#         if X.ndim == 1:
#             X = X.reshape(-1, 1)

#         if self.X_chunk is None:
#             self.X_chunk = np.zeros((self.window_size, X.shape[1]))
#             self.y_chunk = np.zeros(self.window_size)

#         for i in range(len(X)):
#             self.X_chunk[self.p] = X[i]
#             self.y_chunk[self.p] = y[i]
#             self.p += 1

#             if self.p == self.window_size:
#                 self.p = 0
#                 self._process_chunk()

#         return self

#     def _process_chunk(self):
#         classes = np.unique(self.y_chunk)
        
#         # Impute NaN values
#         X_imputed = self.imputer.fit_transform(self.X_chunk)
        
#         new_model = self.train_model(cp.deepcopy(self.base_estimator), X_imputed, self.y_chunk, classes)
#         baseline_score = self.compute_baseline(self.y_chunk)
        
#         new_clf = self.WeightedClassifier(new_model, -1.0, classes)
#         new_clf.weight = self.compute_weight(new_clf, baseline_score, self.n_splits)

#         for model in self.models_pool:
#             model.weight = self.compute_weight(model, baseline_score, None)

#         if len(self.models_pool) < self.n_kept_estimators:
#             self.models_pool.append(new_clf)
#         else:
#             worst_model = min(self.models_pool, key=lambda x: x.weight)
#             if new_clf.weight > worst_model.weight:
#                 self.models_pool.remove(worst_model)
#                 self.models_pool.append(new_clf)

#         self.drift_detected = self._check_drift()

#     def _check_drift(self):
#         if len(self.models_pool) < 2:
#             return False
        
#         weights = [model.weight for model in self.models_pool]
#         weight_variance = np.var(weights)
#         return weight_variance > np.mean(weights) * 0.5  # Adjust this threshold as needed

#     @staticmethod
#     def train_model(model, X, y, classes=None):
#         try:
#             model.fit(X, y)
#         except NotImplementedError:
#             model.partial_fit(X, y, classes)
#         return model

#     def compute_score(self, model, X, y):
#         N = len(y)
#         labels = model.seen_labels
#         X_imputed = self.imputer.transform(X)
#         probabs = model.estimator.predict_proba(X_imputed)

#         sum_error = 0
#         for i, c in enumerate(y):
#             if c in labels:
#                 index_label_c = np.where(labels == c)[0][0]
#                 probab_ic = probabs[i][index_label_c]
#                 sum_error += (1.0 - probab_ic) ** 2
#             else:
#                 sum_error += 1.0

#         return sum_error / N

#     def compute_score_crossvalidation(self, model, n_splits):
#         if n_splits is not None and isinstance(n_splits, int):
#             copy_model = cp.deepcopy(model)
#             copy_model.estimator = cp.deepcopy(self.base_estimator)
#             score = 0
#             kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
#             X_imputed = self.imputer.fit_transform(self.X_chunk)
#             for train_idx, test_idx in kf.split(X=X_imputed, y=self.y_chunk):
#                 X_train, y_train = X_imputed[train_idx], self.y_chunk[train_idx]
#                 X_test, y_test = X_imputed[test_idx], self.y_chunk[test_idx]
#                 copy_model.estimator = self.train_model(copy_model.estimator, X_train, y_train, copy_model.seen_labels)
#                 score += self.compute_score(copy_model, X_test, y_test) / n_splits
#         else:
#             score = self.compute_score(model, self.X_chunk, self.y_chunk)

#         return score

#     def compute_weight(self, model, baseline_score, n_splits=None):
#         score = self.compute_score_crossvalidation(model, n_splits)
#         return max(0.0, baseline_score - score)

#     @staticmethod
#     def compute_baseline(y):
#         classes, class_count = np.unique(y, return_counts=True)
#         class_dist = [class_count[i] / len(y) for i in range(len(classes))]
#         mse_r = np.sum([(class_dist[i] * (1 - class_dist[i]) * (1 - class_dist[i])) for i in range(len(classes))])
#         return mse_r

#     def predict(self, X):
#         if X.ndim == 1:
#             X = X.reshape(-1, 1)

#         if len(self.models_pool) == 0:
#             return np.zeros(len(X), dtype=int)

#         X_imputed = self.imputer.transform(X)
#         ensemble = sorted(self.models_pool, key=lambda clf: clf.weight, reverse=True)[:self.n_estimators]
#         sum_weights = sum(abs(clf.weight) for clf in ensemble) or 1

#         predictions = []
#         for x in X_imputed:
#             votes = {}
#             for model in ensemble:
#                 pred = model.estimator.predict([x])[0]
#                 votes[pred] = votes.get(pred, 0) + model.weight / sum_weights
#             predictions.append(max(votes, key=votes.get))

#         return np.array(predictions)

#     def reset(self):
#         self.models_pool = []
#         self.X_chunk = None
#         self.y_chunk = None
#         self.p = 0
#         self.drift_detected = False
#         self.imputer = SimpleImputer(strategy='mean')


# concept_drift/awe.py
import numpy as np
from collections import deque
from river import base, naive_bayes
from sklearn.metrics import mean_squared_error

class AWEDriftDetector:
    def __init__(self, chunk_size=50, ensemble_size=5):
        base_model=naive_bayes.GaussianNB()
        self.base_model = base_model
        self.chunk_size = chunk_size
        self.ensemble_size = ensemble_size
        self.ensemble = deque(maxlen=ensemble_size)  # Store ensemble of models
        self.weights = deque(maxlen=ensemble_size)   # Store weights for each model
        self.data_chunk = []
        self.drift_detected = False

    def update(self, x, y):
        # Collect data until the chunk is full
        self.data_chunk.append((x, y))
        
        if len(self.data_chunk) >= self.chunk_size:
            self._train_new_model()
            self.data_chunk = []  # Clear data chunk for the next batch

    def _train_new_model(self):
        # Separate features and labels from the collected data chunk
        X_chunk, y_chunk = zip(*self.data_chunk)
        
        # Create and train a new model on the current chunk
        new_model = self.base_model.clone()
        for x, y in zip(X_chunk, y_chunk):
            new_model.learn_one(x, y)
        
        # Evaluate the new model using mean squared error (MSE) or benefits
        predictions = [new_model.predict_one(x) for x in X_chunk]
        mse_new_model = mean_squared_error(y_chunk, predictions)

        # Calculate weights for each model in the ensemble
        mse_values = [mean_squared_error(y_chunk, [model.predict_one(x) for x in X_chunk]) for model in self.ensemble]
        mse_reference = mse_new_model if mse_values else 0.5  # Reference for weights

        # Calculate weights and normalize them
        weights = [(mse_reference - mse) for mse in mse_values]
        weights = [max(w, 0.0) for w in weights]  # Ensure no negative weights
        weights.append(mse_reference - mse_new_model)  # Add weight for new model

        if len(weights) > 0:
            weights = np.array(weights)
            weights = weights / weights.sum() if weights.sum() > 0 else weights

        # Update the ensemble and weights
        self.ensemble.append(new_model)
        self.weights.append(weights[-1])
        self.drift_detected = True if weights[-1] == max(weights) else False

    def predict(self, x):
        # Weighted prediction from the ensemble
        predictions = np.array([model.predict_one(x) for model in self.ensemble])
        weighted_preds = np.dot(predictions, self.weights)
        return 1 if weighted_preds >= 0.5 else 0  # Binary classification threshold at 0.5

    def reset(self):
        # Reset detector by clearing the ensemble and weights
        self.ensemble.clear()
        self.weights.clear()
        self.data_chunk = []
        self.drift_detected = False
