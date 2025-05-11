# # utils/evaluation.py
# from river import metrics
# import numpy as np
# from sklearn.metrics import roc_auc_score, auc, precision_recall_curve
# from sklearn.preprocessing import LabelEncoder

# def evaluate_learner(df_name, model, drift_detector, X, y):

#     metric = metrics.Accuracy()  # Accuracy metric
#     metric_f1 = metrics.F1()  # F1 score
#     metric_precision = metrics.Precision()  # Precision score

#     t = []  # Number of evaluated data points
#     m = []  # Real-time accuracy

#     yt = []  # True labels
#     yp = []  # Predicted labels
#     yp_proba = []  # Predicted probabilities

#     # if df_name == 'kdd':
#     #     if isinstance(X[0], dict):
#     #         feature_names = list(X[0].keys())
#     #     else:
#     #         feature_names = [f'X{i+1}' for i in range(X.shape[1])]

#     for i, (x, y_true) in enumerate(zip(X, y)):

#         # if df_name == 'kdd':
#         #     if isinstance(x, np.ndarray):
#         #         x = {feature_names[j]: x[j] for j in range(len(x))}


#         y_pred = model.predict_one(x)
#         y_proba = model.predict_proba_one(x)  
#         model.learn_one(x, y_true)

#         drift_detector.update(y_pred, y_true)
#         if drift_detector.drift_detected:
#             model = model.clone()  # Reset the model

#         metric.update(y_true, y_pred)
#         metric_f1.update(y_true, y_pred)
#         metric_precision.update(y_true, y_pred)

#         t.append(i)
#         m.append(metric.get() * 100)

#         yt.append(y_true)
#         yp.append(y_pred)
#         class_1_proba = y_proba.get(1, 0.5) if y_proba is not None else 0.5
#         yp_proba.append(class_1_proba)

#     # Check if yt contains at least two classes
#     if len(set(yt)) > 1:
#         auroc_value = roc_auc_score(yt, yp_proba)
#         precision, recall, _ = precision_recall_curve(yt, yp_proba)
#         auc_value = auc(recall, precision)
#     else:
#         auroc_value = None
#         auc_value = None
#         # print("Warning: Only one class present in y_true. ROC AUC score and AUC value are not defined in that case.")

#     return t, m, metric, metric_f1, metric_precision, auc_value, auroc_value



# utils/evaluation.py
from river import metrics
import numpy as np
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve

def evaluate_learner(df_name, model, drift_detector, X, y):
    metric = metrics.Accuracy()  # Accuracy metric
    metric_f1 = metrics.F1()  # F1 score
    metric_precision = metrics.Precision()  # Precision score

    t = []  # Number of evaluated data points
    m = []  # Real-time accuracy

    yt = []  # True labels
    yp = []  # Predicted labels
    yp_proba = []  # Predicted probabilities

    for i, (x, y_true) in enumerate(zip(X, y)):
        y_pred = model.predict_one(x)
        y_proba = model.predict_proba_one(x)  
        model.learn_one(x, y_true)

        # Update the drift detector with x and y_true
        drift_detector.update(x, y_true)
        if drift_detector.drift_detected:
            model = model.clone()  # Reset the model

        # Update metrics
        metric.update(y_true, y_pred)
        metric_f1.update(y_true, y_pred)
        metric_precision.update(y_true, y_pred)

        t.append(i)
        m.append(metric.get() * 100)

        yt.append(y_true)
        yp.append(y_pred)
        class_1_proba = y_proba.get(1, 0.5) if y_proba is not None else 0.5
        yp_proba.append(class_1_proba)

    # Compute AUC and AUROC if yt has at least two classes
    if len(set(yt)) > 1:
        auroc_value = roc_auc_score(yt, yp_proba)
        precision, recall, _ = precision_recall_curve(yt, yp_proba)
        auc_value = auc(recall, precision)
    else:
        auroc_value = None
        auc_value = None
        print("Warning: Only one class present in y_true. ROC AUC score and AUC value are not defined in that case.")

    return t, m, metric, metric_f1, metric_precision, auc_value, auroc_value
