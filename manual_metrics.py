import numpy as np

def calculate_multiclass_weighted_metrics(y_true, y_pred):
    """
    Calcula Accuracy, Precision, Recall y F1 globales
    usando promedio ponderado (weighted), igual que sklearn.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = np.unique(np.concatenate([y_true, y_pred])) # Todos los labels presentes

    # ---- Métricas por clase ----
    precisions, recalls, f1s, supports = list(), list(), list(), list()

    for lab in labels:
        tp = np.sum((y_true == lab) & (y_pred == lab)) # predijo lab y era lab
        fp = np.sum((y_true != lab) & (y_pred == lab)) # predijo lab y no era lab
        fn = np.sum((y_true == lab) & (y_pred != lab)) # no predijo lab y era lab
        support = np.sum(y_true == lab) # numero de ocurrencias de lab en y_true para ponderar

        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * p * r / (p + r)) if (p + r) else 0.0

        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)
        supports.append(support)

    precisions = np.array(precisions)
    recalls = np.array(recalls)
    f1s = np.array(f1s)
    supports = np.array(supports)

    # ---- Promedios ponderados ----
    weights = supports / supports.sum() if supports.sum() else np.zeros_like(supports, dtype=float)
    precision_weighted = np.sum(precisions * weights)
    recall_weighted = np.sum(recalls * weights)
    f1_weighted = np.sum(f1s * weights)

    # ---- Accuracy global ----
    accuracy = np.mean(y_true == y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (weighted): {precision_weighted:.4f}")
    print(f"Recall (weighted): {recall_weighted:.4f}")
    print(f"F1-score (weighted): {f1_weighted:.4f}")


def explain_metrics():
    print("Accuracy: Proportion of correct predictions (both true positives and true negatives) out of all predictions made.")
    print("Precision: Proportion of true positive predictions out of all positive predictions made by the model.")
    print("Recall (Sensitivity): Proportion of true positive predictions out of all actual positive instances.")
    print("F1-score: Harmonic mean of precision and recall, providing a balance between the two metrics.")