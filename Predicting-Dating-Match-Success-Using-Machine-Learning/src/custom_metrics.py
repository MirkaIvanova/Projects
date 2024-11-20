from sklearn.metrics import precision_score, recall_score

BETA = 0.55  # gives precision a weight of 70%


def calculate_fbeta_score(y_true, y_pred):
    """
    Calculate weighted F1 score with more emphasis on precision by default giving 70% (beta=0.55) weight on precision.
    """
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred)
    beta = BETA

    if not precision:
        return 0
    return (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
