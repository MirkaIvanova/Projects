from sklearn.model_selection import StratifiedKFold, cross_val_predict
import numpy as np
from copy import deepcopy
from .custom_metrics import calculate_fbeta_score


def feature_selection(_model, _X, _y, stratify_var, target_features=12, n_splits=5):
    """
    Perform feature selection using cross-validation.

    Parameters:
    model: sklearn model with feature_importances_ attribute
    X: pandas DataFrame with features
    y: target variable
    target_features: minimum number of features to keep
    cv: number of cross-validation folds
    random_state: random seed for reproducibility
    """
    model, X, y = deepcopy(_model), deepcopy(_X), deepcopy(_y)

    n_features_now = len(X.columns)
    selected_features = list(X.columns)  # Start with all features
    least_important_features = []
    best_score = 0
    best_threshold = 0
    best_selected_features = []
    best_model = None

    # Initial fit to get feature importances
    model.fit(X[selected_features], y)
    feature_importances = dict(zip(selected_features, model.feature_importances_))
    feature_importances = [
        (feature, round(importance, 3))
        for feature, importance in sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
    ]
    # print_feature_importances(feature_importances)

    for _ in range(n_features_now - target_features):
        # Get best threshold and score using predict_proba
        current_threshold, current_score = get_predict_proba_best_threshold(
            model, X[selected_features], y, stratify_var, n_splits=n_splits
        )

        print(f"\nNumber of features: {len(selected_features)}")
        # print(f"Current score: {current_score:.4f}, Current threshold: {current_threshold:.4f}")
        print("Removed features:", ", ".join(f"'{item}'" for item in least_important_features))

        # Fit model on full dataset to get feature importances
        model.fit(X[selected_features], y)
        importances = model.feature_importances_

        # Update best scores
        if current_score >= best_score:
            best_model = deepcopy(model)
            best_score = current_score
            best_threshold = current_threshold
            best_selected_features = deepcopy(selected_features)

        # Remove least important feature
        least_important_features.append(selected_features[np.argmin(importances)])
        selected_features.remove(selected_features[np.argmin(importances)])

    # print(f"\nBest score: {best_score:.4f}")

    return best_model, best_score, best_threshold, best_selected_features


def print_feature_importances(feature_importances):
    # Number of features to display per column (10 for >20 features, otherwise half)
    num_features = 20 if len(feature_importances) >= 40 else len(feature_importances) // 2

    # Sort by value in descending order, then round values to 3 decimal places
    sorted_importances = [
        (feature, round(importance, 3))
        for feature, importance in sorted(feature_importances, key=lambda x: x[1], reverse=True)
    ]

    # Split into top and bottom groups
    top_features = sorted_importances[:num_features]
    bottom_features = sorted_importances[-num_features:]

    # Print aligned columns
    print(f"{'Top Features':<20}{'Bottom Features':<20}")
    print("-" * 40)
    for (top_feat, top_imp), (bot_feat, bot_imp) in zip(top_features, bottom_features):
        print(f"{top_feat:<12}: {top_imp:.3f}    {bot_feat:<12}: {bot_imp:.3f}")


def get_predict_proba_best_threshold(model, X, y, stratify_var, n_splits=5, random_state=42):
    """
    Optimize the classification threshold and calculate the precision score using cross_val_predict.
    """
    # Get probability predictions using cross_val_predict

    cv = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)

    # Perform cross-validated predictions with stratified splits on composite variable
    y_pred_proba = cross_val_predict(
        model,
        X,
        y,
        cv=cv.split(X, stratify_var),
        method="predict_proba",
        n_jobs=-1,
    )

    # Define a range of thresholds to try
    thresholds = np.linspace(0.1, 0.9, 9)
    thresholds = [0.5]

    # Initialize variables to store the best threshold and precision score
    best_threshold = 0.5
    best_precision_score = 0

    # Test each threshold
    for threshold in thresholds:
        y_pred = (y_pred_proba[:, 1] > threshold).astype(int)
        precision = calculate_fbeta_score(y, y_pred)

        if precision > best_precision_score:
            best_threshold = threshold
            best_precision_score = precision

    return best_threshold, best_precision_score
