from copy import deepcopy
from sklearn.feature_selection import (
    RFE,
    SelectFromModel,
    SelectKBest,
    SelectPercentile,
)
from sklearn.model_selection import cross_val_predict, KFold

RANDOM_STATE = 42

# @now copy X, y


# 1. Univariate Feature Selection (SelectKBest)
def select_k_best(_model, _X, _y, feature_names, custom_scorer, cv=5, k="all"):
    model, X, y = deepcopy(_model), deepcopy(_X), deepcopy(_y)

    # Prepare cross-validation
    kf = KFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)

    # Perform feature selection
    selector = SelectKBest(k=k)
    selector.fit_transform(X, y)
    mask = selector.get_support()

    # Get selected feature names
    selected_features = [f for f, m in zip(feature_names, mask) if m]

    # Perform cross-validated predictions
    model_copy = type(model)(**model.get_params())
    X_selected = X[:, mask] if not hasattr(X, "iloc") else X.iloc[:, mask]

    # Use cross_val_predict for more robust performance estimation
    y_pred = cross_val_predict(model_copy, X_selected, y, cv=kf)
    score = custom_scorer(y, y_pred)

    return {"score": score, "num_features": len(selected_features), "selected_features": selected_features}


# 2. Percentile-based Feature Selection
def select_percentile(_model, _X, _y, feature_names, custom_scorer, cv=5, percentile=50):
    model, X, y = deepcopy(_model), deepcopy(_X), deepcopy(_y)

    kf = KFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)

    # Perform feature selection
    selector = SelectPercentile(percentile=percentile)
    X_new = selector.fit_transform(X, y)
    mask = selector.get_support()

    # Get selected feature names
    selected_features = [f for f, m in zip(feature_names, mask) if m]

    # Perform cross-validated predictions
    model_copy = type(model)(**model.get_params())
    X_selected = X[:, mask] if not hasattr(X, "iloc") else X.iloc[:, mask]

    # Use cross_val_predict for more robust performance estimation
    y_pred = cross_val_predict(model_copy, X_selected, y, cv=kf)
    score = custom_scorer(y, y_pred)

    return {"score": score, "num_features": len(selected_features), "selected_features": selected_features}


# 3. Recursive Feature Elimination (RFE)
def recursive_feature_elimination(_model, _X, _y, feature_names, custom_scorer, cv=5, n_features_to_select=None):
    model, X, y = deepcopy(_model), deepcopy(_X), deepcopy(_y)
    kf = KFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)

    if n_features_to_select is None:
        n_features_to_select = max(1, X.shape[1] // 2)

    # Perform recursive feature elimination
    selector = RFE(estimator=model, n_features_to_select=n_features_to_select)
    selector = selector.fit(X, y)
    mask = selector.support_

    # Get selected feature names
    selected_features = [f for f, m in zip(feature_names, mask) if m]

    # Perform cross-validated predictions
    model_copy = type(model)(**model.get_params())
    X_selected = X[:, mask] if not hasattr(X, "iloc") else X.iloc[:, mask]

    # Use cross_val_predict for more robust performance estimation
    y_pred = cross_val_predict(model_copy, X_selected, y, cv=kf)
    score = custom_scorer(y, y_pred)

    return {"score": score, "num_features": len(selected_features), "selected_features": selected_features}


# 4. Model-based Feature Selection
def select_from_model(_model, _X, _y, feature_names, custom_scorer, cv=5, threshold="median"):
    model, X, y = deepcopy(_model), deepcopy(_X), deepcopy(_y)
    kf = KFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)

    # Perform feature selection based on model
    selector = SelectFromModel(model, prefit=True, threshold=threshold)
    mask = selector.get_support()

    # Get selected feature names
    selected_features = [f for f, m in zip(feature_names, mask) if m]

    # Perform cross-validated predictions
    model_copy = type(model)(**model.get_params())
    X_selected = X[:, mask] if not hasattr(X, "iloc") else X.iloc[:, mask]

    # Use cross_val_predict for more robust performance estimation
    y_pred = cross_val_predict(model_copy, X_selected, y, cv=kf)
    score = custom_scorer(y, y_pred)

    return {"score": score, "num_features": len(selected_features), "selected_features": selected_features}
