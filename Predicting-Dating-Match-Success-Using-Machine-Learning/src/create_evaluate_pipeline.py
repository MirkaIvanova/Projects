# import copy
import hashlib
import json
import os
import pickle
from copy import deepcopy
from itertools import product
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline

from .color import Color
from .custom_metrics import calculate_fbeta_score


# ❤️MII_21 DELETEME!
def print_info(msg, end=True, dst="automl_debug.log"):
    with open(dst, "a") as f:
        f.write(msg + ("\n" if end else ""))


def evaluate_parameter_grid(param_grid, X, y):
    results = []
    cache = _load_cache(location="./")

    for grid in param_grid:
        param_combinations = _generate_param_combinations(grid)
        total_combinations = len(param_combinations)

        for i, params in enumerate(param_combinations, 1):
            print(f"Evaluating combination {i}/{total_combinations}: ", end="")

            params_hash = _generate_param_hash(params)
            cached_params_filename = f"params_{params_hash}.pkl"

            if params_hash in cache:
                cached_score = deepcopy(cache[params_hash])
                if "weighted_fbeta" in cached_score["scores"]:
                    print(
                        f"Weighted fbeta score: {cached_score['scores']['weighted_fbeta']} (from cache {params_hash})"
                    )
                    cached_params = _load_object(cached_params_filename, location="param_grid_pkl")
                    cached_score["parameters"] = cached_params

                    results.append(cached_score)
                    continue

            pipeline = _create_pipeline(params)

            # Train and evaluate
            pipeline.fit(X, y)
            scores = _evaluate_model(pipeline, X, y, calculate_fbeta_score)
            print(f"Weighted fbeta score: {scores['weighted_fbeta']}")

            result = {
                "scores": scores,
                "parameters": dict(pipeline.steps),
                "selected_featured": _extract_feature_info(pipeline, X)["selected_features"],
            }

            cache[params_hash] = {
                "scores": scores,
                "selected_featured": _extract_feature_info(pipeline, X)["selected_features"],
            }
            _save_object(dict(pipeline.steps), cached_params_filename)  # save model and params
            _save_cache(cache, location="./")

            results.append(result)

    return results


def predict_and_score(best_model, X_test, y_test, best_predict_proba_threshold, selected_features):
    """Evaluate a model's performance on test data and return key metrics."""
    y_pred_proba = best_model.predict_proba(X_test[selected_features])
    y_pred = (y_pred_proba[:, 1] > best_predict_proba_threshold).astype(int)

    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    # fbeta = (1 + BETA**2) * (precision * recall) / ((BETA**2 * precision) + recall)
    fbeta = calculate_fbeta_score(y_test, y_pred)

    return {
        "f1": f1,
        "fbeta": fbeta,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
    }


def print_scores(scores):
    print(f"F1:         {scores["f1"]:.4f}")
    print(f"Fbeta:      {scores["fbeta"]:.4f}")
    print(f"Precision:  {scores["precision"]:.4f}")
    print(f"Recall:     {scores["recall"]:.4f}")
    print(f"Accuracy:   {scores["accuracy"]:.4f}")

    # 💛MII remove!!!
    print_info(f"F1:         {scores["f1"]:.4f}")
    print_info(f"Fbeta:      {scores["fbeta"]:.4f}")
    print_info(f"Precision:  {scores["precision"]:.4f}")
    print_info(f"Recall:     {scores["recall"]:.4f}")
    print_info(f"Accuracy:   {scores["accuracy"]:.4f}")


def display_top_models(
    results,
    total_features,
    top_k=5,
    sort_by="weighted_fbeta",
    show_features_details=False,
):
    """
    Display and return top pipeline results with improved formatting and organization.
    Args:
        results:               List of pipeline results
        total_features:        Total number of available features
        metric:                Metric to sort by (default: f1_score)
        show_features_details: Whether to show detailed feature information
        top_k:                 Number of top results to display, by default 5
    """
    # Sort and get top results
    top_results = sorted(results, key=lambda x: x["scores"][sort_by], reverse=True)[:top_k]

    # Display results
    for rank, result in enumerate(top_results, 1):
        print(Color.style(f"Rank {rank}/{len(results)} - ", Color.MAGENTA), end="")

        # Print scores
        print(_format_metrics(result["scores"]))

        # Print pipeline parameters
        print(Color.BLUE + "Pipeline parameters: \n" + Color.RESET, end="")
        # Color.print_colored_dict(result["parameters"])
        Color.print_colored_dict({"model": result["parameters"]["model"].__class__.__name__})
        Color.print_colored_dict(result["parameters"]["model"].get_params())

        # Print feature selection summary
        selected_features = result.get("selected_features", [])
        if selected_features and (len(selected_features) < total_features):
            print(
                Color.BLUE
                + "Selected features:  "
                + Color.RESET
                + str(len(result.get("selected_features", "")))
                + "/"
                + str(total_features)
                + " "
                + str(", ".join(result.get("selected_features", "")))
            )

        # Print feature selection details
        if show_features_details and "feature_details" in result:
            print(
                Color.style(
                    " Id | Feature Name         | Feature Index | F-Score | P-value",
                    Color.BLUE,
                )
            )
            print(Color.style("-" * 56, Color.BLUE))
            for j, (feature_name, idx, f_score, p_value) in enumerate(result["feature_details"], start=1):
                print(f"{j:3d} | {feature_name:20s} | {idx:13d} | {f_score:7.3f} | {p_value:.3e}")

        print("")

    return top_results


def _create_pipeline(_params):
    """
    Create a sklearn pipeline with optional feature selector.
    """
    steps = []
    params = deepcopy(_params)
    model = deepcopy(params.pop("model"))

    if "feature_selector" in params:
        steps.append(("feature_selector", params.pop("feature_selector")))
    steps.append(("model", model))

    return Pipeline(steps).set_params(**params)


def _evaluate_model(model, X, y, custom_scorer_fn):
    custom_scorer = make_scorer(custom_scorer_fn)
    scoring = {
        "precision": "precision_macro",
        "recall": "recall_macro",
        "weighted_fbeta": custom_scorer,
    }

    scores = cross_validate(model, X, y, scoring=scoring, cv=5)

    return {
        "weighted_fbeta": f"{scores["test_weighted_fbeta"].mean():.4f}",
        "precision": f"{scores["test_precision"].mean():.4f}",
        "recall": f"{scores["test_recall"].mean():.4f}",
    }


def _save_object(obj, filename, location="param_grid_pkl"):
    """
    Saves an object to a specified file and location using pickle.

    Parameters:
    obj (object): The object to be saved.
    filename (str): The name of the file (e.g., 'data.pkl').
    location (str): The directory where the file should be saved.
    """
    os.makedirs(location, exist_ok=True)  # Create directory if it doesn't exist
    filepath = os.path.join(location, filename)
    with open(filepath, "wb") as file:
        pickle.dump(obj, file)


def _load_object(filename, location="param_grid_pkl"):
    """
    Loads an object from a specified file and location using pickle.

    Parameters:
    filename (str): The name of the file (e.g., 'data.pkl').
    location (str): The directory where the file is located.

    Returns:
    object: The loaded object.
    """
    filepath = os.path.join(location, filename)
    with open(filepath, "rb") as file:
        obj = pickle.load(file)
    return obj


def _load_cache(cache_name="params_cache.json", location="param_grid_pkl"):
    filepath = os.path.join(location, cache_name)

    if Path(filepath).exists():
        with open(filepath, "r") as f:
            return json.load(f)
    return {}


def _save_cache(json_data, cache_name="params_cache.json", location="param_grid_pkl"):
    os.makedirs(location, exist_ok=True)  # Create directory if it doesn't exist
    filepath = os.path.join(location, cache_name)

    with open(filepath, "w") as f:
        json.dump(json_data, f, indent=4)


def _generate_param_combinations(param_dict):
    """Generate all combinations of parameters for a parameter grid."""

    keys = list(param_dict.keys())
    values = list(param_dict.values())

    combinations = product(*values)

    param_grid = [dict(zip(keys, combination)) for combination in combinations]

    return param_grid


def _format_metrics(scores):
    """Format the metrics string with color and consistent decimal places."""

    return (
        Color.RED
        + f"Weighted Fbeta: {scores['weighted_fbeta']}, "
        + Color.RESET
        + f"Precision: {scores['precision']}, Recall: {scores['recall']}"
    )


def _generate_param_hash(params):
    """
    Generate a unique hash for a given parameter dictionary.
    Non-serializable objects are converted to their string representations.
    """
    # Convert parameter values to strings if they are not JSON serializable
    serializable_params = {
        k: str(v) if not isinstance(v, (int, float, str, list, dict, tuple)) else v for k, v in params.items()
    }

    # Create a sorted JSON string and hash it
    params_str = json.dumps(serializable_params, sort_keys=True)
    return hashlib.md5(params_str.encode("utf-8")).hexdigest()


def _extract_feature_info(pipeline, X):
    """Extracts feature selection information if available"""

    if "feature_selector" not in pipeline.named_steps:
        return {"selected_features": None, "feature_details": None}

    feature_selector = pipeline.named_steps["feature_selector"]
    mask = feature_selector.get_support()
    feature_names = X.columns.tolist()

    selected_features = np.array(feature_names)[mask].tolist()

    # Create detailed feature information
    feature_details = [
        (
            feature_names[idx],
            idx,
            feature_selector.scores_[idx],
            feature_selector.pvalues_[idx],
        )
        for idx in np.where(mask)[0]
    ]
    feature_details.sort(key=lambda x: x[2], reverse=True)  # Sort by F-score

    return {"selected_features": selected_features, "feature_details": feature_details}


# ####################################################### 💛MII
