import copy
import json

# 💛MII_REF20 remove all comments
# from IPython.display import IFrame
# from category_encoders import CatBoostEncoder, TargetEncoder
from datetime import datetime

import re
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# from joblib import dump, load
# from sklearn.base import BaseEstimator, clone, TransformerMixin
# from sklearn.compose import ColumnTransformer
# from sklearn.datasets import load_iris, make_classification
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import (  # AdaBoostClassifier,; BaggingClassifier,; ExtraTreesClassifier,; RandomForestClassifier,; RandomForestRegressor,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)

# from sklearn.feature_selection import f_classif, RFE, SelectFromModel, SelectKBest
# from sklearn.gaussian_process import GaussianProcessClassifier
# from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier, Perceptron  # Ridge,; SGDClassifier,
from sklearn.metrics import (  # auc,; classification_report,; confusion_matrix,; ConfusionMatrixDisplay,; mean_absolute_error,; mean_squared_error,; r2_score,; roc_auc_score,; roc_curve,; RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

# from sklearn.model_selection import (
#     cross_val_predict,
#     cross_val_score,
#     cross_validate,
#     GridSearchCV,
#     RandomizedSearchCV,
#     StratifiedShuffleSplit,
#     train_test_split,
# )
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier

# from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

# import numpy as np
from sklearn.preprocessing import MaxAbsScaler  # StandardScaler, MinMaxScaler, RobustScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier  # , XGBRegressor


RANDOM_STATE = 42


# TMP
def are_indices_equal(df1, df2):
    """Check if the indices of two DataFrames are equal."""
    return df1.index.equals(df2.index)


def impute_columns_knn(df, columns_to_impute, n_neighbors=4):
    # Create a copy of the dataframe
    df_imputed = df.copy()

    # Select only the columns to impute
    columns_for_imputation = df[columns_to_impute].copy()

    # Scale the data
    # scaler = MinMaxScaler()
    # scaled_data = scaler.fit_transform(columns_for_imputation)
    scaled_data = columns_for_imputation

    # Apply KNN Imputer
    imputer = KNNImputer(n_neighbors=n_neighbors)
    imputed_scaled_data = imputer.fit_transform(scaled_data)

    # Inverse transform back to original scale
    # df_imputed[columns_to_impute] = scaler.inverse_transform(imputed_scaled_data)
    df_imputed = imputed_scaled_data

    return df_imputed


# def impute_values_by_gender(_df, col, new_value_0, new_value_1, old_value=None):
#     df = _df.copy()

#     if old_value is not None:
#         df[col] = df[col].where(~((df["gender"] == 0) & (df[col] == old_value)), new_value_0)
#         df[col] = df[col].where(~((df["gender"] == 1) & (df[col] == old_value)), new_value_1)
#     else:
#         # Replace NA/NaN values
#         df[col] = df[col].fillna(df["gender"].apply(lambda x: new_value_0 if x == 0 else new_value_1))

#     return df


# def tmp_impute_values_by_gender(X_train, X_test, columns, method="median", old_value=None):
#     for col in columns:
#         df = X_train[X_train[col] != old_value] if old_value is not None else X_train
#         value_0, value_1 = df.groupby("gender")[col].agg("median" if method == "median" else "mean")

#         X_train = impute_values_by_gender(X_train, col, value_0, value_1, old_value)
#         X_test = impute_values_by_gender(X_test, col, value_0, value_1, old_value)

#     return X_train, X_test


def impute_column_by_gender(_df, col, new_value_0, new_value_1, old_value=None):
    """
    Impute a column based on gender, with different values for each gender.
    """
    df = _df.copy()

    if old_value is not None:
        df[col] = df[col].where(~((df["gender"] == 0) & (df[col] == old_value)), new_value_0)
        df[col] = df[col].where(~((df["gender"] == 1) & (df[col] == old_value)), new_value_1)
    else:
        # Replace NA/NaN values
        df[col] = df[col].fillna(df["gender"].apply(lambda x: new_value_0 if x == 0 else new_value_1))

    return df


def impute_columns_by_gender(X_train, X_test, columns, method="median", old_value=None):
    """
    Impute multiple columns in the training and test datasets based on gender,
    using either the median or mean for each gender.
    """
    for col in columns:
        # Filter training data to exclude rows with the old value (if provided)
        df = X_train[X_train[col] != old_value] if old_value is not None else X_train

        # Compute the replacement values for each gender
        value_0, value_1 = df.groupby("gender")[col].agg("median" if method == "median" else "mean")

        # Impute values in both datasets
        X_train = impute_column_by_gender(X_train, col, value_0, value_1, old_value)
        X_test = impute_column_by_gender(X_test, col, value_0, value_1, old_value)

    return X_train, X_test


def create_equality_features(_df, columns):
    """
    Create equality features for specified columns by comparing with their '_o' pairs.

    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    columns : list
        List of base column names to compare with their '_o' pairs

    Returns:
    --------
    pandas.DataFrame
        DataFrame with new equality features added

    Raises:
    -------
    ValueError
        If columns don't exist, partner columns don't exist, or NA values are found
    """
    df = _df.copy()

    for col in columns:
        # Check if base column exists
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in dataframe")

        # Check if partner column exists
        partner_col = f"{col}_o"
        if partner_col not in df.columns:
            raise ValueError(f"Partner column '{partner_col}' not found in dataframe")

        # Check for NA values
        if df[col].isna().any() or df[partner_col].isna().any():
            raise ValueError(f"NA values found in '{col}' or '{partner_col}'")

        # Create equality feature
        df[f"{col}_eq"] = (df[col] == df[partner_col]).astype(int)

    return df


def normalize_values_(val, replacement_dict):
    if val in replacement_dict:
        return replacement_dict[val]
    else:
        raise ValueError(f"Value {val} not found in the dictionary")


def questionnaire_impute_0(X_train, X_test, cols_main, cols_other, prefix, min_nan=1, max_nan=5, sum_value=100):
    X_train_copy = X_train.copy()
    X_test_copy = X_test.copy()
    nan_col = f"{prefix}_nan"
    sum_col = f"{prefix}_sum"
    nan_col_o = f"{prefix}_o_nan"
    sum_col_o = f"{prefix}_o_sum"

    X_train_copy[f"{prefix}_sum"] = X_train_copy[cols_main].sum(axis=1, skipna=True)
    X_train_copy[f"{prefix}_nan"] = X_train_copy[cols_main].isna().sum(axis=1)
    X_train_copy[f"{prefix}_o_sum"] = X_train_copy[cols_other].sum(axis=1, skipna=True)
    X_train_copy[f"{prefix}_o_nan"] = X_train_copy[cols_other].isna().sum(axis=1)

    X_test_copy[f"{prefix}_sum"] = X_test_copy[cols_main].sum(axis=1, skipna=True)
    X_test_copy[f"{prefix}_nan"] = X_test_copy[cols_main].isna().sum(axis=1)
    X_test_copy[f"{prefix}_o_sum"] = X_test_copy[cols_other].sum(axis=1, skipna=True)
    X_test_copy[f"{prefix}_o_nan"] = X_test_copy[cols_other].isna().sum(axis=1)

    mask = (
        (X_train_copy[nan_col] >= min_nan) & (X_train_copy[nan_col] <= max_nan) & (X_train_copy[sum_col] == sum_value)
    )
    X_train.loc[mask, cols_main] = X_train_copy.loc[mask, cols_main].fillna(0)

    mask = (
        (X_train_copy[nan_col_o] >= min_nan)
        & (X_train_copy[nan_col_o] <= max_nan)
        & (X_train_copy[sum_col_o] == sum_value)
    )
    X_train.loc[mask, cols_other] = X_train_copy.loc[mask, cols_other].fillna(0)

    mask = (X_test_copy[nan_col] >= min_nan) & (X_test_copy[nan_col] <= max_nan) & (X_test_copy[sum_col] == sum_value)
    X_test.loc[mask, cols_main] = X_test_copy.loc[mask, cols_main].fillna(0)

    mask = (
        (X_test_copy[nan_col_o] >= min_nan)
        & (X_test_copy[nan_col_o] <= max_nan)
        & (X_test_copy[sum_col_o] == sum_value)
    )
    X_test.loc[mask, cols_other] = X_test_copy.loc[mask, cols_other].fillna(0)

    return X_train, X_test


def check_consistency(df, id_column="id", consistent_columns=["age", "gender", "field"]):
    """
    Check if specified columns have consistent values for each unique ID in a DataFrame.

    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame containing the data
    id_column : str, default='id'
        Name of the ID column
    consistent_columns : list, default=['age', 'gender', 'field']
        List of column names that should have consistent values for each ID

    Returns:
    --------
    dict
        Dictionary containing:
        - 'is_consistent': bool indicating if all values are consistent
        - 'inconsistencies': dict with details of any inconsistencies found
    """
    inconsistencies = {}

    # Check each ID group
    for column in consistent_columns:
        # Group by ID and check if all values in the column are the same
        inconsistent_ids = df.groupby(id_column)[column].nunique()
        problematic_ids = inconsistent_ids[inconsistent_ids > 1]

        if not problematic_ids.empty:
            inconsistencies[column] = {"ids": problematic_ids.index.tolist(), "details": {}}

            # Get the actual inconsistent values for each problematic ID
            for id_value in problematic_ids.index:
                values = df[df[id_column] == id_value][column].unique()
                inconsistencies[column]["details"][id_value] = values.tolist()

    return {"is_consistent": len(inconsistencies) == 0, "inconsistencies": inconsistencies}


def compare_columns(df1, col1, df2, col2):
    # Check dtype
    dtype_match = df1[col1].dtype == df2[col2].dtype

    # Check number of elements
    length_match = len(df1[col1]) == len(df2[col2])

    # Check if values differ
    values_match = df1[col1].equals(df2[col2])

    return {"dtype_match": dtype_match, "length_match": length_match, "values_match": values_match}


def check_normality(data, max_samples=5000):
    """
    Check normality using multiple methods and sampling for large datasets.

    Parameters:
    -----------
    data : array-like
        The data to check for normality
    max_samples : int
        Maximum number of samples to use for normality testing

    Returns:
    --------
    bool
        True if the data appears normally distributed
    """
    # Flatten the data if it's 2D
    data = data.ravel()

    # If data is too large, take a random sample
    if len(data) > max_samples:
        data = np.random.choice(data, size=max_samples, replace=False)

    # Multiple tests for normality
    # 1. Skewness test
    skewness = stats.skew(data)
    # 2. Kurtosis test
    kurtosis = stats.kurtosis(data)
    # 3. D'Agostino and Pearson's test
    _, p_value = stats.normaltest(data)

    # Combine results
    is_normal = (
        abs(skewness) < 2  # Not too skewed
        and abs(kurtosis) < 7  # Not too heavy-tailed
        and p_value > 0.05  # Statistically significant
    )

    return is_normal


def auto_scaler(X, return_stats=False):
    """
    Automatically selects and applies the most appropriate scaler based on data characteristics.
    Improved to handle large datasets.

    Parameters:
    -----------
    X : array-like of shape (n_samples, n_features)
        The data to scale
    return_stats : bool, default=False
        If True, returns analysis stats along with scaled data and scaler

    Returns:
    --------
    X_scaled : array-like
        The scaled data
    scaler : object
        The fitted scaler object
    stats_dict : dict, optional
        Dictionary containing analysis statistics (if return_stats=True)
    """

    # Convert to numpy array if not already
    X = np.array(X) if not isinstance(X, np.ndarray) else X

    # Initialize dictionary to store analysis results
    stats_dict = {}

    # Check sparsity
    sparsity = np.sum(X == 0) / X.size
    stats_dict["sparsity"] = sparsity

    # Check for outliers using IQR method
    Q1 = np.percentile(X, 25, axis=0)
    Q3 = np.percentile(X, 75, axis=0)
    IQR = Q3 - Q1
    outlier_mask = ((X < (Q1 - 1.5 * IQR)) | (X > (Q3 + 1.5 * IQR))).any(axis=1)
    outlier_proportion = np.mean(outlier_mask)
    stats_dict["outlier_proportion"] = outlier_proportion

    # Check normality for each feature
    normality_tests = []
    for feature in range(X.shape[1]):
        is_normal = check_normality(X[:, feature])
        normality_tests.append(is_normal)
    normality_proportion = np.mean(normality_tests)
    stats_dict["normality_proportion"] = normality_proportion

    # Check value range
    value_range = np.ptp(X, axis=0)
    max_range_ratio = np.max(value_range) / np.min(value_range) if np.min(value_range) != 0 else np.inf
    stats_dict["max_range_ratio"] = max_range_ratio

    # Decision logic
    if sparsity > 0.3:
        scaler = MaxAbsScaler()
        reason = "High sparsity detected - using MaxAbsScaler"
    elif outlier_proportion > 0.1:
        scaler = RobustScaler()
        reason = "Significant outliers detected - using RobustScaler"
    elif normality_proportion > 0.7 and max_range_ratio > 10:
        scaler = StandardScaler()
        reason = "Approximately normal distribution with different scales - using StandardScaler"
    else:
        scaler = MinMaxScaler()
        reason = "Default case - using MinMaxScaler for general purpose scaling"

    stats_dict["selected_scaler"] = scaler.__class__.__name__
    stats_dict["reason"] = reason

    # Fit and transform the data
    X_scaled = scaler.fit_transform(X)

    if return_stats:
        return X_scaled, scaler, stats_dict
    return X_scaled, scaler


def scale_dataframe_by_column(df):
    """
    Scales each column of a dataframe independently using auto_scaler

    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe with numeric columns

    Returns:
    --------
    scaled_df : pandas.DataFrame
        Scaled dataframe
    scalers : dict
        Dictionary of scalers used for each column
    """
    scaled_data = {}
    scalers = {}

    # for column in df.select_dtypes(include=['float64', 'int64']).columns:
    for column in df.select_dtypes(include=["int32"]).columns:
        # Reshape to 2D array as scalers expect 2D input
        data = df[column].values.reshape(-1, 1)

        # Apply auto_scaler and get results
        scaled_values, scaler, stats = auto_scaler(data, return_stats=True)

        # Store results
        scaled_data[column] = scaled_values.ravel()
        scalers[column] = {"scaler": scaler, "stats": stats}

        print(f"\nColumn: {column}")
        print_scaling_report(stats)

    # Create new dataframe with scaled values
    scaled_df = pd.DataFrame(scaled_data, index=df.index)

    # Add any non-numeric columns back if needed
    non_numeric = df.select_dtypes(exclude=["float64", "int64"]).columns
    if len(non_numeric) > 0:
        scaled_df[non_numeric] = df[non_numeric]

    return scaled_df, scalers


def print_scaling_report(stats_dict):
    """
    Prints a formatted report of the scaling analysis.
    """
    print(f"- Sparsity:                       {stats_dict['sparsity']:.2%}")
    print(f"- Outlier Proportion:             {stats_dict['outlier_proportion']:.2%}")
    print(f"- Normal Distribution Proportion: {stats_dict['normality_proportion']:.2%}")
    print(f"- Maximum Range Ratio:            {stats_dict['max_range_ratio']:.2f}")
    print(f"- Selected Scaler: {stats_dict['selected_scaler']}. Reason: {stats_dict['reason']}")


def one_hot_encode(X_train, X_test, cols_to_encode):
    """
    One-hot encodes the specified categorical columns in the training and test data.

    Parameters:
    X_train (pandas.DataFrame): Training data.
    X_test (pandas.DataFrame): Test data.
    cols_to_encode (list): List of column names to encode.

    Returns:
    pandas.DataFrame, pandas.DataFrame: Encoded training and test data.
    """
    # Fit the encoder on the selected columns in X_train
    ohe = OneHotEncoder()
    X_train_encoded = pd.DataFrame(
        ohe.fit_transform(X_train[cols_to_encode]).toarray(),
        columns=ohe.get_feature_names_out(cols_to_encode),
        index=X_train.index,
    )

    # Transform the X_test data using the same encoder
    X_test_encoded = pd.DataFrame(
        ohe.transform(X_test[cols_to_encode]).toarray(),
        columns=ohe.get_feature_names_out(cols_to_encode),
        index=X_test.index,
    )

    cols_encoded = X_train_encoded.columns.to_list()
    X_train[cols_encoded] = X_train_encoded[cols_encoded]
    X_test[cols_encoded] = X_test_encoded[cols_encoded]

    return X_train, X_test


# feature selection using Random forests
def feature_selection_rf(X_train, y_train):
    rf = RandomForestClassifier()

    # Fit the classifier
    rf.fit(X_train, y_train)

    # Retrieve the feature importances
    rf_importance = pd.Series(rf.feature_importances_, index=rf.feature_names_in_)
    rf_importance = rf_importance.sort_values(ascending=False)
    return rf_importance


# Feature selection using custom xgb + f1 score
def feature_selection_f1_based(model, X_train, y_train, X_test, y_test, target_features=12, step=1):
    n_features_now = len(X_train.columns)
    selected_features = list(X_train.columns)  # Start with all features
    least_important_features = []
    best_f1_based_on_predict = 0
    best_f1_based_on_predict_proba = 0

    model.fit(X_train[selected_features], y_train)
    feature_importances = dict(zip(selected_features, model.feature_importances_))
    feature_importances = [
        (feature, round(importance, 3))
        for feature, importance in sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
    ]
    print_feature_importances(feature_importances)

    for _ in range(n_features_now - target_features):
        # Fit model on current set of selected features
        model.fit(X_train[selected_features], y_train)

        # Make predictions and evaluate F1 score
        y_pred = model.predict(X_test[selected_features])
        current_f1 = f1_score(y_test, y_pred)
        best_predict_proba_threshold, best_predict_proba_f1_score = get_predict_proba_best_threshold_and_f1(
            model, X_test[selected_features], y_test
        )
        print(f"\nNumber of features: {len(selected_features)}")
        print(
            f"F1 Score: {current_f1:.4f}, predict_proba f1: {best_predict_proba_f1_score:.4f}, predict_proba threshold: {best_predict_proba_threshold:.4f}"
        )
        print("Removed features:", ", ".join(f"'{item}'" for item in least_important_features))

        # Calculate feature importances and remove least important
        importances = model.feature_importances_
        least_important_features.append(selected_features[np.argmin(importances)])
        selected_features.remove(selected_features[np.argmin(importances)])

        best_f1_based_on_predict = max(best_f1_based_on_predict, current_f1)
        best_f1_based_on_predict_proba = max(best_f1_based_on_predict_proba, best_predict_proba_f1_score)

    print(f"Best predict/proba scores: {best_f1_based_on_predict:.4f}/{best_f1_based_on_predict_proba:.4f}")
    print_info(f"Best predict/proba scores: {best_f1_based_on_predict:.4f}/{best_f1_based_on_predict_proba:.4f}")

    return None


# RFE2
def feature_selection_rfe_xgb2(X_train, y_train, X_test, y_test):
    xgb_cl = xgb.XGBClassifier(
        max_depth=6,
        min_child_weight=5,
        gamma=0.4,
        learning_rate=0.1,
        n_estimators=500,
        subsample=0.6,
        colsample_bytree=1,
        reg_alpha=0.6,
        reg_lambda=0.2,
        random_state=RANDOM_STATE,
    )
    n_features_now = len(X_train.columns)
    all_features = list(X_train.columns)

    for features in list(range(n_features_now, n_features_now - 12, -1)):
        # Instantiate and fit the RFE
        rfe = RFE(estimator=xgb_cl, n_features_to_select=features)
        rfe.fit(X_train, y_train)

        # Get selected features
        selected_features = X_train.columns[rfe.support_].tolist()
        removed_features = [f for f in all_features if f not in selected_features]

        # Make predictions
        y_pred = rfe.predict(X_test)
        f1 = f1_score(y_test, y_pred)

        print(f"# features: {features}, f1: {f1:.4f}. ", end="")
        # print(f'F1 Score: {f1:.4f}')
        print("Removed features:", ", ".join(removed_features))


# 🧡MII_REF15 explain that X, y are train, why we use stratified split
# 🧡MII_REF15 do not print but return and print in the caller!
def feature_selection_f1_based_cv(
    model, _X, _y, stratify_var, target_features=12, n_splits=5, random_state=RANDOM_STATE
):
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
    X, y = _X.copy(), _y.copy()
    n_features_now = len(X.columns)
    selected_features = list(X.columns)  # Start with all features
    least_important_features = []
    best_f1_based_on_predict_proba = 0
    best_predict_proba_threshold = 0
    best_selected_features = []

    # Initial fit to get feature importances
    model.fit(X[selected_features], y)
    feature_importances = dict(zip(selected_features, model.feature_importances_))
    feature_importances = [
        (feature, round(importance, 3))
        for feature, importance in sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
    ]
    print_feature_importances(feature_importances)

    for _ in range(n_features_now - target_features):
        # Get best threshold and F1 score using predict_proba
        current_predict_proba_threshold, current_predict_proba_f1_score = get_predict_proba_best_threshold_and_f1_cv2(
            model, X[selected_features], y, stratify_var, n_splits=n_splits
        )

        print(f"\nNumber of features: {len(selected_features)}")
        print(
            f"predict_proba f1: {current_predict_proba_f1_score:.4f}, predict_proba threshold: {current_predict_proba_threshold:.4f}"
        )
        print("Removed features:", ", ".join(f"'{item}'" for item in least_important_features))

        # Fit model on full dataset to get feature importances
        model.fit(X[selected_features], y)
        importances = model.feature_importances_

        # Update best scores
        if current_predict_proba_f1_score >= best_f1_based_on_predict_proba:
            best_f1_based_on_predict_proba = current_predict_proba_f1_score
            best_predict_proba_threshold = current_predict_proba_threshold
            best_selected_features = selected_features

        # Remove least important feature
        least_important_features.append(selected_features[np.argmin(importances)])
        selected_features.remove(selected_features[np.argmin(importances)])

    print(f"\nBest predict_proba scores: {best_f1_based_on_predict_proba:.4f}")

    return best_f1_based_on_predict_proba, best_predict_proba_threshold, best_selected_features


def generate_param_combinations(param_dict):
    """Generate all combinations of parameters for a parameter grid."""
    keys, values = zip(*param_dict.items())
    return [dict(zip(keys, v)) for v in product(*values)]


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


# MII_REF09 💛 explain that X, y are train, why we use stratified split
# MII_REF09 💛 do not print but return and print in the caller!
def feature_selection_precision_based_cv(model, _X, _y, stratify_var, target_features=12, n_splits=5, random_state=42):
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
    X, y = _X.copy(), _y.copy()
    n_features_now = len(X.columns)
    selected_features = list(X.columns)  # Start with all features
    least_important_features = []
    best_precision_based_on_predict_proba = 0
    best_predict_proba_threshold = 0
    best_selected_features = []

    # Initial fit to get feature importances
    model.fit(X[selected_features], y)
    feature_importances = dict(zip(selected_features, model.feature_importances_))
    feature_importances = [
        (feature, round(importance, 3))
        for feature, importance in sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
    ]
    print_feature_importances(feature_importances)

    for _ in range(n_features_now - target_features):
        # Get best threshold and precision score using predict_proba
        current_predict_proba_threshold, current_predict_proba_precision_score = (
            get_predict_proba_best_threshold_and_precision_cv(
                model, X[selected_features], y, stratify_var, n_splits=n_splits
            )
        )

        print(f"\nNumber of features: {len(selected_features)}")
        print(
            f"predict_proba precision: {current_predict_proba_precision_score:.4f}, predict_proba threshold: {current_predict_proba_threshold:.4f}"
        )
        print("Removed features:", ", ".join(f"'{item}'" for item in least_important_features))

        # Fit model on full dataset to get feature importances
        model.fit(X[selected_features], y)
        importances = model.feature_importances_

        # Update best scores
        if current_predict_proba_precision_score >= best_precision_based_on_predict_proba:
            best_precision_based_on_predict_proba = current_predict_proba_precision_score
            best_predict_proba_threshold = current_predict_proba_threshold
            best_selected_features = selected_features

        # Remove least important feature
        least_important_features.append(selected_features[np.argmin(importances)])
        selected_features.remove(selected_features[np.argmin(importances)])

    print(f"\nBest predict_proba precision score: {best_precision_based_on_predict_proba:.4f}")

    return best_precision_based_on_predict_proba, best_predict_proba_threshold, best_selected_features


def get_predict_proba_best_threshold_and_precision_cv(model, X, y, stratify_var, n_splits=5, random_state=42):
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
        precision = precision_score(y, y_pred, zero_division=0)

        print(str(y))
        print(str(y_pred))

        if precision > best_precision_score:
            best_threshold = threshold
            best_precision_score = precision

    return best_threshold, best_precision_score


def get_predict_proba_best_threshold_and_f1_cv2(model, X, y, stratify_var, n_splits=5, random_state=42):
    """
    Optimize the classification threshold and calculate the F1 score using cross_val_predict.

    Parameters:
    model (sklearn model): The trained model with a `predict_proba()` method.
    X (numpy array): The feature matrix.
    y (numpy array): The target labels.
    cv (int): Number of folds for cross-validation. Default is 5.
    random_state (int): Random seed for reproducibility. Default is 42.

    Returns:
    float: The best threshold.
    float: The best F1 score.
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

    # Initialize variables to store the best threshold and F1 score
    best_threshold = 0.5
    best_f1_score = 0

    # Test each threshold
    for threshold in thresholds:
        y_pred = (y_pred_proba[:, 1] > threshold).astype(int)
        f1 = f1_score(y, y_pred)

        if f1 > best_f1_score:
            best_threshold = threshold
            best_f1_score = f1

    return best_threshold, best_f1_score


def get_predict_proba_best_threshold_and_f1(model, X_test, y_test):
    """
    Optimize the classification threshold and calculate the F1 score.

    Parameters:
    model (sklearn model): The trained model with a `predict_proba()` method.
    X_test (numpy array): The test features.
    y_test (numpy array): The true test labels.

    Returns:
    float: The best threshold.
    float: The best F1 score.
    """
    # Get the probability estimates
    y_pred_proba = model.predict_proba(X_test)

    # Define a range of thresholds to try
    thresholds = np.linspace(0.1, 0.9, 18)

    # Initialize variables to store the best threshold and F1 score
    best_threshold = 0.5
    best_f1_score = 0

    # Iterate over the thresholds and calculate the F1 score
    for threshold in thresholds:
        # Apply the threshold to get binary predictions
        y_pred = (y_pred_proba[:, 1] > threshold).astype(int)

        # Calculate the F1 score
        f1 = f1_score(y_test, y_pred)

        # Update the best threshold and F1 score if necessary
        if f1 > best_f1_score:
            best_threshold = threshold
            best_f1_score = f1

    return best_threshold, best_f1_score


def optimize_feature_order_sequential2(X_train, X_test, y_train, y_test):
    """
    Determine the optimal order of features for XGBClassifier to achieve the best F1 score using a sequential feature selection approach.

    Parameters:
    X_train (pandas.DataFrame): Feature data for training
    X_test (pandas.DataFrame): Feature data for testing
    y_train (pandas.Series): Target variable for training
    y_test (pandas.Series): Target variable for testing

    Returns:
    list: Optimal order of features
    """
    # Create a list of feature names
    features = list(X_train.columns)

    # Initialize the best F1 score and feature order
    best_f1 = 0
    best_order = []

    # Perform sequential feature selection
    for i in range(len(features)):
        print(f"{i}/{len(features)}")
        best_feature = None
        best_f1_so_far = best_f1
        for feature in features:
            if feature not in best_order:
                # Check if the feature is present in the X_train DataFrame
                if feature in X_train.columns:
                    X_train_ordered = X_train[best_order + [feature]]
                    X_test_ordered = X_test[best_order + [feature]]
                    model = XGBClassifier()
                    model.fit(X_train_ordered, y_train)
                    y_pred = model.predict(X_test_ordered)
                    f1 = f1_score(y_test, y_pred)
                    if f1 > best_f1_so_far:
                        best_feature = feature
                        best_f1_so_far = f1
        if best_feature is not None:
            best_order.append(best_feature)
            best_f1 = best_f1_so_far

    return best_order


def optimize_feature_order_greedy3(X_train, X_test, y_train, y_test):
    """
    Determine the optimal order of features for XGBClassifier to achieve the best F1 score using a greedy approach.

    Parameters:
    X_train (pandas.DataFrame): Feature data for training
    X_test (pandas.DataFrame): Feature data for testing
    y_train (pandas.Series): Target variable for training
    y_test (pandas.Series): Target variable for testing

    Returns:
    list: Optimal order of features
    """
    # Create a list of feature names
    features = list(X_train.columns)

    # Initialize the best F1 score and feature order
    best_f1 = 0
    best_order = features

    # Reorder features greedily
    for i in range(len(features)):
        print(f"{i}/{len(features)}")
        best_feature = None
        best_f1_so_far = best_f1
        for feature in features:
            if feature not in best_order[:i]:
                # Check if the feature is present in the X_train DataFrame
                if feature in X_train.columns:
                    X_train_ordered = X_train[best_order[:i] + [feature]]
                    X_test_ordered = X_test[best_order[:i] + [feature]]
                    model = XGBClassifier()
                    model.fit(X_train_ordered, y_train)
                    y_pred = model.predict(X_test_ordered)
                    f1 = f1_score(y_test, y_pred)
                    if f1 > best_f1_so_far:
                        best_feature = feature
                        best_f1_so_far = f1
        if best_feature is not None:
            best_order = best_order[:i] + [best_feature]
            best_f1 = best_f1_so_far

    return best_order


def visualize_goal_decision(df):
    """
    Visualizes the value counts of 'dec' grouped by 'goal' and gender.

    Parameters:
    df (pandas.DataFrame): The input DataFrame, assumed to have a 'gender' column
    """
    # Group by 'goal', 'dec', and 'gender' then count the values
    goal_dec_gender_counts = df.groupby(["goal", "dec", "gender"])["dec"].count().unstack(fill_value=0)

    # Plot the data as a stacked bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    goal_dec_gender_counts.plot(kind="bar", stacked=True, ax=ax)

    # Add labels to the bars
    for i, p in enumerate(ax.patches):
        width, height = p.get_width(), p.get_height()
        x, y = p.get_xy()

        # Extract the gender and value for this bar
        gender = list(goal_dec_gender_counts.columns)[i % len(goal_dec_gender_counts.columns)]
        value = int(height)

        # Display the value and gender on the bar
        ax.annotate(f"{value}\n{gender}", (x + width / 2, y + height * 1.01), ha="center")

    # Add labels and title
    ax.set_xlabel("Goal")
    ax.set_ylabel("Count")
    ax.set_title("Goal vs. Decision Value Counts by Gender")

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)

    # Display the plot
    plt.show()
