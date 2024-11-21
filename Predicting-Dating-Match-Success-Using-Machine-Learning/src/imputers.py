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
        X_train = _impute_column_by_gender(X_train, col, value_0, value_1, old_value)
        X_test = _impute_column_by_gender(X_test, col, value_0, value_1, old_value)

    return X_train, X_test


def _impute_column_by_gender(_df, col, new_value_0, new_value_1, old_value=None):
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


# @later separate train and test
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
