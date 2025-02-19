import pandas as pd


def compare_dataframes(df1, df2):
    """
    Compare two pandas DataFrames and find differences.

    Args:
        df1 (pd.DataFrame): The first DataFrame.
        df2 (pd.DataFrame): The second DataFrame.

    Returns:
        str or list: If shapes differ, returns an error message. Otherwise,
                     returns a list of up to 10 differences with index, column, and values.
    """
    # Check if shapes differ
    if df1.shape != df2.shape:
        return f"Error: DataFrames have different shapes: {df1.shape} vs {df2.shape}"

    # Collect up to 10 differences
    differences = []
    for idx in df1.index:
        for col in df1.columns:
            val1, val2 = df1.at[idx, col], df2.at[idx, col]
            if pd.isna(val1) and pd.isna(val2):
                continue
            if val1 != val2:
                differences.append({"index": idx, "column": col, "value_in_df1": val1, "value_in_df2": val2})
                if len(differences) == 10:
                    return differences

    return differences if differences else "DataFrames are identical."
