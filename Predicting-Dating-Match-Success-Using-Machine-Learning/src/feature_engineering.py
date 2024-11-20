# @later rename to add_partner_features
def add_partner_columns(_df, columns_to_map, suffix="_o"):
    """
    Adds new columns to the dataframe where values are taken from specified columns of the rows where iid equals the current row's pid.
    Parameters:
        df (pandas.DataFrame): Input dataframe with columns 'iid' and 'pid'
        columns_to_map (list): List of column names to map from iid to pid
        suffix (str): Suffix to append to the new column names (default: '_o')
    Returns: a copy of the dataframe including the new columns
    """
    df = _df.copy()

    for col in columns_to_map:
        # Create a mapping of iid to column value
        iid_to_value = df.set_index("iid")[col].to_dict()

        # Create the new column by mapping pid to the corresponding value
        new_col_name = f"{col}{suffix}"
        df[new_col_name] = df["pid"].map(iid_to_value)

    return df


def create_equality_features(_df, columns):
    """Create equality features for specified columns by comparing with their '_o' pairs."""
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
