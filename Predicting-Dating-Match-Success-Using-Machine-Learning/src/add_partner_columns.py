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
