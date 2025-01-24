import pandas as pd


def filter_rows_containing(_df, column_name, substrings_to_remove):
    df = _df.copy()

    for substring in substrings_to_remove:
        df = df[~df[column_name].str.contains(substring, na=False)]
    return df.reset_index(drop=True)


def trim_characters(text):
    if text is None or pd.isna(text):
        return text

    text = str(text)

    # Continue stripping until no valid enclosing pairs exist
    while True:
        if text.startswith('"') and text.endswith('"') and text.count('"') >= 2:
            text = text[1:-1].strip()
        elif text.startswith("(") and text.endswith(")"):
            text = text[1:-1].strip()
        elif text.startswith("[") and text.endswith("]"):
            text = text[1:-1].strip()
        elif text.startswith(")"):
            text = text[1:].strip()
        elif text.startswith("*"):
            text = text[1:].strip()
        elif text.startswith("-"):
            text = text[1:].strip()
        elif text.startswith('"') and text.count('"') == 1:
            text = text[1:].strip()
        elif text.endswith('"') and text.count('"') == 1:
            text = text[:-1].strip()
        else:
            # Break out of the loop when no further modifications are possible
            break

    return text


def remove_rows_start_end_with(_df, column_name, start_chars, end_chars):
    """
    Filters a DataFrame by removing rows where the specified column starts with specific symbols
    or ends with specific characters. Resets the index after filtering.
    """

    df = _df.copy()
    df = df[~df[column_name].str.startswith(tuple(start_chars))]
    df = df[~df[column_name].str.endswith(tuple(end_chars))].reset_index(drop=True)
    return df


def filter_rows_ending_with(_df, column, pattern):
    """Remove rows from a DataFrame where the specified column's values match a regex pattern."""
    df = _df.copy()
    removed_rows = df[~df[column].str.contains(pattern, regex=True, na=False)]
    df = df[~df.index.isin(removed_rows.index)]

    return df, removed_rows


def clean_all_sentences(_df):
    df = _df.copy()

    # remove rows containing those substrings
    substrings_to_remove = ["\n", "Б.пр.", "Бел.пр.", "Бележка:"]
    df = filter_rows_containing(df, "text", substrings_to_remove)
    df.shape

    # remove quotes/brackets from beginning/ending of sentence
    df["text"] = df["text"].apply(trim_characters)

    # remove rows starting or ending in any of these special characters
    start_chars = r'!&?$#({+,./:;<[=]_~|@^>"\\\''
    end_chars = "])"
    df = remove_rows_start_end_with(df, "text", start_chars, end_chars)

    # Remove rows where the 'text' column starts with a number
    df = df[~df["text"].str.match(r"^\d")]

    # remove duplicate rows and sort
    df = df.drop_duplicates()
    df = df.sort_values(by=["text"])

    # filter rows not ending with the usual punctuation.
    pattern = r'[.!?;"]$'
    df, removed_rows = filter_rows_ending_with(df, "text", pattern)

    return df
