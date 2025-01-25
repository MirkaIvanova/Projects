import ast
import pandas as pd
import numpy as np
from enum import Enum
import re
from typing import List, Tuple, Set, Dict

from compare_dataframes import compare_dataframes


class BgError(Enum):
    NA = 0
    NO_ERROR = 1
    DEFINITE_ARTICLE = 2
    MISSING_COMMA = 3


# Precompile regex patterns for punctuation adjustment
QUOTES_REGEX = re.compile(r'(["\'({\[])\s+([^\]\)\'"]+)\s+(["\')}\]])')
COMMA_REGEX = re.compile(r"\s+([,.!?;])")


def join_words_with_punctuation(words: List[str]) -> str:
    """Join words and adjust punctuation spacing using precompiled regex patterns."""
    res = " ".join(words)
    res = QUOTES_REGEX.sub(r"\1\2\3", res)
    res = COMMA_REGEX.sub(r"\1", res)
    return res


def find_opposite_case_by_rules(word: str, pos: str) -> str:
    """Rule-based switching between definite article forms."""
    if pos == "NOUN":
        if word in ["баща", "съдия"]:
            return word + "та"
        elif word in ["бащата", "съдията"]:
            return word[:-2]
        elif word.endswith("ът"):
            return word[:-2] + "а"
        elif word.endswith("ят"):
            return word[:-2] + "я"
        elif word.endswith("а"):
            return word[:-1] + "ът"
        elif word.endswith("я"):
            return word[:-1] + "ят"
    elif pos == "ADJ":
        if word.endswith("ият"):
            return word[:-1]
        elif word.endswith("ия"):
            return word + "т"
    return None


def get_opposite_case_word(word: str, lemma: str, pos: str, case: str) -> str:
    """Wrapper for rule-based article conversion."""
    return find_opposite_case_by_rules(word, pos)


def is_eligible_for_definite_article(pos: str, gender: str, number: str, case: str) -> bool:
    """Check if a word is eligible for definite article error."""
    return pos in ["NOUN", "ADJ"] and gender == "M" and number == "S" and case in "fh"


def create_error_row(sentence: str, error_type: BgError, parent_row: int) -> Dict:
    """Create a standardized error row dictionary."""
    return {"sentence": sentence, "error_type": error_type.value, "parent_row": parent_row}


def add_error_definite_article(index: int, row: pd.Series) -> Tuple[List[Dict], Set[int]]:
    """Generate definite article errors for a row."""
    new_rows = []
    modified_indices = set()

    pos_list = row["pos"]
    words = row["words"]
    lemmas = row["lemmas"]
    genders = row["gender"]
    numbers = row["number"]
    cases = row["case"]

    for i in range(len(pos_list)):
        if is_eligible_for_definite_article(pos_list[i], genders[i], numbers[i], cases[i]):
            new_word = get_opposite_case_word(words[i], lemmas[i], pos_list[i], cases[i])
            if new_word:
                new_words = words.copy()
                new_words[i] = new_word
                new_sentence = join_words_with_punctuation(new_words)
                new_rows.append(create_error_row(new_sentence, BgError.DEFINITE_ARTICLE, index))
                modified_indices.add(index)

    return new_rows, modified_indices


def add_error_missing_comma(index: int, row: pd.Series) -> Tuple[List[Dict], Set[int]]:
    """Generate missing comma errors for a row."""
    new_rows = []
    modified_indices = set()
    words = row["words"]

    comma_indices = [idx for idx, word in enumerate(words) if word == ","]
    for i in reversed(comma_indices):
        new_words = words.copy()
        del new_words[i]
        new_sentence = join_words_with_punctuation(new_words)
        new_rows.append(create_error_row(new_sentence, BgError.MISSING_COMMA, index))
        modified_indices.add(index)

    return new_rows, modified_indices


def create_error_dataframe(all_new_rows: List[Dict], original_columns: pd.Index) -> pd.DataFrame:
    """Create DataFrame from error rows with original columns structure."""
    if not all_new_rows:
        return pd.DataFrame()

    new_df = pd.DataFrame(all_new_rows)
    for col in original_columns:
        if col not in new_df:
            new_df[col] = np.nan
    return new_df


def update_error_types(df: pd.DataFrame, modified_indices: Set[int]) -> pd.DataFrame:
    """Update error types for modified original rows."""
    if modified_indices:
        df.loc[df.index.isin(modified_indices), "error_type"] = BgError.NO_ERROR.value
    return df


def adjust_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure correct data types for specific columns."""
    df["error_type"] = df["error_type"].astype("int32")
    df["parent_row"] = df["parent_row"].astype("int32")
    return df


def add_errors_all_rows(_df: pd.DataFrame) -> pd.DataFrame:
    """Process DataFrame to generate error variants."""
    df = _df.copy()
    all_new_rows = []
    modified_indices = set()

    for index, row in df.iterrows():
        definite_rows, definite_modified = add_error_definite_article(index, row)
        comma_rows, comma_modified = add_error_missing_comma(index, row)
        all_new_rows.extend(definite_rows + comma_rows)
        modified_indices.update(definite_modified | comma_modified)

    new_df = create_error_dataframe(all_new_rows, df.columns)
    if not new_df.empty:
        df = pd.concat([df, new_df], ignore_index=True)

    df = update_error_types(df, modified_indices)
    return adjust_data_types(df)


def convert_list_columns(df: pd.DataFrame, list_cols: List[str]) -> pd.DataFrame:
    """Convert string-represented lists to actual lists in DataFrame."""
    for col in list_cols:
        if col not in df.columns:
            continue
        try:
            sample_value = df[col].dropna().iloc[0] if not df[col].empty else None
            if isinstance(sample_value, str):
                df[col] = df[col].apply(ast.literal_eval)
        except (IndexError, ValueError, SyntaxError):
            continue
    return df


def initialize_error_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Initialize error tracking columns."""
    df["error_type"] = BgError.NA.value
    df["parent_row"] = -1
    return df


def load_and_preprocess_data(file_path: str, list_cols: List[str]) -> pd.DataFrame:
    """Load data and preprocess list columns."""
    df = pd.read_csv(file_path, nrows=10001)
    df = convert_list_columns(df, list_cols)
    return initialize_error_columns(df)


if __name__ == "__main__":
    root_dir = "."
    data_processed_dir = f"{root_dir}/data/processed"
    sentences_csv = f"{data_processed_dir}/sent_wikipedia_nlp_features_stanza_final_10000_tmp.csv"

    df = load_and_preprocess_data(sentences_csv, ["pos", "words", "lemmas", "gender", "number", "case"])

    start_time = pd.Timestamp.now()
    df = add_errors_all_rows(df)

    print(f"Processed {len(df)} rows in {pd.Timestamp.now() - start_time}")
