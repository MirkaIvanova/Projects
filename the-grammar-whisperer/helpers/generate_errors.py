import ast
import pandas as pd
import numpy as np
from enum import Enum
from copy import deepcopy
import re
import sys


class BgError(Enum):
    NA = 0
    NO_ERROR = 1
    DEFINITE_ARTICLE = 2
    MISSING_COMMA = 3


class VocabularyCase(Enum):
    FULL = 1  # "непълен член"
    SHORT = 2  # "пълен член"
    DEFINITE = 3  # "членувано"     (neither short nor long, for example for feminine/neuter nouns)


def map_pos_sentence_to_vocabulary(pos):
    pos_map = {"ADJ": 3, "NOUN": 2}
    if pos in pos_map:
        return pos_map[pos]

    print(f"❌Error: POS {pos} not found in vocabulary")
    return None


def join_words_with_punctuation(words):
    res = " ".join(word for word in words)

    quotes_pattern = r'["\'\[\(] ([^\]\)\'"]+) ["\'\]\)]'  # Pattern 1: Matches quoted/bracketed text with spaces
    comma_pattern = r"\s+([,.!?;])"  # Pattern 2: Matches spaces before punctuation marks

    res = re.sub(quotes_pattern, lambda m: f"{m.group(0)[0]}{m.group(1)}{m.group(0)[-1]}", res)
    res = re.sub(comma_pattern, r"\1", res)
    return res


def find_opposite_case_by_lemma(df_vocabulary, lemma, other_case, pos_in_vocab):
    df_vocabulary_search = df_vocabulary[
        (df_vocabulary.lemma == lemma)
        & (df_vocabulary.definite_article == other_case)
        & (df_vocabulary.number == 1)
        & (df_vocabulary.participle == 0)
        & (df_vocabulary.pos_encoded == pos_in_vocab)
    ]

    num_search_results = df_vocabulary_search.shape[0]

    if num_search_results == 1:
        other_word = df_vocabulary_search.iloc[0]["word"]
        return other_word

    return None


def find_opposite_case_by_word(df_vocabulary, word, case_vocab, other_case_vocab, pos_vocab):
    df_vocabulary_search = df_vocabulary[
        (df_vocabulary.word == word) & (df_vocabulary.definite_article == case_vocab) & (df_vocabulary.participle == 0)
    ]

    num_search_results = df_vocabulary_search.shape[0]

    if num_search_results == 1:
        lemma_vocab = df_vocabulary_search.iloc[0]["lemma"]
        other_word = find_word_opposite_case_by_lemma(df_vocabulary, lemma_vocab, other_case_vocab, pos_vocab)
        if other_word:
            return other_word

    return None


def find_opposite_case_by_rules(word: str, pos: str) -> str:
    """
    Switches between short and long definite article forms for Bulgarian nouns and adjectives.

    Parameters:
        word (str): The word with definite article
        pos (str): Part of speech - 'NOUN' or 'ADJ'

    Returns:
        str: Word with swapped definite article form

    Rules for definite article forms in Bulgarian:
    Masculine nouns use:
        Long form (-ът/-ят) for subjects
        Short form (-а/-я) for objects
        Choice depends on final consonant:
        Use -ът/-а for hard consonants (столът → стола)
        Use -ят/-я for soft consonants (героят → героя)

    Masculine adjectives use:
        Long form (-ият) for subject position
        Short form (-ия) for object position

    Exceptions:
        Some masculine nouns ending in -а/-я use feminine articles:
        баща → бащата (not *бащаът)
        съдия → съдията (not *съдият)
    """
    # Noun rules
    if pos == "NOUN":
        # Exceptions for masculine nouns ending in -а/-я
        if word in ["баща", "съдия"]:
            return word + "та"
        elif word in ["бащата", "съдията"]:
            return word[:-2]

        # Long to short conversion
        elif word.endswith("ът"):
            return word[:-2] + "а"
        elif word.endswith("ят"):
            return word[:-2] + "я"
        # Short to long conversion
        elif word.endswith("а"):
            return word[:-1] + "ът"
        elif word.endswith("я"):
            return word[:-1] + "ят"

    # Adjective rules
    elif pos == "ADJ":
        # Long to short conversion
        if word.endswith("ият"):
            return word[:-1]  # remove final 'т'
        # Short to long conversion
        elif word.endswith("ия"):
            return word + "т"

    # Return original if no patterns match
    return None


def get_opposite_case_word(df_vocabulary, word, lemma, pos, case):
    # if this is the full definite article, find the short form of the word
    # if this is the short definite article, find the full form of the word
    if case == "f":
        case_vocab = VocabularyCase.SHORT.value
        other_case_vocab = VocabularyCase.FULL.value
    elif case == "h":
        case_vocab = VocabularyCase.FULL.value
        other_case_vocab = VocabularyCase.SHORT.value
    else:
        print("❌Error: case is not 'f'/full or 'h'/short")
        return None

    # First search in vocabulary by lemma:
    pos_vocab = map_pos_sentence_to_vocabulary(pos)
    if not pos_vocab:
        print(f"POS not found in vocabulary for word {word}")

    # other_word = find_opposite_case_by_lemma(df_vocabulary, lemma, other_case_vocab, pos_vocab)
    # if other_word:
    #     return other_word

    # # If not found, search in vocabulary by word
    # other_word = find_opposite_case_by_word(df_vocabulary, word, case_vocab, other_case_vocab, pos_vocab)
    # if other_word:
    #     return other_word

    # If not found, use rule-based conversion
    other_word = find_opposite_case_by_rules(word, pos)
    if other_word:
        # print(f"⚠️ Using rule-based conversion: {word}/{lemma}/{pos}/{case} -> {other_word}")
        return other_word

    print(f"❌Error: {word}/{lemma}/{pos}/{case}: other word not found in vocabulary.")
    return None


indices_with_errors = []


def add_error_definite_article(df, index, row, pos, words, lemmas, genders, numbers, cases):
    length = len(genders)
    new_rows = []

    for i in range(length):
        # search for nouns or adjectives in masculine singular form, definite article either short or full
        if pos[i] in ["NOUN", "ADJ"] and genders[i] == "M" and numbers[i] == "S" and cases[i] in "fh":
            new_word = get_opposite_case_word(df_vocabulary, words[i], lemmas[i], pos[i], cases[i])
            if new_word:
                new_words = deepcopy(words)
                new_words[i] = new_word
                new_words_list = join_words_with_punctuation(new_words)
                new_index = df.index.max() + 1
                df.loc[new_index] = np.nan

                df.loc[new_index, "sentence"] = "".join(new_words_list)
                df.loc[new_index, "error_type"] = BgError.DEFINITE_ARTICLE.value
                df.loc[new_index, "parent_row"] = index

                df.loc[index, "error_type"] = BgError.NO_ERROR.value
            else:
                indices_with_errors.append(index)
    return new_rows


def add_error_missing_comma(df, index, row, words):
    new_rows = []

    for i, word in enumerate(words):
        if word == ",":
            new_words = deepcopy(words)
            del new_words[i]
            new_words_list = join_words_with_punctuation(new_words)
            new_index = df.index.max() + 1

            df.loc[new_index, "sentence"] = "".join(new_words_list)
            df.loc[new_index, "error_type"] = BgError.MISSING_COMMA.value
            df.loc[new_index, "parent_row"] = index

            df.loc[index, "error_type"] = BgError.NO_ERROR.value

    return new_rows


def add_errors_all_rows(_df, df_vocabulary):
    df = _df.copy(deep=True)
    new_rows = []
    for index, row in df.iterrows():
        pos = ast.literal_eval(row["pos"])
        words = ast.literal_eval(row["words"])
        lemmas = ast.literal_eval(row["lemmas"])
        genders = row["gender"]
        numbers = row["number"]
        cases = row["case"]
        # new_rows = new_rows + add_error_definite_article(df, index, row, pos, words, lemmas, genders, numbers, cases)
        new_rows = new_rows + add_error_missing_comma(df, index, row, words)

    # Concatenate the original DataFrame and the new rows that contain errors
    df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=False)

    columns_to_convert = ["error_type", "parent_row"]
    df[columns_to_convert] = df[columns_to_convert].astype("int32")

    return df


if __name__ == "__main__":

    root_dir = "."
    data_processed_dir = f"{root_dir}/data/processed"

    sentences_csv = f"{data_processed_dir}/sent_wikipedia_nlp_features_stanza_final_10000_tmp.csv"
    # sentences_csv = r"C:\mirka\git\softuni\04.deep_learning\Project\tmp\deleteme_sent_fiction_nlp_features_part1_v1_checkpoint124.tsv"
    vocabulary_csv = f"{data_processed_dir}/bg_vocabulary_final2.csv"
    df = pd.read_csv(sentences_csv)
    df = df.iloc[0:10001]
    # df = df.iloc[0:1001]
    # df.iloc[0:10000].to_csv(f"{data_processed_dir}/sent_wikipedia_nlp_features_stanza_final_10000_tmp.csv", index=False)
    # sys.exit(1)
    df_vocabulary = pd.read_csv(vocabulary_csv)
    df_vocabulary = df_vocabulary[df_vocabulary["participle"] == 0]

    start_time = pd.Timestamp.now()

    df["error_type"] = BgError.NA.value
    df["parent_row"] = -1

    df1 = add_errors_all_rows(df, df_vocabulary)
    for i, iterrow in df1.iloc[0:10].iterrows():
        print(f"[{i}] e={iterrow['error_type']} p={iterrow['parent_row']} {iterrow['sentence']}")

    print(f"New df size: {df1.shape}")
    print("Time elapsed: ", pd.Timestamp.now() - start_time)
    print("Indices with errors: ", indices_with_errors)
