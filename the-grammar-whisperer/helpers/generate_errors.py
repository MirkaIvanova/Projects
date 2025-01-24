import ast
import pandas as pd
from enum import Enum


class BgError(Enum):
    NO_ERROR = 1
    DEFINITE_ARTICLE = 2
    GREEN = 2
    BLUE = 3


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


def find_word_opposite_case_by_lemma(df_vocabulary, lemma, other_case, pos_in_vocab):
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


indices_with_errors = []


def get_opposite_case_word(df_vocabulary, word, lemma, pos, case):
    # print("Current word: ", word, pos, lemma, case)

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

    # search in vocabulary by lemma:
    pos_vocab = map_pos_sentence_to_vocabulary(pos)
    if not pos_vocab:
        print(f"POS not found in vocabulary for word {word}")

    other_word = find_word_opposite_case_by_lemma(df_vocabulary, lemma, other_case_vocab, pos_vocab)
    if other_word:
        return other_word

    # search in vocabulary by word
    df_vocabulary_search = df_vocabulary[
        (df_vocabulary.word == word) & (df_vocabulary.definite_article == case_vocab) & (df_vocabulary.participle == 0)
    ]

    num_search_results = df_vocabulary_search.shape[0]

    if num_search_results == 1:
        lemma_vocab = df_vocabulary_search.iloc[0]["lemma"]
        other_word = find_word_opposite_case_by_lemma(df_vocabulary, lemma_vocab, other_case_vocab, pos_vocab)
        if other_word:
            return other_word
    else:
        print(
            f"❌Error: {word}/{lemma}: other word not found in vocabulary, returned search results (by word): {num_search_results}, case: {case_vocab}/{other_case_vocab}, pos: {pos}/{pos_vocab}"
        )

    return None


def add_error_definite_article(df, df_vocabulary, index, row):
    pos = ast.literal_eval(row["pos"])
    words = ast.literal_eval(row["words"])
    lemmas = ast.literal_eval(row["lemmas"])
    genders = row["gender"]
    numbers = row["number"]
    cases = row["case"]
    length = len(genders)
    new_rows = []

    for i in range(length):
        if pos[i] in ["NOUN", "ADJ"] and genders[i] == "M" and numbers[i] == "S" and cases[i] in "fh":
            new_word = get_opposite_case_word(df_vocabulary, words[i], lemmas[i], pos[i], cases[i])
            # print("Index: ", index, words[i], lemmas[i], new_word)
            if new_word:
                new_row = row.copy()
                new_words = words.copy()
                new_words[i] = new_word

                new_words_list = [
                    " " + new_words[idx] if pos[idx] != "PUNCT" else new_words[idx] for idx in range(len(words))
                ]
                new_row["words"] = new_words
                new_row["sentence"] = "".join(new_words_list)
                new_row["error_type"] = BgError.DEFINITE_ARTICLE.value
                new_rows.append(new_row)
                df.loc[index, "error_type"] = BgError.NO_ERROR.value
            else:
                indices_with_errors.append(index)
    return new_rows


def add_errors_all_rows(_df, df_vocabulary):
    df = _df.copy()
    new_rows = []
    for index, row in df.iterrows():
        new_rows = new_rows + add_error_definite_article(df, df_vocabulary, index, row)

    # Concatenate the original DataFrame and the new rows that contain errors
    df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    return df


if __name__ == "__main__":
    from tabulate import tabulate

    root_dir = "."

    data_processed_dir = f"{root_dir}/data/processed"

    sentences_csv = f"{data_processed_dir}/sent_wikipedia_nlp_features_stanza_final_10000_tmp.csv"
    vocabulary_csv = f"{data_processed_dir}/bg_vocabulary_final2.csv"
    df = pd.read_csv(sentences_csv)
    df_vocabulary = pd.read_csv(vocabulary_csv)
    df_vocabulary = df_vocabulary[df_vocabulary["participle"] == 0]

    df["error_type"] = 0

    start_time = pd.Timestamp.now()

    df1 = add_errors_all_rows(df, df_vocabulary)
    # print(tabulate(df1[["sentence", "error_type"]], headers="keys", tablefmt="psql"))

    print("Time elapsed: ", pd.Timestamp.now() - start_time)

    print("Indices with errors: ", indices_with_errors)

    # df1 = df.iloc[0:10000]
    # df1.to_csv(f"{data_processed_dir}/sent_wikipedia_nlp_features_stanza_final_10000_tmp.csv", index=False)
