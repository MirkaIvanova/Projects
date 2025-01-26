import ast
import pandas as pd


def parse_nlp_morphtag(tag: str) -> dict:
    """
    Parse Bulgarian Universal Dependencies notation.

    Examples:
    'Ncmsf' - Noun common masculine singular full
    'Ppxta' - Pronoun
    'Vpitf-r3s' - Verb personal indicative present full-reflexive 3rd singular
    'punct' - Punctuation
    """

    pos_map = {
        "N": "noun",
        "V": "verb",
        "A": "adjective",
        "R": "adverb",
        "M": "numeral",
        "P": "pronoun",
        "punct": "punctuation",
    }
    gender_map = {"m": "masculine", "f": "feminine", "n": "neuter"}
    number_map = {"s": "singular", "p": "plural"}
    case_map = {"d": "definite", "i": "indefinite", "f": "full", "h": "short"}  # Adding short form case
    verb_form_map = {"f": "finite", "o": "participle", "c": "converb"}

    pos = tag[0]
    features = {}

    features["xpos"] = tag
    features["pos"] = pos_map.get(tag[0])
    if len(tag) > 1:
        if pos == "N":  # Noun
            if len(tag) >= 2:
                features["type"] = "common" if tag[1] == "c" else "proper"
            if len(tag) >= 3:
                features["gender"] = gender_map.get(tag[2])
            if len(tag) >= 4:
                features["number"] = number_map.get(tag[3])
            if len(tag) >= 5:
                if tag[4] != "-":
                    features["case"] = case_map.get(tag[4])

        elif pos == "V":  # Verb
            if len(tag) > 4:
                features["type"] = {"p": "personal", "n": "impersonal"}.get(tag[1])
                features["mood"] = {"i": "indicative", "p": "participle"}.get(tag[2])
                features["tense"] = {"t": "present", "p": "past"}.get(tag[3])
                features["form"] = verb_form_map.get(tag[4])

                if "-" in tag:
                    remaining = tag.split("-")[1]
                    if len(remaining) >= 1:
                        features["voice"] = "reflexive" if remaining[0] == "r" else remaining[0]
                    if len(remaining) >= 2:
                        features["person"] = remaining[1]
                    if len(remaining) >= 3:
                        features["number"] = number_map.get(remaining[2])

        elif pos == "A":  # Adjective
            if len(tag) >= 2:
                features["gender"] = gender_map.get(tag[1])
            if len(tag) >= 3:
                features["number"] = number_map.get(tag[2])
            if len(tag) >= 4:
                if tag[3] != "-":
                    features["case"] = case_map.get(tag[3])

        elif pos == "P":  # Pronoun
            if len(tag) >= 2:
                features["type"] = {"p": "personal", "x": "reflexive"}.get(tag[1])
            if len(tag) >= 3:
                features["person"] = tag[2]
            # if len(tag) >= 4:
            # features["case"] = case_map.get(tag[3])  # <-- This is not correct!

        elif pos == "p":  # punct
            features["pos"] = "punctuation"

    return features


def parse_nlp_morphtags(sentence_xpostags: list) -> tuple:

    case = []
    # here add other xpostags

    case_map = {"definite": "d", "indefinite": "i", "full": "f", "short": "h"}

    for xpostags in sentence_xpostags:
        xpostag_obj = parse_nlp_morphtag(xpostags)

        if "case" in xpostag_obj:
            case.append(case_map.get(xpostag_obj["case"], "i"))
        else:
            case.append(".")

    return ("".join(case),)


if __name__ == "__main__":
    import time

    file_path = "./data/processed/sent_fiction_nlp_features_part1_v2.tsv"
    output_path = "./tmp/processed_fiction_case.csv"
    df = pd.read_csv(file_path, low_memory=True, sep="\t")
    df = df.dropna(subset=["morph"])

    start_time = time.time()

    df["morph"] = df["morph"].apply(ast.literal_eval)

    results = df["morph"].apply(parse_nlp_morphtags).apply(pd.Series)

    results.columns = ["case"]
    df[results.columns] = results

    print(df.shape)

    for i, row in df.iterrows():
        print(i, row.sentence, row["case"])
        if i > 5:
            break

    df.to_csv(output_path, index=False, sep="\t")

    print(f"Execution time: {time.time() - start_time} seconds")
