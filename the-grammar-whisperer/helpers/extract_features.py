import ast
import pandas as pd


def extract_feature(features_obj, feature_name):
    if feature_name in features_obj:
        return features_obj[feature_name][0]
    return ""


def extract_features(sentence_features_str):
    sentence_features = ast.literal_eval(sentence_features_str)
    gender = []
    number = []
    person = []

    for features in sentence_features:
        if features == "_":
            gender.append("")
            number.append("")
            person.append("")
            continue

        features_obj = dict(pair.split("=") for pair in features.split("|"))

        gender.append(extract_feature(features_obj, "Gender"))
        number.append(extract_feature(features_obj, "Number"))
        person.append(extract_feature(features_obj, "Person"))

    return gender, number, person


if __name__ == "__main__":
    file_path = "../tmp/sent_wikipedia_nlp_features_checkpoint6.csv"
    df = pd.read_csv(file_path, low_memory=False)
    df = df.dropna()

    results = df["features"].apply(extract_features).apply(pd.Series)

    results.columns = ["gender", "number", "person"]
    df[results.columns] = results

    print(df.shape)
