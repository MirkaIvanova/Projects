import ast
import pandas as pd


def parse_nlp_features2(sentence_features):
    # sentence_features = ast.literal_eval(sentence_features_str)
    gender, number, person = [], [], []

    for features in sentence_features:
        if features in ("_", ""):
            gender.append(".")
            number.append(".")
            person.append(".")
            continue

        g_val = n_val = p_val = "."
        for pair in features.split("|"):
            if "=" in pair:
                key, val = pair.split("=", 1)
                if key == "Gender":
                    g_val = val or "."
                elif key == "Number":
                    n_val = val or "."
                elif key == "Person":
                    p_val = val or "."
        gender.append(g_val)
        number.append(n_val)
        person.append(p_val)

    return gender, number, person


if __name__ == "__main__":
    import time

    file_path = "./tmp/sent_wikipedia_nlp_features_checkpoint6.csv"
    df = pd.read_csv(file_path, low_memory=False).dropna()
    df["features"] = df["features"].apply(ast.literal_eval)
    df = df.iloc[:100000]

    start_time = time.time()  # Process in chunks to reduce memory overhead
    chunk_size = 100000  # Adjust based on system's memory
    results = []
    for start in range(0, len(df), chunk_size):
        end = start + chunk_size
        features_chunk = df["features"].iloc[start:end]
        results.extend(parse_nlp_features2(x) for x in features_chunk)

    # Assign results to DataFrame
    df[["gender", "number", "person"]] = pd.DataFrame(results, index=df.index)
    print(df.shape)

    print(f"Execution time: {time.time() - start_time} seconds")
