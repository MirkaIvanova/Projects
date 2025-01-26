import ast
import pandas as pd


def parse_nlp_features(sentence_features):
    gender, number, person = [], [], []

    for features in sentence_features:
        g_val = n_val = p_val = "."

        # udpipe: "_", spacy_stanza: ""
        if features not in ("_", ""):
            for pair in features.split("|"):
                if "=" in pair:
                    key, val = pair.split("=", 1)
                    if key == "Gender":
                        g_val = val[0] or "."
                    elif key == "Number":
                        n_val = val[0] or "."
                    elif key == "Person":
                        p_val = val[0] or "."
        gender.append(g_val)
        number.append(n_val)
        person.append(p_val)

    return "".join(gender), "".join(number), "".join(person)


def process_chunk(chunk):
    """Process a chunk of data and return enhanced DataFrame"""
    chunk = chunk

    # Process features in list comprehension for better memory efficiency
    chunk["features"] = chunk["features"].apply(ast.literal_eval)
    results = [parse_nlp_features(x) for x in chunk["features"]]

    # Convert results to DataFrame columns
    result_df = pd.DataFrame(results, columns=["gender", "number", "person"], index=chunk.index)

    return pd.concat([chunk, result_df], axis=1)


if __name__ == "__main__":
    file_path = "./data/processed/sent_fiction_nlp_features_part1_v1.tsv"
    output_path = "./tmp/processed_features.tsv"
    chunk_size = 100_000

    # Initialize CSV reader with chunks
    reader = pd.read_csv(file_path, chunksize=chunk_size, low_memory=True, sep="\t")

    # Process and write chunks incrementally
    for i, chunk in enumerate(reader):
        processed_chunk = process_chunk(chunk)
        # Write header only for the first chunk
        mode = "w" if i == 0 else "a"
        header = i == 0
        processed_chunk.to_csv(output_path, mode=mode, header=header, index=False, sep="\t")
        print(f"Processed chunk {i+1} with {len(processed_chunk)} rows")
