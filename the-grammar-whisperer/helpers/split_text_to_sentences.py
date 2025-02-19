from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
import csv


def split_bulgarian_text_into_sentences(text):
    # Define common Bulgarian abbreviations
    bulgarian_abbreviations = [
        "г",
        "д-р",
        "м",
        "ул",
        "жк",
        "п-л",
        "проф",
        "с",
        "б",
        "д",
        "акад",
        "н",
        "гр",
        "др",
        "доц",
        "инж",
        "бр",
    ]

    # Configure Punkt tokenizer to recognize these abbreviations
    punkt_param = PunktParameters()

    # Create the tokenizer
    tokenizer = PunktSentenceTokenizer(punkt_param)

    # punkt_param.abbrev_types = set(bulgarian_abbreviations)
    tokenizer._params.abbrev_types.update(bulgarian_abbreviations)

    # Tokenize the text into sentences
    sentences = tokenizer.tokenize(text)

    sentences = [sentence.strip() for sentence in sentences]

    return sentences


def save_sentences_to_csv(sentences, output_path):
    with open(output_path, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter="\t")
        writer.writerow(["text"])  # simplified header
        writer.writerows([[sentence] for sentence in sentences])  # write sentences directly


def split_and_save_sentences(bg_text_path):

    with open(bg_text_path, "r", encoding="utf-8") as file:
        bulgarian_text = file.read()

    sentences = split_bulgarian_text_into_sentences(bulgarian_text)

    return sentences


if __name__ == "__main__":
    root_dir = "."
    data_processed_dir = r"C:\mirka\git\softuni\04.deep_learning\Project\data\processed"
    data_raw_dir = r"C:\mirka\git\softuni\04.deep_learning\Project\tmp\chitanka\chitanka_info chunks"
    bg_text_path = r"{data_raw_dir}\chunk_1.txt"
    output_path = f"{data_processed_dir}/lit_chunk1_v1.tsv"

    raw_texts = [
        "chunk_1.txt",
        "chunk_2.txt",
        "chunk_3.txt",
        "chunk_4.txt",
        "chunk_5.txt",
        "chunk_6.txt",
        "chunk_7.txt",
        "chunk_8.txt",
    ]
    split_texts = [
        "lit_chunk1_v1.tsv",
        "lit_chunk2_v1.tsv",
        "lit_chunk3_v1.tsv",
        "lit_chunk4_v1.tsv",
        "lit_chunk5_v1.tsv",
        "lit_chunk6_v1.tsv",
        "lit_chunk7_v1.tsv",
        "lit_chunk8_v1.tsv",
    ]

    for raw_text, split_text in zip(raw_texts, split_texts):
        bg_text_path = f"{data_raw_dir}\\{raw_text}"
        output_path = f"{data_processed_dir}/{split_text}"
        sentences = split_and_save_sentences(bg_text_path, output_path)

    # TODO: read a tsv file into a pandas dataframe
    # df = pd.read_csv(output_path, sep="\t")
    # print(df.head())

    # for i, sentence in enumerate(sentences, 1):
    #     print(f"Sentence {i}: {sentence}")
    #     if i > 10:
    #         break
