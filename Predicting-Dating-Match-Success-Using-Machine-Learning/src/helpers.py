import json, re


def load_model_params(file_path):
    """
    Loads a JSON file, removes comments, and returns the parsed JSON object.
    Args:
        file_path (str): Path to the JSON file with comments.
    Returns:
        dict: Parsed JSON object.
    """
    try:
        with open(file_path, "r") as file:
            raw_json = file.read()
            # Remove single-line comments (//) only till the end of the line
            raw_json = re.sub(r"//.*$", "", raw_json, flags=re.MULTILINE)
            # Remove multi-line comments (/* */)
            raw_json = re.sub(r"/\*.*?\*/", "", raw_json, flags=re.DOTALL)

            return json.loads(raw_json)
    except Exception as e:
        raise ValueError(f"Error loading JSON file: {e}")
