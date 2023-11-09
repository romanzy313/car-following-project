import re
import numpy as np


def extract_brain_name(input_string: str):
    match = re.search(r"\/([^/]+)\.pth$", input_string)

    if match:
        extracted_text = match.group(1)
        return extracted_text
    else:
        raise Exception(f"failed to extract brain name from {input_string}")
