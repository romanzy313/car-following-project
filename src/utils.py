import pickle
import re
import numpy as np


def extract_brain_name(input_string: str):
    match = re.search(r"\/([^/]+)\.pth$", input_string)

    if match:
        extracted_text = match.group(1)
        return extracted_text
    else:
        raise Exception(f"failed to extract brain name from {input_string}")


def create_sequences_2(data, n_steps_in, n_steps_out, step):
    X, y = [], []
    for i in range(0, len(data) - n_steps_in - n_steps_out + 5, step):
        seq_x = data[i : i + n_steps_in]
        seq_y = data[i + n_steps_in : i + n_steps_in + n_steps_out]
        if seq_x.shape[0] == n_steps_in and seq_y.shape[0] == n_steps_out:
            X.append(seq_x)
            y.append(seq_y)
    return np.array(X), np.array(y)
