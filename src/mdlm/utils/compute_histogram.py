# tools/length_histogram.py
import json
import pathlib
import re

import numpy as np
import pandas as pd

TOKEN_PATTERN = re.compile(r"\[[^\]]+\]")


def build_histogram(data_path: str, column: str = "selfies") -> dict[int, int]:
    """Returns {length : count} for the whole corpus."""
    df = pd.read_csv(data_path)
    lengths = df[column].str.count(r"\[")
    values, counts = np.unique(lengths, return_counts=True)
    return {int(k): int(v) for k, v in zip(values, counts)}


def save_histogram(hist: dict[int, int], out_file: str):
    pathlib.Path(out_file).write_text(json.dumps(hist, indent=2))
    print(f"Histogram saved to {out_file}")


if __name__ == "__main__":
    hist = build_histogram(
        "/scratch/s3905845/thesis_final_coding/data/kraken/training_data/filtered_selfies.csv"
    )
    save_histogram(
        hist,
        "/scratch/s3905845/thesis_final_coding/data/kraken/training_data/length_histogram.json",
    )
