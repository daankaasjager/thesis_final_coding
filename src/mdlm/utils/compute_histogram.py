"""
Utility for building and saving a histogram of SELFIES sequence lengths.
"""

import json
import re
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

logger = __import__("logging").getLogger(__name__)

TOKEN_PATTERN = re.compile(r"\[[^\]]+\]")


def _build_histogram(data_path: str, column: str = "selfies") -> Dict[int, int]:
    """
    Compute a histogram of token counts for a specified column in a CSV.

    Args:
        data_path: path to the CSV file containing SELFIES strings.
        column: name of the column with SELFIES sequences.

    Returns:
        A dict mapping sequence length (number of tokens) to count.
    """
    df = pd.read_csv(data_path)
    lengths = df[column].str.count(r"\[")
    values, counts = np.unique(lengths, return_counts=True)
    return {int(length): int(count) for length, count in zip(values, counts)}


def _save_histogram(hist: Dict[int, int], out_file: str) -> None:
    """
    Write a length histogram to a JSON file.

    Args:
        hist: mapping of sequence lengths to counts.
        out_file: filesystem path where JSON will be saved.
    """
    path = Path(out_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(hist, indent=2), encoding="utf-8")
    logger.info(f"Histogram saved to {out_file}")


if __name__ == "__main__":
    """
    Build and save a SELFIES length histogram for the training corpus.
    """
    input_csv = "/scratch/s3905845/thesis_final_coding/data/kraken/training_data/filtered_selfies.csv"
    output_json = "/scratch/s3905845/thesis_final_coding/data/kraken/training_data/length_histogram.json"
    histogram = _build_histogram(input_csv)
    _save_histogram(histogram, output_json)
