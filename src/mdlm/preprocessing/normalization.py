from math import log
from venv import logger
import pandas as pd
from omegaconf import DictConfig
from tqdm import tqdm
import json
import logging
from ..utils import select_numeric_columns

logger = logging.getLogger(__name__)

def apply_normalization(config: DictConfig, df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes (z-score) specified columns in the DataFrame based on the configuration.
    Operates in memory. Normalization is applied to columns that are specified
    and the normalized values are stored in new columns with the same name.
    suffixed with '_norm'. Also save the mean and std of the original column
    in a JSON file for each column.

    Args:
        df (pd.DataFrame): Input DataFrame.
        config (DictConfig): Configuration object with preprocessing settings
                             (e.g., preprocessing.normalize, preprocessing.normalize_cols).

    Returns:
        pd.DataFrame: DataFrame with normalized columns. If normalization is
                      disabled or no columns are normalized, returns the original DataFrame.
    """

    cols_to_normalize = select_numeric_columns(df, ["smiles", "selfies", "tokenized_selfies"])
    mean_std_dict = {}
    for col in tqdm(cols_to_normalize, desc="Normalizing columns", unit="column"):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame. Please check your configuration.")
        mean = df[col].mean()
        std = df[col].std()
        df[f"{col}_norm"] = (df[col] - mean) / std

        mean_std_dict[col] = {
            "mean": mean,
            "std": std
        }
    try:
        with open (config.paths.mean_std, "w") as f:
            json.dump(mean_std_dict, f) 
        logger.info(f"Mean and std values saved to {config.paths.mean_std}")
    except Exception as e:
        logger.error(f"Error saving mean and std values: {e}")
        raise
    return df


def normalize_scalar_target_properties(target_properties: dict, mean_std_path: str) -> dict:
    """
    Normalize a dictionary of target properties using mean and std from a JSON file.

    Args:
        target_properties (dict): Raw user-provided target property values.
        mean_std_path (str): Path to the JSON file containing mean/std for each property.

    Returns:
        dict: Normalized property values.
    """
    with open(mean_std_path, "r") as f:
        mean_std = json.load(f)

    normalized = {}
    for prop, value in target_properties.items():
        if prop not in mean_std:
            raise ValueError(f"Property '{prop}' not found in mean_std file.")
        mean = mean_std[prop]["mean"]
        std = mean_std[prop]["std"]
        normalized[prop] = (value - mean) / std

    return normalized
