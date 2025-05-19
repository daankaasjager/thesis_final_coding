import logging
import pandas as pd
from omegaconf import DictConfig
from tqdm import tqdm

from utils import select_numeric_columns

logger = logging.getLogger(__name__)


def _bin_column(series: pd.Series, col_name: str, num_bins: int = 3) -> pd.Series:
    """
    Discretise a numeric column into quantile bins.
    Returns a Series of bracket-wrapped labels (e.g., [col_low], [col_mid], [col_high]).
    Handles NaNs gracefully.
    """
    if num_bins <= 1 :
         logger.warning(f"Number of bins must be >= 2 for column '{col_name}'. Skipping binning.")
         return pd.Series(index=series.index, dtype="string") 

    labels = [f"[{col_name}_bin_{i+1}|{num_bins}]" for i in range(num_bins)]

    binned_series = pd.Series(index=series.index, dtype="string")

    numeric_series = pd.to_numeric(series, errors='coerce')
    valid_idx = numeric_series.notna()

    if not valid_idx.any():
        logger.warning(f"Column '{col_name}' contains no valid numeric data for binning.")
        return binned_series 

    valid_numeric_series = numeric_series[valid_idx]

    try:
        binned_values = pd.qcut(valid_numeric_series, q=num_bins, labels=labels, duplicates="drop")
        binned_series.loc[valid_idx] = binned_values.astype("string")
    except ValueError as e_qcut:
        logger.warning(f"Quantile binning failed for column '{col_name}' (reason: {e_qcut}). Falling back to equal-width bins.")
        try:
            binned_values = pd.cut(valid_numeric_series, bins=num_bins, labels=labels, include_lowest=True, duplicates='drop')
            binned_series.loc[valid_idx] = binned_values.astype("string")
        except ValueError as e_cut:
            logger.error(f"Equal-width binning also failed for column '{col_name}' (reason: {e_cut}). Skipping binning for this column.")
    return binned_series

def apply_discretization(config: DictConfig, df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds binned columns ('[col_name_binX]') to the DataFrame based on config.
    Operates in memory.

    Args:
        df (pd.DataFrame): Input DataFrame.
        config (DictConfig): Configuration object with preprocessing settings
                              (e.g., preprocessing.discretize, preprocessing.discretize_cols).

    Returns:
        pd.DataFrame: DataFrame with added binned columns. If discretization is
                      disabled or no columns are binned, returns the original DataFrame.
    """

    num_bins = config.preprocessing.discretize_num_bins
    cols_to_bin = select_numeric_columns(df, ["smiles", "selfies", "tokenized_selfies"])

    logger.info(f"Discretising {len(cols_to_bin)} columns into {num_bins} bins: {cols_to_bin}")
    df_processed = df.copy()
    for col in tqdm(cols_to_bin, desc="Discretizing columns"):
        bin_col_name = f"{col}_bin"
        if bin_col_name in df_processed.columns:
            logger.warning(f"Bin column '{bin_col_name}' already exists. Skipping discretization for '{col}'.")
            continue
        df_processed[bin_col_name] = _bin_column(df_processed[col], col, num_bins)
    logger.info("Data discretization step complete.")
    return df_processed