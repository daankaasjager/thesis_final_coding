import logging
from typing import List, Optional
import pandas as pd
from omegaconf import DictConfig
from tqdm import tqdm

logger = logging.getLogger(__name__)

def _select_numeric_columns(df: pd.DataFrame, exclude: set) -> List[str]:
    numeric_cols = []
    for c, dtype in df.dtypes.items():
        if c not in exclude and pd.api.types.is_numeric_dtype(dtype) and not c.endswith("_bin"):
             if df[c].notna().any() and pd.to_numeric(df[c], errors='coerce').notna().any():
                numeric_cols.append(c)
             else:
                logger.debug(f"Column '{c}' looks numeric by dtype but contains non-numeric data or only NaNs. Skipping for auto-binning.")

    logger.debug(f"Identified numeric columns for potential binning: {numeric_cols}")
    return numeric_cols


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

def apply_discretization(df: pd.DataFrame, config: DictConfig) -> pd.DataFrame:
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
    cols_to_bin = _select_numeric_columns(df, ["smiles", "selfies", "tokenized_selfies"])

    logger.info(f"Discretising {len(cols_to_bin)} columns into {num_bins} bins: {cols_to_bin}")
    df_processed = df.copy()
    for col in tqdm(cols_to_bin, desc="Discretizing columns"):
        bin_col_name = f"{col}_bin"
        if bin_col_name in df_processed.columns:
            logger.warning(f"Bin column '{bin_col_name}' already exists. Skipping discretization for '{col}'.")
            continue
        df_processed[bin_col_name] = _bin_column(df_processed[col], col, num_bins)
    print(f"Discretized columns: {df_processed.columns[df_processed.columns.str.endswith('_bin')]}")
    # print first 10 row elements of each discretized column
    for col in df_processed.columns[df_processed.columns.str.endswith('_bin')]:
        print(f"First 10 elements of '{col}': {df_processed[col].head(10).tolist()}")
    logger.info("Data discretization step complete.")
    return df_processed