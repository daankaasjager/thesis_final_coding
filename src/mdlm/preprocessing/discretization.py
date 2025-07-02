import logging
import pandas as pd
from omegaconf import DictConfig
from tqdm import tqdm
import json
import bisect


from ..utils import select_numeric_columns

logger = logging.getLogger(__name__)


def _bin_column(series: pd.Series, col_name: str, num_bins: int = 3):
    """
    Discretize a numeric column into quantile bins.
    Returns:
        - binned_series: Series of bracket-wrapped labels
        - bin_edges: bin boundaries used for qcut or cut
    """
    if num_bins <= 1:
        logger.warning(f"Number of bins must be >= 2 for column '{col_name}'. Skipping binning.")
        return pd.Series(index=series.index, dtype="string"), None

    labels = [f"[{col_name}_bin_{i+1}|{num_bins}]" for i in range(num_bins)]
    binned_series = pd.Series(index=series.index, dtype="string")

    numeric_series = pd.to_numeric(series, errors='coerce')
    valid_idx = numeric_series.notna()
    if not valid_idx.any():
        logger.warning(f"Column '{col_name}' contains no valid numeric data for binning.")
        return binned_series, None

    valid_numeric_series = numeric_series[valid_idx]

    try:
        binned_values, bin_edges = pd.qcut(valid_numeric_series, q=num_bins, labels=labels, retbins=True, duplicates="drop")
        binned_series.loc[valid_idx] = binned_values.astype("string")
    except ValueError as e_qcut:
        logger.warning(f"qcut failed for '{col_name}' (reason: {e_qcut}). Falling back to pd.cut.")
        try:
            binned_values, bin_edges = pd.cut(valid_numeric_series, bins=num_bins, labels=labels, include_lowest=True, retbins=True, duplicates="drop")
            binned_series.loc[valid_idx] = binned_values.astype("string")
        except ValueError as e_cut:
            logger.error(f"cut also failed for '{col_name}' (reason: {e_cut}).")
            return binned_series, None

    return binned_series, bin_edges


def apply_discretization(config: DictConfig, df: pd.DataFrame):
    num_bins = config.preprocessing.discretize_num_bins
    cols_to_bin = select_numeric_columns(df, ["smiles", "selfies", "tokenized_selfies"])

    logger.info(f"Discretising {len(cols_to_bin)} columns into {num_bins} bins: {cols_to_bin}")
    df_processed = df.copy()
    bin_edges_dict = {}  # <-- save here

    for col in tqdm(cols_to_bin, desc="Discretizing columns"):
        bin_col_name = f"{col}_bin"
        if bin_col_name in df_processed.columns:
            logger.warning(f"Bin column '{bin_col_name}' already exists. Skipping '{col}'.")
            continue
        binned_series, bin_edges = _bin_column(df_processed[col], col, num_bins)
        df_processed[bin_col_name] = binned_series
        if bin_edges is not None:
            bin_edges_dict[col] = bin_edges.tolist()  # convert to JSON-serializable

    logger.info("Data discretization step complete.")


    # Save bin edges for mapping during sampling
    with open(config.paths.bin_edges, "w") as f:
        json.dump(bin_edges_dict, f, indent=2)
    logger.info(f"Bin edges saved to {config.paths.bin_edges}")
    return df_processed



def map_target_properties_to_bins(config, target_properties: dict, tokenizer) -> list[int]:
        """
        This function is used during sampling, 
        it maps numeric unnormalized target property values to 
        bin tokens using bin edges and tokenizer.
        
        Args:
            target_properties: dict like {"logP": 1.23, "QED": 0.61}
        
        Returns:
            List of token IDs corresponding to bin labels like "[logP_bin_2|5]"
        """
        with open(config.paths.bin_edges, "r") as f:
            bin_edges_dict = json.load(f)
        bin_token_ids = []
        for prop_name, prop_value in target_properties.items():
            bin_edges = bin_edges_dict.get(prop_name)
            if bin_edges is None:
                raise ValueError(f"No bin edges found for property '{prop_name}'.")

            # If its lower or higher than the max bin edge then assign the first or last bin
            bin_index = bisect.bisect_right(bin_edges, prop_value) - 1
            num_bins = len(bin_edges) - 1
            bin_index = max(0, min(bin_index, num_bins - 1))

            bin_token = f"[{prop_name}_bin_{bin_index+1}|{num_bins}]"
            bin_token_id = tokenizer.convert_tokens_to_ids(bin_token)
            if bin_token_id is None or bin_token_id == tokenizer.unk_token_id:
                raise ValueError(f"Token '{bin_token}' not in tokenizer vocabulary.")
            bin_token_ids.append(bin_token_id)

        return bin_token_ids