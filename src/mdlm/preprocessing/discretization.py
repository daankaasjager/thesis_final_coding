"""
Discretization utilities for numeric SELFIES properties: binning and token mapping.

Provides functions to:
1. Discretize numeric DataFrame columns into quantile or uniform bins.
2. Save and load bin edges for reproducible mappings.
3. Map raw property values to corresponding conditioning token IDs.
"""

import bisect
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from omegaconf import DictConfig

from src.common.utils import select_numeric_columns

logger = logging.getLogger(__name__)


def _bin_column(
    series: pd.Series, col_name: str, num_bins: int = 3
) -> Tuple[pd.Series, Optional[List[float]]]:
    """
    Discretize a numeric Series into labeled bins.

    Attempts quantile binning (pd.qcut) and falls back to uniform binning (pd.cut)
    if necessary.

    Args:
        series: numeric values to bin.
        col_name: base name for bin labels.
        num_bins: number of bins (must be >= 2).

    Returns:
        A tuple of:
        - binned_series: pd.Series of labels like "[col_name_bin_2|5]".
        - bin_edges: list of float boundaries, or None if binning failed.
    """
    labels = [f"[{col_name}_bin_{i+1}|{num_bins}]" for i in range(num_bins)]
    binned = pd.Series(index=series.index, dtype="string")
    if num_bins < 2:
        logger.warning(f"Skipping binning for '{col_name}': num_bins < 2")
        return binned, None

    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric.notna()
    if not valid.any():
        logger.warning(f"No valid numeric data for column '{col_name}'")
        return binned, None

    values = numeric[valid]
    try:
        bins, edges = pd.qcut(values, q=num_bins, labels=labels, retbins=True, duplicates="drop")
    except ValueError as e:
        logger.warning(f"qcut failed for '{col_name}': {e}; falling back to cut")
        try:
            bins, edges = pd.cut(
                values,
                bins=num_bins,
                labels=labels,
                include_lowest=True,
                retbins=True,
                duplicates="drop",
            )
        except ValueError as e2:
            logger.error(f"cut failed for '{col_name}': {e2}")
            return binned, None

    binned.loc[valid] = bins.astype("string")
    return binned, edges.tolist()


def apply_discretization(config: DictConfig, df: pd.DataFrame) -> pd.DataFrame:
    """
    Discretize all eligible numeric columns and persist bin edges.

    Args:
        config: configuration with `preprocessing.discretize_num_bins` and paths.
        df: DataFrame containing numeric and SELFIES columns.

    Returns:
        A copy of df with new "<col>_bin" columns appended.
    """
    num_bins = config.preprocessing.discretize_num_bins
    cols = select_numeric_columns(df, exclude=["smiles", "selfies", "tokenized_selfies"])
    logger.info(f"Discretizing {len(cols)} columns into {num_bins} bins: {cols}")

    result = df.copy()
    edges: Dict[str, List[float]] = {}

    for col in cols:
        bin_col = f"{col}_bin"
        if bin_col in result:
            logger.warning(f"Column '{bin_col}' already exists; skipping '{col}'")
            continue
        series = result[col]
        binned, bin_edges = _bin_column(series, col, num_bins)
        result[bin_col] = binned
        if bin_edges:
            edges[col] = bin_edges

    logger.info("Discretization complete")
    bin_path = Path(config.paths.bin_edges)
    bin_path.parent.mkdir(parents=True, exist_ok=True)
    bin_path.write_text(json.dumps(edges, indent=2), encoding="utf-8")
    logger.info(f"Saved bin edges to {bin_path}")
    return result


def map_target_properties_to_bins(
    config: DictConfig,
    target_properties: Dict[str, float],
    tokenizer: Any,
) -> List[int]:
    """
    Map raw property values to conditioning token IDs based on saved bin edges.

    Args:
        config: configuration with path to bin edges file.
        target_properties: dict of {property_name: numeric_value}.
        tokenizer: SelfiesTokenizer (or similar) with `convert_tokens_to_ids`.

    Returns:
        List of integer token IDs for each property in the order of target_properties.

    Raises:
        ValueError: if bin edges or tokens are missing for any property.
    """
    bin_path = Path(config.paths.bin_edges)
    if not bin_path.is_file():
        raise FileNotFoundError(f"Bin edges file not found: {bin_path}")

    edges: Dict[str, List[float]] = json.loads(bin_path.read_text(encoding="utf-8"))
    token_ids: List[int] = []

    for prop, value in target_properties.items():
        if prop not in edges:
            raise ValueError(f"No bin edges for property '{prop}'")
        bins = edges[prop]
        index = bisect.bisect_right(bins, value) - 1
        max_index = len(bins) - 2
        index = max(0, min(index, max_index))
        num_bins = len(bins) - 1
        token = f"[{prop}_bin_{index+1}|{num_bins}]"
        tok_id = tokenizer.convert_tokens_to_ids(token)
        if tok_id is None or tok_id == tokenizer.unk_token_id:
            raise ValueError(f"Token '{token}' not found in tokenizer vocabulary")
        token_ids.append(tok_id)

    return token_ids
