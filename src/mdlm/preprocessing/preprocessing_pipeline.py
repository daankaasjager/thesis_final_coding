"""
Preprocessing pipeline for SELFIES molecular data: loading, tokenization, filtering,
augmentation, discretization, normalization, and saving artifacts.
"""

import logging
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import selfies as sf
from omegaconf import DictConfig
from tqdm import tqdm

from .augmentation import apply_augmentation
from .csv_reader import read_csv
from .discretization import apply_discretization
from .normalization import apply_normalization

logger = logging.getLogger(__name__)


def _optimize_memory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Downcast float64 columns to float32 and pack list-valued tokens as space-separated strings.

    Args:
        df: DataFrame with preprocessing output.

    Returns:
        DataFrame with reduced memory footprint.
    """
    float_cols = df.select_dtypes(include=["float64"]).columns
    if not float_cols.empty:
        df[float_cols] = df[float_cols].apply(pd.to_numeric, downcast="float")
    if (
        "tokenized_selfies" in df.columns
        and df["tokenized_selfies"].apply(lambda x: isinstance(x, list)).any()
    ):
        df["tokenized_selfies"] = df["tokenized_selfies"].apply(
            lambda lst: " ".join(lst) if isinstance(lst, list) else lst
        )
    return df


def _load_alphabet_from_txt(path: Path) -> List[str]:
    """
    Load one-token-per-line SELFIES alphabet from text file.

    Args:
        path: Path to alphabet file.

    Returns:
        List of tokens.

    Raises:
        FileNotFoundError: if file does not exist.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Alphabet file not found: {path}")
    return path.read_text(encoding="utf-8").splitlines()


def _load_preprocessed_data(
    preproc_path: Path, alphabet_path: Path
) -> Tuple[List[str], pd.DataFrame]:
    """
    Load cached preprocessed DataFrame (Parquet) and alphabet.

    Args:
        preproc_path: Path to Parquet file.
        alphabet_path: Path to alphabet text file.

    Returns:
        Tuple of alphabet list and loaded DataFrame.

    Raises:
        FileNotFoundError: if files are missing.
        ValueError: if format unsupported.
    """
    if not preproc_path.is_file():
        raise FileNotFoundError(f"Preprocessed data file not found: {preproc_path}")
    if not alphabet_path.is_file():
        raise FileNotFoundError(f"Alphabet file not found: {alphabet_path}")

    if preproc_path.suffix != ".parquet":
        raise ValueError(f"Unsupported format: {preproc_path.suffix}")

    table = pq.ParquetFile(preproc_path)
    df = table.read().to_pandas()
    logger.info("Loaded preprocessed data (Parquet).")

    alphabet = _load_alphabet_from_txt(alphabet_path)
    logger.info("Loaded alphabet and preprocessed data successfully.")
    return alphabet, df


def _save_selfies_alphabet(
    alphabet_path: Path, alphabet: List[str], include_special_tokens: bool = True
) -> None:
    """
    Save SELFIES alphabet tokens, ensuring special tokens appear first.

    Args:
        alphabet_path: target file path.
        alphabet: list of tokens.
        include_special_tokens: whether to include PAD/BOS/EOS/UNK.
    """
    special = ["[PAD]", "[BOS]", "[EOS]", "[UNK]"] if include_special_tokens else []
    present = [s for s in special if s in alphabet]
    missing = [s for s in special if s not in alphabet]
    others = sorted(tok for tok in alphabet if tok not in special)
    ordered = present + missing + others

    alphabet_path.parent.mkdir(parents=True, exist_ok=True)
    alphabet_path.write_text("\n".join(ordered), encoding="utf-8")
    logger.info(f"Saved SELFIES alphabet ({len(ordered)} tokens) to {alphabet_path}")


def _tokenize_selfies_and_filter(
    config: DictConfig, df: pd.DataFrame, selfies_col: str = "selfies"
) -> Tuple[List[str], pd.DataFrame]:
    """
    Tokenize and length-filter SELFIES strings, build alphabet, and save filtered CSV.

    Args:
        config: preprocessing configuration.
        df: raw DataFrame with SELFIES column.
        selfies_col: name of SELFIES column.

    Returns:
        Tuple of alphabet list and filtered DataFrame.

    Raises:
        ValueError: if column missing or no valid sequences remain.
    """
    if selfies_col not in df.columns:
        raise ValueError(f"Column '{selfies_col}' not found")

    max_len = config.preprocessing.permitted_selfies_length
    df[selfies_col] = df[selfies_col].astype(str).fillna("")
    unique_strs = df.loc[df[selfies_col] != "", selfies_col].unique().tolist()

    if unique_strs:
        alphabet = list(sf.get_alphabet_from_selfies(unique_strs))
        logger.info(f"Built alphabet with {len(alphabet)} tokens.")
    else:
        alphabet = []
        logger.warning("No non-empty SELFIES; alphabet is empty.")

    token_map: dict[int, List[str]] = {}
    skipped = errors = 0
    for idx, seq in tqdm(
        df[selfies_col].items(), total=len(df), desc="Tokenizing & filtering"
    ):
        if not seq:
            continue
        try:
            toks = list(sf.split_selfies(seq))
            if 0 < len(toks) <= max_len:
                token_map[idx] = toks
            elif len(toks) > max_len:
                skipped += 1
        except Exception:
            errors += 1

    if not token_map:
        raise ValueError("All sequences filtered out or errored")

    filtered = df.loc[list(token_map.keys())].copy()
    filtered["tokenized_selfies"] = pd.Series(token_map)
    logger.info(
        f"Tokenized: {len(token_map)}/{len(df)} retained, {skipped} too long, {errors} errors"
    )

    filtered.to_csv(config.paths.filtered_original_data, index=False)
    logger.info(f"Saved filtered CSV to {config.paths.filtered_original_data}")
    return alphabet, filtered


def prepare_data_for_training(config: DictConfig) -> Tuple[List[str], pd.DataFrame]:
    """
    Execute full preprocessing with optional caching and artifact saving.

    Args:
        config: configuration with paths and flags.

    Returns:
        Tuple of alphabet and processed DataFrame.

    Raises:
        ValueError: if preprocessing yields no data.
    """
    preproc_path = Path(config.paths.pre_processed_data)
    alphabet_path = Path(config.paths.selfies_alphabet)
    raw_path = Path(config.paths.original_data)

    if not config.checkpointing.fresh_data:
        try:
            return _load_preprocessed_data(preproc_path, alphabet_path)
        except FileNotFoundError:
            logger.warning("Cache miss; recomputing")
        except Exception:
            logger.warning("Cache load failed; recomputing")

    df = read_csv(raw_path, row_limit=config.row_limit)
    alphabet, df = _tokenize_selfies_and_filter(config, df)

    if config.preprocessing.discretize:
        df = apply_discretization(config, df)
    if config.preprocessing.normalize:
        df = apply_normalization(config, df)
    if config.preprocessing.augment:
        df = apply_augmentation(config, df)

    if df.empty:
        raise ValueError("Preprocessing produced empty dataset")

    df = _optimize_memory(df)
    preproc_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pandas(df, preserve_index=False), preproc_path, compression="zstd"
    )
    _save_selfies_alphabet(alphabet_path, alphabet)

    logger.info("Preprocessing pipeline completed successfully")
    return alphabet, df
