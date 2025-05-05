from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

from networkx import dfs_edges
import pandas as pd
import selfies as sf
from tqdm import tqdm
import torch
from omegaconf import DictConfig

from .csv_reader import read_csv
from .discretization import apply_discretization
from .augmentation import apply_augmentation

logger = logging.getLogger(__name__)


def _load_alphabet_from_txt(path: str | Path) -> List[str]:
    """Loads alphabet from a text file (one token per line)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Alphabet file not found: {path}")
    with open(path, encoding="utf-8") as f:
        return f.read().splitlines()

def _load_preprocessed_data(
    preproc_path: str | Path,
    alphabet_path: str | Path,
) -> Tuple[List[str], pd.DataFrame]:
    """
    Load cached preprocessed DataFrame and alphabet.
    Handles legacy dict format as well.
    """
    preproc_path = Path(preproc_path)
    alphabet_path = Path(alphabet_path)

    if not preproc_path.is_file():
        raise FileNotFoundError(f"Preprocessed data file not found: {preproc_path}")
    if not alphabet_path.is_file():
         raise FileNotFoundError(f"Alphabet file not found: {alphabet_path}")

    try:
        map_location = 'cuda' if torch.cuda.is_available() else 'cpu'
        obj = torch.load(preproc_path, map_location=map_location, weights_only=False)

        if isinstance(obj, dict) and "raw_data" in obj and isinstance(obj["raw_data"], pd.DataFrame):
            data = obj["raw_data"]
            logger.info("Loaded pre-processed data (legacy dict format).")
        elif isinstance(obj, pd.DataFrame):
            data = obj
            logger.info("Loaded pre-processed data (DataFrame format).")
        else:
            raise TypeError(f"Expected DataFrame or legacy dict, but loaded type {type(obj)} from {preproc_path}")

        alphabet = _load_alphabet_from_txt(alphabet_path)
        logger.info("Pre-processed data and alphabet loaded successfully.")
        return alphabet, data

    except Exception as e:
        logger.error(f"Could not load pre-processed data from {preproc_path} or alphabet from {alphabet_path}: {e!s}", exc_info=True)
        raise


def _save_selfies_alphabet(
    alphabet_path: str | Path,
    alphabet: list[str],
    *,
    include_special_tokens: bool = True,
) -> None:
    """Dump alphabet (one token/line) to the specified path."""
    path = Path(alphabet_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Define standard special tokens
    specials = ["[PAD]", "[BOS]", "[EOS]", "[UNK]"] if include_special_tokens else []

    # Combine special tokens and learned alphabet tokens, ensuring uniqueness and order
    present_specials = [s for s in specials if s in alphabet]
    new_specials = [s for s in specials if s not in alphabet]
    other_tokens = sorted([token for token in alphabet if token not in specials])

    # Final alphabet order: Specified specials first, then sorted others
    final_alphabet = present_specials + new_specials + other_tokens

    logger.info("Saving SELFIES alphabet (%d tokens) -> %s", len(final_alphabet), path)
    try:
        path.write_text("\n".join(final_alphabet), encoding="utf-8")
    except IOError as e:
        logger.error(f"Failed to write alphabet file to {path}: {e}")
        raise

def _write_selfies_txt(
    lines_path: str | Path,
    df: pd.DataFrame,
    selfies_col: str = "tokenized_selfies",
    *,
    whitespace: bool,
) -> None:
    """Write tokenized SELFIES from DataFrame column to a text file."""
    path = Path(lines_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if selfies_col not in df.columns:
        logger.error(f"Column '{selfies_col}' not found in DataFrame for writing SELFIES text.")
        return

    num_lines = len(df)
    logger.info("Writing %d tokenized SELFIES sequences -> %s", num_lines, path)

    try:
        with open(path, "w", encoding="utf-8") as fh:
            sep = " " if whitespace else ""
            count = 0
            for tok_list in df[selfies_col]:
                # Check if it's a valid list of strings (tokens)
                if isinstance(tok_list, list) and all(isinstance(t, str) for t in tok_list):
                    fh.write(sep.join(tok_list) + "\n")
                    count += 1
                else:
                    logger.warning(f"Skipping invalid or non-list entry in '{selfies_col}' at index {df[df[selfies_col].isnotnull()].index[count] if count < len(df[df[selfies_col].isnotnull()]) else 'unknown'}: type={type(tok_list)}") # Log type
            if count != num_lines:
                 logger.warning(f"Wrote {count} lines, but DataFrame had {num_lines} rows. This might be due to invalid entries in '{selfies_col}'.")

    except IOError as e:
        logger.error(f"Failed to write SELFIES text file to {path}: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred while writing {path}: {e}", exc_info=True)
        raise


def _tokenize_selfies_and_filter(
    df: pd.DataFrame,
    config: DictConfig,
    selfies_col: str = "selfies"
) -> Tuple[list[str], pd.DataFrame]:
    """
    Tokenizes SELFIES, builds alphabet, filters by length, and returns alphabet and filtered DataFrame.
    """
    if selfies_col not in df.columns:
        logger.error(f"'{selfies_col}' column not found in DataFrame for tokenization.")
        raise ValueError(f"'{selfies_col}' column not found.")

    max_len = config.preprocessing.permitted_selfies_length
    logger.info(f"Tokenizing '{selfies_col}' column and filtering sequences longer than {max_len} tokens...")

    if pd.api.types.is_numeric_dtype(df[selfies_col]):
         logger.warning(f"Column '{selfies_col}' is numeric. Attempting to convert to string.")
    df[selfies_col] = df[selfies_col].astype(str).fillna('') # convert NaNs to empty strings

    all_selfies = df[selfies_col][df[selfies_col] != ''].unique().tolist() # Use unique for efficiency
    if not all_selfies:
        logger.warning("No valid non-empty SELFIES strings found to build alphabet.")
        alphabet = []
    else:
        logger.info(f"Building alphabet from {len(all_selfies)} unique non-empty SELFIES strings...")
        try:
            alphabet = list(sf.get_alphabet_from_selfies(all_selfies))
            logger.info(f"Generated alphabet with {len(alphabet)} unique tokens.")
        except Exception as e:
            logger.error(f"Failed to generate alphabet from SELFIES: {e}", exc_info=True)
            raise 

    tokenized_results = {}
    skipped_count = 0
    error_count = 0

    for index, s in tqdm(df[selfies_col].items(), total=len(df), desc="Tokenizing & Filtering SELFIES"):
        if not isinstance(s, str) or not s:
             # Skip empty strings directly
             continue

        try:
            toks = list(sf.split_selfies(s))
            if 0 < len(toks) <= max_len:
                tokenized_results[index] = toks
            elif len(toks) > max_len:
                skipped_count += 1
        except Exception as e:
            logger.debug(f"Failed to tokenize SELFIES string at index {index}: '{s}'. Error: {e}. Skipping.")
            error_count += 1

    valid_indices = list(tokenized_results.keys())
    if not valid_indices:
         logger.error("No SELFIES sequences remained after tokenization and filtering.")
         return alphabet, pd.DataFrame(columns=df.columns.tolist() + ['tokenized_selfies'])

    filtered_df = df.loc[valid_indices].copy() 
    filtered_df['tokenized_selfies'] = pd.Series(tokenized_results)

    original_rows = len(df)
    final_rows = len(filtered_df)
    logger.info(f"Tokenization complete. Original rows: {original_rows}. Successfully tokenized & length-filtered: {final_rows}. Too long: {skipped_count}. Errors: {error_count}.")
    if original_rows > 0 :
        logger.info(f"Retention rate: {final_rows / original_rows:.2%}")

    if final_rows > 0:
        actual_max_len = filtered_df["tokenized_selfies"].apply(len).max()
        logger.info(f"Longest SELFIES sequence retained: {actual_max_len} tokens.")
    else:
        logger.warning("No sequences remained after filtering. Check data and max_len setting.")

    return alphabet, filtered_df


def prepare_data_for_training(config: DictConfig) -> Tuple[List[str], pd.DataFrame]:
    """
    Orchestrates the full preprocessing pipeline:
    optional: load preprocessed data, otherwise:
    1. Load raw data.
    2. (Optional) Augment data using `apply_augmentation`.
    3. (Optional) Discretize data using `apply_discretization`.
    4. Tokenize SELFIES, build alphabet, filter by length.
    5. Save final processed data (DataFrame.pt) and artifacts (alphabet.txt, selfies.txt).

    Returns the alphabet and the final processed DataFrame.
    Handles caching: Loads preprocessed data if it exists and fresh_data=False.
    """
    try:
        preproc_path = Path(config.local_paths.pre_processed_data)
        alphabet_path = Path(config.local_paths.selfies_alphabet)
        raw_data_path = Path(config.local_paths.original_data)
        selfies_nospace_path = config.local_paths.selfies_nospace_txt
        selfies_whitespace_path = config.local_paths.selfies_whitespace_txt
    except AttributeError as e:
        logger.error(f"Missing path configuration in `config.local_paths`: {e}")
        raise ValueError(f"Missing path configuration: {e}")

    if not config.checkpointing.fresh_data:
        logger.info("Attempting to load existing pre-processed data (fresh_data=False).")
        try:
            return _load_preprocessed_data(preproc_path, alphabet_path)
        except FileNotFoundError:
            logger.warning(f"Cached data not found ({preproc_path} or {alphabet_path}). Will recompute.")
        except Exception as e:
            logger.warning(f"Could not load cached pre-processed data: {e!s}. Will recompute.")


    logger.info("Starting fresh data preprocessing pipeline (fresh_data=True or cache miss)...")

    # 1. Load Raw Data
    df = read_csv(raw_data_path, row_limit=config.row_limit)

    try:
        #2. Tokenize selfies and filter by length
        alphabet, df = _tokenize_selfies_and_filter(df, config)

        #3 Apply discretization
        if config.preprocessing.discretize:
            logger.info("Applying discretization to DataFrame...")
            df = apply_discretization(df, config)
        #4. Apply augmentation
        if config.preprocessing.augment:
            logger.info("Applying data augmentation to DataFrame...")
            df = apply_augmentation(df, config)

    except Exception as e:
         logger.error(f"Error during preprocessing pipeline (augmentation/discretization/tokenization): {e}", exc_info=True)
         raise

    if df.empty:
        logger.error("Preprocessing resulted in an empty DataFrame. Cannot proceed.")
        raise ValueError("No data remaining after preprocessing steps.")
    
    #5. Save final preprocessed artifacts
    logger.info("Saving final preprocessed artifacts...")
    preproc_path.parent.mkdir(parents=True, exist_ok=True) 
    try:
        torch.save(df, preproc_path)
        logger.info(f"Final pre-processed DataFrame saved -> {preproc_path}")

        _save_selfies_alphabet(alphabet_path, alphabet)

        if selfies_nospace_path:
             _write_selfies_txt(selfies_nospace_path, df, whitespace=False)
        if selfies_whitespace_path:
             _write_selfies_txt(selfies_whitespace_path, df, whitespace=True)

    except Exception as e:
        logger.error(f"Failed to save one or more preprocessing artifacts: {e}", exc_info=True)
        raise 
    logger.info("Preprocessing pipeline finished successfully.")
    return alphabet, df