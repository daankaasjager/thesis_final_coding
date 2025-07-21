"""
Tokenization utilities for SELFIES datasets, with caching and conditioning support.
"""

import math
import os
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import datasets
import pandas as pd
import torch
from datasets import Dataset

from .selfies_tokenizer import SelfiesTokenizer

logger = logging.getLogger(__name__)


def _load_cached_tokenized_data(config: Any, tokenizer: SelfiesTokenizer) -> Optional[datasets.Dataset]:
    """
    Load tokenized dataset from disk if it exists.

    Args:
        config: configuration object with paths.
        tokenizer: tokenizer instance for reporting vocab size.

    Returns:
        Loaded Dataset or None if not found or on error.
    """
    path = config.paths.train_data_encoding
    logger.info(f"Looking for cached SELFIES training data at {path}")
    try:
        if os.path.exists(path):
            tokenized_data = datasets.load_from_disk(path, keep_in_memory=True)
            logger.info(f"SELFIES data loaded; vocab size: {tokenizer.vocab_size}")
            sample_ids = [tokenized_data[i]["input_ids"] for i in range(min(5, len(tokenized_data)))]
            logger.info(f"First 5 tokenized sequences: {sample_ids}")
            return tokenized_data
        logger.info(f"No cached data found at {path}")
        return None
    except Exception as e:
        logger.error(f"Error loading SELFIES data: {e}")
        return None


def _save_tokenized_data(config: Any, tokenized_data: Any) -> datasets.Dataset:
    """
    Save tokenized data to disk in HuggingFace format.

    Args:
        config: configuration with output paths.
        tokenized_data: dict or Dataset to save.

    Returns:
        The saved Dataset.
    """
    path = Path(config.paths.train_data_encoding)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(tokenized_data, dict):
        dataset = Dataset.from_dict(tokenized_data)
    else:
        dataset = tokenized_data
    dataset.save_to_disk(str(path))
    logger.info(f"Tokenized data saved to {path}")
    return dataset


def _prepend_conditioning_tokens(config: Any, raw_data: pd.DataFrame) -> List[str]:
    """
    Prepend discretized conditioning tokens to each SELFIES string.

    Args:
        config: configuration containing conditioning properties.
        raw_data: DataFrame with SELFIES and bin columns.

    Returns:
        List of conditioned SELFIES strings.

    Raises:
        ValueError: if expected bin columns are missing.
    """
    props = config.conditioning.properties
    bin_cols = [f"{p}_bin" for p in props]
    missing = [c for c in bin_cols if c not in raw_data.columns]
    if missing:
        raise ValueError(f"Missing discretized columns: {missing}")

    sequences = []
    for _, row in raw_data.iterrows():
        bins = [row[f"{p}_bin"] for p in props]
        if any(pd.isna(b) for b in bins):
            continue
        prefix = "".join(str(int(b)) for b in bins)
        sequences.append(f"{prefix}{row['selfies']}")

    if config.debug and sequences:
        logger.info(f"Conditioned {len(sequences)} sequences; examples: {sequences[:5]}")
    return sequences


def _extract_cond_vectors(config: Any, df: pd.DataFrame) -> List[List[float]]:
    """
    Extract normalized conditioning vectors from DataFrame.

    Args:
        config: configuration with conditioning properties.
        df: DataFrame containing normalized columns.

    Returns:
        List of conditioning vectors per row.

    Raises:
        ValueError: if normalized columns are missing.
    """
    props = list(config.conditioning.properties)
    cols = [f"{p}_norm" for p in props]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing normalized columns: {missing}")
    return df[cols].astype("float32").values.tolist()


def _greedy_selfies_splitter(selfies_string: str, vocab_keys: List[str]) -> List[str]:
    """
    Split a SELFIES string into tokens using a greedy longest-match strategy.

    Args:
        selfies_string: raw SELFIES string (e.g., "[C][O]").
        vocab_keys: vocabulary tokens sorted by descending length.

    Returns:
        List of SELFIES tokens.

    Raises:
        ValueError: if any part of the string cannot be matched.
    """
    tokens: List[str] = []
    i = 0
    while i < len(selfies_string):
        match = next((k for k in vocab_keys if selfies_string.startswith(k, i)), None)
        if not match:
            remainder = selfies_string[i:]
            raise ValueError(
                f"No vocab match for remainder '{remainder}'; ensure vocabulary completeness."
            )
        tokens.append(match)
        i += len(match)
    return tokens


def tokenize_selfies_vocab(
    config: Any,
    tokenizer: SelfiesTokenizer,
    raw_data: pd.DataFrame,
    chunk_size: int = 50000,
    max_length: int = 310,
) -> Any:
    """
    Tokenize SELFIES dataset with optional caching and conditioning.

    Args:
        config: configuration object with tokenizer and conditioning settings.
        tokenizer: tokenizer instance.
        raw_data: DataFrame containing 'selfies'.
        chunk_size: number of sequences per processing chunk.
        max_length: maximum token sequence length.

    Returns:
        Tokenized data dict or loaded Dataset.
    """
    if os.path.exists(config.paths.train_data_encoding) and not config.checkpointing.retrain_tokenizer:
        return _load_cached_tokenized_data(config, tokenizer)

    if "selfies" not in raw_data.columns:
        logger.warning("'selfies' column not found in raw_data.")
        raise KeyError("Missing 'selfies' column.")

    if config.conditioning.properties:
        sequences = (
            _prepend_conditioning_tokens(config, raw_data)
            if config.conditioning.prepend
            else raw_data["selfies"].tolist()
        )
        cond_vectors = (
            _extract_cond_vectors(config, raw_data)
            if config.conditioning.embeddings or config.conditioning.cfg
            else None
        )
    else:
        sequences = raw_data["selfies"].tolist()
        cond_vectors = None

    vocab = tokenizer.get_vocab()
    vocab_keys = sorted(vocab.keys(), key=len, reverse=True)
    total = len(sequences)
    logger.info(f"Tokenizing {total} sequences in chunks of {chunk_size}.")
    chunks = math.ceil(total / chunk_size)
    tokenized: Dict[str, List[Any]] = defaultdict(list)

    for idx in range(chunks):
        start = idx * chunk_size
        end = min((idx + 1) * chunk_size, total)
        batch = sequences[start:end]
        logger.info(f"Processing chunk {idx+1}/{chunks} with {len(batch)} sequences.")

        if config.tokenizer.tokenizer_type.lower() == "ape":
            split_batch = [_greedy_selfies_splitter(s, vocab_keys) for s in batch]
            tk = tokenizer(
                split_batch,
                is_split_into_words=True,
                max_length=max_length,
                padding="longest",
                truncation=False,
                add_special_tokens=True,
            )
        else:
            tk = tokenizer(
                batch,
                is_split_into_words=False,
                max_length=max_length,
                padding="longest",
                truncation=False,
                add_special_tokens=True,
            )

        tokenized["input_ids"].extend(tk["input_ids"])
        tokenized["attention_mask"].extend(tk["attention_mask"])
        tokenized["token_type_ids"].extend(
            tk.get("token_type_ids", [[0] * len(ids) for ids in tk["input_ids"]])
        )
        if cond_vectors:
            tokenized["cond_props"].extend(cond_vectors[start:end])
        else:
            zeros = [0] * len(config.conditioning.properties or [])
            tokenized["cond_props"].extend([zeros] * len(batch))

    _save_tokenized_data(config, tokenized)
    logger.info(f"Tokenization complete; vocab size: {tokenizer.vocab_size}")
    return tokenized
