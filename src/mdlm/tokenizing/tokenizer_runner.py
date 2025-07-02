from functools import cache
import math
import os
from collections import defaultdict

from typing import Dict, List, Any
from pathlib import Path
import pandas as pd
import torch
import logging
import pyarrow.parquet as pq
import pyarrow as pa
import datasets
from datasets import Dataset


logger = logging.getLogger(__name__)


def _load_cached_tokenized_data(config, tokenizer):
    """Load tokenized data from disk if available."""
    path = config.paths.train_data_encoding
    logger.info(f"Looking for cached SELFIES training data at {path}")
    try:
        if os.path.exists(path):
            tokenized_data = datasets.load_from_disk(path, keep_in_memory=True)
            logger.info(f"SELFIES data loaded successfully. Vocab size: {tokenizer.vocab_size}") 
            print(f"Print first 5 tokenized sequences: {[tokenized_data[i]['input_ids'] for i in range(5)]}")
            return tokenized_data
        else:
            logger.info(f"No cached data found at {path}")
            return None
    except Exception as e:
        logger.error(f"Error loading SELFIES data: {e}")
        return None

def _save_tokenized_data(config, tokenized_data):
    """Save tokenized data as a datasets.Dataset to disk in an efficient format."""
    path = Path(config.paths.train_data_encoding)
    try:
        # Create directory if it doesn't exist
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert dictionary to datasets.Dataset
        if isinstance(tokenized_data, dict):
            dataset = Dataset.from_dict(tokenized_data)
        else:
            dataset = tokenized_data
            
        # Save dataset to disk
        dataset.save_to_disk(str(path))
        logger.info(f"Tokenized data saved to {path}")
        return dataset
    except Exception as e:
        logger.error(f"Error saving tokenized SELFIES data: {e}")
        raise


def _prepend_conditioning_tokens(config, raw_data):
    input_selfies = []
    bin_column_names = [
        f"{prop}_bin" for prop in config.conditioning.properties
    ]
    for col in bin_column_names:
        if col not in raw_data.columns:
            raise ValueError(f"Expected discretized column '{col}' not found in data")
    for _, row in raw_data.iterrows():
        bin_tokens = [
            str(row[f"{prop}_bin"]) for prop in config.conditioning.properties
        ]
        if any(pd.isna(tok) for tok in bin_tokens):
            continue
        prefix = "".join(bin_tokens)
        full_sequence = f"{prefix}{row['selfies']}"
        input_selfies.append(full_sequence)
    if config.debug:
        logger.info(
            f"Conditioning tokens added. Number of sequences: {len(input_selfies)}"
        )
        if len(input_selfies) > 5:
            logger.info(f"show some examples of conditioning tokens: {input_selfies[:5]}")
    return input_selfies

def _extract_cond_vectors(cfg, df):
    props = list(cfg.conditioning.properties)
    cols = [f"{p}_norm" for p in props]
    if any(c not in df.columns for c in cols):
        raise ValueError(f"Missing normalised columns {cols}")
    return df[cols].astype("float32").values.tolist()

def greedy_selfies_splitter(selfies_string: str, vocab_keys: list) -> list[str]:
    """
    Splits a SELFIES string using a greedy, longest-match algorithm based on a vocabulary.

    Args:
        selfies_string: The raw SELFIES string to split (e.g., "[C][C][O]").
        vocab_keys: A list of all tokens in the vocabulary, pre-sorted by length descending.

    Returns:
        A list of SELFIES tokens (e.g., ['[C][C]', '[O]']).
    """
    tokens = []
    i = 0
    while i < len(selfies_string):
        found_match = False
        for key in vocab_keys:
            if selfies_string.startswith(key, i):
                tokens.append(key)
                i += len(key)
                found_match = True
                break  #
        if not found_match:
            remaining_str = selfies_string[i:]
            raise ValueError(
                f"Could not find a vocab match for the remainder of the string: '{remaining_str}'. "
                f"Please ensure your vocabulary is complete."
            )
    return tokens

def tokenize_selfies_vocab(
    config, tokenizer, raw_data=None, chunk_size=50000, max_length=310
):
    if (
        os.path.exists(config.paths.train_data_encoding)
        and not config.checkpointing.retrain_tokenizer
    ):
        return _load_cached_tokenized_data(config, tokenizer) # this is none if tokenizer is being retrained and is a datasets dict othewise
    
    if "selfies" not in raw_data.columns:
        logger.warning("'selfies' column not found in raw_data.")
        raise
    
    if config.conditioning.properties is not None:
        input_selfies = _prepend_conditioning_tokens(config, raw_data) if config.conditioning.prepend else raw_data["selfies"].tolist()
        cond_vectors_full = _extract_cond_vectors(config, raw_data) if config.conditioning.embeddings or config.conditioning.cfg else None
        
    vocab_dict = tokenizer.get_vocab()
    sorted_vocab_keys = sorted(vocab_dict.keys(), key=len, reverse=True)
    total_samples = len(input_selfies)
    logger.info(f"Tokenizing {total_samples} SELFIES in chunks of {chunk_size}...")
    num_chunks = math.ceil(total_samples / chunk_size)
    tokenized_data = defaultdict(list)
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, total_samples)
        chunk = input_selfies[start_idx:end_idx]
        logger.info(
            f"Processing chunk {chunk_idx+1}/{num_chunks}: {len(chunk)} sequences"
        )
        if config.tokenizer.tokenizer_type == "ape" or config.tokenizer.tokenizer_type == "APE": 
            pre_split_chunk = [greedy_selfies_splitter(s, sorted_vocab_keys) for s in chunk]
            tokenized_chunk = tokenizer(
                pre_split_chunk,
                is_split_into_words=True,
                max_length=max_length,
                padding="longest",
                truncation=False,
                add_special_tokens=True,
            )
        else: # Wordlevel
            tokenized_chunk = tokenizer(
                chunk,
                is_split_into_words=False,
                max_length=max_length,
                padding="longest",
                truncation=False,
                add_special_tokens=True,
            )
        tokenized_data["input_ids"].extend(tokenized_chunk["input_ids"])
        tokenized_data["attention_mask"].extend(tokenized_chunk["attention_mask"])
        token_type_ids = tokenized_chunk.get(
            "token_type_ids",
            [[0] * len(ids) for ids in tokenized_chunk["input_ids"]],
        )
        tokenized_data["token_type_ids"].extend(token_type_ids)
        if cond_vectors_full and config.conditioning.properties:
            tokenized_data["cond_props"].extend(
                cond_vectors_full[start_idx:end_idx]
            )
        else:
            tokenized_data["cond_props"].extend(
                [[0] * len(config.conditioning.properties)] * len(chunk)
            )

    _save_tokenized_data(config, tokenized_data)
    logger.info(f"Done tokenizing. Vocab size: {tokenizer.vocab_size}")
    return tokenized_data
