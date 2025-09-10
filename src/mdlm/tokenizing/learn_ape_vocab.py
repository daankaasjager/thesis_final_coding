"""
Train a SELFIES APE (Byte-Pair Encoding–like) vocabulary from a corpus of SELFIES strings.

This module provides:
- A pre-tokenizer splitting SELFIES into atomic symbols.
- A BPE-style merge algorithm to grow the vocabulary by frequent pair merges.
- Saving of the final vocabulary to disk.
"""

import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


def _selfies_pre_tokenizer(molecule: str) -> List[str]:
    """
    Split a SELFIES string into its atomic tokens.

    Args:
        molecule: SELFIES string, e.g. "[C][O][Branch1]".

    Returns:
        List of SELFIES symbols.

    Raises:
        None. Logs a warning if the full string cannot be reconstructed.
    """
    tokens = re.findall(r"(\[[^\]]+\])", molecule)
    if "".join(tokens) != molecule:
        logger.warning(f"Pre-tokenization may be incomplete for: {molecule}")
    return tokens


def train_selfies_bpe_vocab(
    config: Any,
    corpus: List[str],
    initial_vocab: Dict[str, int],
    max_vocab_size: int,
    min_freq_for_merge: int = 2,
    verbose: bool = True,
) -> Dict[str, int]:
    """
    Train a BPE-like SELFIES vocabulary by iteratively merging the most frequent token pairs.

    Args:
        config: configuration object with paths and flags.
        corpus: list of SELFIES strings to use for training.
        initial_vocab: starting token-to-ID mapping (including special tokens).
        max_vocab_size: desired maximum vocabulary size.
        min_freq_for_merge: threshold frequency for merging pairs.
        verbose: whether to log training progress.

    Returns:
        Final token-to-ID mapping including merged tokens.
    """
    if verbose:
        logger.info("Starting APE vocabulary training")
        logger.info(f"Initial vocab size: {len(initial_vocab)}")
        logger.info(f"Target vocab size: {max_vocab_size}")
        logger.info(f"Min merge frequency: {min_freq_for_merge}")

    vocab = dict(initial_vocab)
    sequences = [_selfies_pre_tokenizer(s) for s in corpus]
    merges_needed = max_vocab_size - len(vocab)

    if merges_needed <= 0:
        logger.warning("No merges to perform: max_vocab_size ≤ initial vocab size")
        return vocab

    if verbose:
        logger.info(f"Planning {merges_needed} merges")

    for merge_idx in range(merges_needed):
        pair_counts: Dict[Tuple[str, str], int] = defaultdict(int)
        for seq in sequences:
            for i in range(len(seq) - 1):
                pair = (seq[i], seq[i + 1])
                if pair[0] in vocab and pair[1] in vocab:
                    pair_counts[pair] += 1

        if not pair_counts:
            logger.info("No more mergeable pairs found")
            break

        best_pair, freq = max(pair_counts.items(), key=lambda item: item[1])
        merged_token = "".join(best_pair)

        if freq < min_freq_for_merge or merged_token in vocab:
            logger.info("Stopping merges: threshold not met or token exists")
            break

        vocab[merged_token] = len(vocab)
        if verbose:
            logger.info(
                f"Merge {merge_idx + 1}/{merges_needed}: "
                f"{best_pair} -> '{merged_token}' (freq={freq})"
            )

        new_sequences: List[List[str]] = []
        for seq in sequences:
            new_seq: List[str] = []
            idx = 0
            while idx < len(seq):
                if idx < len(seq) - 1 and (seq[idx], seq[idx + 1]) == best_pair:
                    new_seq.append(merged_token)
                    idx += 2
                else:
                    new_seq.append(seq[idx])
                    idx += 1
            new_sequences.append(new_seq)
        sequences = new_sequences

    if verbose:
        logger.info(f"Completed training: final vocab size = {len(vocab)}")

    vocab_path = Path(config.paths.selfies_ape_vocab)
    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=4)
    logger.info(f"Saved vocabulary to {vocab_path}")

    return vocab
