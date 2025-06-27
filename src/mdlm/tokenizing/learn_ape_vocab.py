import json
import logging
from os import path
import re
from collections import defaultdict
from typing import Dict, List
from pathlib import Path

from tqdm import tqdm

logger = logging.getLogger(__name__)


"""This code is based on the training functionality of https://github.com/mikemayuare/apetokenizer/tree/main/src
and adapted to work as a separate function for APE tokenization. """


def _selfies_pre_tokenizer(molecule: str) -> List[str]:
    """
    Basic pre-tokenizer for SELFIES strings.
    Splits the string into individual SELFIES symbols like '[C]', '[Branch1]', etc.
    Args:
        molecule: The SELFIES string.
    Returns:
        A list of SELFIES symbol strings.
    """
    pattern = r"(\[[^\]]+\])"
    # Alternative pattern if non-bracketed symbols are also expected:
    # pattern = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    words = re.findall(pattern, molecule)
    if "".join(words) != molecule:
        logger.warning(f"Pre-tokenization might be incomplete for: {molecule}")
    return words


def train_selfies_bpe_vocab(
    config,
    corpus: List[str],
    initial_vocab: Dict[str, int],
    max_vocab_size: int,
    min_freq_for_merge: int = 2,
    verbose: bool = True,
) -> Dict[str, int]:
    """
    Trains a vocabulary by merging frequent pairs of SELFIES symbols (BPE-like).

    Starts with an initial vocabulary and iteratively merges the most frequent
    adjacent pairs in the corpus until max_vocab_size is reached or no pairs
    meet the minimum frequency threshold.

    Args:
        corpus: A list of SELFIES strings to train on.
        initial_vocab: A dictionary mapping initial tokens (base symbols + special) to IDs.
        max_vocab_size: The target maximum size for the final vocabulary.
        min_freq_for_merge: The minimum frequency a pair must have to be merged.
        verbose: Whether to print progress information.

    Returns:
        A dictionary representing the final vocabulary (token -> ID map),
        including the merged tokens (motifs).
    """
    if verbose:
        logger.info("Starting SELFIES APE vocabulary training...")
        logger.info(f"Initial vocab size: {len(initial_vocab)}")
        logger.info(f"Target max vocab size: {max_vocab_size}")
        logger.info(f"Min frequency for merge: {min_freq_for_merge}")

    current_vocab = dict(initial_vocab)

    word_sequences = [_selfies_pre_tokenizer(sentence) for sentence in corpus]

    num_merges_to_do = max_vocab_size - len(current_vocab)
    if num_merges_to_do <= 0:
        logger.warning(
            "Target max_vocab_size is not larger than initial vocab size. No merges will be performed."
        )
        return current_vocab

    if verbose:
        logger.info(f"Aiming for approximately {num_merges_to_do} merges.")

    for i in range(num_merges_to_do):
        pair_counts = defaultdict(int)
        for sequence in word_sequences:
            for j in range(len(sequence) - 1):
                pair = (sequence[j], sequence[j + 1])
                if pair[0] in current_vocab and pair[1] in current_vocab:
                    pair_counts[pair] += 1
        # Check if any pairs were found
        if not pair_counts:
            if verbose:
                logger.info(
                    f"Merge iteration {i+1}: No more pairs found in the corpus representation."
                )
            break 
        # Find the best pair that hasn't been merged into a token already present
        best_pair = (None, None)
        max_freq = -1
        sorted_pairs = sorted(
            pair_counts.items(), key=lambda item: item[1], reverse=True
        )

        for pair, freq in sorted_pairs:
            merged_token_candidate = "".join(pair)
            if merged_token_candidate not in current_vocab:
                # Found the best valid pair
                if freq >= min_freq_for_merge:
                    best_pair = pair
                    max_freq = freq
                    break  # Use this pair
                else:
                    if verbose:
                        logger.info(
                            f"Merge iteration {i+1}: Most frequent pair {pair} frequency ({freq}) is below threshold ({min_freq_for_merge}). Stopping merges."
                        )
                    best_pair = (None, None)
                    break

        # 3. Check if a valid merge was found
        if best_pair == (None, None) or max_freq < min_freq_for_merge:
            if best_pair == (None, None) and max_freq >= min_freq_for_merge:
                logger.info(
                    f"Merge iteration {i+1}: All frequent pairs already resulted in existing tokens. Stopping merges."
                )
            break

        merged_word = "".join(best_pair)
        new_token_id = len(current_vocab)
        current_vocab[merged_word] = new_token_id
        if verbose:
            print(
                f"Merge {i+1}/{num_merges_to_do}: Merging {best_pair} -> '{merged_word}' (freq: {max_freq})"
            )
        new_word_sequences = []
        for sequence in word_sequences:
            new_sequence = []
            j = 0
            while j < len(sequence):
                if (
                    j < len(sequence) - 1
                    and (sequence[j], sequence[j + 1]) == best_pair
                ):
                    new_sequence.append(merged_word)
                    j += 2  # skip both tokens that were merged
                else:
                    new_sequence.append(sequence[j])
                    j += 1

            new_word_sequences.append(new_sequence)
        word_sequences = new_word_sequences

    if verbose:
        logger.info(
            f"APE vocabulary training complete. Final vocabulary size: {len(current_vocab)}"
        )
    if not path.exists(config.paths.selfies_ape_vocab):
        Path(config.paths.selfies_ape_vocab).parent.mkdir(parents=True, exist_ok=True)
    with open(
        config.paths.selfies_ape_vocab, "w", encoding="utf-8)"
    ) as vocab_file:
        json.dump(current_vocab, vocab_file, ensure_ascii=False, indent=4)
    logger.info(f"Final vocabulary saved to {config.paths.selfies_ape_vocab}")
    return current_vocab
