from __future__ import annotations
import logging
from pathlib import Path
from typing import List, Tuple, Optional

import torch
import selfies
import pandas as pd

logger = logging.getLogger(__name__)


def _load_alphabet_from_txt(path: str | Path) -> List[str]:
    with open(path, encoding="utf-8") as f:
        return f.read().splitlines()


def load_preprocessed_data(
    preproc_path: str | Path,
    alphabet_path: str | Path,
) -> Tuple[List[str], pd.DataFrame]:
    """
    Load cached dataframe (+ alphabet txt).

    Accepts *both* the new format (file == DataFrame) **and** the legacy
    dict format ``{"alphabet": …, "raw_data": …}``.
    """
    try:
        obj = torch.load(preproc_path, weights_only=False)

        # unwrap legacy dict
        if isinstance(obj, dict) and "raw_data" in obj:
            data = obj["raw_data"]
        else:
            data = obj  # already a DataFrame

        alphabet = _load_alphabet_from_txt(alphabet_path)
        logger.info("Pre-processed data loaded successfully.")
        return alphabet, data

    except Exception as e:  # pragma: no cover
        logger.warning(
            f"Could not load pre-processed data: {e!s}. Will recompute."
        )
        raise


def save_selfies_alphabet(
    config,
    alphabet: list[str],
    *,
    include_special_tokens: bool = True,
) -> None:
    """Dump ``alphabet`` (one token/line) to *config.local_paths.selfies_alphabet*."""
    path = Path(config.local_paths.selfies_alphabet)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        logger.info("%s already exists – skipping.", path)
        return

    specials = ["[PAD]", "[BOS]", "[EOS]", "[UNK]"] if include_special_tokens else []
    vocab_lines = specials + sorted(alphabet)

    logger.info("Saving SELFIES alphabet (%d tokens) → %s", len(vocab_lines), path)
    path.write_text("\n".join(vocab_lines), encoding="utf-8")


def _write_selfies_txt(
    lines_path: str | Path,
    df: pd.DataFrame,
    *,
    whitespace: bool,
) -> None:
    """Write every list in `df["tokenized_selfies"]` to *lines_path*."""
    if Path(lines_path).exists():
        logger.info("%s already exists – skipping.", lines_path)
        return

    logger.info("Writing SELFIES data → %s", lines_path)
    with open(lines_path, "w", encoding="utf-8") as fh:
        for tok_list in df["tokenized_selfies"]:
            fh.write((" " if whitespace else "").join(tok_list) + "\n")

def preprocess_data(
    config,
    raw_data: Optional[pd.DataFrame] = None,
) -> Tuple[List[str], pd.DataFrame]:
    """
    Returns
    -------
    alphabet : list[str]
    dataframe : pd.DataFrame   (with column “tokenized_selfies”)
    """
    preproc_path = Path(config.local_paths.pre_processed_data)
    alphabet_path = Path(config.local_paths.selfies_alphabet)

    if (
        not config.checkpointing.fresh_data
        and preproc_path.exists()
        and alphabet_path.exists()
    ):
        return load_preprocessed_data(preproc_path, alphabet_path)

    if not config.checkpointing.fresh_data and not preproc_path.exists():
        raise ValueError("Data load requested but no pre-processed data found.")
    if raw_data is None:
        raise ValueError("raw_data must be provided when fresh_data=True")

    if "selfies" not in raw_data.columns:
        raise ValueError("'selfies' column not found in raw_data")

    logger.info("Creating vocabulary from SELFIES column …")
    alphabet = selfies.get_alphabet_from_selfies(raw_data["selfies"])

    # TOKENISE & FILTER ------------------------------------------------------
    def _tok_if_valid(s: str) -> Optional[list[str]]:
        toks = list(selfies.split_selfies(s))
        return toks if len(toks) <= config.permitted_selfies_length else None

    raw_data["tokenized_selfies"] = raw_data["selfies"].apply(_tok_if_valid)
    df = raw_data[raw_data["tokenized_selfies"].notnull()].copy()

    max_len = df["tokenized_selfies"].apply(len).max()
    logger.info("Longest SELFIES sequence kept: %d tokens", max_len)

    # CACHE ------------------------------------------------------------------
    preproc_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        torch.save(df, preproc_path)        # ⚠️  save *only* the dataframe
        logger.info("Pre-processed data saved → %s", preproc_path)
    except Exception as e:  # pragma: no cover
        logger.warning("Could not save pre-processed data: %s", e)

    # AUXILIARY TXT CORPORA --------------------------------------------------
    _write_selfies_txt(config.local_paths.selfies_nospace_txt, df, whitespace=False)
    _write_selfies_txt(config.local_paths.selfies_whitespace_txt, df, whitespace=True)

    # ALPHABET TXT -----------------------------------------------------------
    save_selfies_alphabet(config, alphabet)

    return alphabet, df
