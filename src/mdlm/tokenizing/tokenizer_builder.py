"""
Tokenizer management for SELFIES sequences, supporting WordLevel and APE tokenizers
with optional conditioning tokens and vocabulary caching.
"""

import json
import logging
import os
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

from tokenizers import Regex, Tokenizer, pre_tokenizers, processors
from tokenizers.models import WordLevel

from .learn_ape_vocab import train_selfies_bpe_vocab
from .selfies_tokenizer import SelfiesTokenizer

logger = logging.getLogger(__name__)


def _load_selfies_vocab(
    alphabet_path: str, special_tokens: List[str]
) -> Tuple[Dict[str, int], List[str]]:
    """
    Load the SELFIES alphabet and prepend special tokens.

    Args:
        alphabet_path: Path to the file containing SELFIES tokens, one per line.
        special_tokens: List of tokens to reserve at the beginning (e.g., BOS, EOS).

    Returns:
        A tuple of:
        - vocab: mapping from token to integer index.
        - unique_tokens: list of tokens in index order.
    """
    logger.info(f"Loading alphabet from: {alphabet_path}")
    with open(alphabet_path, encoding="utf-8") as f:
        raw_tokens = [line.strip() for line in f if line.strip()]
    unique_tokens = list(OrderedDict.fromkeys(special_tokens + raw_tokens))
    vocab = {tok: idx for idx, tok in enumerate(unique_tokens)}
    return vocab, unique_tokens


def _build_wordlevel_tokenizer(vocab: Dict[str, int]) -> Tokenizer:
    """
    Construct a WordLevel tokenizer that splits on SELFIES atom patterns.

    Args:
        vocab: token-to-index mapping for WordLevel model.

    Returns:
        Configured Tokenizer instance.
    """
    logger.info("Building WordLevel tokenizer")
    atom_pattern = Regex(r"\[[^\]]+\]")
    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Split(atom_pattern, behavior="isolated")
    return tokenizer


def _build_ape_tokenizer(config: Any, data: Any, vocab: Dict[str, int]) -> Tokenizer:
    """
    Construct or load an APE (BPE) tokenizer for SELFIES.

    Args:
        config: configuration object with paths and training flags.
        data: DataFrame or dict containing raw 'selfies' list.
        vocab: base vocabulary for conditioning tokens.

    Returns:
        A Tokenizer instance using WordLevel over the learned BPE vocab.

    Raises:
        ValueError: if no existing vocab file and retrain flag is False.
    """
    logger.info("Building APE tokenizer")
    vocab_path = config.paths.selfies_ape_vocab
    if config.checkpointing.retrain_ape_vocab:
        logger.info("Training APE tokenizer from scratch")
        selfies_vocab = train_selfies_bpe_vocab(
            config,
            data["selfies"].tolist(),
            vocab,
            config.tokenizer.max_vocab_size,
            config.tokenizer.min_freq_for_merge,
            verbose=True,
        )
        return Tokenizer(WordLevel(vocab=selfies_vocab, unk_token="[UNK]"))

    if os.path.exists(vocab_path):
        logger.info(f"Loading APE vocab from {vocab_path}")
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab_json = json.load(f)
        for token in vocab:
            if token not in vocab_json:
                vocab_json[token] = len(vocab_json)
        return Tokenizer(WordLevel(vocab=vocab_json, unk_token="[UNK]"))

    raise ValueError("No APE vocab found and retrain_ape_vocab is False")


def _configure_tokenizer(tokenizer: Tokenizer, vocab: Dict[str, int]) -> Tokenizer:
    """
    Add special tokens and padding configuration to a tokenizer.

    Args:
        tokenizer: base Tokenizer instance.
        vocab: token-to-index mapping for special tokens.

    Returns:
        The same Tokenizer instance with post-processing and padding enabled.
    """
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[BOS] $A [EOS]",
        special_tokens=[("[BOS]", vocab["[BOS]"]), ("[EOS]", vocab["[EOS]"])],
    )
    tokenizer.enable_padding(
        direction="right", pad_id=vocab["[PAD]"], pad_token="[PAD]"
    )
    return tokenizer


def _add_conditioning_tokens(config: Any, vocab: Dict[str, int]) -> Dict[str, int]:
    """
    Extend the vocabulary with discretized conditioning tokens.

    Args:
        config: configuration with conditioning.properties and discretize_num_bins.
        vocab: existing token-to-index mapping to modify.

    Returns:
        Updated vocab including new conditioning tokens.
    """
    logger.info("Adding conditioning tokens to the vocabulary")
    props = config.conditioning.properties
    bins = config.preprocessing.discretize_num_bins
    for prop in props:
        for i in range(bins):
            token = f"[{prop}_bin_{i+1}|{bins}]"
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab


def _train_tokenizer(config: Any, data: Any) -> SelfiesTokenizer:
    """
    Train and save a SELFIES tokenizer according to config.

    Args:
        config: configuration object with paths and tokenizer settings.
        data: dataset containing 'selfies' column for training APE vocab if needed.

    Returns:
        A SelfiesTokenizer instance saved to disk.
    """
    logger.info("Training tokenizer from alphabet")
    special_tokens = ["[BOS]", "[EOS]", "[SEP]", "[CLS]", "[PAD]", "[MASK]", "[UNK]"]
    vocab, _ = _load_selfies_vocab(config.paths.selfies_alphabet, special_tokens)
    if config.conditioning.properties:
        vocab = _add_conditioning_tokens(config, vocab)

    ttype = config.tokenizer.tokenizer_type.lower()
    if ttype == "wordlevel":
        tok = _build_wordlevel_tokenizer(vocab)
    elif "ape" in ttype:
        tok = _build_ape_tokenizer(config, data, vocab)
    else:
        raise ValueError(
            f"Unsupported tokenizer_type: {config.tokenizer.tokenizer_type}"
        )

    tok = _configure_tokenizer(tok, vocab)
    selfies_tok = SelfiesTokenizer(
        tokenizer_object=tok,
        bos_token="[BOS]",
        eos_token="[EOS]",
        pad_token="[PAD]",
        unk_token="[UNK]",
        sep_token="[SEP]",
        cls_token="[CLS]",
        mask_token="[MASK]",
    )
    logger.info(f"Saving tokenizer to: {config.paths.tokenizer}")
    selfies_tok.save_pretrained(config.paths.tokenizer)
    return selfies_tok


def _load_tokenizer(config: Any) -> SelfiesTokenizer:
    """
    Load a saved SelfiesTokenizer from disk.

    Args:
        config: configuration object with tokenizer path.

    Returns:
        Loaded SelfiesTokenizer instance.
    """
    logger.info(f"Loading tokenizer from {config.paths.tokenizer}")
    return SelfiesTokenizer.from_pretrained(config.paths.tokenizer)


def get_tokenizer(config: Any, data: Optional[Any] = None) -> SelfiesTokenizer:
    """
    Get or train a SELFIES tokenizer based on config.

    Args:
        config: configuration object with checkpointing.retrain_tokenizer flag.
        data: optional raw data for training APE vocab.

    Returns:
        SelfiesTokenizer instance, loaded or newly trained.
    """
    should_load = (
        os.path.isdir(config.paths.tokenizer)
        and os.listdir(config.paths.tokenizer)
        and not config.checkpointing.retrain_tokenizer
    )
    logger.info(f"Should load tokenizer: {should_load}")
    return _load_tokenizer(config) if should_load else _train_tokenizer(config, data)
