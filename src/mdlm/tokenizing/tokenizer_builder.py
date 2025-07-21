import json
import logging
import os
from collections import OrderedDict

# Make sure necessary imports are present
from tokenizers import Regex, Tokenizer, pre_tokenizers, processors
from tokenizers.models import WordLevel

from .learn_ape_vocab import train_selfies_bpe_vocab
from .selfies_tokenizer import SelfiesTokenizer

logger = logging.getLogger(__name__)


def _load_selfies_vocab(alphabet_path: str, special_tokens: list):
    logger.info(f"Loading alphabet from: {alphabet_path}")
    try:
        with open(alphabet_path, encoding="utf-8") as f:
            raw_tokens = [line.strip() for line in f if line.strip()]
        unique_tokens = list(OrderedDict.fromkeys(special_tokens + raw_tokens))
        vocab = {tok: idx for idx, tok in enumerate(unique_tokens)}
        return vocab, unique_tokens
    except Exception as e:
        logger.error(f"Error loading SELFIES alphabet: {e}")
        raise


def _build_wordlevel_tokenizer(vocab: dict):
    logger.info("Building WordLevel tokenizer")
    atom_rgx = Regex(r"\[[^\]]+\]")
    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Split(atom_rgx, behavior="isolated")
    return tokenizer


def _build_ape_tokenizer(config, data, vocab):
    logger.info("Building APE tokenizer")
    vocab_path = config.paths.selfies_ape_vocab
    if config.checkpointing.retrain_ape_vocab:
        logger.info("Training APE tokenizer from scratch using raw data")
        selfies_vocab = train_selfies_bpe_vocab(
            config,
            data["selfies"].tolist(),
            vocab,
            config.tokenizer.max_vocab_size,
            config.tokenizer.min_freq_for_merge,
            verbose=True,
        )
        return Tokenizer(WordLevel(vocab=selfies_vocab, unk_token="[UNK]"))
    elif os.path.exists(vocab_path):
        logger.info(f"Loading APE vocab from {vocab_path}")
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab_json = json.load(f)
        # if each token of the vocab is not in the json dictionary version, we need to update the tokenizer vocab
        for token in vocab:
            if token not in vocab_json:
                vocab_json[token] = len(vocab_json)
        return Tokenizer(WordLevel(vocab=vocab_json, unk_token="[UNK]"))
    raise ValueError("No valid APE vocab found and retrain flag is not set.")


def _configure_tokenizer(tokenizer: Tokenizer, vocab: dict) -> Tokenizer:
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[BOS] $A [EOS]",
        special_tokens=[("[BOS]", vocab["[BOS]"]), ("[EOS]", vocab["[EOS]"])],
    )
    tokenizer.enable_padding(
        direction="right",
        pad_id=vocab["[PAD]"],
        pad_token="[PAD]",
    )
    return tokenizer


def _add_conditioning_tokens(config, vocab: dict) -> dict:
    logger.info("Adding conditioning tokens to the vocabulary")
    number_of_properties = len(config.conditioning.properties)
    for i in range(number_of_properties):
        for j in range(config.preprocessing.discretize_num_bins):
            token = f"[{config.conditioning.properties[i]}_bin_{j+1}|{config.preprocessing.discretize_num_bins}]"
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab


def _train_tokenizer(config, data=None):
    logger.info("Training tokenizer from alphabet")
    special_tokens = ["[BOS]", "[EOS]", "[SEP]", "[CLS]", "[PAD]", "[MASK]", "[UNK]"]
    vocab, _ = _load_selfies_vocab(config.paths.selfies_alphabet, special_tokens)

    if config.conditioning.properties is not None:
        vocab = _add_conditioning_tokens(config, vocab)

    tokenizer_type = config.tokenizer.tokenizer_type
    if tokenizer_type == "wordlevel":
        tokenizer = _build_wordlevel_tokenizer(vocab)
    elif tokenizer_type == "APE" or "ape" in tokenizer_type.lower():
        tokenizer = _build_ape_tokenizer(config, data, vocab)
    else:
        raise ValueError(f"Unsupported tokenizer_type: {tokenizer_type}")

    tokenizer = _configure_tokenizer(tokenizer, vocab)

    selfies_tokenizer = SelfiesTokenizer(
        tokenizer_object=tokenizer,
        bos_token="[BOS]",
        eos_token="[EOS]",
        pad_token="[PAD]",
        unk_token="[UNK]",
        sep_token="[SEP]",
        cls_token="[CLS]",
        mask_token="[MASK]",
    )
    logger.info("Saving tokenizer to: %s", config.paths.tokenizer)
    selfies_tokenizer.save_pretrained(config.paths.tokenizer)
    return selfies_tokenizer


def _load_tokenizer(config):
    tokenizer_dir = config.paths.tokenizer
    logger.info(f"Loading tokenizer from {tokenizer_dir}")
    return SelfiesTokenizer.from_pretrained(tokenizer_dir)


def get_tokenizer(config, data=None):
    try:
        should_load = (
            os.path.isdir(config.paths.tokenizer)
            and os.listdir(config.paths.tokenizer)
            and not config.checkpointing.retrain_tokenizer
        )

        logger.info(f"Should load tokenizer: {should_load}")
        if should_load:
            return _load_tokenizer(config)
        else:
            return _train_tokenizer(config, data)
    except Exception:
        logger.exception("Failed to get tokenizer")  # includes traceback
        raise
