import os
import logging
import torch
import math
from collections import defaultdict
import json
# Make sure necessary imports are present
from tokenizers import Tokenizer, Regex
from tokenizers.models import WordLevel
from collections import OrderedDict
from tokenizers import Tokenizer, pre_tokenizers, processors
from src.tokenizing.learn_ape_vocab import train_selfies_bpe_vocab
from src.tokenizing.selfies_tokenizer import SelfiesTokenizer

logger = logging.getLogger(__name__)


def load_tokenizer(config):
    tokenizer_dir = config.local_paths.tokenizer
    logger.info(f"Loading tokenizer from {tokenizer_dir}")
    return SelfiesTokenizer.from_pretrained(tokenizer_dir)

def load_selfies_vocab(alphabet_path: str, special_tokens: list):
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

def build_wordlevel_tokenizer(vocab: dict):
    atom_rgx = Regex(r"\[[^\]]+\]")
    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Split(atom_rgx, behavior="isolated")
    return tokenizer


def build_ape_tokenizer(config, data, vocab):
    vocab_path = config.local_paths.selfies_ape_vocab

    if os.path.exists(vocab_path):
        logger.info(f"Loading JSON vocab from {vocab_path}")
        with open(vocab_path, "r", encoding="utf-8") as f:
            tokenizer_json = json.load(f)
        return Tokenizer(WordLevel.from_json(tokenizer_json))

    if config.checkpointing.retrain_ape_vocab:
        logger.info("Training APE tokenizer from scratch using raw data")
        selfies_vocab = train_selfies_bpe_vocab(
            config,
            data["selfies"].tolist(),
            vocab,
            config.tokenizer.max_vocab_size,
            config.tokenizer.min_freq_for_merge,
            verbose=True
        )
        return Tokenizer(WordLevel(vocab=selfies_vocab, unk_token="[UNK]"))

    raise ValueError("No valid APE vocab found and retrain flag is not set.")


def configure_tokenizer(tokenizer: Tokenizer, vocab: dict) -> Tokenizer:
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


def train_tokenizer(config, data=None):
    special_tokens = ["[BOS]", "[EOS]", "[SEP]", "[CLS]", "[PAD]", "[MASK]", "[UNK]"]
    vocab, _ = load_selfies_vocab(config.local_paths.selfies_alphabet, special_tokens)

    tokenizer_type = config.tokenizer.tokenizer_type
    if tokenizer_type == "wordlevel":
        tokenizer = build_wordlevel_tokenizer(vocab)
    elif tokenizer_type == "APE":
        tokenizer = build_ape_tokenizer(config, data, vocab)
    else:
        raise ValueError(f"Unsupported tokenizer_type: {tokenizer_type}")

    tokenizer = configure_tokenizer(tokenizer, vocab)

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

    logger.info(f"Tokenizer vocabulary is: {selfies_tokenizer.get_vocab()}")
    selfies_tokenizer.save_pretrained(config.local_paths.tokenizer)

    return selfies_tokenizer



def get_tokenizer(config, data=None):
    try:
        should_load = (
            os.path.isdir(config.local_paths.tokenizer)
            and os.listdir(config.local_paths.tokenizer)
            and not config.checkpointing.retrain_tokenizer
        )
        if should_load:
            return load_tokenizer(config)
        else:
            return train_tokenizer(config, data)
    except Exception as e:
        logger.exception("Failed to get tokenizer")  # includes traceback
        raise


def tokenize_selfies_vocab(config, tokenizer, raw_data=None, chunk_size=50000, max_length=310):
    if os.path.exists(config.local_paths.train_data_encoding) and not config.checkpointing.retrain_tokenizer:
        logger.info(f"SELFIES training data encoding found at {config.local_paths.train_data_encoding}")
        try:
            tokenized_data = torch.load(config.local_paths.train_data_encoding, map_location="cpu", weights_only = False)
            logger.info(f"SELFIES data loaded successfully. Vocab size: {tokenizer.vocab_size}")
            return tokenized_data
        except Exception as e:
            logger.error(f"Error loading SELFIES data: {e}")
            return None
    if 'selfies' not in raw_data.columns:
        logger.info("'selfies' column not found in raw_data.")
        return None
    
    input_selfies = raw_data['selfies'].tolist()
    total_samples = len(input_selfies)
    logger.info(f"Tokenizing {total_samples} SELFIES in chunks of {chunk_size}...")
    num_chunks = math.ceil(total_samples / chunk_size)
    tokenized_data = defaultdict(list) 
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, total_samples)
        chunk = input_selfies[start_idx:end_idx]
        logger.info(f"Processing chunk {chunk_idx+1}/{num_chunks}: {len(chunk)} sequences")
        tokenized_chunk = tokenizer(
            chunk,
            is_split_into_words=False,
            max_length=max_length,
            padding='longest',
            truncation=False,
            add_special_tokens=True
        )
        tokenized_data['input_ids'].extend(tokenized_chunk['input_ids'])
        tokenized_data['attention_mask'].extend(tokenized_chunk['attention_mask'])
        token_type_ids = tokenized_chunk.get(
            "token_type_ids",
            [[0] * len(ids) for ids in tokenized_chunk["input_ids"]],
        )
        tokenized_data["token_type_ids"].extend(token_type_ids)
    try:
        torch.save(tokenized_data, config.local_paths.train_data_encoding, pickle_protocol=4)
        logger.info(f"Tokenized data saved to {config.local_paths.train_data_encoding}")
    except Exception as e:
        logger.error(f"Error saving tokenized SELFIES data: {e}")
    logger.info(f"Done tokenizing. Vocab size: {tokenizer.vocab_size}")
    return tokenized_data