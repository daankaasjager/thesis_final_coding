from curses import raw
import os
import logging
import re
import selfies as sf
import torch
import math
from collections import defaultdict

from transformers import PreTrainedTokenizerFast
# Make sure necessary imports are present
from tokenizers import Tokenizer, AddedToken, Regex
from tokenizers.models import WordLevel, BPE # Import models if used below
from tokenizers.trainers import WordLevelTrainer, UnigramTrainer # Import trainers if used belo
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast
from collections import OrderedDict
from pathlib import Path
from tokenizers import Tokenizer, models, pre_tokenizers, processors

logger = logging.getLogger(__name__)

class SelfiesTokenizer(PreTrainedTokenizerFast):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def decode(
        self,
        token_ids,
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = False, # Added default argument to match base class
        **kwargs,
    ) -> str:
        """
        Decodes a list of token IDs back into a SELFIES string.

        Args:
            token_ids: List of token IDs to decode.
            skip_special_tokens: Whether to remove special tokens (like BOS, EOS, PAD)
                                 from the output string. Defaults to True.
            clean_up_tokenization_spaces: Whether to clean up spaces before/after tokens.
                                          Defaults to False for SELFIES to keep tokens like '[C]' intact.
            **kwargs: Additional decoding arguments passed to the underlying tokenizer.

        Returns:
            The decoded SELFIES string.
        """
        # ➊ Convert ids → token strings using the base class method
        #    but ensure skip_special_tokens is False initially so we can handle it manually
        tokens = self.convert_ids_to_tokens(token_ids, skip_special_tokens=False)

        # ➋ Filter out special tokens if requested
        if skip_special_tokens:
            # Use the standard all_special_tokens attribute from the base class
            special_tokens_set = set(self.all_special_tokens)
            tokens = [tok for tok in tokens if tok not in special_tokens_set]

        # ➌ Join tokens without spaces, suitable for SELFIES format '[C][Branch1][C]'
        #    The clean_up_tokenization_spaces=False argument in the signature helps
        #    signal that we handle spacing manually here.
        text = "".join(tokens)

        # Optional: If needed, implement clean_up_tokenization_spaces logic here,
        # but for SELFIES, direct joining is usually desired.

        return text


def train_or_load_selfies_tokenizer(config):
    tokenizer_dir = config.directory_paths.tokenizer
    if os.path.isdir(tokenizer_dir) and os.listdir(tokenizer_dir) and not config.checkpointing.retrain_tokenizer:
        logger.info(f"Loading tokenizer from {tokenizer_dir}")
        return SelfiesTokenizer.from_pretrained(tokenizer_dir)

    selfies_alphabet_path = config.directory_paths.selfies_alphabet

    if not os.path.exists(selfies_alphabet_path):
        logger.error(f"Selfies text file not found or not provided: {selfies_alphabet_path}")
        raise FileNotFoundError(f"Required selfies data file not found: {selfies_alphabet_path}")
    else:
        special_tokens = ["[BOS]", "[EOS]", "[SEP]", "[CLS]", "[PAD]", "[MASK]", "[UNK]"]
        logger.info(f"Training tokenizer from scratch using {selfies_alphabet_path}")
        # this is a path to a txt file, where each line is a token. Read appropriately.
        # 1️⃣  read one token per line, keep original order, drop duplicates

        with open(selfies_alphabet_path, encoding="utf-8") as f:
            raw_tokens = [line.strip() for line in f if line.strip()]
            unique_tokens = list(OrderedDict.fromkeys(special_tokens + raw_tokens))
        vocab = {tok: idx for idx, tok in enumerate(unique_tokens)}
        if config.tokenizer_type == "wordlevel":
            atom_rgx = Regex(r"\[[^\]]+\]") 
            tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
            tokenizer.pre_tokenizer = pre_tokenizers.Split(
                atom_rgx, behavior="isolated"
            )  
        elif config.tokenizer_type == "unigram":
            tokenizer = Tokenizer(models.Unigram())            # <──  only line that really changes
            tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()   # atoms already have spaces
            corpus_txt = Path(config.directory_paths.selfies_whitespace_txt)
            trainer = UnigramTrainer(
                vocab_size=len(vocab) + 50,              # learn ≈50 new motifs
                initial_alphabet=[],                     # not used by Unigram
                special_tokens=unique_tokens,   # just the framework specials
                unk_token="[UNK]",
            )
            tokenizer.train([str(corpus_txt)], trainer)
        else:
             raise ValueError(f"Unsupported tokenizer_type: {config.tokenizer_type}")

        tokenizer.post_processor = processors.TemplateProcessing(
            single="[BOS] $A [EOS]",
            special_tokens=[("[BOS]", vocab["[BOS]"]), ("[EOS]", vocab["[EOS]"])],
        )

        tokenizer.enable_padding(direction="right",
                        pad_id=vocab["[PAD]"],
                        pad_token="[PAD]")

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
        print(f"vocab of tokenizer: {selfies_tokenizer.get_vocab()}")

        #selfies_tokenizer._real_specials = set(special_tokens)   # only BOS/EOS/… – not atoms

        selfies_tokenizer.save_pretrained(config.directory_paths.tokenizer)
        example = "[C][Branch1][=C]"
        out = selfies_tokenizer(example)
        print(out.tokens())          # ← need parentheses; .tokens is a method
        # ['[BOS]', '[C]', '[Branch1]', '[=C]', '[EOS]']
        print(selfies_tokenizer.decode(out["input_ids"], skip_special_tokens=True))
        # [C][Branch1][=C]
        atom_set = set(raw_tokens)
        return selfies_tokenizer
    
def get_tokenizer(config):
    try:
        tokenizer = train_or_load_selfies_tokenizer(config)
        return tokenizer
    except Exception as e:
        logger.error(f"Failed to get tokenizer: {e}", exc_info=True) # Log traceback
        # It's helpful to see the original exception type as well
        logger.error(f"Original exception type: {type(e).__name__}")
        raise

def tokenize_selfies_vocab(config, tokenizer, raw_data=None, chunk_size=50000, max_length=310):
    if os.path.exists(config.directory_paths.train_data_encoding) and not config.checkpointing.retrain_tokenizer:
        logger.info(f"SELFIES training data encoding found at {config.directory_paths.train_data_encoding}")
        try:
            tokenized_data = torch.load(config.directory_paths.train_data_encoding, map_location="cpu", weights_only = False)
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
        torch.save(tokenized_data, config.directory_paths.train_data_encoding, pickle_protocol=4)
        logger.info(f"Tokenized data saved to {config.directory_paths.train_data_encoding}")
    except Exception as e:
        logger.error(f"Error saving tokenized SELFIES data: {e}")
    logger.info(f"Done tokenizing. Vocab size: {tokenizer.vocab_size}")
    return tokenized_data