from curses import raw
import os
import logging
import re
import selfies as sf
import torch

from transformers import PreTrainedTokenizerFast
# Make sure necessary imports are present
from tokenizers import Tokenizer, AddedToken
from tokenizers.models import WordLevel, BPE # Import models if used below
from tokenizers.trainers import WordLevelTrainer, BpeTrainer # Import trainers if used belo
from transformers import PreTrainedTokenizerFast
from collections import OrderedDict
from pathlib import Path
from tokenizers import Tokenizer, models, pre_tokenizers, processors

logger = logging.getLogger(__name__)

class SelfiesTokenizer(PreTrainedTokenizerFast):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def decode(self, token_ids, skip_special_tokens=True, **kwargs):
        """
        Decodes token IDS into a string without spaces. E.g., "[C][Branch1]"
        """
        decoded = super().decode(token_ids, skip_special_tokens=skip_special_tokens, **kwargs)
        return decoded.replace(" ", "")

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
            unique_tokens = list(OrderedDict.fromkeys(special_tokens + raw_tokens))  # preserves order

        vocab = {tok: idx for idx, tok in enumerate(unique_tokens)}

        if config.tokenizer_type == "wordlevel":
            tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
            tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
        elif config.tokenizer_type == "bpe":
            tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
            tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
            tokenizer.add_special_tokens(special_tokens)
            tokenizer.add_tokens([AddedToken(t, normalized=False) for t in raw_tokens])
            corpus_txt = Path(config.directory_paths.selfies_nospace_txt)   # one molecule per line
            merges_to_learn = 50             # change in config later
            trainer = BpeTrainer(
                vocab_size=len(raw_tokens) + merges_to_learn,
                special_tokens=special_tokens,             
                show_progress=True
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
        )
        selfies_tokenizer.save_pretrained("selfies_wordlevel")
        example = "[C][Branch1][=C]"
        out = selfies_tokenizer(example)
        print(out.tokens())          # ← need parentheses; .tokens is a method
        # ['[BOS]', '[C]', '[Branch1]', '[=C]', '[EOS]']
        print(selfies_tokenizer.decode(out["input_ids"]))
        # [C][Branch1][=C]
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

# Placeholder
def tokenize_selfies_vocab(config, tokenizer, raw_data=None, chunk_size=50000, max_length=310):
    pass