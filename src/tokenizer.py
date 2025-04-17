import os
import logging
import re
import selfies as sf
import torch

from transformers import PreTrainedTokenizerFast
# Make sure necessary imports are present
from tokenizers import Tokenizer, NormalizedString, PreTokenizedString, Regex
from tokenizers.pre_tokenizers import PreTokenizer
from tokenizers.models import WordLevel, BPE # Import models if used below
from tokenizers.trainers import WordLevelTrainer, BpeTrainer # Import trainers if used below


logger = logging.getLogger(__name__)

# --- Corrected SelfiesPreTokenizer ---
class SelfiesPreTokenizer:
    def __init__(self):
        # Store the raw regex STRING - THIS is what tokenizers needs
        self.pattern_str: str = r"\[[^\[\]]+?\]|\."
        # Keep the compiled version for potential use with Python's 're' module (e.g., in pre_tokenize_str)
        self.compiled_pattern: re.Pattern = re.compile(self.pattern_str)

    def pre_tokenize(self, pretok: PreTokenizedString):
        """
        Applies the SELFIES splitting logic to the PreTokenizedString in-place.
        """
        def split_function(i: int, normalized_string: NormalizedString) -> list[NormalizedString]:
            # *** Use the raw string pattern (self.pattern_str) here ***
            return normalized_string.split(self.pattern_str, behavior='isolated')

        # Apply the split function using the PreTokenizedString's split method
        pretok.split(split_function)

    def pre_tokenize_str(self, sequence: str) -> list[tuple[str, tuple[int, int]]]:
        """
        Pre-tokenizes a raw string, returning tokens and their offsets.
        Uses the compiled pattern for standard Python 're' operations.
        """
        tokens_with_offsets = []
        # Use the compiled pattern here with re.finditer
        for match in self.compiled_pattern.finditer(sequence):
            start, end = match.span()
            tokens_with_offsets.append((match.group(0), (start, end)))
        return tokens_with_offsets

# --- The rest of your code remains the same as the previous version ---
# (SelfiesTokenizer, SPECIAL_TOKENS, train_or_load_selfies_tokenizer, get_tokenizer, tokenize_selfies_vocab)

# Example snippets from the rest of your code that should now work:

class SelfiesTokenizer(PreTrainedTokenizerFast):
    # ... (as before) ...
    pass

SPECIAL_TOKENS = ["[BOS]", "[EOS]", "[SEP]", "[CLS]", "[PAD]", "[MASK]", "[UNK]"]

def train_or_load_selfies_tokenizer(config):
    tokenizer_dir = config.directory_paths.tokenizer
    if os.path.isdir(tokenizer_dir) and os.listdir(tokenizer_dir) and not config.checkpointing.retrain_tokenizer:
        logger.info(f"Loading tokenizer from {tokenizer_dir}")
        return SelfiesTokenizer.from_pretrained(tokenizer_dir)

    selfies_txt_file_path = config.directory_paths.selfies_txt
    if not selfies_txt_file_path or not os.path.exists(selfies_txt_file_path):
        logger.error(f"Selfies text file not found or not provided: {selfies_txt_file_path}")
        raise FileNotFoundError(f"Required selfies data file not found: {selfies_txt_file_path}")
    else:
        logger.info(f"Training tokenizer from scratch using {selfies_txt_file_path}")
        files = [selfies_txt_file_path]

        # Instantiate the custom pre-tokenizer (this is fine)
        selfies_pre_tokenizer = SelfiesPreTokenizer()

        if config.tokenizer_type == "wordlevel":
            tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
            tokenizer.pre_tokenizer = PreTokenizer.custom(selfies_pre_tokenizer) # Assign instance
            trainer = WordLevelTrainer(
                special_tokens=SPECIAL_TOKENS,
                vocab_size=config.vocab_size, # Ensure these are in config
                min_frequency=config.min_frequency # Ensure these are in config
            )
        elif config.tokenizer_type == "bpe":
            tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
            tokenizer.pre_tokenizer = PreTokenizer.custom(selfies_pre_tokenizer) # Assign instance
            trainer = BpeTrainer(
                special_tokens=SPECIAL_TOKENS,
                vocab_size=config.vocab_size, # Ensure these are in config
                min_frequency=config.min_frequency # Ensure these are in config
            )
        else:
             raise ValueError(f"Unsupported tokenizer_type: {config.tokenizer_type}")

        # THIS CALL caused the error, but should now work because the pre_tokenizer uses the string pattern internally
        logger.info(f"Starting tokenizer training with trainer: {trainer!r}") # Use !r for repr
        tokenizer.train(files, trainer)
        logger.info("Tokenizer training complete.")

        # ... (rest of the function: creating Fast tokenizer, testing, saving) ...
        fast_tokenizer = SelfiesTokenizer( # Use your custom class here!
            tokenizer_object=tokenizer,
            # ... other tokens ...
        )
        # ... testing ...
        fast_tokenizer.save_pretrained(tokenizer_dir)
        return fast_tokenizer

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