import os
import math
import torch
import logging
import pandas as pd
from typing import Set
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

logger = logging.getLogger(__name__)

class SelfiesTokenizer(PreTrainedTokenizerFast):
    """
    A custom tokenizer that inherits from PreTrainedTokenizerFast and
    builds a vocabulary from a SELFIES alphabet plus special tokens.
    """
    def __init__(
        self,
        selfies_vocab: Set[str] = None,
        bos_token="[BOS]",
        eos_token="[EOS]",
        sep_token="[SEP]",
        cls_token="[CLS]",
        pad_token="[PAD]",
        mask_token="[MASK]",
        unk_token="[UNK]",
        tokenizer_file=None,
        **kwargs
    ):
        if tokenizer_file is not None:
            super().__init__(
                tokenizer_file=tokenizer_file,
                bos_token=bos_token,
                eos_token=eos_token,
                sep_token=sep_token,
                cls_token=cls_token,
                pad_token=pad_token,
                mask_token=mask_token,
                unk_token=unk_token,
                **kwargs
            )
            self.vocab_dict = self.get_vocab()
        else:
            if selfies_vocab is None:
                raise ValueError("Must provide either `selfies_vocab` or `tokenizer_file`.")
            vocab_dict = {}
            idx = 0
            special_tokens = [
                bos_token, eos_token, sep_token, cls_token,
                pad_token, mask_token, unk_token
            ]
            for token in special_tokens:
                if token not in vocab_dict:
                    vocab_dict[token] = idx
                    idx += 1
            for symbol in selfies_vocab:
                if symbol not in vocab_dict:
                    vocab_dict[symbol] = idx
                    idx += 1
            tokenizer_backend = Tokenizer(WordLevel(vocab=vocab_dict, unk_token=unk_token))
            super().__init__(
                tokenizer_object=tokenizer_backend,
                bos_token=bos_token,
                eos_token=eos_token,
                sep_token=sep_token,
                cls_token=cls_token,
                pad_token=pad_token,
                mask_token=mask_token,
                unk_token=unk_token,
                **kwargs
            )
            self.vocab_dict = vocab_dict
    
    def decode(self, token_ids, skip_special_tokens=True, **kwargs):
        """
        Decodes token IDS into a string without spaces. E.g., "[C][Branch1]"
        """
        decoded = super().decode(token_ids, skip_special_tokens=skip_special_tokens, **kwargs)
        return decoded.replace(" ", "")

def get_tokenizer(config, selfies_vocab):
    """
    Loads an existing tokenizer from config.directory_paths.tokenizer
    if it exists, otherwise builds a new one from selfies_vocab.
    Ensures BOS/EOS/PAD tokens exist.
    """
    if os.path.exists(config.directory_paths.tokenizer) and config.checkpointing.fresh_data == False:
        logger.info("Tokenizer folder found. Loading...")
        try:
            tokenizer = SelfiesTokenizer.from_pretrained(config.directory_paths.tokenizer)
        except Exception as e:
            logger.error(f"Error loading tokenizer: {e}")
            exit()
    elif not os.path.exists(config.directory_paths.tokenizer) and config.pointing.fresh_data == False:
        logger.error(f"No tokenizer found at {config.directory_paths.tokenizer}. But config.resh_data is {config.nfig.resh_data}.")
        exit()
    else:
        logger.info(f"config.pointing.fresh_data is {config.checkpointing.fresh_data}.  Creating new tokenizer...")
        tokenizer = SelfiesTokenizer(selfies_vocab=selfies_vocab)
        tokenizer.save_pretrained(config.directory_paths.tokenizer)
        logger.info(f"Tokenizer saved to {config.directory_paths.tokenizer}")

    if tokenizer.bos_token is None:
        if tokenizer.cls_token is None:
            raise AttributeError('Tokenizer must have a bos_token or cls_token.')
        tokenizer.bos_token = tokenizer.cls_token
    if tokenizer.eos_token is None:
        if tokenizer.sep_token is None:
            raise AttributeError('Tokenizer must have a eos_token or sep_token.')
        tokenizer.eos_token = tokenizer.sep_token
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    return tokenizer

def tokenize_selfies_vocab(config, tokenizer, raw_data=None, chunk_size=50000, max_length=310):
    """
    Chunk-based tokenization for SELFIES data. Merges all sequences into a single dictionary 
    with 'input_ids', 'attention_mask', and 'token_type_ids' stored as lists of lists of ints.
    """
    if os.path.exists(config.directory_paths.train_data_encoding) and config.checkpointing.fresh_data == False:
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
    all_input_ids = []
    all_attention_masks = []
    all_token_type_ids = []
    num_chunks = math.ceil(total_samples / chunk_size)
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    # REMOVE LATER
    max_seq_len = 0 

    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, total_samples)
        chunk = input_selfies[start_idx:end_idx]
        logger.info(f"Processing chunk {chunk_idx+1}/{num_chunks}: {len(chunk)} sequences")
        tokenized_chunk = tokenizer(
            chunk,
            is_split_into_words=True,
            max_length=max_length,
            padding='longest',
            truncation=False,
            return_tensors=None
        )
        chunk_input_ids = tokenized_chunk["input_ids"]
        chunk_attention_masks = tokenized_chunk["attention_mask"]
        for seq_ids, seq_mask in zip(chunk_input_ids, chunk_attention_masks):
            updated_ids = [bos_id] + seq_ids + [eos_id]
            updated_mask = [1] + seq_mask + [1]
            all_input_ids.append(updated_ids)
            all_attention_masks.append(updated_mask)
            all_token_type_ids.append([0]*len(updated_ids))

            # Update max sequence length
            current_len = len(updated_ids)
            if current_len > max_seq_len:
                max_seq_len = current_len
            # REMOVE LATER

    tokenized_data = {
        "input_ids": all_input_ids,
        "attention_mask": all_attention_masks,
        "token_type_ids": all_token_type_ids
    }

    try:
        torch.save(tokenized_data, config.directory_paths.train_data_encoding, pickle_protocol=4)
        logger.info(f"Tokenized data saved to {config.directory_paths.train_data_encoding}")
    except Exception as e:
        logger.error(f"Error saving tokenized SELFIES data: {e}")
    logger.info(f"Done tokenizing. Vocab size: {tokenizer.vocab_size}")
    logger.info(f"Max sequence length in the final tokenized data: {max_seq_len}")
    # REMOVE LATER
    return tokenized_data
