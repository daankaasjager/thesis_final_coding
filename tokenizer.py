import os
from idna import decode
import torch
import hydra
import logging
import pandas as pd
import selfies as sf
from omegaconf import DictConfig, OmegaConf
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers
from transformers import PreTrainedTokenizerFast
import selfies
from typing import Iterable, Set, List
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
            #build from the raw selfies_vocab
            if selfies_vocab is None:
                raise ValueError("Must provide either `selfies_vocab` or `tokenizer_file`.")
            vocab_dict = {}
            idx = 0
            
            # Add special tokens first
            special_tokens = [
                bos_token, eos_token, sep_token, cls_token,
                pad_token, mask_token, unk_token
            ]
            for token in special_tokens:
                if token not in vocab_dict:
                    vocab_dict[token] = idx
                    idx += 1

            # Add SELFIES symbols
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
        """Decodes token IDS into a string without spaces. E.g., "[C][Branch1]" """
        decoded = super().decode(token_ids, skip_special_tokens=skip_special_tokens, **kwargs)
        decoded_no_spaces = decoded.replace(" ", "")
        return decoded_no_spaces
    

def get_tokenizer(config, selfies_vocab):
    if os.path.exists(config.directory_paths.tokenizer):
        logger.info("Tokenizer folder found. Loading...")
        try:
            tokenizer = SelfiesTokenizer.from_pretrained(config.directory_paths.tokenizer)
        except Exception as e:
            logger.error(f"Error loading tokenizer: {e}")
            exit()
    else:
        logger.info(f"No tokenizer found at {config.directory_paths.tokenizer}. Creating...")
        tokenizer = SelfiesTokenizer(selfies_vocab=selfies_vocab)
        tokenizer.save_pretrained(config.directory_paths.tokenizer)
        logger.info(f"Tokenizer saved to {config.directory_paths.tokenizer}")

    """ This logic makes sure that tokenizer is compatible with MDLM model"""
    if tokenizer.bos_token is None:
        if tokenizer.cls_token is None:
            raise AttributeError(
                'Tokenizer must have a bos_token or '
                f'cls_token: {tokenizer}')
        tokenizer.bos_token = tokenizer.cls_token
    if tokenizer.eos_token is None:
        if tokenizer.sep_token is None:
            raise AttributeError(
            'Tokenizer must have a eos_token '
            f'or sep_token: {tokenizer}')
        tokenizer.eos_token = tokenizer.sep_token
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    return tokenizer



""" This is not part of the functionality that initializes the tokenizer. 
This is a separate function that tokenizes the SELFIES data. """

def add_bos_and_eos_tokens(tokenized_data, tokenizer):
    """
    Adds BOS and EOS tokens to each sequence in the BatchEncoding.
    Assumes tokenized_data['input_ids'] is a torch.Tensor of shape (batch_size, seq_length).
    Returns a new BatchEncoding with updated 'input_ids'.
    """
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    # Convert each sequence to list, add special tokens, and collect
    new_input_ids = []
    for seq in tokenized_data['input_ids']:
        # Convert tensor to list of ints
        seq_list = seq.tolist()
        # Prepend BOS and append EOS
        new_seq = [bos_id] + seq_list + [eos_id]
        new_input_ids.append(new_seq)
    # Optionally, you could pad these new sequences to a fixed length if needed.
    # Here we simply convert them back to a tensor (this might result in a ragged tensor if lengths differ)
    # For simplicity, we'll pad to the max length in the new batch.
    max_len = max(len(seq) for seq in new_input_ids)
    padded_input_ids = [seq + [tokenizer.pad_token_id] * (max_len - len(seq)) for seq in new_input_ids]
    tokenized_data['input_ids'] = torch.tensor(padded_input_ids, dtype=torch.long)
    return tokenized_data


def tokenize_selfies_vocab(tokenizer, config, raw_data):
    # If encoded inputs are not found, tokenize the SELFIES data, else, load the tokenized data
    if os.path.exists(config.directory_paths.train_data_encoding):
        logger.info(f"SELFIES training data encoding found, loading data from {config.directory_paths.train_data_encoding}")
        try :
            tokenized_data = torch.load(config.directory_paths.train_data_encoding, weights_only=False)
            logger.info(f"SELFIES data loaded successfully. Vocabulary size is: {tokenizer.vocab_size}")
            return tokenized_data, tokenizer.vocab_size
        except Exception as e:
            logger.error(f"Error loading SELFIES data: {e}")
            return
    else: 
        logger.info(f"No SELFIES training data encoding found at {config.directory_paths.train_data_encoding}. Tokenizing SELFIES training data...")
        # Check if 'selfies' column exists in the raw data.
        if 'selfies' not in raw_data.columns:
            logger.info("'selfies' column not found")
            return
        else:
            logger.info("'selfies' column found. Tokenizing SELFIES data...")
            # Prepare the list of SELFIES strings lists
            input_selfies = raw_data['selfies'].to_list()
            try:
                # Tokenize the SELFIES data
                tokenized_data = tokenizer.batch_encode_plus(
                    input_selfies,
                    is_split_into_words=True, # IMPORTANT! This is required because the data is pre split into tokens ["[C]"", "[=C]"", "[F]"]
                    max_length=256, #Change this to actual max length of training data
                    padding='longest',
                    truncation=False,
                    return_tensors='pt'
                )
                tokenized_data = add_bos_and_eos_tokens(tokenized_data, tokenizer)
                logger.info("SELFIES data tokenized successfully")
                try:
                    # Save the tokenized data
                    torch.save(tokenized_data, config.directory_paths.train_data_encoding)
                    logger.info(f"Tokenized data saved to {config.directory_paths.train_data_encoding}")
                except Exception as e:
                    logger.error(f"Error saving tokenized SELFIES data: {e}")
                    exit()
            except Exception as e: 
                logger.error(f"Error tokenizing SELFIES data: {e}")
                exit()
    logger.info(f"Vocabulary size: {tokenizer.vocab_size}")
    return tokenized_data, tokenizer.vocab_size
