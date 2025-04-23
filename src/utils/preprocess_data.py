import logging
import os
import torch
import selfies

logger = logging.getLogger(__name__)

def load_preprocessed_data(preproc_path):
    logger.info(f"Pre-processed data found at {preproc_path}. Loading instead of processing anew.")
    try:
        saved_data = torch.load(preproc_path, weights_only=False)
        alphabet = saved_data["alphabet"]
        processed_data = saved_data["raw_data"]
        logger.info("Pre-processed data loaded successfully.")
        return alphabet, processed_data
    except Exception as e:
        logger.warning(f"Could not load pre-processed data: {e}. Proceeding with re-processing.")
        exit()


def save_selfies_alphabet(config, alphabet, *, include_special_tokens=True):
    """
    Write one token per line to `config.directory_paths.selfies_alphabet`
    (creates the parent folder if needed).  Skips writing when the file
    is already present.

    The resulting text file can be fed straight into HuggingFace's
    WordLevelTrainer / BpeTrainer as `vocab_file=…`.
    """
    path = config.directory_paths.selfies_alphabet
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if os.path.exists(path):
        logger.info(f"{path} already exists skipping alphabet dump.")
        return

    # Add the usual special tokens up front, if desired
    specials = ["[PAD]", "[BOS]", "[EOS]", "[UNK]"] if include_special_tokens else []
    vocab_lines = specials + sorted(alphabet)

    logger.info(f"Saving SELFIES alphabet ({len(vocab_lines)} tokens) → {path}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(vocab_lines))
    logger.info("Alphabet saved.")


def save_selfies_text_file(output_path,  df, whitespace=True):
    """
    Writes each list of tokens in df['selfies'] as a space-delimited line
    to config.directory_paths.selfies_txt, skipping if file already exists.
    """
    if os.path.exists(output_path):
        logger.info(f"{output_path} already exists. Skipping creation.")
        return
    
    logger.info(f"Writing SELFIES data to {output_path}")
    if whitespace:
        with open(output_path, "w", encoding="utf-8") as f:
            for token_list in df["selfies"]:
                f.write(" ".join(token_list) + "\n")
    else:
        with open(output_path, "w", encoding="utf-8") as f:
            for token_list in df["selfies"]:
                f.write("".join(token_list) + "\n")
    logger.info(f"SELFIES text file saved to {output_path}")


def preprocess_selfies_data(config, raw_data=None):
    """
    This function preprocesses raw data by 
    splitting the selfies data and returning the alphabet of the selfies data.
    If a preprocessed file is found at config.directory_paths.pre_processed_data,
    it will load that file instead of reprocessing.
    """
    preproc_path = config.directory_paths.pre_processed_data

    if os.path.exists(preproc_path) and config.checkpointing.fresh_data == False:
        return load_preprocessed_data(preproc_path)
    else: 
        logger.info(f"Starting fresh preprocessing at: {preproc_path}. Proceeding with processing.")
        
    if 'selfies' not in raw_data.columns:
        logger.warning("'selfies' column not found in raw_data. Cannot create vocabulary.")
        exit()
    else:
        logger.info("'selfies' column found. Creating vocabulary from SELFIES data...")
        
        alphabet = selfies.get_alphabet_from_selfies(raw_data['selfies'])
        
        # splits and checks the SELFIES length.
        def tokenize_if_valid(s):
            tokenized = list(selfies.split_selfies(s))
            return tokenized if len(tokenized) <= config.permitted_selfies_length else None

        raw_data['tokenized_selfies'] = raw_data['selfies'].apply(tokenize_if_valid)
        filtered_data = raw_data[raw_data['tokenized_selfies'].notnull()].copy()

        # compute max_lengt (might remove later)
        max_selfies_length = filtered_data['tokenized_selfies'].apply(len).max()
        logger.info(f"max length = {max_selfies_length}")

        # Replace the original selfies column with the tokenized version
        #filtered_data['selfies'] = filtered_data['tokenized_selfies']
        #filtered_data = filtered_data.drop(columns=['tokenized_selfies'])
        
        try:
            torch.save({"alphabet": alphabet, "raw_data": filtered_data}, preproc_path)
            logger.info(f"Pre-processed data saved to {preproc_path}.")
        except Exception as e:
            logger.warning(f"Could not save pre-processed data: {e}.")
        save_selfies_text_file(config.directory_paths.selfies_nospace_txt, filtered_data, whitespace=False)
        save_selfies_text_file(config.directory_paths.selfies_whitespace_txt, filtered_data, whitespace=True)
        save_selfies_alphabet(config, alphabet)
        print(f"selfies column: {filtered_data['selfies']}")
        print(f"alphabet: {alphabet}")
        return alphabet, filtered_data
