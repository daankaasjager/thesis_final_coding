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
        filtered_data['selfies'] = filtered_data['tokenized_selfies']
        filtered_data = filtered_data.drop(columns=['tokenized_selfies'])
        
        try:
            torch.save({"alphabet": alphabet, "raw_data": filtered_data}, preproc_path)
            logger.info(f"Pre-processed data saved to {preproc_path}.")
        except Exception as e:
            logger.warning(f"Could not save pre-processed data: {e}.")

        return alphabet, filtered_data
