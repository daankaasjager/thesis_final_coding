import logging
import os
import torch
import selfies

logger = logging.getLogger(__name__)

def preprocess_selfies_data(config, raw_data):
    """
    This function preprocesses raw data by 
    splitting the selfies data and returning the alphabet of the selfies data.
    If a preprocessed file is found at config.directory_paths.pre_processed_data,
    it will load that file instead of reprocessing.
    """
    preproc_path = config.directory_paths.pre_processed_data

    if os.path.exists(preproc_path):
        logger.info(f"Pre-processed data found at {preproc_path}. Loading instead of processing anew.")
        try:
            saved_data = torch.load(preproc_path, weights_only=False)
            alphabet = saved_data["alphabet"]
            processed_data = saved_data["raw_data"]
            logger.info("Pre-processed data loaded successfully.")
            return alphabet, processed_data
        except Exception as e:
            logger.warning(f"Could not load pre-processed data: {e}. Proceeding with re-processing.")

    if 'selfies' not in raw_data.columns:
        logger.warning("'selfies' column not found in raw_data. Cannot create vocabulary.")
        exit()
    else:
        logger.info("'selfies' column found. Creating vocabulary from SELFIES data...")
        alphabet = selfies.get_alphabet_from_selfies(raw_data['selfies'])
        tokenized_list_col = []
        max_selfies_length = 0
        for row_str in raw_data['selfies']:
            my_selfies_list = list(selfies.split_selfies(row_str))
            tokenized_list_col.append(my_selfies_list)
            if len(my_selfies_list) > max_selfies_length:
                max_selfies_length = len(my_selfies_list)
        logger.info(f"max length = {max_selfies_length}")
        raw_data['selfies'] = tokenized_list_col

        try:
            torch.save({"alphabet": alphabet, "raw_data": raw_data}, preproc_path)
            logger.info(f"Pre-processed data saved to {preproc_path}.")
        except Exception as e:
            logger.warning(f"Could not save pre-processed data: {e}.")

        return alphabet, raw_data
