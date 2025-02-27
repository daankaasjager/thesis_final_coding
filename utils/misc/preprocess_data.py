import logging
import selfies

logger = logging.getLogger(__name__)

def preprocess_selfies_data(raw_data):
    """ This function preprocesses raw data by 
    splitting the selfies data and returning the alphabet of the selfies data"""
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
        return alphabet, raw_data
    