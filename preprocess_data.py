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
        for row_str in raw_data['selfies']:
            tokenized_list_col.append(list(selfies.split_selfies(row_str)))
        raw_data['selfies'] = tokenized_list_col
        return alphabet, raw_data
    