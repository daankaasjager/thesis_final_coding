import logging

import pandas as pd

logger = logging.getLogger(__name__)


def read_csv(file_path: str, row_limit) -> pd.DataFrame:
    """
    Gets the column names from a large CSV file and then loads the rest of the data.

    Args:
        file_path (str): Path to the CSV file.
        row_limit (int): Maximum number of rows to read excluding column headers(default 10).

    Returns:
        pd.df: Dataframe with the column names and the first 10 rows of data. Exclude the column names from the data.
    """
    # Read only the header to get column names
    data = pd.read_csv(file_path, nrows=1)
    column_names = data.columns

    if row_limit is None:
        data = pd.read_csv(file_path, names=column_names, skiprows=1)
    else:
        data = pd.read_csv(file_path, names=column_names, skiprows=1, nrows=row_limit)

    logger.info(
        f"Data loaded from {file_path}. row limit = {row_limit}. Number of rows read: {len(data)}"
    )
    return data
