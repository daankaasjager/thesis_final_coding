import logging
from typing import List

import pandas as pd

logger = logging.getLogger(__name__)


def select_numeric_columns(df: pd.DataFrame, exclude: set) -> List[str]:
    """
    Returns columns that can be coerced to numeric (excluding specified columns).
    """
    numeric_cols = []
    for col in df.columns:
        if col in exclude:
            continue
        coerced = pd.to_numeric(df[col], errors="coerce")
        if coerced.notna().any():
            numeric_cols.append(col)
    return numeric_cols
