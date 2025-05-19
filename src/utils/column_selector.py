import pandas as pd
import logging
from typing import List

logger = logging.getLogger(__name__)

def select_numeric_columns(df: pd.DataFrame, exclude: set) -> List[str]:
    numeric_cols = []
    for c, dtype in df.dtypes.items():
        if c not in exclude and pd.api.types.is_numeric_dtype(dtype) and not c.endswith("_bin"):
             if df[c].notna().any() and pd.to_numeric(df[c], errors='coerce').notna().any():
                numeric_cols.append(c)
             else:
                logger.debug(f"Column '{c}' looks numeric by dtype but contains non-numeric data or only NaNs. Skipping for auto-binning.")

    logger.debug(f"Identified numeric columns for potential binning: {numeric_cols}")
    return numeric_cols