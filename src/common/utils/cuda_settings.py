import logging
import os

import torch

logger = logging.getLogger(__name__)


def setup_cuda() -> None:
    """
    Configure CUDA settings for optimal performance.
    """
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7:
        try:
            torch.set_float32_matmul_precision("high")
            logger.info("Enabled high-precision float32 matmul for Tensor Cores.")
        except Exception as e:
            logger.warning(f"Could not set float32 matmul precision: {e}")
