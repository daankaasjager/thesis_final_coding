"""
Custom tokenizer for SELFIES sequences, extending HuggingFace's PreTrainedTokenizerFast.
"""

import logging
from typing import Any, List, Union

from transformers import PreTrainedTokenizerFast

logger = logging.getLogger(__name__)


class SelfiesTokenizer(PreTrainedTokenizerFast):
    """
    Tokenizer that decodes SELFIES token IDs into contiguous SELFIES strings.

    Inherits all methods from PreTrainedTokenizerFast, overriding decode to
    remove inter-token spaces.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the SELFIES tokenizer with underlying Fast tokenizer settings."""
        super().__init__(*args, **kwargs)

    def decode(
        self,
        token_ids: Union[List[int], Any],
        skip_special_tokens: bool = True,
        **kwargs: Any
    ) -> str:
        """
        Convert a sequence of token IDs into a SELFIES string without spaces.

        Args:
            token_ids: list or tensor of token ID integers.
            skip_special_tokens: whether to omit special tokens (BOS, EOS, PAD, etc.).
            **kwargs: additional arguments passed to the base decode method.

        Returns:
            A SELFIES string, e.g., "[C][Branch1]" with no spaces between tokens.
        """
        decoded = super().decode(
            token_ids, skip_special_tokens=skip_special_tokens, **kwargs
        )
        return decoded.replace(" ", "")
