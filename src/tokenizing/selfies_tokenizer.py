from venv import logger
from transformers import PreTrainedTokenizerFast
import logging

logger  = logging.getLogger(__name__)

class SelfiesTokenizer(PreTrainedTokenizerFast):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def decode(self, token_ids, skip_special_tokens=True, **kwargs):
        """
        Decodes token IDS into a string without spaces. E.g., "[C][Branch1]"
        """
        print(token_ids)
        decoded = super().decode(token_ids, skip_special_tokens=skip_special_tokens, **kwargs)
        return decoded.replace(" ", "")
