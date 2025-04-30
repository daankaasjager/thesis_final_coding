from transformers import PreTrainedTokenizerFast

class SelfiesTokenizer(PreTrainedTokenizerFast):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def decode(
        self,
        token_ids,
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = False, # Added default argument to match base class
        **kwargs,
    ) -> str:
        """
        Decodes a list of token IDs back into a SELFIES string.

        Args:
            token_ids: List of token IDs to decode.
            skip_special_tokens: Whether to remove special tokens (like BOS, EOS, PAD)
                                 from the output string. Defaults to True.
            clean_up_tokenization_spaces: Whether to clean up spaces before/after tokens.
                                          Defaults to False for SELFIES to keep tokens like '[C]' intact.
            **kwargs: Additional decoding arguments passed to the underlying tokenizer.

        Returns:
            The decoded SELFIES string.
        """
        tokens = self.convert_ids_to_tokens(token_ids, skip_special_tokens=False)
        if skip_special_tokens:
            special_tokens_set = set(self.all_special_tokens)
            tokens = [tok for tok in tokens if tok not in special_tokens_set]
        text = "".join(tokens)
        return text