"""
Sampler that draws sequence lengths from an empirical histogram distribution.

The histogram is loaded from a JSON file mapping length strings to counts.
Optionally truncates lengths above a maximum and renormalizes the distribution.
"""

import json
import logging
from pathlib import Path
from typing import Optional, Union

import torch

logger = logging.getLogger(__name__)


class LengthSampler:
    """
    Empirical length sampler using a Categorical distribution.

    Args:
        hist_file: path to JSON file with {"length": count, ...}.
        max_len: optional maximum sequence length to include.
        device: torch device (e.g., "cpu" or "cuda").
        dtype: integer dtype for sampled lengths (default: torch.long).
    """

    def __init__(
        self,
        hist_file: Union[str, Path],
        max_len: Optional[int] = None,
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.long,
    ) -> None:
        # Load and normalize histogram
        data = json.loads(Path(hist_file).read_text(encoding="utf-8"))
        items = sorted((int(length), int(count)) for length, count in data.items())
        lengths, counts = zip(*items)

        lengths_tensor = torch.tensor(lengths, dtype=dtype, device=device)
        probs = torch.tensor(counts, dtype=torch.float32, device=device)
        probs /= probs.sum()

        if max_len is not None:
            mask = lengths_tensor <= max_len
            lengths_tensor = lengths_tensor[mask]
            probs = probs[mask]
            probs /= probs.sum()

        self.supp = lengths_tensor
        self.dist = torch.distributions.Categorical(probs)

        logger.info(
            "Initialized LengthSampler with support sizes %d and device %s",
            self.supp.numel(),
            device,
        )

    def sample(self, n: int) -> torch.LongTensor:
        """
        Draw samples of sequence lengths.

        Args:
            n: number of lengths to sample.

        Returns:
            Tensor of shape (n,) containing sampled lengths.
        """
        samples = self.dist.sample((n,))
        return self.supp[samples]
