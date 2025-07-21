import json

import torch


class LengthSampler:
    def __init__(self, hist_file, max_len=None, device="cpu", dtype=torch.long):
        hist = json.load(open(hist_file))

        # convert keys → int  + keep them sorted
        items = sorted((int(k), int(v)) for k, v in hist.items())
        lengths, counts = zip(*items)

        lengths = torch.tensor(lengths, dtype=dtype, device=device)
        probs = torch.tensor(counts, dtype=torch.float32, device=device)
        probs /= probs.sum()

        if max_len is not None:
            keep = lengths <= max_len
            lengths, probs = lengths[keep], probs[keep]
            probs /= probs.sum()

        self.supp = lengths
        self.dist = torch.distributions.Categorical(probs)

    def sample(self, n: int) -> torch.LongTensor:
        return self.supp[self.dist.sample((n,))]
