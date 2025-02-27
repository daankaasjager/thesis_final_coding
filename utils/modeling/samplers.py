from typing import Iterator
import math
import torch
from torch.utils.data import RandomSampler, DistributedSampler

# Adapted from https://github.com/Lightning-AI/lightning/blob/2845e7565dbe6b765ae32870e7d2bc456529c30a/tests/tests_pytorch/utilities/test_auto_restart.py#L1397

class RandomFaultTolerantSampler(RandomSampler):
    """
    A random sampler that ensures fault tolerance by keeping track of the sampling state.
    If a training job is interrupted and restarted, it resumes sampling from where it left off.
    """
    def __init__(self, *args, generator=None, **kwargs):
        """
        Initializes the sampler with a random seed for reproducibility.
        If a generator is not provided, a new random seed is generated.
        """
        if generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator().manual_seed(seed)
        super().__init__(*args, generator=generator, **kwargs)
        self.counter = 0  # Tracks the number of sampled elements
        self.restarting = False  # Flag indicating if the sampler is resuming from a saved state

    def state_dict(self):
        """
        Saves the current state of the sampler, including the random generator state and the sampling counter.
        """
        return {"random_state": self.state, "counter": self.counter}

    def load_state_dict(self, state_dict):
        """
        Restores the sampler's state from a previously saved checkpoint.
        """
        self.generator.set_state(state_dict.get("random_state"))
        self.counter = state_dict["counter"]
        self.restarting = True  # Indicate that we are restarting and need to adjust the sample sequence

    def __iter__(self) -> Iterator[int]:
        """
        Iterates over the dataset in a randomly permuted order.
        If resuming from a checkpoint, it continues from the last sampled index.
        """
        n = len(self.data_source)
        self.state = self.generator.get_state()
        indices = torch.randperm(n, generator=self.generator).tolist()
        
        if self.restarting:
            indices = indices[self.counter:]
            self.restarting = False
        else:
            self.counter = 0

        for index in indices:
            self.counter += 1
            yield index
        
        self.counter = 0


class FaultTolerantDistributedSampler(DistributedSampler):
    """
    A distributed sampler that supports fault tolerance by tracking sampling progress.
    Ensures consistent sample distribution across multiple GPUs/nodes.
    """
    def __init__(self, *args, **kwargs):
        """
        Initializes the distributed sampler.
        Keeps track of the sampling progress to allow resumption after interruptions.
        """
        super().__init__(*args, **kwargs)
        self.counter = 0  # Tracks the number of sampled elements
        self.restarting = False  # Flag indicating if the sampler is resuming from a saved state

    def state_dict(self):
        """
        Saves the current state of the sampler, including the epoch and the sampling counter.
        """
        return {"epoch": self.epoch, "counter": self.counter}

    def load_state_dict(self, state_dict):
        """
        Restores the sampler's state from a previously saved checkpoint.
        """
        self.epoch = state_dict["epoch"]
        self.counter = state_dict["counter"]
        self.restarting = True  # Indicate that we are restarting and need to adjust the sample sequence

    def __iter__(self):
        """
        Iterates over the dataset in a distributed manner, ensuring equal sample distribution across workers.
        If resuming from a checkpoint, it continues from the last sampled index.
        """
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # Adjust indices to ensure even distribution among distributed workers
        if not self.drop_last:
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # Subsample the dataset for the current rank
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        if self.restarting:
            indices = indices[self.counter:]
            self.restarting = False
        else:
            self.counter = 0

        for index in indices:
            self.counter += 1
            yield index
        
        self.counter = 0
