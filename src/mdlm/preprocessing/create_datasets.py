import logging

import datasets
import torch
from transformers import DataCollatorWithPadding

logger = logging.getLogger(__name__)


def _check_gpu_compatibility(config) -> None:
    """
    Check if the batch sizes are consistent with the configured number of GPUs.
    Do NOT use all visible GPUs, only what Lightning is instructed to use.
    """
    logger.info("Checking GPU compatibility")
    requested_gpus = config.trainer.devices
    accumulate = config.trainer.accumulate_grad_batches
    nodes = config.trainer.num_nodes
    batch_size = config.loader.batch_size
    expected_global = requested_gpus * nodes * accumulate * batch_size
    logger.info(
        f"Configured GPUs: {requested_gpus} | Accumulate: {accumulate} | Nodes: {nodes} | Batch size: {batch_size}"
    )
    logger.info(
        f"Expected global batch size: {expected_global} | Actual: {config.loader.global_batch_size}"
    )
    assert (
        config.loader.global_batch_size == expected_global
    ), f"Mismatch: expected global_batch_size={expected_global}, got {config.loader.global_batch_size}"
    if config.loader.global_batch_size % (requested_gpus * accumulate) != 0:
        raise ValueError(
            f"Train Batch Size {config.loader.global_batch_size} is not divisible by GPUs * accumulation ({requested_gpus} * {accumulate})."
        )
    if config.loader.eval_global_batch_size % requested_gpus != 0:
        raise ValueError(
            f"Eval Batch Size {config.loader.eval_global_batch_size} not divisible by number of GPUs ({requested_gpus})."
        )


class CondPropertyCollator(DataCollatorWithPadding):
    """
    Pads the usual sequence fields and stacks the conditional properties
    into a float tensor of shape (Batch_size, Properties).
    """

    def __call__(self, features):
        padded = super().__call__(features)
        if "cond_props" in features[0]:
            props = [f["cond_props"] for f in features]
            padded["cond_props"] = torch.tensor(props, dtype=torch.float32)
        return padded


def _create_train_val_dataloaders(config, tokenized_selfies_data, tokenizer) -> tuple:
    """
    Creates a Hugging Face Dataset from tokenized_selfies_data,
    splits into train and validation sets, then returns PyTorch DataLoaders.
    """
    if isinstance(tokenized_selfies_data, dict):
        data = datasets.Dataset.from_dict(tokenized_selfies_data)
        logger.info("Converted dictionary to datasets.Dataset")
    elif isinstance(tokenized_selfies_data, datasets.Dataset):
        data = tokenized_selfies_data
        logger.info("Using provided datasets.Dataset")
    else:
        raise TypeError(
            f"Expected dict or datasets.Dataset, got {type(tokenized_selfies_data)}"
        )
    if config.train_test_split.train < 1.0:
        split_dataset = data.train_test_split(
            test_size=1 - config.train_test_split.train, seed=config.seed, shuffle=True
        )
        train_dataset = split_dataset["train"]
        val_dataset = split_dataset["test"]
    else:
        logger.warning("train_test_split.train=1.0, no separate validation dataset.")
        train_dataset = data
        val_dataset = None
    data_collator = CondPropertyCollator(tokenizer=tokenizer, padding="longest")
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.loader.batch_size,
        num_workers=config.loader.num_workers,
        pin_memory=config.loader.pin_memory,
        shuffle=not config.data.streaming,
        collate_fn=data_collator,
        persistent_workers=True,
    )
    if val_dataset is not None and len(val_dataset) > 0:
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=config.loader.eval_batch_size,
            num_workers=config.loader.num_workers,
            pin_memory=config.loader.pin_memory,
            shuffle=False,
            collate_fn=data_collator,
        )
    else:
        val_loader = None
    logger.info("Train and validation DataLoaders created succesfully.")
    return (train_loader, val_loader)


def get_dataloaders(config, tokenized_selfies_data, tokenizer):
    """
    This function creates the train and validation dataloaders.
    It also checks GPU compatibility and sets up the DataLoader.
    It is only called during training.
    """
    _check_gpu_compatibility(config)
    train_set, valid_set = _create_train_val_dataloaders(
        config, tokenized_selfies_data, tokenizer
    )
    return (train_set, valid_set)
