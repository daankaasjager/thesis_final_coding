from functools import cache
import torch
import os
import logging
import datasets
from transformers import DataCollatorWithPadding

logger = logging.getLogger(__name__)

def check_gpu_compatibility(config):
    logger.info("Checking GPU compatibility")
    num_gpus = config.trainer.devices # CHANGE LATER to torch.device.count() or somth
    logger.info(f"Number of GPUs: {num_gpus}")
    """assert (config.loader.global_batch_size ==
            config.loader.batch_size * config.trainer.num_nodes
            * num_gpus * config.trainer.accumulate_grad_batches)"""
    if config.loader.global_batch_size % (num_gpus * config.trainer.accumulate_grad_batches) != 0:
        raise ValueError(
            f'Train Batch Size {config.loader.global_batch_size}'
            f' not divisible by {num_gpus * config.trainer.accumulate_grad_batches} gpus with accumulation.'
        )
    if config.loader.eval_global_batch_size % num_gpus != 0:
        raise ValueError(
            f'Eval Batch Size for {config.loader.eval_global_batch_size}'
            f' not divisible by {num_gpus}.'
        )

def create_train_val_dataloaders(config, tokenized_selfies_data, tokenizer):
    """
    Creates a Hugging Face Dataset from tokenized_selfies_data,
    splits into train and validation sets, then returns PyTorch DataLoaders.
    """
    dataset = datasets.Dataset.from_dict(tokenized_selfies_data)
    # Potentially also include the column "conditioning". First requires adding that to the tokenized selfies data
    
    if config.train_test_split.train < 1.0:
        split_dataset = dataset.train_test_split(
            test_size=1 - config.train_test_split.train,
            seed=getattr(config, 'seed', 42),
            shuffle=True
        )
        train_dataset = split_dataset['train']
        val_dataset = split_dataset['test']
    else:
        logger.warning("train_test_split.train=1.0, no separate validation dataset.")
        train_dataset = dataset
        val_dataset = None

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.loader.batch_size,
        num_workers=config.loader.num_workers,
        pin_memory=config.loader.pin_memory,
        shuffle=not config.data.streaming,
        collate_fn=data_collator,
        persistent_workers=True
    )

    if val_dataset is not None and len(val_dataset) > 0:
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=config.loader.eval_batch_size,
            num_workers=config.loader.num_workers,
            pin_memory=config.loader.pin_memory,
            shuffle=False,
            collate_fn=data_collator
        )
    else:
        val_loader = None

    # The torch generator controls the randomness of the DataLoader. If a specific seed is passed,
    # it guarantees that the order is reproducible across runs. For validation data, shuffling isn't
    # strictly necessary, but if we do shuffle, we might want to use a seed.
    logger.info("Train and validation DataLoaders created succesfully.")
    return train_loader, val_loader

def get_dataloaders(config, tokenized_selfies_data, tokenizer):
    """
    This function creates the train and validation dataloaders.
    It also checks GPU compatibility and sets up the DataLoader.
    It is only called during training.
    """
    check_gpu_compatibility(config)
    train_set, valid_set = create_train_val_dataloaders(config, tokenized_selfies_data, tokenizer)
    return train_set, valid_set
