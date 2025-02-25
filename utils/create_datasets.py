from functools import cache
import torch
import os
import logging
import datasets

logger = logging.getLogger(__name__)

def check_gpu_compatibility(config):
    num_gpus = torch.cuda.device_count()
    assert (config.loader.global_batch_size == (config.loader.batch_size
                                              * config.trainer.num_nodes
                                              * num_gpus
                                              * config.trainer.accumulate_grad_batches))
    if config.loader.global_batch_size % (
        num_gpus * config.trainer.accumulate_grad_batches) != 0:
        raise ValueError(
        f'Train Batch Size {config.loader.global_batch_size}'
        f'not divisible by {num_gpus * config.trainer.accumulate_grad_batches} gpus with accumulation '
        f'{config.trainer.accumulate_grad_batches}.')
    if config.loader.eval_global_batch_size % num_gpus != 0:
        raise ValueError(
        f'Eval Batch Size for {config.loader.eval_global_batch_size} '
        f'not divisible by {num_gpus}.')

def create_train_val_dataloaders(config, tokenized_selfies_data, pin_memory=True, valid_seed=None):
    # Create a Hugging Face Dataset
    dataset = datasets.Dataset.from_dict(tokenized_selfies_data)
    
    # Potentially also include the column "conditioning". First requires adding that to the tokenized selfies data
    dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask'])
    
    # Split the data
    split_dataset = dataset.train_test_split(test_size=1 - config.train_test_split.train)
    train_dataset = split_dataset['train']
    val_dataset = split_dataset['test']
    
    # Create PyTorch DataLoaders for training and validation.
    train_loader = torch.utils.data.DataLoader(
      train_dataset,
      batch_size=config.loader.batch_size,
      num_workers=config.loader.num_workers,
      pin_memory=config.loader.pin_memory,
      shuffle=not config.data.streaming,
      persistent_workers=True)
    
    """The torch generator controls the randomness of the DataLoader. If a specific seed is passed,
        it guaranetees that the order is reproducible across runs. For validation data, shuffling isn't 
         strictly necessary, but if we do shuffle, we might want to use a seed.  """
    if valid_seed is None:
      shuffle_valid = False
      generator = None
    else:
      shuffle_valid = True
      generator = torch.Generator().manual_seed(valid_seed)
    val_loader = torch.utils.data.DataLoader(
      val_dataset,
      batch_size=config.loader.eval_batch_size,
      num_workers=config.loader.num_workers,
      pin_memory=config.loader.pin_memory,
      shuffle=shuffle_valid,
      generator=generator)
    
    return train_loader, val_loader

def get_dataloaders(config,tokenized_selfies_data, tokenizer):
    check_gpu_compatibility(config)
    train_set, valid_set = create_train_val_dataloaders(config, tokenized_selfies_data)
    return train_set, valid_set