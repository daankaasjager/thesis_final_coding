import hydra
from omegaconf import OmegaConf
import lightning as L
from pathlib import Path
from omegaconf import DictConfig

def resolve_paths(config: DictConfig):
    """
    Recursively resolves all paths in a Hydra configuration
    while ignoring non-path fields like metric names.
    """
    def _resolve(obj, key=None):
        # Skip resolving 'monitor' or other non-path fields
        if key in ["monitor", "id", "tags"]:
            return obj  # Do not modify metric names
        
        # If it's a string and looks like a path, resolve it
        if isinstance(obj, str) and ("/" in obj or "\\" in obj):
            return str(Path(obj).resolve())  
        
        # Recursively process dictionaries
        if isinstance(obj, DictConfig) or isinstance(obj, dict):
            return {k: _resolve(v, k) for k, v in obj.items()}
        
        # Process lists of paths
        if isinstance(obj, list):
            return [_resolve(v) for v in obj]
        
        return obj  # Leave everything else unchanged

    return DictConfig(_resolve(config))


def setup_training_logging(config):
    """Sets up wandb logging for training. Also checks for any checkpoints to resume from and implements callbacks"""
    wandb_logger = None
    if config.get('wandb', None) is not None:
        wandb_logger = L.pytorch.loggers.WandbLogger(
        config=OmegaConf.to_object(config),
        ** config.wandb)

    if (config.checkpointing.resume_from_ckpt
        and config.checkpointing.resume_ckpt_path is not None
        ):
        ckpt_path = config.checkpointing.resume_ckpt_path
    else:
        ckpt_path = None

    """ Defines Lighntning callbacks in YAML config file. 
    Can be stuff like early_stopping, lr_monitor, model saving"""
    callbacks = []
    if 'callbacks' in config:
        for _, callback in config.callbacks.items():
            callbacks.append(hydra.utils.instantiate(callback))
    return wandb_logger, ckpt_path, callbacks

@L.pytorch.utilities.rank_zero_only
def print_batch(train_ds, valid_ds, tokenizer, k=8):
  for dl_type, dl in [
    ('train', train_ds), ('valid', valid_ds)]:
    print(f'Printing {dl_type} dataloader batch.')
    batch = next(iter(dl))
    print('Batch input_ids.shape', batch['input_ids'].shape)
    first = batch['input_ids'][0, :k]
    last = batch['input_ids'][0, -k:]
    print(f'First {k} tokens:', tokenizer.decode(first, skip_special_tokens=True))
    print('ids:', first)
    print(f'Last {k} tokens:', tokenizer.decode(last, skip_special_tokens=False))
    print('ids:', last)