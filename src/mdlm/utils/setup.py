import logging
from pathlib import Path

import hydra
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


def resolve_paths(config: DictConfig) -> DictConfig:
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


def setup_training_logging(config) -> tuple:
    """Sets up wandb logging for training. Also checks for any checkpoints to resume from and implements callbacks"""
    wandb_logger = None
    if config.get("wandb", None) is not None:
        wandb_logger = WandbLogger(config=OmegaConf.to_object(config), **config.wandb)

    """ Defines Lighntning callbacks in YAML config file. 
    Can be stuff like early_stopping, lr_monitor, model saving"""
    callbacks = []
    if "callbacks" in config:
        for _, callback in config.callbacks.items():
            callbacks.append(hydra.utils.instantiate(callback))
    return wandb_logger, callbacks
