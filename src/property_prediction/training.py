# train.py

import logging
import os

import hydra
import lightning as L
import pandas as pd
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

from .gcnn import MolPropModule
from .graph_utils import prepare_graph_dataset, split_and_load


def _setup_cuda():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if (
        torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7
    ):  # Ampere GPUs (like A100) are capability 8.0+
        try:
            # Use 'high' for TF32 on Ampere/Hopper, 'medium' might be faster but uses BFloat16 too
            torch.set_float32_matmul_precision("high")
            logger.info(
                "Success torch.set_float32_matmul_precision('high') for Tensor Core utilization."
            )
        except Exception as e:
            logger.warning(f"Could not set float32 matmul precision: {e}")


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


def train_property_predictor(prop_pred_config: DictConfig):
    """
    Trains the Lightning model for property prediction.

    Args:
        config: A DictConfig object containing all configuration parameters.
    """
    import wandb

    wandb.login()
    _setup_cuda()
    wandb_logger, callbacks = setup_training_logging(prop_pred_config)

    logger.info("Loading and preparing dataset...")
    df = pd.read_csv(prop_pred_config.data.path)

    logger.info(f"Dataset contains {len(df)} molecules. Normalizing properties...")
    data_list, props_mean, props_std = prepare_graph_dataset(
        df,
        prop_columns=prop_pred_config.data.prop_columns,
        normalize=True,
        stats_path=prop_pred_config.inference.normalization_stats_file,
    )

    if data_list:
        prop_pred_config.model.node_dim = data_list[0].num_node_features
        prop_pred_config.model.edge_dim = data_list[0].num_edge_features
        prop_pred_config.model.out_dim = len(prop_pred_config.data.prop_columns)
    else:
        raise ValueError("Dataset is empty after processing. Check your SMILES data.")

    logger.info("Creating data loaders...")
    train_loader, val_loader = split_and_load(
        data_list,
        batch_size=prop_pred_config.training.batch_size,
        val_ratio=prop_pred_config.training.val_ratio,
        num_workers=prop_pred_config.training.num_workers,
    )

    logger.info("Initializing model...")
    model = MolPropModule(prop_pred_config.model)
    model.configure_model()

    trainer = L.Trainer(
        max_epochs=prop_pred_config.training.max_epochs,
        accelerator=prop_pred_config.training.accelerator,
        devices=prop_pred_config.training.devices,
        callbacks=callbacks,
        logger=wandb_logger,
    )

    print("Starting training...")
    trainer.fit(model, train_loader, val_loader)
    print("Training finished.")
