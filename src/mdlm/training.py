"""
Training entry point for the MDLM diffusion model.

This module sets up CUDA, data, tokenization, and launches model training via Hydra.
"""

import logging
import os

import hydra
import torch
import wandb

from .preprocessing import get_dataloaders, prepare_data_for_training
from .tokenizing import get_tokenizer, tokenize_selfies_vocab
from src.common.utils import setup_cuda, setup_training_logging

logger = logging.getLogger(__name__)


def _run_model(
    config,
    tokenizer,
    train_dataloader,
    val_dataloader,
    callbacks,
    wandb_logger,
    ckpt_path: str | None = None,
) -> None:
    """
    Instantiate and train the diffusion model.

    Args:
        config: Hydra configuration object for model and training.
        tokenizer: tokenizer instance for SELFIES sequences.
        train_dataloader: DataLoader for the training set.
        val_dataloader: DataLoader for the validation set.
        callbacks: list of Lightning callbacks.
        wandb_logger: Weights & Biases logger.
        ckpt_path: optional path to resume from a checkpoint.
    """
    import src.mdlm.diffusion as diffusion  # noqa: F401

    model = diffusion.Diffusion(config, tokenizer=tokenizer)
    logger.info("Starting training with provided callbacks.")
    trainer = hydra.utils.instantiate(
        config.trainer,
        default_root_dir=os.getcwd(),
        callbacks=callbacks,
        strategy=config.trainer.strategy,
        logger=wandb_logger,
    )
    trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=ckpt_path)


def train(config) -> None:
    """
    Main training routine.

    1. Logs into Weights & Biases.
    2. Sets up CUDA.
    3. Initializes logging and callbacks.
    4. Prepares data and tokenizer.
    5. Builds DataLoaders.
    6. Executes the training loop.

    Args:
        config: Hydra configuration object containing all training settings.
    """
    wandb.login()
    setup_cuda()
    wandb_logger, callbacks = setup_training_logging(config)
    ckpt_path = (
        config.checkpointing.resume_ckpt_path
        if config.checkpointing.resume_from_ckpt
        else None
    )
    logger.info(f"Resuming from checkpoint: {ckpt_path}")
    selfies_vocab, data = prepare_data_for_training(config)
    tokenizer = get_tokenizer(config, data)
    tokenized_data = tokenize_selfies_vocab(config, tokenizer, data)
    train_dataloader, val_dataloader = get_dataloaders(
        config, tokenized_data, tokenizer
    )
    _run_model(
        config,
        tokenizer,
        train_dataloader,
        val_dataloader,
        callbacks,
        wandb_logger,
        ckpt_path,
    )
