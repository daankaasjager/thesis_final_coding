import logging
import os

import hydra
import torch

from .preprocessing import get_dataloaders, prepare_data_for_training
from .tokenizing import get_tokenizer, tokenize_selfies_vocab
from .utils import setup_training_logging

logger = logging.getLogger(__name__)


def _run_model(
    config,
    tokenizer,
    train_dataloader,
    val_dataloader,
    callbacks,
    wandb_logger,
    ckpt_path=None,
):
    import src.mdlm.diffusion as diffusion

    model = diffusion.Diffusion(config, tokenizer=tokenizer)
    # print_batch(train_dataloader, val_dataloader, tokenizer) # takes a a long time so only run if necessary.
    # Print the resolved Hydra config (optional, for full context)

    # Inspect the callbacks being passed to the function
    logging.info("Callbacks received in _run_model:")

    trainer = hydra.utils.instantiate(
        config.trainer,
        default_root_dir=os.getcwd(),
        callbacks=callbacks,
        strategy=config.trainer.strategy,
        logger=wandb_logger,
    )
    trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=ckpt_path)


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


def train(config):
    import wandb

    wandb.login()

    _setup_cuda()
    wandb_logger, callbacks = setup_training_logging(config)

    ckpt_path = (
        config.checkpointing.resume_ckpt_path
        if config.checkpointing.resume_from_ckpt
        else None
    )
    logger.info(f"Checkpoint path: {ckpt_path}")
    selfies_vocab, data = prepare_data_for_training(config)
    # Passes the selfies data to the tokenizer, so that it can train from scratch if it doesn't already exist
    # and the retrain_tokenizer flag is set to True
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
