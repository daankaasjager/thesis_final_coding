import logging
import os

import hydra
import torch

from preprocessing import get_dataloaders, preprocess_data, read_csv
from tokenizing import get_tokenizer, tokenize_selfies_vocab
from utils import setup_training_logging

logger = logging.getLogger(__name__)


def run_model(
    config,
    tokenizer,
    train_dataloader,
    val_dataloader,
    callbacks,
    wandb_logger,
    ckpt_path=None,
):
    import src.diffusion as diffusion

    model = diffusion.Diffusion(config, tokenizer=tokenizer)
    # print_batch(train_dataloader, val_dataloader, tokenizer) # takes a a long time so only run if necessary.
    trainer = hydra.utils.instantiate(
        config.trainer,
        default_root_dir=os.getcwd(),
        callbacks=callbacks,
        strategy=config.trainer.strategy,
        logger=wandb_logger,
    )
    trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=ckpt_path)


def resume_training_from_ckpt(config):
    logger.info(
        f"Resuming training from checkpoint: {config.checkpointing.resume_ckpt_path}"
    )
    # This just loads the preprocessed data if it can find the path
    selfies_vocab, data = preprocess_data(config)
    # save selfies vocab somewhere and load that
    tokenizer = get_tokenizer(config)
    tokenized_data = tokenize_selfies_vocab(config, tokenizer)
    train_dataloader, val_dataloader = get_dataloaders(
        config, tokenized_data, tokenizer
    )
    return (
        tokenizer,
        train_dataloader,
        val_dataloader,
        config.checkpointing.resume_ckpt_path,
    )


def train_model_from_scratch(config) -> tuple:
    if config.checkpointing.fresh_data:
        logger.info("Training model from scratch. Data will be reprocessed.")
        # read in the raw data
        raw_data = read_csv(
            config.local_paths.augmented_data, row_limit=config.row_limit
        )
        selfies_vocab, data = preprocess_data(config, raw_data)
    else:
        logger.info("Training model from scratch. Tokenized data will be loaded.")
        # This just loads the preprocessed data if it can find the path
        selfies_vocab, data = preprocess_data(config)
        print("selfies_vocab: ", selfies_vocab)
        print("data: ", data)

    ckpt_path = None

    # Passes the selfies data to the tokenizer, so that it can train from scratch if it doesn't already exist
    tokenizer = get_tokenizer(config, data)
    tokenized_data = tokenize_selfies_vocab(config, tokenizer, data)
    train_dataloader, val_dataloader = get_dataloaders(
        config, tokenized_data, tokenizer
    )
    return tokenizer, train_dataloader, val_dataloader, ckpt_path


def train(config):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    wandb_logger, callbacks = setup_training_logging(config)
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
    if config.checkpointing.resume_from_ckpt:
        tokenizer, train_dataloader, val_dataloader, ckpt_path = (
            resume_training_from_ckpt(config)
        )
    else:
        tokenizer, train_dataloader, val_dataloader, ckpt_path = (
            train_model_from_scratch(config)
        )
    run_model(
        config,
        tokenizer,
        train_dataloader,
        val_dataloader,
        callbacks,
        wandb_logger,
        ckpt_path,
    )
