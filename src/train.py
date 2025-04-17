
import logging

logger = logging.getLogger(__name__)
import os
import hydra

from src.utils.preprocess_data import preprocess_selfies_data
from src.utils.create_datasets import get_dataloaders
from src.utils.csv_data_reader import fast_csv_to_df_reader
from src.tokenizer import tokenize_selfies_vocab, get_tokenizer
from src.utils.setup import setup_training_logging
from src.utils.plot_dist import plot_selfies_length_distribution



def run_model(config, tokenizer, train_dataloader, val_dataloader, ckpt_path, callbacks, wandb_logger):
    import src.diffusion as diffusion
    model = diffusion.Diffusion(config, tokenizer=tokenizer)
    # print_batch(train_dataloader, val_dataloader, tokenizer) # takes a a long time so only run if necessary.
    trainer = hydra.utils.instantiate(
      config.trainer,
      default_root_dir=os.getcwd(),
      callbacks=callbacks,
      strategy=config.trainer.strategy,
      logger=wandb_logger
    )
    trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=ckpt_path)


def resume_training_from_ckpt(config, callbacks, wandb_logger):
    logger.info(f"Resuming training from checkpoint: {config.pointing.resume_ckpt_path}")
    # This just loads the preprocessed data if it can find the path
    selfies_vocab, data = preprocess_selfies_data(config)
    # save selfies vocab somewhere and load that
    tokenizer = get_tokenizer(config)
    tokenized_data = tokenize_selfies_vocab(config, tokenizer)
    train_dataloader, val_dataloader = get_dataloaders(config, tokenized_data, tokenizer)
    run_model(config, tokenizer, train_dataloader, val_dataloader, config.pointing.resume_ckpt_path, callbacks, wandb_logger)

def train_model_from_scratch(config, callbacks, wandb_logger):
    if config.checkpointing.fresh_data == True:
        logger.info("Training model from scratch. Data will be reprocessed.")
        # read in the raw data
        raw_data = fast_csv_to_df_reader(config.directory_paths.raw_data, row_limit=config.row_limit)
        selfies_vocab, data = preprocess_selfies_data(config, raw_data)
    else:
        logger.info("Training model from scratch. Tokenized data will be loaded.")
        # This just loads the preprocessed data if it can find the path
        selfies_vocab, data = preprocess_selfies_data(config)

    ckpt_path = None

    if config.plot_dist:
      plot_selfies_length_distribution(data)
    
    # Passes the selfies data to the tokenizer, so that it can train from scratch if it doesn't already exist
    tokenizer = get_tokenizer(config)
    tokenized_data = tokenize_selfies_vocab(config, tokenizer, data)
    train_dataloader, val_dataloader = get_dataloaders(config, tokenized_data, tokenizer)
    run_model(config, tokenizer, train_dataloader, val_dataloader, ckpt_path, callbacks, wandb_logger)

def train(config):
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    wandb_logger, callbacks = setup_training_logging(config)
    if config.checkpointing.resume_from_ckpt:
        resume_training_from_ckpt(config, callbacks, wandb_logger)
    else:
        train_model_from_scratch(config, callbacks, wandb_logger)