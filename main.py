from venv import logger
import hydra
from omegaconf import OmegaConf, DictConfig
import lightning as L
from preprocess_data import preprocess_selfies_data
import logging
import torch
import os
from tokenizer import get_tokenizer, tokenize_selfies_vocab
from utils.setup import setup_training_logging, resolve_paths, print_batch
from utils.create_datasets import get_dataloaders
from utils.csv_data_reader import fast_csv_to_df_reader
from utils.logging_config import configure_logging



"""Most of the code of this project is based on the original implementation
of Masked Diffusion Language Models (MDLM). 
doi: 10.48550/arXiv.2406.07524
repository: https://github.com/kuleshov-group/mdlm/tree/bbc8fb61077a3ca38eab2423d0f81a5484754f51"""


OmegaConf.register_new_resolver(
  'cwd', os.getcwd)
OmegaConf.register_new_resolver(
  'device_count', torch.cuda.device_count)
OmegaConf.register_new_resolver(
  'eval', eval)
OmegaConf.register_new_resolver(
  'div_up', lambda x, y: (x + y - 1) // y)

configure_logging()
logger = logging.getLogger(__name__)

def _train(config, tokenizer, data):
    logger.info('Starting Training.')
    wandb_logger, ckpt_path, callbacks = setup_training_logging(config)
    tokenized_data, vocab_size = tokenize_selfies_vocab(tokenizer, config, data)
    train_dataloader, val_dataloader = get_dataloaders(config, tokenized_data, tokenizer)
    print_batch(train_dataloader, val_dataloader, tokenizer)
    # TO DO: What is the max length of the training data? Maybe set the batch size
    #  to something in the neighborhood. Take a look at initial training of the diffusion model. tokenizer
    # is all setup. Try to pass the normal tokenizer to the diffusion model. Why does the other code use valid_ds.tokenizer?


@hydra.main(version_base=None, config_path='hydra_configs',
            config_name='config')
def run(config: DictConfig):
    # log config settings
    logger.info(OmegaConf.to_yaml(config))

    L.seed_everything(config.seed, verbose=False)

    config = resolve_paths(config)

    raw_data = fast_csv_to_df_reader(config.directory_paths.raw_data, row_limit=10)

    # Data  goes from "[C][=C][C]" to ['[C]', '[=C]', '[C]'] and obtain alphabet
    selfies_vocab, data = preprocess_selfies_data(raw_data)

    # Passes selfies_vocab in case the tokenizer needs to be trained.
    tokenizer = get_tokenizer(config, selfies_vocab)
    
    if config.mode == 'train':
        _train(config, tokenizer, data)


if __name__ == "__main__":
    print("Program initiated")
    print("PyTorch version:", torch.__version__)
    print("CUDA version (compiled):", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())
    
    run()