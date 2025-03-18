from venv import logger
import hydra
from omegaconf import OmegaConf, DictConfig
import lightning as L
from utils.misc.preprocess_data import preprocess_selfies_data
import logging
import torch
import os
from tokenizer import tokenize_selfies_vocab, get_tokenizer
from utils.misc.setup import setup_training_logging, resolve_paths, print_batch
from utils.misc.create_datasets import get_dataloaders
from utils.misc.csv_data_reader import fast_csv_to_df_reader
from utils.misc.logging_config import configure_logging
import diffusion
from tqdm import tqdm


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

def _load_from_checkpoint(config, tokenizer):
    logger.info("Loading model from checkpoint to CPU.")
    
    model = diffusion.Diffusion.load_from_checkpoint(
        config.eval.checkpoint_path,
        tokenizer=tokenizer,
        config=config,
        map_location="cpu",  # Load to CPU first
        strict=False
    )
    torch.cuda.empty_cache()
    logger.info("Model loaded to CPU. Now moving selective parts to GPU.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Move only encoder and backbone to GPU (if necessary)
    try:
        model.backbone.to(device, non_blocking=True)  
        logger.info("Moved backbone to GPU.")
    except RuntimeError as e:
        logger.error(f"Backbone OOM: {e}")
    try:
        model.noise.to(device, non_blocking=True)  
        logger.info("Moved noise module to GPU.")
    except RuntimeError as e:
        logger.error(f"Noise OOM: {e}")
    return model


def _generate_samples(config, logger, tokenizer):
    logger.info('Generating samples.')
    
    model = _load_from_checkpoint(config=config, tokenizer=tokenizer)

    model.gen_ppl_metric.reset()

    if config.eval.disable_ema:
        logger.info('Disabling EMA.')
        model.ema = None

    stride_length = config.sampling.stride_length
    num_strides = config.sampling.num_strides
    text_samples = []

    # Add tqdm progress bar for sample batches
    for _ in tqdm(range(config.sampling.num_sample_batches), desc="Sampling batches", unit="batch"):
        with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.bfloat16):
            if config.sampling.semi_ar:
                _, intermediate_samples, _ = model.restore_model_and_semi_ar_sample(
                    stride_length=stride_length,
                    num_strides=num_strides,
                    dt=1 / config.sampling.steps
                )
                text_samples = intermediate_samples[-1]
            else:
                samples = model.restore_model_and_sample(num_steps=config.sampling.steps)
                text_samples = model.tokenizer.batch_decode(samples)

    print("\nGenerated Text Samples:")
    for i, sample in enumerate(text_samples):
        print(f"Sample {i+1}: {sample.strip()}")
    return text_samples



def _train(config, tokenizer, data):
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    logger.info('Starting Training.')
    wandb_logger, ckpt_path, callbacks = setup_training_logging(config)
    tokenized_data, vocab_size = tokenize_selfies_vocab(tokenizer, config, data)
    train_dataloader, val_dataloader = get_dataloaders(config, tokenized_data, tokenizer)
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
    
    # TO DO: What is the max length of the training data? Maybe set the batch size
    #  to something in the neighborhood. Take a look at initial training of the diffusion model. tokenizer
    # is all setup. Try to pass the normal tokenizer to the diffusion model. Why does the other code use valid_ds.tokenizer?


@hydra.main(version_base=None, config_path='hydra_configs',
            config_name='config')
def run(config: DictConfig):
    #This only prints the config once to reduce clutter in the logs because it is distributed.
    rank = int(os.environ.get("LOCAL_RANK", 0))
    if rank == 0:
      logger.info(OmegaConf.to_yaml(config))

    L.seed_everything(config.seed, verbose=False)

    config = resolve_paths(config)

    raw_data = fast_csv_to_df_reader(config.directory_paths.raw_data, row_limit=1000)

    # Data  goes from "[C][=C][C]" to ['[C]', '[=C]', '[C]'] and obtain alphabet
    selfies_vocab, data = preprocess_selfies_data(config, raw_data)

    # Passes selfies_vocab in case the tokenizer needs to be trained.
    tokenizer = get_tokenizer(config, selfies_vocab)
    
    if config.mode == 'train':
      _train(config, tokenizer, data)
    elif config.mode == "sample_eval":
      _generate_samples(config, logger, tokenizer)


if __name__ == "__main__":
    print("Program initiated")
    print("PyTorch version:", torch.__version__)
    print("CUDA version (compiled):", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())
    run()

    # TO DO: Check even fast GPUs, check configuration for different loader sizes, do more epochs and faster training.
    # Perhaps even try different models and then just run it on a job