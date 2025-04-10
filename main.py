from venv import logger
import hydra
from omegaconf import OmegaConf, DictConfig
import lightning as L
from src.utils.preprocess_data import preprocess_selfies_data
import logging
import torch
import os
from src.tokenizer import tokenize_selfies_vocab, get_tokenizer
from src.utils.setup import setup_training_logging, resolve_paths, print_batch
from src.utils.create_datasets import get_dataloaders
from src.utils.csv_data_reader import fast_csv_to_df_reader
from src.utils.logging_config import configure_logging
from src.utils.plot_dist import plot_selfies_length_distribution
from tqdm import tqdm
from datetime import datetime
import json
from src.evaluate.plot import analyze_bos_eos_tokens, plot_token_frequency_histogram, plot_molecule_length_histogram
from src.evaluate.metrics import calculate_and_plot_metrics

"""The coding framework of this project is based on the original implementation
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

def _evaluate_samples(config, original_selfies):
    with open(config.directory_paths.sampled_data, 'r') as f:
        data = json.load(f)
        samples = data['samples']
    # Example usage:
    metrics_to_compute = ['sascore']
    generated_results = calculate_and_plot_metrics(config, samples, metrics_to_compute, name="generated", use_moses=True)
    #original_results = calculate_and_plot_metrics(config, original_selfies, metrics_to_compute, name="original")
    
    """plot_token_frequency_histogram(config, samples, "generated_samples")
    plot_token_frequency_histogram(config, original_selfies, "original_selfies")
    plot_molecule_length_histogram(config, samples, "generated_samples")
    plot_molecule_length_histogram(config, original_selfies, "original_selfies")
    analyze_bos_eos_tokens(config, samples, "generated_samples")"""


def _load_from_checkpoint(config, tokenizer):
    import src.diffusion as diffusion
    logger.info("Loading model from checkpoint to CPU.")
    
    model = diffusion.Diffusion.load_from_checkpoint(
        config.eval.checkpoint_path,
        tokenizer=tokenizer,
        config=config,
        map_location="cpu",  # Load to CPU first
        strict=False
    ).half()
    torch.cuda.empty_cache()
    logger.info("Model loaded to CPU. Now moving selective parts to GPU.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Move only encoder and backbone to GPU (if necessary)
    model.to(device)

    return model


def _generate_samples(config):
    logger.info('Generating samples.')
    
    model = _load_from_checkpoint(config=config, tokenizer=tokenizer)
    model.gen_ppl_metric.reset()

    if config.eval.disable_ema:
        logger.info('Disabling EMA.')
        model.ema = None

    stride_length = config.sampling.stride_length
    num_strides = config.sampling.num_strides
    timestamp = datetime.utcnow().isoformat()
    flat_config = OmegaConf.to_container(config, resolve=True)

    # Temp file to store intermediate samples in case of crash
    temp_path = os.path.join(config.directory_paths.sampled_data, "generated_samples.tmp")
    os.makedirs(config.directory_paths.sampled_data, exist_ok=True)

    sample_count = 0
    with open(temp_path, 'a', encoding='utf-8') as temp_file:
        for _ in tqdm(range(config.sampling.num_sample_batches), desc=f"Sampling batches of size: {config.mode.loader.eval_batch_size}", unit="batch"):
            with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.bfloat16):
                if config.sampling.semi_ar:
                    _, intermediate_samples, _ = model.restore_model_and_semi_ar_sample(
                        stride_length=stride_length,
                        num_strides=num_strides,
                        dt=1 / config.sampling.steps
                    )
                    batch_text_samples = intermediate_samples[-1]
                else:
                    samples = model.restore_model_and_sample(num_steps=config.sampling.steps)
                    batch_text_samples = model.tokenizer.batch_decode(samples)

            for sample in batch_text_samples:
                sample = sample.strip()
                print(f"Sample {sample_count + 1}: {sample}")
                temp_file.write(json.dumps(sample) + '\n')
                sample_count += 1

    # Now flush everything into the final JSON
    final_path = os.path.join(config.directory_paths.sampled_data, "generated_samples.json")
    with open(temp_path, 'r', encoding='utf-8') as temp_file:
        samples = [json.loads(line.strip()) for line in temp_file]

    sample_record = {
        "timestamp": timestamp,
        "config": flat_config,
        "samples": samples
    }

    with open(final_path, 'w', encoding='utf-8') as f:
        json.dump(sample_record, f, indent=2)

    os.remove(temp_path)
    logger.info(f"Saved {sample_count} samples to {final_path} with metadata.")

def run_model(config, tokenizer, train_dataloader, val_dataloader, ckpt_path, callbacks, wandb_logger):
    import src.diffusion as diffusion
    model = diffusion.Diffusion(config, tokenizer=tokenizer)
    # print_batch(train_dataloader, val_dataloader, tokenizer) # takes a a long time so only run if necessary.
    trainer = hydra.utils.instantiate(
      config.mode.trainer,
      default_root_dir=os.getcwd(),
      callbacks=callbacks,
      strategy=config.mode.trainer.strategy,
      logger=wandb_logger
    )
    trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=ckpt_path)

def resume_training_from_ckpt(config):
    logger.info(f"Resuming training from checkpoint: {config.mode.checkpointing.resume_ckpt_path}")
    # This just loads the preprocessed data if it can find the path
    selfies_vocab, data = preprocess_selfies_data(config)
    # save selfies vocab somewhere and load that
    tokenizer = get_tokenizer(config, selfies_vocab)

def train_model_from_scratch(config, callbacks, wandb_logger):
    if config.mode.checkpointing.fresh_data== True:
        logger.info("Training model from scratch. Data will be reprocessed.")
        # read in the raw data
        raw_data = fast_csv_to_df_reader(config.directory_paths.raw_data, row_limit=config.mode.row_limit)
        selfies_vocab, data = preprocess_selfies_data(config, raw_data)
    else:
        logger.info("Training model from scratch. Tokenized data will be loaded.")
        # This just loads the preprocessed data if it can find the path
        selfies_vocab, data = preprocess_selfies_data(config)

    ckpt_path = None

    if config.plot_dist:
      plot_selfies_length_distribution(data)
    
    # Passes selfies_vocab in case the tokenizer needs to be trained.
    tokenizer = get_tokenizer(config, selfies_vocab)
    tokenized_data, vocab_size = tokenize_selfies_vocab(tokenizer, config, data)
    train_dataloader, val_dataloader = get_dataloaders(config, tokenized_data, tokenizer)
    run_model(config, tokenizer, train_dataloader, val_dataloader, ckpt_path, callbacks, wandb_logger)
    # TO DO: What is the max length of the training data? Maybe set the batch size
    #  to something in the neighborhood. Take a look at initial training of the diffusion model. tokenizer
    # is all setup. Try to pass the normal tokenizer to the diffusion model. Why does the other code use valid_ds.tokenizer?

@hydra.main(version_base=None, config_path='configs',
            config_name='config')
def run(config: DictConfig):
    """Main function to run the script.
    Args:
        config (DictConfig): Configuration object.
    """

    """sets up seeding, logging, and the config file"""
    rank = int(os.environ.get("LOCAL_RANK", 0))
    if rank == 0:
      logger.info(OmegaConf.to_yaml(config))
    L.seed_everything(config.seed, verbose=False)
    config = resolve_paths(config)
    if config.mode.name == 'train':
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        wandb_logger, callbacks = setup_training_logging(config)
        if config.mode.checkpointing.resume_from_ckpt:
            resume_training_from_ckpt(config, callbacks, wandb_logger)
        else:
            train_model_from_scratch(config, callbacks, wandb_logger)
    elif config.mode.name == "sample":
        _generate_samples(config)
    elif config.mode.name == "evaluate_samples":
        _evaluate_samples(config)


if __name__ == "__main__":
    print("Program initiated")
    print("PyTorch version:", torch.__version__)
    print("CUDA version (compiled):", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())
    run()

    # TO DO: Check even fast GPUs, check configuration for different loader sizes, do more epochs and faster training.
    # Perhaps even try different models and then just run it on a job