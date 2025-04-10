import torch.cuda
import logging
import os
from src.tokenizer import SelfiesTokenizer
from datetime import datetime
import json
from tqdm import tqdm
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)

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


def generate_samples(config):
    logger.info('Generating samples.')
    if os.path.exists(config.directory_paths.tokenizer):
        logger.info("Tokenizer folder found. Loading...")
        try:
            tokenizer = SelfiesTokenizer.from_pretrained(config.directory_paths.tokenizer)
        except Exception as e:
            logger.error(f"Error loading tokenizer: {e}")
            exit()
    model = _load_from_checkpoint(config=config, tokenizer=tokenizer)
    model.gen_ppl_metric.reset()

    if config.eval.disable_ema:
        logger.info('Disabling EMA.')
        model.ema = None

    stride_length = config.sampling.stride_length
    num_strides = config.sampling.num_strides
    timestamp = datetime.isoformat()
    flat_config = OmegaConf.to_container(config, resolve=True)

    # Temp file to store intermediate samples in case of crash
    temp_path = os.path.join(config.directory_paths.sampled_data, "generated_samples.tmp")
    os.makedirs(config.directory_paths.sampled_data, exist_ok=True)

    sample_count = 0
    with open(temp_path, 'a', encoding='utf-8') as temp_file:
        for _ in tqdm(range(config.sampling.num_sample_batches), desc=f"Sampling batches of size: {config.loader.eval_batch_size}", unit="batch"):
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