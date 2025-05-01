import json
import logging
import os
from datetime import datetime

import torch.cuda
from omegaconf import OmegaConf
from tqdm import tqdm

from tokenizing import SelfiesTokenizer

logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_tokenizer(config):
    logger.info("Tokenizer folder found. Loading...")
    try:
        return SelfiesTokenizer.from_pretrained(config.local_paths.tokenizer)
    except Exception as e:
        logger.error(f"Error loading tokenizer: {e}")
        exit()


def _load_from_checkpoint(config, tokenizer):
    from src import Diffusion

    logger.info("Loading model from checkpoint to CPU.")

    model = Diffusion.load_from_checkpoint(
        config.eval.checkpoint_path,
        tokenizer=tokenizer,
        config=config,
        map_location="cpu",
        strict=False,
    ).half()

    torch.cuda.empty_cache()
    logger.info("Model loaded to CPU. Now moving selective parts to GPU.")
    model.to(device)
    return model


def _sample_batch(model, config):
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        if config.sampling.semi_ar:
            _, intermediate_samples, _ = model.restore_model_and_semi_ar_sample(
                stride_length=config.sampling.stride_length,
                num_strides=config.sampling.num_strides,
                dt=1 / config.sampling.steps,
            )
            return intermediate_samples[-1]
        else:
            samples = model.restore_model_and_sample(num_steps=config.sampling.steps)
            return model.tokenizer.batch_decode(samples)


def _write_temp_sample(temp_file, sample, sample_count):
    logger.info(f"Sample {sample_count}: {sample}")
    temp_file.write(json.dumps(sample.strip()) + "\n")


def _save_final_output(temp_path, final_path, flat_config, timestamp):
    with open(temp_path, "r", encoding="utf-8") as temp_file:
        samples = [json.loads(line.strip()) for line in temp_file]

    sample_record = {"timestamp": timestamp, "config": flat_config, "samples": samples}

    with open(final_path, "w", encoding="utf-8") as f:
        json.dump(sample_record, f, indent=2)

    os.remove(temp_path)
    logger.info(f"Saved {len(samples)} samples to {final_path} with metadata.")


def generate_samples(config):
    logger.info("Generating samples.")

    tokenizer = _load_tokenizer(config)
    model = _load_from_checkpoint(config, tokenizer)
    model.gen_ppl_metric.reset()

    if config.eval.disable_ema:
        logger.info("Disabling EMA.")
        model.ema = None

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    flat_config = OmegaConf.to_container(config, resolve=True)
    temp_path = config.local_paths.temp_path
    sample_count = 0

    with open(temp_path, "a", encoding="utf-8") as temp_file:
        for _ in tqdm(
            range(config.sampling.num_sample_batches),
            desc=f"Sampling batches of size: {config.loader.eval_batch_size}",
            unit="batch",
        ):
            batch_samples = _sample_batch(model, config)
            for sample in batch_samples:
                sample_count += 1
                _write_temp_sample(temp_file, sample, sample_count)

    _save_final_output(
        temp_path=temp_path,
        final_path=config.local_paths.sampled_data,
        flat_config=flat_config,
        timestamp=timestamp,
    )
