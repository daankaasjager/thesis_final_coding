"""
Generate molecular SELFIES samples using a trained diffusion model.

This module handles tokenizer and model loading, sampling loops, and robust I/O
to produce JSON output files with generated samples and metadata.
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from .tokenizing import SelfiesTokenizer

logger = logging.getLogger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_tokenizer(config: Any) -> SelfiesTokenizer:
    """Load the SELFIES tokenizer from the configured path."""
    path = config.paths.tokenizer
    logger.info(f"Loading tokenizer from: {path}")
    try:
        return SelfiesTokenizer.from_pretrained(path)
    except Exception as e:
        logger.error(f"Error loading tokenizer: {e}")
        sys.exit(1)


def _load_from_checkpoint(
    checkpoint_path: str, tokenizer: SelfiesTokenizer, config: Any
) -> torch.nn.Module:
    """
    Load and prepare the diffusion model from a checkpoint.

    Args:
        checkpoint_path: filesystem path to the .ckpt file
        tokenizer: loaded SelfiesTokenizer instance
        config: full Hydra configuration object

    Returns:
        The Diffusion model moved to the appropriate device and in half precision.

    Raises:
        FileNotFoundError: if checkpoint_path does not exist
    """
    from src.mdlm import Diffusion

    logger.info(f"Loading model from checkpoint: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    model = Diffusion.load_from_checkpoint(
        checkpoint_path,
        tokenizer=tokenizer,
        config=config,
        map_location="cpu",
        strict=False,
    ).half()
    torch.cuda.empty_cache()
    model.to(DEVICE)
    logger.info("Model loaded and moved to device.")
    return model


def _sample_batch(model: torch.nn.Module, config: Any) -> list[str]:
    """
    Generate a batch of SELFIES samples from the model.

    Uses semi-autoregressive sampling if enabled in config.

    Args:
        model: the diffusion model instance
        config: Hydra config containing sampling parameters

    Returns:
        A list of generated SELFIES strings for the batch.
    """
    with torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16):
        if config.sampling.semi_ar:
            _, intermediate, _ = model.restore_model_and_semi_ar_sample(
                stride_length=config.sampling.stride_length,
                num_strides=config.sampling.num_strides,
                dt=1 / config.sampling.steps,
                target_properties=config.sampling.target_properties,
            )
            return intermediate[-1]
        samples = model.restore_model_and_sample(
            num_steps=config.sampling.steps,
            target_properties=config.sampling.target_properties,
            guidance_scale=config.conditioning.guidance_scale,
        )
        return model.tokenizer.batch_decode(samples, skip_special_tokens=False)


def _init_temp_json(file_obj: Any, timestamp: str, flat_config: dict[str, Any]) -> None:
    """
    Write the JSON header and metadata to the temporary output file.

    Args:
        file_obj: open file object for writing
        timestamp: ISO-formatted run timestamp
        flat_config: flat dict of resolved config values
    """
    file_obj.write("{\n")
    file_obj.write(f'"timestamp": {json.dumps(timestamp)},\n')
    file_obj.write(f'"config": {json.dumps(flat_config, indent=2)},\n')
    file_obj.write('"samples": [\n')


def _write_temp_sample(
    file_obj: Any, sample: str, count: int, is_last: bool = False
) -> None:
    """
    Append a single sample entry to the temporary JSON file.

    Args:
        file_obj: open file object for writing
        sample: SELFIES string to write
        count: sequential sample index
        is_last: whether this is the final sample in the batch
    """
    logger.info(f"Sample {count}: {sample}")
    entry = json.dumps(sample.strip())
    suffix = "\n" if is_last else ",\n"
    file_obj.write(entry + suffix)


def generate_samples(config: Any) -> list[str]:
    """
    Main entry point to generate SELFIES samples and save them to disk.

    Args:
        config: Hydra configuration containing paths, sampling, and model settings

    Returns:
        The list of generated samples (for optional further processing)
    """
    # Determine output paths
    base_dir = config.paths.sampled_data
    if config.conditioning.cfg:
        base_dir = base_dir.replace(
            config.experiment.name, f"{config.conditioning.guidance_scale}_cfg"
        )
    os.makedirs(base_dir, exist_ok=True)

    stem = (
        "hist_generated_samples"
        if config.model.sample_length_mode == "histogram"
        else "generated_samples"
    )
    temp_path = Path(base_dir) / f"{stem}.json.tmp"
    final_path = Path(base_dir) / f"{stem}.json"

    logger.info(f"Output directory: {base_dir}")
    logger.info(f"Writing temporary file: {temp_path}")
    logger.info(f"Final output will be: {final_path}")

    # Load resources
    tokenizer = _load_tokenizer(config)
    model = _load_from_checkpoint(
        config.checkpointing.resume_ckpt_path, tokenizer, config
    )
    if config.eval.disable_ema:
        logger.info("Disabling EMA.")
        model.ema = None

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    flat_cfg = OmegaConf.to_container(config, resolve=True)
    samples: list[str] = []
    count = 0

    try:
        with open(temp_path, "w", encoding="utf-8") as tmp:
            _init_temp_json(tmp, timestamp, flat_cfg)
            for batch_idx in tqdm(
                range(config.sampling.num_sample_batches),
                desc=f"Sampling batches (size={config.loader.eval_batch_size})",
            ):
                for sample in _sample_batch(model, config):
                    count += 1
                    is_last = (
                        batch_idx == config.sampling.num_sample_batches - 1
                        and count % config.loader.eval_batch_size == 0
                    )
                    _write_temp_sample(tmp, sample, count, is_last)
                    samples.append(sample)
        with open(temp_path, "a", encoding="utf-8") as tmp:
            tmp.write("]\n}\n")
        os.replace(temp_path, final_path)
        logger.info(f"Generated {count} samples; saved to {final_path}")
    except (Exception, KeyboardInterrupt) as e:
        logger.error(f"Sampling error: {e}")
        logger.warning(f"Partial results in {temp_path}")
        raise

    return samples
