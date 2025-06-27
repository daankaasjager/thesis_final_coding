import json
import logging
import os
from datetime import datetime

import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from .tokenizing import SelfiesTokenizer
from .preprocessing import normalize_scalar_target_properties

logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_tokenizer(config):
    """Loads tokenizer from the path specified in the config."""
    tokenizer_path = config.paths.tokenizer
    logger.info(f"Loading tokenizer from: {tokenizer_path}")
    try:
        return SelfiesTokenizer.from_pretrained(tokenizer_path)
    except Exception as e:
        logger.error(f"Error loading tokenizer: {e}")
        exit()

def _load_from_checkpoint(checkpoint_path: str, tokenizer: SelfiesTokenizer, config):
    """Loads a model from a specific checkpoint path."""
    from src.mdlm import Diffusion  # Local import
    logger.info(f"Loading model from checkpoint: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    model = Diffusion.load_from_checkpoint(
        checkpoint_path, tokenizer=tokenizer, config=config, map_location="cpu", strict=False,
    ).half()
    torch.cuda.empty_cache()
    model.to(device)
    logger.info("Model loaded successfully and moved to device.")
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
            return model.tokenizer.batch_decode(samples, skip_special_tokens=False)


def _init_temp_json(temp_file, timestamp, flat_config):
    """Initializes the temporary JSON file. (Unchanged)"""
    temp_file.write('{\n')
    temp_file.write(f'"timestamp": {json.dumps(timestamp)},\n')
    temp_file.write(f'"config": {json.dumps(flat_config, indent=2)},\n')
    temp_file.write('"samples": [\n')

def _write_temp_sample(temp_file, sample, sample_count, is_last=False):
    """Writes a single sample to the temporary file. (Unchanged)"""
    logger.info(f"Sample {sample_count}: {sample}")
    json_str = json.dumps(sample.strip())
    if not is_last:
        temp_file.write(json_str + ",\n")
    else:
        temp_file.write(json_str + "\n")

def _save_final_output(temp_path, final_path):
    """
    Finalizes the output by closing the JSON structure in the temp file
    and renaming it to the final path. This is an atomic operation on most systems.
    """
    # This is a slightly improved version that avoids re-reading the whole file.
    # It just appends the closing brackets and renames.
    with open(temp_path, "a", encoding="utf-8") as temp_file:
        temp_file.write("]\n}\n")
    
    os.rename(temp_path, final_path)
    logger.info(f"Finalized samples saved to {final_path}")


# --- Main generate_samples Function (Adapted for Hydra Paths) ---

def generate_samples(config):
    """
    Main function to generate samples, using robust I/O and Hydra path management.
    """
    # --- 1. Determine Dynamic Paths using Hydra ---
    output_dir = config.hydra.run.dir
    samples_dir = os.path.join(output_dir, "synthesized_molecules")
    os.makedirs(samples_dir, exist_ok=True)
    
    # The temporary file and final file now live inside the unique run directory
    temp_path = os.path.join(samples_dir, "generated_samples.json.tmp")
    final_output_path = os.path.join(samples_dir, "generated_samples.json")
    
    logger.info(f"Run output directory: {output_dir}")
    logger.info(f"Temporary samples will be written to: {temp_path}")
    logger.info(f"Final samples will be saved as: {final_output_path}")

    # --- 2. Load Model and Tokenizer ---
    if config.mode == 'eval':
        checkpoint_path = config.eval.checkpoint_path
    else:
        checkpoint_path = config.checkpointing.resume_ckpt_path

    tokenizer = _load_tokenizer(config)
    model = _load_from_checkpoint(checkpoint_path, tokenizer, config)
    
    # ... (Your EMA logic)

    # --- 3. Generate Samples using Your Robust I/O Method ---
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    flat_config = OmegaConf.to_container(config, resolve=True)
    sample_count = 0
    
    # This `with` block ensures the file is handled correctly
    try:
        with open(temp_path, "w", encoding="utf-8") as temp_file:
            _init_temp_json(temp_file, timestamp, flat_config)
            
            for b_idx in tqdm(
                range(config.sampling.num_sample_batches),
                desc=f"Sampling batches of size: {config.loader.eval_batch_size}",
            ):
                batch_samples = _sample_batch(model, config)
                for s_idx, sample in enumerate(batch_samples):
                    sample_count += 1
                    is_last = (
                        b_idx == config.sampling.num_sample_batches - 1
                        and s_idx == len(batch_samples) - 1
                    )
                    _write_temp_sample(temp_file, sample, sample_count, is_last=is_last)
        
        # --- 4. Finalize the Output ---
        # This part now runs only if the loop completes successfully.
        # I've modified _save_final_output to be a bit simpler.
        with open(temp_path, "a", encoding="utf-8") as temp_file:
            temp_file.write("]\n}\n")
        
        os.rename(temp_path, final_output_path)
        logger.info(f"Successfully generated {sample_count} samples and saved to {final_output_path}")

    except (Exception, KeyboardInterrupt) as e:
        logger.error(f"An error occurred during sampling: {e}")
        logger.warning(f"Sampling interrupted. Partial results may be available in the temporary file: {temp_path}")
        # The script will exit, but the .tmp file remains for manual recovery.
        raise