import torch
import json
import pandas as pd
from pathlib import Path
import logging
import re
from typing import List, Set

from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from rdkit import Chem
import selfies as sf

from .gcnn import MolPropModule
from .gnn_dataset import _smiles_to_graph
from ..mdlm.analysis.bos_eos_analyzer import bos_eos_analysis

logger = logging.getLogger(__name__)

# A precise pattern for splitting SELFIES strings into tokens
TOKEN_PATTERN = re.compile(r"(\[[^\]]*\])")

# Explicitly define the special tokens that should always be removed before decoding
SPECIAL_TOKENS: Set[str] = {"[BOS]", "[EOS]", "[PAD]", "[UNK]"}


# --- HELPER FUNCTIONS ---

def load_selfies_alphabet(path: str) -> Set[str]:
    """
    Loads the SELFIES alphabet from a file (one token per line) into a set.
    """
    alphabet_path = Path(path)
    if not alphabet_path.exists():
        logger.error(f"Alphabet file not found at {path}. Cannot perform token validation.")
        raise FileNotFoundError(f"Alphabet file not found: {path}")
    try:
        with open(alphabet_path, "r", encoding="utf-8") as f:
            # Read lines, strip whitespace/newlines, and filter out empty lines
            tokens = {line.strip() for line in f if line.strip()}
        logger.info(f"Loaded alphabet with {len(tokens)} tokens from {path}.")
        return tokens
    except Exception as e:
        logger.error(f"Failed to load or parse alphabet from {path}: {e}")
        raise


def load_generated_samples(path: str) -> list[str]:
    """Loads generated SELFIES samples from a JSON file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("samples", [])
    except Exception as e:
        logger.error(f"Failed to load generated samples from {path}: {e}")
        return []

def load_generated_json(path: str) -> dict:
    """Loads the full generated JSON (metadata + samples)."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_samples_with_metadata(original_json: dict, new_samples: list[str], save_path: str):
    """
    Saves samples together with the original metadata under a new path.
    """
    # Update the samples field with the new samples
    original_json["samples"] = new_samples

    # Optionally, update timestamp or add processing information
    import datetime
    original_json["processed_timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(original_json, f, indent=4)



def load_normalization_stats(path: str) -> tuple[list[str], pd.Series, pd.Series] | tuple[None, None, None]:
    """Loads normalization statistics from a JSON file."""
    stats_path = Path(path)
    if not stats_path.exists():
        logger.warning(f"Normalization stats file not found at {path}. Predictions will not be denormalized.")
        return None, None, None
    try:
        with open(stats_path, "r") as f:
            stats = json.load(f)
        prop_names = list(stats.keys())
        mean_dict = {prop: data['mean'] for prop, data in stats.items()}
        std_dict = {prop: data['std'] for prop, data in stats.items()}
        mean_series = pd.Series(mean_dict)[prop_names]
        std_series = pd.Series(std_dict)[prop_names]
        logger.info(f"Loaded normalization stats for {len(prop_names)} properties.")
        return prop_names, mean_series, std_series
    except Exception as e:
        logger.error(f"Could not load or parse normalization stats from {path}: {e}")
        return None, None, None


def clean_generated_data(sample: str, alphabet: Set[str]) -> str:
    """
    Cleans a single SELFIES string by removing special tokens and any
    tokens not present in the provided alphabet.
    """
    # Split the string by our token pattern, keeping the delimiters
    tokens = [tok for tok in TOKEN_PATTERN.split(sample) if tok]

    # Filter tokens: must be in the alphabet and not a special token
    cleaned_tokens = [
        tok
        for tok in tokens
        if tok in alphabet and tok not in SPECIAL_TOKENS
    ]

    return "".join(cleaned_tokens)


def selfies_to_graph(cleaned_selfies_string: str) -> Data | None:
    """Converts a cleaned SELFIES string to a PyTorch Geometric Data object."""
    if not cleaned_selfies_string:
        return None
    try:
        smiles = sf.decoder(cleaned_selfies_string)
        # Assuming props are not needed at this stage, providing a placeholder
        return _smiles_to_graph(smiles, props=pd.Series([0.0]))
    except Exception as e:
        # It's useful to log the problematic SELFIES string
        logger.error(f"Failed to convert SELFIES '{cleaned_selfies_string}' to graph: {e}")
        return None


def predict_properties(config):
    """
    Main function to run property prediction on generated molecules.
    """
    logger.info("--> Starting property prediction...")

    # 1. Load Model, Stats, and Alphabet
    model_path = Path(config.inference.model_path)
    lightning_module = MolPropModule.load_from_checkpoint(model_path)
    model = lightning_module.model
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    prop_columns, props_mean, props_std = load_normalization_stats(config.inference.normalization_stats_file)

    # Load the SELFIES alphabet for token validation
    try:
        valid_alphabet = load_selfies_alphabet(config.inference.selfies_alphabet)
    except (FileNotFoundError, AttributeError) as e:
        logger.error(f"Could not load SELFIES alphabet due to config error or missing file: {e}. Aborting.")
        return

    # 2. Load and Trim SELFIES
    selfies_path = config.inference.sampled_selfies_file
    original_json = load_generated_json(selfies_path)
    selfies_samples = original_json.get("samples", [])

    if not selfies_samples:
        logger.warning("No SELFIES samples found to process.")
        return

    logger.info("--> Step 1: Trimming sequences with BOS/EOS analysis...")
    name = config.experiment.name if hasattr(config.experiment, 'name') else "default"
    trimmed_selfies_list = bos_eos_analysis(
        selfies_samples, name=name, output_path=config.inference.plotting_path
    )

    # 3. Clean SELFIES using the alphabet
    logger.info("--> Step 2: Cleaning special tokens and validating against alphabet...")
    cleaned_selfies_list = [clean_generated_data(s, valid_alphabet) for s in trimmed_selfies_list]

    # 4. Convert to Graphs
    logger.info("--> Step 3: Converting to graph objects...")
    data_list = [selfies_to_graph(s) for s in cleaned_selfies_list]

    # Synchronize all lists to keep only valid entries
    valid_data = [(cleaned_selfies_list[i], data) for i, data in enumerate(data_list) if data is not None]
    if not valid_data:
        logger.error("No valid molecules could be created from the input SELFIES.")
        return
    valid_selfies_samples, valid_data_list = zip(*valid_data)
    logger.info(f"--> Successfully converted {len(valid_data_list)} molecules.")

    # 5. Inference
    loader = DataLoader(list(valid_data_list), batch_size=config.training.batch_size, shuffle=False)
    predictions = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            predictions.append(model(batch).cpu())
    predictions = torch.cat(predictions, dim=0).numpy()

    # 6. Denormalize and Save
    if props_mean is not None and props_std is not None:
        predictions = (predictions * props_std.values) + props_mean.values

    results = []
    for i in range(len(predictions)):
        results.append({
            "selfies": valid_selfies_samples[i],
            "predicted_properties": {
                prop_name: float(predictions[i, j])
                for j, prop_name in enumerate(prop_columns)
            }
        })

    output_path = Path(config.inference.output_path)
    output_path = output_path / f"{config.experiment.name}_predictions.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_samples_with_metadata(original_json, cleaned_selfies_list, save_path=output_path)
    logger.info(f"--> Predictions for {len(results)} molecules saved to {output_path}")