import json
import logging
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Set

import pandas as pd
import selfies as sf
import torch
from rdkit import Chem
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from ..mdlm.analysis.bos_eos_analyzer import bos_eos_analysis
from .gcnn import MolPropModule
from .graph_utils import get_molecule_graph

logger = logging.getLogger(__name__)
HARD_BOUNDS = {
    "mol_wt": (0.0, None),
    "volume": (0.0, None),
    "dipolemoment": (0.0, None),
    "sa_score": (1.0, 10.5),
    "sterimol_L": (0.0, None),
    "sterimol_B1": (0.0, None),
    "sterimol_B5": (0.0, None),
    "vbur_vbur": (0.0, None),
    "nbo_bd_occ_avg": (0.0, None),
    "nbo_lp_P_occ": (0.0, None),
    "nbo_bds_occ_max": (0.0, None),
}
TOKEN_PATTERN = re.compile("(\\[[^\\]]*\\])")
SPECIAL_TOKENS: Set[str] = {"[BOS]", "[EOS]", "[PAD]", "[UNK]"}


def _load_selfies_alphabet(path: str) -> Set[str]:
    alphabet_path = Path(path)
    if not alphabet_path.exists():
        logger.error(
            f"Alphabet file not found at {path}. Cannot perform token validation."
        )
        raise FileNotFoundError(f"Alphabet file not found: {path}")
    with open(alphabet_path, "r", encoding="utf-8") as f:
        tokens = {line.strip() for line in f if line.strip()}
    logger.info(f"Loaded alphabet with {len(tokens)} tokens from {path}.")
    return tokens


def _load_generated_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_samples_with_metadata(
    original_json: dict, new_samples: list[str], save_path: str
):
    original_json["samples"] = new_samples
    import datetime

    original_json["processed_timestamp"] = datetime.datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S"
    )
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(original_json, f, indent=4)


def _load_scaling_stats(path: str):
    """
    Loads normalization statistics from a JSON file.
    Expects a file with top-level "mean" and "std" keys,
    each containing a dictionary of property-wise stats.
    """
    stats_path = Path(path)
    if not stats_path.exists():
        logger.warning(
            f"Normalization stats file not found at {path}. Predictions will not be denormalized."
        )
        return (None, None, None)
    with open(stats_path, "r") as f:
        stats = json.load(f)
    if "mean" in stats and "std" in stats:
        mean_dict = stats["mean"]
        std_dict = stats["std"]
        prop_names = list(mean_dict.keys())
        mu = pd.Series(mean_dict)[prop_names]
        sigma = pd.Series(std_dict)[prop_names]
        return ("zscore", prop_names, mu, sigma, None)
    if "min_max_vals" in stats:
        available_atoms = stats["available_atoms"]
        mm_dict = stats["min_max_vals"]
        prop_names = list(mm_dict.keys())
        mins = pd.Series({k: v[0] for k, v in mm_dict.items()})[prop_names]
        maxs = pd.Series({k: v[1] for k, v in mm_dict.items()})[prop_names]
        return ("minmax", prop_names, mins, maxs, available_atoms)
    raise KeyError(
        f"File at {path} is malformed. Expected keys 'mean/std' or 'min_max_vals', found: {{list(stats.keys())}}"
    )


def _clean_generated_data(sample: str, alphabet: Set[str]) -> str:
    tokens = [tok for tok in TOKEN_PATTERN.split(sample) if tok]
    cleaned_tokens = [
        tok for tok in tokens if tok in alphabet and tok not in SPECIAL_TOKENS
    ]
    return "".join(cleaned_tokens)


def _selfies_to_graph(
    cleaned_selfies_string: str, available_atoms: list[str]
) -> Data | None:
    """
    Converts a SELFIES string to a detailed graph object using the
    project's specific featurization.
    """
    if not cleaned_selfies_string:
        return None
    try:
        smiles = sf.decoder(cleaned_selfies_string)
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.error(f"RDKit could not process SMILES: {smiles}")
            return None
        graph = get_molecule_graph(mol, available_atoms)
        graph.y = torch.tensor([0.0], dtype=torch.float32)
        return graph
    except Exception as e:
        logger.error(
            f"Failed to convert SELFIES '{cleaned_selfies_string}' to graph: {e}"
        )
        return None


def predict_properties(config):
    logger.info("--> Starting property prediction...")
    cutoff_counters = defaultdict(Counter)
    model_path = Path(config.inference.model_path)
    lightning_module = MolPropModule.load_from_checkpoint(model_path)
    model = lightning_module.model
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    scale_type, prop_columns, a_series, b_series, available_atoms = _load_scaling_stats(
        config.inference.normalization_stats_file
    )
    try:
        valid_alphabet = _load_selfies_alphabet(config.inference.selfies_alphabet)
    except Exception as e:
        logger.error(f"Could not load SELFIES alphabet: {e}")
        return
    selfies_path = (
        config.inference.hist_sampled_selfies_file
        if config.inference.hist
        else config.inference.sampled_selfies_file
    )
    original_json = _load_generated_json(selfies_path)
    selfies_samples = original_json.get("samples", [])
    if not selfies_samples:
        logger.warning("No SELFIES samples found to process.")
        return
    logger.info("--> Step 1: Trimming sequences with BOS/EOS analysis...")
    name = config.experiment.name if hasattr(config.experiment, "name") else "default"
    trimmed_selfies_list = bos_eos_analysis(
        selfies_samples, name=name, output_path=config.inference.plotting_path
    )
    logger.info(
        "--> Step 2: Cleaning special tokens and validating against alphabet..."
    )
    cleaned_selfies_list = [
        _clean_generated_data(s, valid_alphabet) for s in trimmed_selfies_list
    ]
    logger.info("--> Step 3: Converting to graph objects...")
    data_list = [_selfies_to_graph(s, available_atoms) for s in cleaned_selfies_list]
    valid_data = [
        (cleaned_selfies_list[i], data)
        for i, data in enumerate(data_list)
        if data is not None
    ]
    if not valid_data:
        logger.error("No valid molecules could be created from the input SELFIES.")
        return
    valid_selfies_samples, valid_data_list = zip(*valid_data)
    logger.info(f"--> Successfully converted {len(valid_data_list)} molecules.")
    loader = DataLoader(
        list(valid_data_list), batch_size=config.training.batch_size, shuffle=False
    )
    predictions = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            predictions.append(model(batch).cpu())
    predictions = torch.cat(predictions, dim=0).numpy()
    if scale_type == "zscore":
        predictions = predictions * b_series.values + a_series.values
    elif scale_type == "minmax":
        predictions = (
            predictions * (b_series.values - a_series.values) + a_series.values
        )
    total_molecules = len(predictions)
    model_name = config.experiment.name
    results = []
    for i in range(total_molecules):
        props = {
            prop_name: float(predictions[i, j])
            for j, prop_name in enumerate(prop_columns)
        }

        # Collect all out-of-bounds properties for this molecule
        oob_props = []
        for key, (min_val, max_val) in HARD_BOUNDS.items():
            if key not in props:
                continue
            val = props[key]
            if (min_val is not None and val < min_val) or (
                max_val is not None and val > max_val
            ):
                oob_props.append((key, val))

        if oob_props:
            # Count every OOB property (no break -> no skew)
            for key, val in oob_props:
                cutoff_counters[model_name][key] += 1
                logger.warning(
                    f"Property {key} out of bounds ({val}) for molecule {valid_selfies_samples[i]}"
                )
            # Molecule is invalid if any property is OOB
            continue

        # Keep only molecules that are within bounds for all properties
        results.append(
            {"selfies": valid_selfies_samples[i], "predicted_properties": props}
        )

    output_path = (
        Path(config.inference.hist_output_path)
        if config.inference.hist
        else Path(config.inference.output_path)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _save_samples_with_metadata(original_json, results, save_path=output_path)
    logger.info(f"--> Predictions for {len(results)} molecules saved to {output_path}")
    records = []
    for prop in HARD_BOUNDS.keys():
        count = cutoff_counters[model_name].get(prop, 0)
        percent = count / total_molecules * 100 if total_molecules > 0 else 0.0
        records.append(
            {
                "model": model_name,
                "property": prop,
                "cutoff_count": count,
                "total_molecules": total_molecules,
                "percent_filtered": percent,
            }
        )
    df = pd.DataFrame(records)
    cutoff_path = output_path.parent / "hard_bound_cutoff_summary.csv"
    df.to_csv(cutoff_path, index=False)
    logger.info(f"Saved hard bound cutoff summary to {cutoff_path}")
    total_filtered = total_molecules - len(results)
    filtering_summary_path = output_path.parent / "total_filtering_summary.txt"
    with open(filtering_summary_path, "w") as f:
        f.write(f"Total valid molecules before filtering: {total_molecules}\n")
        f.write(f"Total retained after filtering: {len(results)}\n")
        f.write(
            f"Total filtered out: {total_filtered} ({total_filtered / total_molecules * 100:.2f}%)\n"
        )
    logger.info(f"Saved total filtering summary to {filtering_summary_path}")
    logger.info(
        f"Top filtered properties:\n{df.sort_values('percent_filtered', ascending=False).head(5)}"
    )
