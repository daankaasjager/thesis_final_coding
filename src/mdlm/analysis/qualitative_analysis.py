import os
import random
import logging
from pathlib import Path
from typing import List, Dict, Tuple

import pandas as pd
import matplotlib.pyplot as plt
import scienceplots
import selfies as sf
from rdkit import Chem
from rdkit.Chem import Draw

try:
    from .metrics_core import METRIC_REGISTRY
except ImportError:
    from metrics_core import METRIC_REGISTRY

logger = logging.getLogger(__name__)
plt.style.use(['science', 'no-latex'])


def _selfies_to_mol(selfies_string: str) -> Chem.Mol | None:
    """Converts a SELFIES string to an RDKit Mol object."""
    try:
        smiles = sf.decoder(selfies_string)
        mol = Chem.MolFromSmiles(smiles)
        return mol if mol else None
    except Exception:
        return None

def _calculate_metric_on_the_fly(mol: Chem.Mol, metric: str) -> float | None:
    """Calculates a registered metric for a single molecule."""
    if not mol: return None
    if metric in METRIC_REGISTRY:
        try:
            return METRIC_REGISTRY[metric](mol)
        except Exception as e:
            logger.warning(f"Could not calculate metric '{metric}' on the fly: {e}")
    return None

def _find_best_matches(
    samples: List[Dict],
    prop: str,
    num_samples: int,
    conditioning_targets: Dict[str, float]
) -> List[Tuple[Chem.Mol, Dict]]:
    """Finds the N samples with the smallest error against a global target value."""
    target_val = conditioning_targets.get(prop)
    if target_val is None:
        logger.warning(f"Property '{prop}' not found in conditioning targets. Cannot find best matches.")
        return []

    matched_samples = []
    for sample in samples:
        mol = _selfies_to_mol(sample.get("selfies", ""))
        if not mol: continue

        predicted_props = sample.get("predicted_properties", {})
        predicted_val = predicted_props.get(prop)
        if predicted_val is None:
            predicted_val = _calculate_metric_on_the_fly(mol, prop)

        if predicted_val is not None:
            error = abs(float(predicted_val) - float(target_val))
            sample_with_target = sample.copy()
            sample_with_target['conditioning_properties'] = {prop: target_val}
            matched_samples.append((error, sample_with_target, mol))

    matched_samples.sort(key=lambda x: x[0])
    unique_mols = []
    seen_smiles = set()
    for _, sample, mol in matched_samples:
        if len(unique_mols) >= num_samples: break
        smiles = Chem.MolToSmiles(mol)
        if smiles not in seen_smiles:
            unique_mols.append((mol, sample))
            seen_smiles.add(smiles)

    return unique_mols

def generate_qualitative_comparison_grid(
    sample_sources: Dict[str, List[Dict]],
    baseline_model_name: str,
    properties_to_visualize: List[str],
    output_dir: str,
    run_type: str,
    conditioning_targets_path: str,
    num_samples: int = 4
):
    """
    Generates and saves a grid of molecule images for qualitative comparison,
    with color-coded, boxed row titles for maximum clarity.
    """
    logger.info(f"🎨 Starting qualitative analysis for run '{run_type}'.")
    Path(output_dir).mkdir(exist_ok=True)

    try:
        df_targets = pd.read_csv(conditioning_targets_path)
        conditioning_targets = pd.Series(
            df_targets.bottom33_median.values, index=df_targets.properties
        ).to_dict()
        logger.info(f"✅ Successfully loaded conditioning targets from {conditioning_targets_path}")
    except FileNotFoundError:
        logger.error(f"❌ FATAL: Conditioning targets file not found at {conditioning_targets_path}.")
        return

    model_order = ["Original data", baseline_model_name] + sorted(
        [name for name in sample_sources.keys() if name not in ["Original data", baseline_model_name]]
    )
    
    color_map = {
        "Original data": "darkorange",
        baseline_model_name: "royalblue",
    }
    conditioned_color = "#B266FF"

    for prop in properties_to_visualize:
        logger.info(f"--- Generating grid for property: {prop} ---")
        plot_data_map = {}

        for name in model_order:
            if name not in sample_sources or not sample_sources[name]:
                continue
            matches_found = _find_best_matches(sample_sources[name], prop, num_samples, conditioning_targets)
            if matches_found:
                plot_data_map[name] = matches_found
                logger.info(f"✅ Found {len(matches_found)} best matches for '{name}'.")
            else:
                logger.warning(f"❌ No best matches found for '{name}' on '{prop}'.")
        
        models_to_plot = [name for name in model_order if name in plot_data_map]
        if not models_to_plot:
            logger.error(f"No models had valid molecules to plot for '{prop}'. Skipping grid.")
            continue

        num_rows = len(models_to_plot)
        fig, axes = plt.subplots(
            nrows=num_rows, ncols=num_samples,
            figsize=(3.5 * num_samples, 3.2 * num_rows), squeeze=False
        )
        
        target_value = conditioning_targets.get(prop)
        target_str_title = f"{target_value:.2f}" if target_value is not None else "N/A"
        fig.suptitle(
            f"Qualitative Comparison for Property: '{prop}' (Target: {target_str_title})",
            fontsize=18, weight='bold'
        )

        for i, model_name in enumerate(models_to_plot):
            mol_data = plot_data_map[model_name]
            for j in range(num_samples):
                ax = axes[i, j]
                ax.axis('off')
                if j < len(mol_data):
                    mol, sample_info = mol_data[j]
                    img = Draw.MolToImage(mol, size=(300, 300))
                    ax.imshow(img)

                    pred_val = _calculate_metric_on_the_fly(mol, prop)
                    pred_str = f"{pred_val:.2f}" if pred_val is not None else "N/A"
                    title = f"Pred: {pred_str}\nTarget: {target_str_title}"
                    ax.set_title(title, fontsize=11)
        
        fig.canvas.draw()
        for i, model_name in enumerate(models_to_plot):
            color = color_map.get(model_name, conditioned_color)

            ax0 = axes[i, 0]

            ax0.text(
                -1, 0.5,
                f" {model_name} ",
                transform=ax0.transAxes,
                ha='left',
                va='center',
                fontsize=14,
                fontweight='bold',
                color='white',
                bbox=dict(
                    boxstyle='round,pad=0.4',
                    fc=color,
                    ec='black',
                    lw=1
                )
            )

        fig.tight_layout(rect=[0.1, 0, 1, 0.95])
        
        save_path = Path(output_dir) / f"{run_type}_qualitative_comparison_{prop}.png"
        plt.savefig(save_path, dpi=300)
        logger.info(f"📊 Grid saved to {save_path}")
        plt.close(fig)