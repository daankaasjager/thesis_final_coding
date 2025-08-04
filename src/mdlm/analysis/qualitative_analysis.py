import logging
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import selfies as sf
from rdkit import Chem
from rdkit.Chem import Draw

try:
    from .metrics_core import METRIC_REGISTRY
except ImportError:
    from metrics_core import METRIC_REGISTRY

logger = logging.getLogger(__name__)
plt.style.use(["science", "no-latex"])


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
    if not mol:
        return None
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
    conditioning_targets: Dict[str, float],
) -> List[Tuple[Chem.Mol, Dict]]:
    """Finds the N samples with the smallest error against a global target value."""
    target_val = conditioning_targets.get(prop)
    if target_val is None:
        logger.warning(
            f"Property '{prop}' not found in conditioning targets. Cannot find best matches."
        )
        return []

    matched_samples: List[Tuple[float, Dict, Chem.Mol]] = []
    for sample in samples:
        mol = _selfies_to_mol(sample.get("selfies", ""))
        if mol is None:
            continue

        # 1) try reading from dict, 2) fallback to on-the-fly calc
        pred_dict = sample.get("predicted_properties", {})
        predicted_val = pred_dict.get(prop)
        if predicted_val is None:
            predicted_val = _calculate_metric_on_the_fly(mol, prop)

        if predicted_val is None:
            logger.warning(
                f"Sample missing value for '{prop}' and could not calculate on the fly. Skipping."
            )
            continue

        error = abs(float(predicted_val) - float(target_val))
        # annotate both target and actual prediction
        sample_with_vals = sample.copy()
        sample_with_vals["conditioning_properties"] = {prop: target_val}
        sample_with_vals["best_match_val"] = float(predicted_val)

        matched_samples.append((error, sample_with_vals, mol))

    # sort by error and pick unique smiles
    matched_samples.sort(key=lambda x: x[0])
    unique: List[Tuple[Chem.Mol, Dict]] = []
    seen_smiles: set = set()
    for _, sample_info, mol in matched_samples:
        if len(unique) >= num_samples:
            break
        smi = Chem.MolToSmiles(mol)
        if smi in seen_smiles:
            continue
        unique.append((mol, sample_info))
        seen_smiles.add(smi)

    return unique


def generate_qualitative_comparison_grid(
    sample_sources: Dict[str, List[Dict]],
    baseline_model_name: str,
    properties_to_visualize: List[str],
    output_dir: str,
    run_type: str,
    conditioning_targets_path: str,
    num_samples: int = 4,
):
    """
    Generates and saves a grid of molecule images for qualitative comparison,
    splitting into multiple figures if the number of rows exceeds 6.
    """
    logger.info(f"🎨 Starting qualitative analysis for run '{run_type}'.")
    Path(output_dir).mkdir(exist_ok=True)

    try:
        df_targets = pd.read_csv(conditioning_targets_path)
        conditioning_targets = pd.Series(
            df_targets.bottom33_median.values, index=df_targets.properties
        ).to_dict()
        logger.info(
            f"✅ Successfully loaded conditioning targets from {conditioning_targets_path}"
        )
    except FileNotFoundError:
        logger.error(
            f"❌ FATAL: Conditioning targets file not found at {conditioning_targets_path}."
        )
        return

    model_order = ["Original data", baseline_model_name] + sorted(
        [
            name
            for name in sample_sources.keys()
            if name not in ["Original data", baseline_model_name]
        ]
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
            matches_found = _find_best_matches(
                sample_sources[name], prop, num_samples, conditioning_targets
            )
            if matches_found:
                plot_data_map[name] = matches_found
                logger.info(f"✅ Found {len(matches_found)} best matches for '{name}'.")
            else:
                logger.warning(f"❌ No best matches found for '{name}' on '{prop}'.")

        models_to_plot = [name for name in model_order if name in plot_data_map]
        if not models_to_plot:
            logger.error(
                f"No models had valid molecules to plot for '{prop}'. Skipping grid."
            )
            continue

        # Split into chunks if rows exceed max
        max_rows = 6
        total_rows = len(models_to_plot)
        num_figs = (total_rows + max_rows - 1) // max_rows
        chunk_size = (total_rows + num_figs - 1) // num_figs
        chunks = [
            models_to_plot[i : i + chunk_size] for i in range(0, total_rows, chunk_size)
        ]

        target_value = conditioning_targets.get(prop)
        target_str_title = f"{target_value:.2f}" if target_value is not None else "N/A"

        for idx, chunk in enumerate(chunks, start=1):
            rows = len(chunk)
            fig, axes = plt.subplots(
                nrows=rows,
                ncols=num_samples,
                figsize=(3.5 * num_samples, 3.2 * rows),
                squeeze=False,
            )

            fig.suptitle(
                f"Qualitative Comparison for Property: '{prop}' (Target: {target_str_title})"
                + (f" - Part {idx}/{len(chunks)}" if num_figs > 1 else ""),
                fontsize=18,
                weight="bold",
            )

            for i, model_name in enumerate(chunk):
                mol_data = plot_data_map[model_name]
                for j in range(num_samples):
                    ax = axes[i, j]
                    ax.axis("off")
                    if j < len(mol_data):
                        mol, sample_info = mol_data[j]
                        img = Draw.MolToImage(mol, size=(300, 300))
                        ax.imshow(img)

                        pred_val = sample_info.get("best_match_val")
                        pred_str = f"{pred_val:.2f}" if pred_val is not None else "N/A"

                        title = f"Pred: {pred_str}\nTarget: {target_str_title}"
                        ax.set_title(title, fontsize=11)

            fig.canvas.draw()
            for i, model_name in enumerate(chunk):
                color = color_map.get(model_name, conditioned_color)
                ax0 = axes[i, 0]
                ax0.text(
                    -1,
                    0.5,
                    f" {model_name} ",
                    transform=ax0.transAxes,
                    ha="left",
                    va="center",
                    fontsize=14,
                    fontweight="bold",
                    color="white",
                    bbox=dict(boxstyle="round,pad=0.4", fc=color, ec="black", lw=1),
                )

            fig.tight_layout(rect=[0.1, 0, 1, 0.95])

            part_suffix = f"_part{idx}" if num_figs > 1 else ""
            save_path = (
                Path(output_dir)
                / f"{run_type}_qualitative_comparison_{prop}{part_suffix}.png"
            )
            plt.savefig(save_path, dpi=300)
            logger.info(f"📊 Grid saved to {save_path}")
            plt.close(fig)
