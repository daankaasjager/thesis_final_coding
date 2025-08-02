"""
Evaluate and compare generated SELFIES samples using predefined metrics.

This module provides functions to load generated and original samples,
and to run metric comparisons across different sample sources.
"""

import json
import logging
from typing import Any, Dict, List, Optional

import pandas as pd

from .analysis import MetricRunner, generate_qualitative_comparison_grid
from .preprocessing import read_csv

logger = logging.getLogger(__name__)

PROPERTY_COLS: List[str] = [
    "nbo_P", "nmr_P", "pyr_P", "fmo_mu", "vmin_r", "volume", "fmo_eta",
    "fukui_m", "fukui_p", "nuesp_P", "somo_rc", "nbo_P_rc", "pyr_alpha",
    "qpole_amp", "vbur_vbur", "Pint_P_min", "sterimol_L", "sterimol_B1",
    "sterimol_B5", "dipolemoment", "efgtens_xx_P", "efgtens_yy_P",
    "nbo_bd_e_max", "nbo_lp_P_occ", "qpoletens_yy", "E_solv_elstat",
    "nbo_bds_e_avg", "sterimol_burL", "nbo_bd_occ_avg", "sterimol_burB5",
    "vbur_ovbur_min", "vbur_qvbur_min", "nbo_bds_occ_max",
    "vbur_ratio_vbur_vtot",
]

GROUP_123_PROPS = ["sascore", "molweight", "volume", "vbur_vbur", "vmin_r", 
                   "sterimol_L", "sterimol_B1", "dipolemoment"]

METRICS: List[str] = [
    "validity", "uniqueness", "novelty", "token_frequency",
    "length_distribution", "sascore", "num_rings",
    "tetrahedral_carbons", "logp", "molweight", "tpsa",
]


def _load_generated_samples(path: str) -> List[str]:
    """
    Load generated samples from a JSON file.

    Args:
        path: filesystem path to the JSON containing a "samples" list.

    Returns:
        List of sample strings; empty list on error.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("samples", [])
    except Exception as e:
        logger.error(f"Failed to load generated samples from {path}: {e}")
        return []


def _load_original_samples(
    path: str, row_limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Load original samples and their properties from a CSV.

    Args:
        path: filesystem path to the CSV with a 'selfies' column.
        row_limit: optional maximum number of rows to read.

    Returns:
        List of dicts with keys 'selfies' and 'predicted_properties'.
    """
    df = read_csv(path, row_limit=row_limit)
    if "selfies" not in df.columns:
        logger.error("Missing 'selfies' column in original data.")
        return []
    samples: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        props = {
            col: float(row[col])
            for col in PROPERTY_COLS
            if col in row and pd.notna(row[col])
        }
        samples.append({"selfies": row["selfies"], "predicted_properties": props})
    return samples


def _evaluate_by_comparison(
    config: Any,
    sample_sources: Dict[str, List[Any]],
    reference_name: str,
    run_type: str,
) -> None:
    """
    Run metrics comparison across multiple sample sources.

    Args:
        config: Hydra configuration object.
        sample_sources: mapping of source names to sample lists.
        reference_name: name of the reference dataset for comparison.
        run_type: identifier for this evaluation run (e.g. 'prelim', 'conditioning').
    """
    logger.info(f"Evaluating samples: run_type={run_type}")
    filtered = {k: v for k, v in sample_sources.items() if v}
    runner = MetricRunner(config)
    runner.run_multi(
        filtered, METRICS + PROPERTY_COLS, PROPERTY_COLS, reference_name, run_type
    )


def evaluate_preliminaries(config: Any) -> None:
    """
    Evaluate preliminary experiments: compare original data vs. various models.
    """
    sources = {
        "Original data": _load_original_samples(config.paths.filtered_original_data, 10000),
        "Tiny WordLevel": _load_generated_samples(config.paths.tiny_wordlevel),
        "Small WordLevel": _load_generated_samples(config.paths.small_wordlevel),
        "Small APE 70": _load_generated_samples(config.paths.ape_70),
        "Small APE 80": _load_generated_samples(config.paths.ape_80),
        "Small APE 110": _load_generated_samples(config.paths.ape_110),
    }
    _evaluate_by_comparison(
        config, sources, reference_name="Original data", run_type="prelim"
    )


def evaluate_conditioning(config: Any, baseline_model_name: str) -> None:
    """
    Evaluate conditioning experiments: compare baseline vs. conditioning strategies.

    Args:
        config: Hydra configuration object.
        baseline_model_name: name of the baseline model source.
    """
    original_samples = _load_original_samples(config.paths.filtered_original_data, 10000)
    sample_sources_1 = {
        "Original data": original_samples,
        baseline_model_name: _load_generated_samples(config.paths.baseline_model_path),
        "Prepend 1": _load_generated_samples(config.paths.prepend_1),
        "Prepend 3": _load_generated_samples(config.paths.prepend_3),
        "Prepend 8": _load_generated_samples(config.paths.prepend_8),
        "Prepend all": _load_generated_samples(config.paths.prepend_all),
        "Embedding 1": _load_generated_samples(config.paths.embedding_1),
        "Embedding 3": _load_generated_samples(config.paths.embedding_3),
        "Embedding 8": _load_generated_samples(config.paths.embedding_8),
        "Embedding all": _load_generated_samples(config.paths.embedding_all)
    }
    #_evaluate_by_comparison(config, sample_sources_1, reference_name=baseline_model_name, run_type="conditioning")

    # --- Second comparison set (CFG) ---
    sample_sources_2 = {
        "Original data": original_samples,
        baseline_model_name: _load_generated_samples(config.paths.baseline_model_path),
        "Embedding 3": _load_generated_samples(config.paths.embedding_3), # A good one to compare against
        "CFG 0.3": _load_generated_samples(config.paths.cfg_03),
        "CFG 1.0": _load_generated_samples(config.paths.cfg_10),
        "CFG 4.0": _load_generated_samples(config.paths.cfg_40)
    }
    #_evaluate_by_comparison(config, sample_sources_2, reference_name=baseline_model_name, run_type="conditioning_cfg")


    
    logger.info("--- Starting Qualitative Analysis for Prepend/Embedding ---")
    generate_qualitative_comparison_grid(
        sample_sources=sample_sources_1,
        baseline_model_name=baseline_model_name,
        properties_to_visualize=GROUP_123_PROPS,
        output_dir=config.paths.metrics_dir, # Save images alongside other metrics
        num_samples=4, # You can set this to 3, 4, or 5
        run_type="conditioning",
        conditioning_targets_path=config.paths.median_percentile # Path to the CSV with global targets
    )
    
    logger.info("--- Starting Qualitative Analysis for CFG ---")
    generate_qualitative_comparison_grid(
        sample_sources=sample_sources_2,
        baseline_model_name=baseline_model_name,
        properties_to_visualize=GROUP_123_PROPS,
        output_dir=config.paths.metrics_dir,
        num_samples=4,
        run_type="conditioning_cfg",
        conditioning_targets_path=config.paths.median_percentile # Path to the CSV with global targets
    )
