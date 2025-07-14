import json
import logging
import pandas as pd

from .analysis import bos_eos_analysis, MetricRunner
from .preprocessing import read_csv
from .analysis import MetricRunner
from src.property_prediction.inference import predict_properties

PROPERTY_COLS = [  
    'nbo_P', 'nmr_P', 'pyr_P', 'fmo_mu', 'vmin_r', 'volume',
    'fmo_eta', 'fukui_m', 'fukui_p', 'nuesp_P', 'somo_rc', 'nbo_P_rc',
    'pyr_alpha', 'qpole_amp', 'vbur_vbur', 'Pint_P_min', 'sterimol_L',
    'sterimol_B1', 'sterimol_B5', 'dipolemoment', 'efgtens_xx_P',
    'efgtens_yy_P', 'nbo_bd_e_max', 'nbo_lp_P_occ', 'qpoletens_yy',
    'E_solv_elstat', 'nbo_bds_e_avg', 'sterimol_burL', 'nbo_bd_occ_avg',
    'sterimol_burB5', 'vbur_ovbur_min', 'vbur_qvbur_min',
    'nbo_bds_occ_max', 'vbur_ratio_vbur_vtot', 'mol_wt', 'sa_score'
]

METRICS = [
    "validity", "uniqueness", "novelty", "token_frequency", "length_distribution",
    "sascore", "num_rings", "tetrahedral_carbons", "logp", "molweight", "tpsa"
]

logger = logging.getLogger(__name__)


def load_generated_samples(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("samples", [])
    except Exception as e:
        logger.error(f"Failed to load generated samples from {path}: {e}")
        return []


def load_original_samples(path: str, row_limit: int | None = None) -> list[dict]:
    """
    Returns a list of *sample objects* that look exactly like the generated ones:
        {"selfies": <str>, "predicted_properties": {<prop>: <float>, …}}
    """
    df = read_csv(path, row_limit=row_limit)

    if "selfies" not in df.columns:
        logger.error("Missing 'selfies' column in original data.")
        return []

    samples = []
    for _, row in df.iterrows():
        prop_dict = {
            col: float(row[col])
            for col in PROPERTY_COLS
            if col in row and pd.notna(row[col])
        }
        samples.append(
            {
                "selfies": row["selfies"],
                "predicted_properties": prop_dict,   # <- name stays the same
            }
        )
    return samples

def evaluate_by_comparison(config, sample_sources: dict, reference_name: str, run_type: str):
    """
    General evaluation function for any set of samples.
    
    Args:
        config: Hydra config.
        sample_sources: dict mapping names to loaded samples.
        reference_name: name of the reference dataset (e.g. 'Original data' or baseline model name).
        run_type: 'prelim' or 'conditioning' for output file naming.
    """
    logger.info(f"Evaluating samples: run_type={run_type}")

    # Remove empty datasets
    sample_sources = {k: v for k, v in sample_sources.items() if v}

    runner = MetricRunner(config)
    aggregated_results, fcd_scores = runner.run_multi(sample_sources, METRICS + PROPERTY_COLS, PROPERTY_COLS, reference_name, run_type)
    runner.plotter.display_statistical_summary(aggregated_results, fcd_scores, run_type)


def evaluate_preliminaries(config):
    """
    Evaluates preliminary experiments: Original vs all models.
    """
    sample_sources = {
        "Original data": load_original_samples(config.paths.filtered_original_data, 10000),
        "Small WordLevel": load_generated_samples(config.paths.small_wordlevel)
    }
    """
        "Tiny WordLevel": load_generated_samples(config.paths.tiny_wordlevel),
        "Small APE 70": load_generated_samples(config.paths.ape_70),
        "Small APE 80": load_generated_samples(config.paths.ape_80),
        "Small APE 110": load_generated_samples(config.paths.ape_110)"""
    evaluate_by_comparison(config, sample_sources, reference_name="Original data", run_type="prelim")


def evaluate_conditioning(config, baseline_model_name):
    """
    Evaluates conditioning experiments: Baseline vs conditioning strategies.
    """
    sample_sources = {
        "Original data": load_original_samples(config.paths.filtered_original_data, None),
        baseline_model_name: load_generated_samples(config.paths.baseline_model_path),
        "Prepend 1": load_generated_samples(config.paths.prepend_1),
        "Prepend 3": load_generated_samples(config.paths.prepend_3),
        "Prepend 8": load_generated_samples(config.paths.prepend_8),
        "Prepend all": load_generated_samples(config.paths.prepend_all),
        "Emedding 1": load_generated_samples(config.paths.embedding_1),
        "Embedding 3": load_generated_samples(config.paths.embedding_3),
        "Embedding 8": load_generated_samples(config.paths.embedding_8),
        "Embedding all": load_generated_samples(config.paths.embedding_all),
        "CFG 0.3": load_generated_samples(config.paths.cfg_03),
        "CFG 1.0": load_generated_samples(config.paths.cfg_10),
        "CFG 4.0": load_generated_samples(config.paths.cfg_40)
    }

    evaluate_by_comparison(config, sample_sources, reference_name=baseline_model_name, run_type="conditioning")
