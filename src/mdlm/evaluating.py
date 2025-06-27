import json
import logging

from .analysis import (bos_eos_analysis, MetricRunner)
from .preprocessing import read_csv
from .analysis import MetricRunner

logger = logging.getLogger(__name__)


def load_generated_samples(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("samples", [])
    except Exception as e:
        logger.error(f"Failed to load generated samples from {path}: {e}")
        return []


def load_original_samples(path, row_limit):
    df = read_csv(path, row_limit=row_limit)
    if "selfies" not in df.columns:
        logger.error("Missing 'selfies' column in original data.")
        return []
    return df["selfies"].tolist()


def evaluate_samples(config):
    """Main function to evaluate generated vs original SELFIES samples."""
    logger.info("Evaluating...")

    sample_sources = {
        "Original data": load_original_samples(config.paths.original_data, 100),
        "No conditioning (WordLevel)": load_generated_samples(config.paths.no_sampled_data),
        "No conditioning (APE)": load_generated_samples(config.paths.no_ape_sampled_data),
        "Prepend conditioning (WordLevel)": load_generated_samples(config.paths.prepend_sampled_data),
        "Prepend conditioning (APE)": load_generated_samples(config.paths.prepend_ape_sampled_data),
        "Embed conditioning (WordLevel)": load_generated_samples(config.paths.embed_sampled_data),
        "Embed conditioning (APE)": load_generated_samples(config.paths.embed_ape_sampled_data),
        "CFG conditioning (WordLevel)": load_generated_samples(config.paths.cfg_sampled_data),
        "CFG conditioning (APE)": load_generated_samples(config.paths.cfg_ape_sampled_data)
    }

    sample_sources = {k: v for k, v in sample_sources.items() if v}

    processed_samples = {}
    for name, samples in sample_sources.items():
        if name == "Original data":
            processed_samples[name] = samples
        else:
            processed_samples[name] = bos_eos_analysis(samples, config, name=name)
    
    metrics = ["validity", "uniqueness", "novelty", "token_frequency", "length_distribution", "sascore", 
               "num_rings", "tetrahedral_carbons", "logp", "molweight", 
                "tpsa"]
    
    runner = MetricRunner(config)

    aggregated_results, fcd_scores = runner.run_multi(processed_samples, metrics)
    runner.plotter.display_statistical_summary(aggregated_results, fcd_scores)

