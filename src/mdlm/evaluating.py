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

    processed_samples = {}
    for name, samples in sample_sources.items():
        if name == reference_name:
            processed_samples[name] = samples
        else:
            processed_samples[name] = bos_eos_analysis(samples, config, name=name)

    metrics = ["validity", "uniqueness", "novelty", "token_frequency", "length_distribution",
               "sascore", "num_rings", "tetrahedral_carbons", "logp", "molweight", "tpsa"]

    runner = MetricRunner(config)
    aggregated_results, fcd_scores = runner.run_multi(processed_samples, metrics)

    # Generate split violin plots per metric
    for metric, data_dict in aggregated_results.items():
        if metric not in ['validity', 'uniqueness', 'novelty', 'token_frequency', 'length_distribution']:
            runner.plotter.plot_split_violin(metric, data_dict,
                                             reference_name=reference_name,
                                             comparison_prefix="",
                                             output_suffix=run_type)

    runner.plotter.display_statistical_summary(aggregated_results, fcd_scores)


def evaluate_preliminaries(config):
    """
    Evaluates preliminary experiments: Original vs all models.
    """
    sample_sources = {
        "Original data": load_original_samples(config.paths.original_data, 100),
        "Tiny WordLevel": load_generated_samples(config.paths.test_sample),
        "Small WordLevel": load_generated_samples(config.paths.test_sample),
        "Small APE 70": load_generated_samples(config.paths.test_sample),
        "Small APE 80": load_generated_samples(config.paths.test_sample),
        "Small APE 110": load_generated_samples(config.paths.test_sample)
    }

    evaluate_by_comparison(config, sample_sources, reference_name="Original data", run_type="prelim")


def evaluate_conditioning(config, baseline_model_name):
    """
    Evaluates conditioning experiments: Baseline vs conditioning strategies.
    """
    sample_sources = {
        baseline_model_name: load_generated_samples(config.paths.baseline_model_path),
        "Prepend 1": load_generated_samples(config.paths.prepend_sampled_data),
        "Prepend 3": load_generated_samples(config.paths.prepend_3_sampled_data),
        "Prepend 8": load_generated_samples(config.paths.prepend_8_sampled_data),
        "Prepend all": load_generated_samples(config.paths.prepend_all_sampled_data),
        "Emedding 1": load_generated_samples(config.paths.embedding_1_sampled_data),
        "Embedding 3": load_generated_samples(config.paths.embedding_3_sampled_data),
        "Embedding 8": load_generated_samples(config.paths.embedding_8_sampled_data),
        "Embedding all": load_generated_samples(config.paths.embedding_all_sampled_data),
        "CFG 0.1": load_generated_samples(config.paths.cfg_01_sampled_data),
        "CFG 0.2": load_generated_samples(config.paths.cfg_02_sampled_data),
        "CFG 0.3": load_generated_samples(config.paths.cfg_03_sampled_data)
    }

    evaluate_by_comparison(config, sample_sources, reference_name=baseline_model_name, run_type="conditioning")
