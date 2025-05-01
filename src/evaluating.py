import json
import logging

from evaluate import (bos_eos_analysis, calculate_and_plot_metrics,
                      calculate_and_plot_metrics_multi)
from preprocessing import read_csv

logger = logging.getLogger(__name__)


def load_generated_samples(path):
    """Load generated samples from a JSON file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("samples", [])
    except Exception as e:
        logger.error(f"Failed to load generated samples from {path}: {e}")
        return []


def load_original_samples(path, row_limit):
    """Load original SELFIES from a CSV file."""
    df = read_csv(path, row_limit=row_limit)
    if "selfies" not in df.columns:
        logger.error("Missing 'selfies' column in original data.")
        return []
    return df["selfies"].tolist()


def evaluate_samples(config):
    """Main function to evaluate generated vs original SELFIES samples."""
    logger.info("Evaluating...")

    generated_samples = load_generated_samples(config.local_paths.sampled_data)
    if not generated_samples:
        logger.warning("No generated samples found.")
        return

    # This takes the path of the original data because it has the same distribution as the augmented data
    original_samples = load_original_samples(
        config.local_paths.original_data, config.row_limit
    )
    if not original_samples:
        logger.warning("No original samples found.")
        return

    metrics = ["token_frequency", "length_distribution", "sascore", "num_rings"]
    if config.eval.overlay:
        logger.info("Evaluating original and generated samples with overlay...")
        calculate_and_plot_metrics_multi(
            config,
            {"original": original_samples, "generated": generated_samples},
            metrics,
        )
    else:
        logger.info("Evaluating original samples...")
        calculate_and_plot_metrics(config, original_samples, metrics, name="original")

        logger.info("Evaluating generated samples...")
        trimmed_generated = bos_eos_analysis(
            generated_samples, config, name="generated"
        )
        calculate_and_plot_metrics(config, trimmed_generated, metrics, name="generated")

    logger.info("✅ Evaluation complete.")
