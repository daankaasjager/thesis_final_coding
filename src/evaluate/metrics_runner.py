import logging

from .metrics_core import compute_standard_metric, get_valid_molecules
from .metrics_plotting import (plot_length_distribution,
                               plot_metric_distribution,
                               plot_overlay_distribution, plot_token_frequency)

logger = logging.getLogger(__name__)


def calculate_and_plot_metrics(config, samples, metrics, name: str):
    results = {}

    if "token_frequency" in metrics:
        plot_token_frequency(config, samples, name)

    if "length_distribution" in metrics:
        plot_length_distribution(config, samples, name)

    mols = get_valid_molecules(samples, name)

    for metric in metrics:
        if metric in ("token_frequency", "length_distribution"):
            continue
        values = [compute_standard_metric(mol, metric) for mol in mols]
        values = [v for v in values if v is not None]
        results[metric] = values
        logger.info(
            f"[{name}] {metric}: avg={sum(values)/len(values):.3f}, n={len(values)}"
        )
        plot_metric_distribution(config, values, metric, name)

    return results


def calculate_and_plot_metrics_multi(config, sample_dict, metrics):
    all_results = {metric: {} for metric in metrics}

    for name, samples in sample_dict.items():
        results = calculate_and_plot_metrics(config, samples, metrics, name)
        for metric, vals in results.items():
            all_results[metric][name] = vals

    for metric in all_results:
        if metric not in ("token_frequency", "length_distribution"):
            plot_overlay_distribution(config, metric, all_results[metric])

    return all_results
