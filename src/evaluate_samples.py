import json
import os

from src.utils.csv_data_reader import fast_csv_to_df_reader
from src.evaluate.bos_eos_analysis import analyze_bos_eos_and_trim
from src.evaluate.metrics import calculate_and_plot_metrics

def evaluate_samples(config):
    # Load generated samples
    file_path = config.directory_paths.sampled_data
    with open(file_path, "r") as f:
        data = json.load(f)
        generated_samples = data["samples"]

    # Load original samples
    original_df = fast_csv_to_df_reader(config.directory_paths.raw_data, row_limit=config.row_limit)
    original_samples = original_df["selfies"].tolist()

    # 3) Decide which set to pass to metrics
    #    If config.overwrite_original is True, the original_samples 
    #    is already trimmed, so we can pass original_samples directly.
    #    Otherwise pass 'trimmed_original'.
    metrics_to_compute = [
        "token_frequency",       # Pseudo-metric -> token frequency plot
        "length_distribution",   # Pseudo-metric -> length distribution plot
        "sascore",               # standard
        "num_rings"              # standard
    ]
    
    calculate_and_plot_metrics(config, original_samples, metrics_to_compute, name="original")

    trimmed_generated = analyze_bos_eos_and_trim(generated_samples, config, name="generated")
    calculate_and_plot_metrics(config, trimmed_generated, metrics_to_compute, name="generated")


