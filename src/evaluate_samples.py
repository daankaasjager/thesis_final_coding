from src.evaluate.plot import analyze_bos_eos_tokens, plot_token_frequency_histogram, plot_molecule_length_histogram
from src.evaluate.metrics import calculate_and_plot_metrics
from src.utils.csv_data_reader import fast_csv_to_df_reader
import json
import os

def evaluate_samples(config):
    file_path = os.path.join(config.directory_paths.sampled_data, "generated_samples.json")
    with open(file_path, 'r') as f:
        data = json.load(f)
        samples = data['samples']
    # Example usage:
    metrics_to_compute = ['sascore']
    original_selfies = fast_csv_to_df_reader(config.directory_paths.raw_data, row_limit=config.row_limit)
    generated_results = calculate_and_plot_metrics(config, samples, metrics_to_compute, name="generated", use_moses=False)
    #original_results = calculate_and_plot_metrics(config, original_selfies, metrics_to_compute, name="original")
    
    """plot_token_frequency_histogram(config, samples, "generated_samples")
    plot_token_frequency_histogram(config, original_selfies, "original_selfies")
    plot_molecule_length_histogram(config, samples, "generated_samples")
    plot_molecule_length_histogram(config, original_selfies, "original_selfies")
    analyze_bos_eos_tokens(config, samples, "generated_samples")"""