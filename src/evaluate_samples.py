from src.evaluate.plot import analyze_bos_eos_tokens, plot_token_frequency_histogram, plot_molecule_length_histogram
from src.evaluate.metrics import calculate_and_plot_metrics
import json

def evaluate_samples(config, original_selfies):
    with open(config.directory_paths.sampled_data, 'r') as f:
        data = json.load(f)
        samples = data['samples']
    # Example usage:
    metrics_to_compute = ['sascore']
    generated_results = calculate_and_plot_metrics(config, samples, metrics_to_compute, name="generated", use_moses=True)
    #original_results = calculate_and_plot_metrics(config, original_selfies, metrics_to_compute, name="original")
    
    """plot_token_frequency_histogram(config, samples, "generated_samples")
    plot_token_frequency_histogram(config, original_selfies, "original_selfies")
    plot_molecule_length_histogram(config, samples, "generated_samples")
    plot_molecule_length_histogram(config, original_selfies, "original_selfies")
    analyze_bos_eos_tokens(config, samples, "generated_samples")"""