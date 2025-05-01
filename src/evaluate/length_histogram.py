import matplotlib.pyplot as plt
import os
import logging

logger = logging.getLogger(__name__)

def plot_length(processed_data, column='tokenized_selfies', bins=50, save_path='molecule_length_distribution.png'):
    """
    Plots and saves a histogram of molecule lengths from preprocessed SELFIES data.

    Parameters:
    - processed_data (DataFrame): Preprocessed data containing tokenized SELFIES.
    - column (str): Column name in processed_data containing SELFIES tokens.
    - bins (int): Number of bins for the histogram.
    - save_path (str): Path where the plot image will be saved.
    """
    logger.info("Plotting data")
    print(processed_data)
    molecule_lengths = processed_data[column].apply(len)

    plt.figure(figsize=(10, 6))
    plt.hist(molecule_lengths, bins=bins, color='skyblue', edgecolor='black')
    plt.xlabel('Molecule Length (Number of SELFIES tokens)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Molecule Lengths in Preprocessed SELFIES Data')
    plt.grid(True)

    # Explicitly save figure
    plt.savefig(save_path, bbox_inches='tight')
    logger.info(f"Plot saved to {os.path.abspath(save_path)}")