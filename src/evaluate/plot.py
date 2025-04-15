import re
from collections import Counter
import matplotlib.pyplot as plt

def plot_token_frequency_histogram(config, samples, name="default"):
    """
    Plots a bar chart of normalized token frequencies across all samples.
    Tokens are extracted using regex (e.g., [C], [=C], [Branch1], etc.).
    The normalized frequency is computed as count / total_tokens.
    Saves the figure as '{config.directory_paths.images_dir}token_frequency_histogram_{name}.png'.
    """
    token_counts = Counter()
    for sample in samples:
        # Extract tokens using regex
        tokens = re.findall(r'\[[^\]]*\]', sample)
        token_counts.update(tokens)
    
    # Sort tokens by frequency (most_common returns in descending order)
    tokens, counts = zip(*token_counts.most_common())
    total = sum(counts)
    # Normalize counts to get probabilities
    norm_counts = [count / total for count in counts]
    
    plt.figure(figsize=(12, 6))
    plt.bar(tokens, norm_counts)
    plt.xticks(rotation=90)
    plt.title(f"Normalized Token Frequency Distribution ({name})")
    plt.ylabel("Normalized Frequency")
    plt.tight_layout()
    
    plt.savefig(f"{config.directory_paths.images_dir}token_frequency_histogram_{name}.png")
    plt.show()
    plt.close()


def plot_molecule_length_histogram(config, samples, name="default"):
    """
    Plots a histogram of the lengths (number of tokens) of molecules in 'samples'.
    Each molecule is a SELFIES string; tokens are extracted using regex.
    Saves the figure as '{config.directory_paths.images_dir}molecule_length_histogram_{name}.png'.
    """
    lengths = []
    token_pattern = re.compile(r'\[[^\]]*\]')
    for sample in samples:
        tokens = token_pattern.findall(sample)
        lengths.append(len(tokens))
    
    plt.figure(figsize=(10, 5))
    # Create bins from the minimum to maximum length
    bins = range(min(lengths), max(lengths) + 2)
    plt.hist(lengths, bins=bins, align='left', edgecolor='black')
    plt.title(f"Histogram of Molecule Lengths (number of tokens) ({name})")
    plt.xlabel("Number of Tokens")
    plt.ylabel("Frequency")
    plt.tight_layout()
    
    save_path = f"{config.directory_paths.images_dir}molecule_length_histogram_{name}.png"
    plt.savefig(save_path)
    plt.show()
    plt.close()

def analyze_bos_eos_tokens(config, samples, name="default"):
    """
    Given a list of SELFIES strings (samples):
      1) Counts how many molecules start with [BOS] vs. do not.
      2) Counts how many molecules end with [EOS] vs. do not.
      3) Counts how many molecules have [BOS] or [EOS] in the middle.
    
    Produces 3 separate bar charts.
    """

    # Initialize counters
    total_mols = len(samples)
    bos_start_count = 0
    eos_end_count = 0
    bos_eos_middle_count = 0

    # Regex to extract tokens like [C], [Branch1], [BOS], [EOS], etc.
    token_pattern = re.compile(r'\[[^\]]*\]')

    for sample in samples:
        tokens = token_pattern.findall(sample)
        if not tokens:
            continue  # skip empty or invalid strings

        # Check if starts with [BOS]
        if tokens[0] == "[BOS]":
            bos_start_count += 1
        
        # Check if ends with [EOS]
        if tokens[-1] == "[EOS]":
            eos_end_count += 1
        
        # Check if there is [BOS] or [EOS] in the middle (not first or last)
        # i.e. any place in tokens[1:-1]
        middle_tokens = tokens[1:-1]
        # We'll see if [BOS] or [EOS] is in the middle
        if any(tok in ("[BOS]", "[EOS]") for tok in middle_tokens):
            bos_eos_middle_count += 1

    # 1) Molecules that start vs. do not start with [BOS]
    plt.figure(figsize=(4, 4))
    plt.bar(
        ["Starts w/ [BOS]", "No [BOS] at start"],
        [bos_start_count, total_mols - bos_start_count],
        color=["orange", "blue"]
    )
    plt.title("Molecules: Start w/ [BOS] vs. No [BOS]")
    plt.tight_layout()
    plt.savefig(f"{config.directory_paths.images_dir}bos_start_bar_{name}.png")
    plt.show()
    plt.close()
    
    # 2) Molecules that end vs. do not end with [EOS]
    plt.figure(figsize=(4, 4))
    plt.bar(
        ["Ends w/ [EOS]", "No [EOS] at end"],
        [eos_end_count, total_mols - eos_end_count],
        color=["green", "red"]
    )
    plt.title("Molecules: End w/ [EOS] vs. No [EOS]")
    plt.tight_layout()
    plt.savefig(f"{config.directory_paths.images_dir}eos_end_bar_{name}.png")
    plt.show()
    plt.close()

    # 3) Molecules that have [BOS]/[EOS] in the middle vs. not
    plt.figure(figsize=(4, 4))
    plt.bar(
        ["Has [BOS]/[EOS] in middle", "No [BOS]/[EOS] in middle"],
        [bos_eos_middle_count, total_mols - bos_eos_middle_count],
        color=["purple", "gray"]
    )
    plt.title("Molecules: [BOS] or [EOS] in the Middle")
    plt.tight_layout()
    plt.savefig(f"{config.directory_paths.images_dir}bos_eos_middle_bar_{name}.png")
    plt.show()
    plt.close()
