import os
import re
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)

def analyze_bos_eos_and_trim(samples, config, name="default"):
    """
    1) Analyze how often molecules start with [BOS], end with [EOS],
       or have [BOS]/[EOS] in the middle.
    2) Plot bar charts if config.plot_dist is True.
    3) Trim each SELFIES after the first [EOS] token (inclusive).
    4) If config.overwrite_original == True, replace 'samples' in-place,
       otherwise, return a new list 'trimmed_samples'.
    
    Returns:
        (list) The final "active" samples (trimmed).
    """
    total_mols = len(samples)
    bos_start_count = 0
    eos_end_count = 0
    bos_eos_middle_count = 0

    token_pattern = re.compile(r'\[[^\]]*\]')
    trimmed_samples = []

    for sample in samples:
        tokens = token_pattern.findall(sample)

        if not tokens:
            trimmed_samples.append("")  # keep empty if no tokens
            continue

        # Count starts/ends w/ BOS/EOS
        if tokens[0] == "[BOS]":
            bos_start_count += 1
        if tokens[-1] == "[EOS]":
            eos_end_count += 1

        # Middle check
        middle_tokens = tokens[1:-1]
        if any(tok in ("[BOS]", "[EOS]") for tok in middle_tokens):
            bos_eos_middle_count += 1

        # Trim after first [EOS]
        new_tokens = []
        for tok in tokens:
            new_tokens.append(tok)
            if tok == "[EOS]":
                break
        trimmed_samples.append("".join(new_tokens))

    # Plot bar charts if config.plot_dist
    if config.plot_dist:
        # i) Molecules that start vs. do not start with [BOS]
        plt.figure(figsize=(4, 4))
        plt.bar(
            ["Starts w/ [BOS]", "No [BOS] at start"],
            [bos_start_count, total_mols - bos_start_count],
            color=["orange", "blue"]
        )
        plt.title(f"Molecules: Start w/ [BOS] vs. No [BOS] ({name})")
        plt.tight_layout()
        plt.savefig(os.path.join(config.local_paths.metrics_dir, f"bos_start_bar_{name}.png"))
        plt.close()

        # ii) Molecules that end vs. do not end with [EOS]
        plt.figure(figsize=(4, 4))
        plt.bar(
            ["Ends w/ [EOS]", "No [EOS] at end"],
            [eos_end_count, total_mols - eos_end_count],
            color=["green", "red"]
        )
        plt.title(f"Molecules: End w/ [EOS] vs. No [EOS] ({name})")
        plt.tight_layout()
        plt.savefig(os.path.join(config.local_paths.metrics_dir, f"eos_end_bar_{name}.png"))
        plt.close()

        # iii) Molecules that have [BOS]/[EOS] in the middle
        plt.figure(figsize=(4, 4))
        plt.bar(
            ["Has [BOS]/[EOS] in middle", "No [BOS]/[EOS] in middle"],
            [bos_eos_middle_count, total_mols - bos_eos_middle_count],
            color=["purple", "gray"]
        )
        plt.title(f"Molecules: [BOS] or [EOS] in Middle ({name})")
        plt.tight_layout()
        plt.savefig(os.path.join(config.local_paths.metrics_dir, f"bos_eos_middle_bar_{name}.png"))
        plt.close()

    # Possibly overwrite the original data
    if config.overwrite_original:
        logger.error("implement this")
    else:
        # Return the trimmed version as a new list
        return trimmed_samples
