import logging
import os
import re

import matplotlib.pyplot as plt
import scienceplots

plt.style.use(['science', 'no-latex'])

logger = logging.getLogger(__name__)


def bos_eos_analysis(samples, config, name="default"):
    """
    1) Analyze how often molecules start with [BOS], end with [EOS],
       or have [BOS]/[EOS] in the middle.
    2) Plot bar charts if config.eval.plot_dist is True.
    3) Trim each SELFIES after the first [EOS] token (inclusive).
    4) If config.eval.overwrite_original == True, replace 'samples' in-place,
       otherwise, return a new list 'trimmed_samples'.

    Returns:
        (list) The final "active" samples (trimmed).
    """
    total_mols = len(samples)
    bos_start_count = 0
    eos_end_count = 0
    bos_eos_middle_count = 0

    token_pattern = re.compile(r"\[[^\]]*\]")
    trimmed_samples = []

    for sample in samples:
        tokens = token_pattern.findall(sample)

        if not tokens:
            trimmed_samples.append("")  # keep empty if no tokens
            continue

        # Count starts or ends with BOS/EOS
        if tokens[0] == "[BOS]":
            bos_start_count += 1

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

    if config.eval.plot_dist:
        # 1) Molecules that start vs. do not start with [BOS]
        plt.figure(figsize=(4, 4))
        plt.bar(
            ["Starts w/ [BOS]", "No [BOS] at start"],
            [bos_start_count, total_mols - bos_start_count],
            color=["orange", "blue"],
        )
        plt.title(f"Molecules: Start w/ [BOS] vs. No [BOS] ({name})")
        plt.tight_layout()
        plt.savefig(
            os.path.join(config.paths.metrics_dir, f"bos_start_bar_{name}.png")
        )
        plt.close()



        # 3) Molecules that have [BOS]/[EOS] in the middle
        plt.figure(figsize=(4, 4))
        plt.bar(
            ["Has [BOS]/[EOS] in middle", "No [BOS]/[EOS] in middle"],
            [bos_eos_middle_count, total_mols - bos_eos_middle_count],
            color=["purple", "gray"],
        )
        plt.title(f"Molecules: [BOS] or [EOS] in Middle ({name})")
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                config.paths.metrics_dir, f"bos_eos_middle_bar_{name}.png"
            )
        )
        plt.close()

    if config.eval.overwrite_original:
        logger.error("implement this")
    else:
        return trimmed_samples
