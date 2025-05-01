import os
import re
import matplotlib.pyplot as plt
from collections import Counter


def plot_token_frequency(config, samples, name):
    if not config.eval.plot_dist:
        return

    token_pattern = re.compile(r'\[[^\]]+\]')
    token_counts = Counter()

    for s in samples:
        token_counts.update(token_pattern.findall(s))

    if not token_counts:
        return

    tokens, counts = zip(*token_counts.most_common())
    total = sum(counts)
    norm_counts = [c / total for c in counts]

    plt.figure(figsize=(12, 6))
    plt.bar(tokens, norm_counts)
    plt.xticks(rotation=90)
    plt.title(f"Token Frequency ({name})")
    plt.ylabel("Normalized Frequency")
    plt.tight_layout()

    _save_plot(config, f"token_frequency_{name}.png")


def plot_length_distribution(config, samples, name):
    if not config.eval.plot_dist:
        return

    token_pattern = re.compile(r'\[[^\]]+\]')
    lengths = [len(token_pattern.findall(s)) for s in samples]
    if not lengths:
        return

    plt.figure(figsize=(10, 5))
    plt.hist(lengths, bins=range(min(lengths), max(lengths) + 2), align='left', edgecolor='black')
    plt.title(f"Length Distribution ({name})")
    plt.xlabel("Tokens")
    plt.ylabel("Frequency")
    plt.tight_layout()

    _save_plot(config, f"length_distribution_{name}.png")


def plot_metric_distribution(config, values, metric, name):
    if not config.eval.plot_dist or not values:
        return

    avg = sum(values) / len(values)

    plt.figure(figsize=(10, 5))
    plt.hist(values, bins=50, alpha=0.75, edgecolor='black', density=True)
    plt.axvline(avg, color='red', linestyle='dotted', linewidth=2, label=f"Mean={avg:.2f}")
    plt.title(f"{metric.title()} Distribution ({name})")
    plt.xlabel(metric.title())
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()

    _save_plot(config, f"{metric}_distribution_{name}.png")


def plot_overlay_distribution(config, metric, results_by_name):
    if not config.eval.plot_dist:
        return

    plt.figure(figsize=(10, 5))
    for name, values in results_by_name.items():
        if values:
            plt.hist(values, bins=50, alpha=0.5, label=name, density=True)

    plt.title(f"{metric.title()} Overlay Distribution")
    plt.xlabel(metric.title())
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()

    _save_plot(config, f"{metric}_overlay_distribution.png")


def _save_plot(config, filename):
    path = os.path.join(config.local_paths.metrics_dir, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path)
    plt.close()
