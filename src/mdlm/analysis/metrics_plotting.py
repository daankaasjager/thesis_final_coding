import os
from venv import logger
import matplotlib.pyplot as plt
import scienceplots
from collections import defaultdict, Counter
from typing import Dict, List, Counter
import numpy as np
from pathlib import Path
import logging
import pandas as pd

logger = logging.getLogger(__name__)

plt.style.use(['science', 'no-latex'])

class MetricPlotter:
    def __init__(self, config):
        self.config = config
        self.metrics_dir = Path(config.paths.metrics_dir)
        self.global_metric_bins = {}

    def _save(self, filename: str):
        path = self.metrics_dir / filename
        os.makedirs(path.parent, exist_ok=True)
        plt.savefig(path, bbox_inches='tight')
        print(f"Saved plot to {path}")
        plt.close()

    def set_global_bins(self, all_metric_values: Dict[str, List[float]]):
        """Calculates and stores global min/max for each chemical metric."""
        for metric, values in all_metric_values.items():
            if values:
                min_val = min(values)
                max_val = max(values)
                self.global_metric_bins[metric] = np.linspace(min_val, max_val, 50)
            else:
                self.global_metric_bins[metric] = np.linspace(0, 1, 50)

    def plot_token_frequency(self, counts: Dict[str, Counter], top_n: int = 20):
        plt.figure(figsize=(12, 6))
        all_tokens_combined = Counter()
        for counter in counts.values():
            all_tokens_combined.update(counter)
        global_top_tokens = [item[0] for item in all_tokens_combined.most_common(top_n)]
        
        for name, counter in counts.items():
            freqs = [counter.get(token, 0) for token in global_top_tokens]
            ranks = range(1, len(freqs) + 1)
            plt.plot(ranks, freqs, label=name, alpha=0.7, marker='o')
            
        plt.xlabel(f'Token Rank (among Global Top {top_n})')
        plt.ylabel('Frequency')
        plt.title(f"Token Frequency Comparison (Global Top {top_n} Tokens)")
        plt.legend()
        plt.grid(True)
        self._save("token_frequency_overlay.png")

    def plot_length_distribution(self, lengths: Dict[str, List[int]]):
        plt.figure(figsize=(10, 5))
        all_lengths = [l for lst in lengths.values() for l in lst]
        if not all_lengths:
            logger.warning("No length data to plot.")
            return
        
        min_len, max_len = min(all_lengths), max(all_lengths)
        bins = range(min_len, max_len + 2)
        
        for name, vals in lengths.items():
            if vals:
                plt.hist(vals, bins=bins, alpha=0.5, label=name, density=True, align='left')
        
        plt.title("Length Distribution Comparison")
        plt.xlabel('Length')
        plt.ylabel('Density')
        plt.legend()
        self._save("length_distribution_overlay.png")

    def _draw_split_violin(self, ax, data1, data2, label1, label2, color1, color2, position):
        """Helper function to draw a split violin plot."""
        violin_width = 0.8

        def draw_half(plot_data, pos, plot_color, is_left):
            if not plot_data: return
            vp = ax.violinplot(plot_data, positions=[pos], widths=violin_width, 
                             showmeans=False, showextrema=False, showmedians=False)
            for body in vp['bodies']:
                path = body.get_paths()[0]
                mean_x = np.mean(path.vertices[:, 0])
                if is_left:
                    path.vertices[:, 0] = np.clip(path.vertices[:, 0], -np.inf, mean_x)
                else:
                    path.vertices[:, 0] = np.clip(path.vertices[:, 0], mean_x, np.inf)
                body.set_color(plot_color)
                body.set_alpha(0.7)
                body.set_edgecolor('black')
                body.set_linewidth(1)

        draw_half(data1, position, color1, is_left=True)
        draw_half(data2, position, color2, is_left=False)

    def plot_baseline_violin(self, metric: str, data: Dict[str, List[float]]):
        """
        Generates split violin plots to compare Original Data vs. Unconditioned models.
        This addresses RQ1: Can the model replicate the original data distribution? 
        """
        original_data = data.get("Original data", [])
        no_cond_wordlevel = data.get("No conditioning (WordLevel)", [])
        no_cond_ape = data.get("No conditioning (APE)", [])

        if not original_data or (not no_cond_wordlevel and not no_cond_ape):
            logger.warning(f"Skipping baseline plot for {metric}: missing data.")
            return

        fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
        fig.suptitle(f"{metric.replace('_', ' ').title()}: Baseline Generation Fidelity", fontsize=14)

        colors = plt.cm.get_cmap('tab10').colors
        
        # Plot 1: Original vs WordLevel
        ax[0].set_title("Original vs. WordLevel")
        self._draw_split_violin(ax[0], original_data, no_cond_wordlevel, "Original", "WordLevel", colors[0], colors[1], position=1)
        ax[0].set_xticks([1])
        ax[0].set_xticklabels(["Original | WordLevel"])
        ax[0].set_ylabel(metric.replace('_', ' ').title())

        # Plot 2: Original vs APE
        ax[1].set_title("Original vs. APE")
        self._draw_split_violin(ax[1], original_data, no_cond_ape, "Original", "APE", colors[0], colors[2], position=1)
        ax[1].set_xticks([1])
        ax[1].set_xticklabels(["Original | APE"])
        
        for axis in ax:
            axis.grid(True, axis='y', linestyle=':', alpha=0.6)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        self._save(f"{metric}_baseline_comparison.png")

    def plot_conditioning_violin(self, metric: str, data: Dict[str, List[float]]):
        """
        Generates split violin plots to compare unconditioned vs. conditioned models.
        This addresses RQ2: How effective is conditioning at steering generation? 
        """
        model_name_mapping = {
            "Prepend conditioning": "Prepend",
            "Embed conditioning": "Embed",
            "CFG conditioning": "CFG"
        }
        no_cond_wordlevel = data.get("No conditioning (WordLevel)", [])
        no_cond_ape = data.get("No conditioning (APE)", [])

        for model_key, short_name in model_name_mapping.items():
            cond_wordlevel_data = data.get(f"{model_key} (WordLevel)", [])
            cond_ape_data = data.get(f"{model_key} (APE)", [])

            if not cond_wordlevel_data and not cond_ape_data:
                continue

            fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
            fig.suptitle(f"{metric.replace('_', ' ').title()}: Effect of {short_name} Conditioning", fontsize=14)
            
            colors = plt.cm.get_cmap('tab10').colors

            # Plot 1: No Cond WordLevel vs Cond WordLevel
            ax[0].set_title("WordLevel: No Cond vs. Conditioned")
            self._draw_split_violin(ax[0], no_cond_wordlevel, cond_wordlevel_data, "No Cond", "Conditioned", colors[1], colors[3], position=1)
            ax[0].set_xticks([1])
            ax[0].set_xticklabels(["No Cond | Conditioned"])
            ax[0].set_ylabel(metric.replace('_', ' ').title())
            
            # Plot 2: No Cond APE vs Cond APE
            ax[1].set_title("APE: No Cond vs. Conditioned")
            self._draw_split_violin(ax[1], no_cond_ape, cond_ape_data, "No Cond", "Conditioned", colors[2], colors[4], position=1)
            ax[1].set_xticks([1])
            ax[1].set_xticklabels(["No Cond | Conditioned"])
            
            for axis in ax:
                axis.grid(True, axis='y', linestyle=':', alpha=0.6)

            plt.tight_layout(rect=[0, 0, 1, 0.96])
            self._save(f"{metric}_conditioning_effect_{short_name}.png")

    def display_statistical_summary(self, aggregated_results: Dict[str, Dict[str, any]], fcd_scores: Dict[str, float]):
        """
        Generates and prints a summary table of key statistics for all metrics and models.
        """
        summary_data = defaultdict(dict)
        all_model_names = set(aggregated_results.get("validity", {}).keys())
        all_model_names.update(fcd_scores.keys())

        for model_name in all_model_names:
            summary_data[model_name]['Validity'] = aggregated_results.get("validity", {}).get(model_name, np.nan)
            summary_data[model_name]['Uniqueness'] = aggregated_results.get("uniqueness", {}).get(model_name, np.nan)
            
            if model_name != "Original data":
                summary_data[model_name]['Novelty'] = aggregated_results.get("novelty", {}).get(model_name, np.nan)
                if "Original data" in all_model_names:
                    summary_data[model_name]['FCD'] = fcd_scores.get(model_name, np.nan)
                else:
                    summary_data[model_name]['FCD'] = np.nan
            else:
                summary_data[model_name]['Novelty'] = np.nan
                summary_data[model_name]['FCD'] = np.nan

        metrics_for_summary = ["sascore", "logp", "molweight", "num_rings", "tetrahedral_carbons", "tpsa"]
        for metric in metrics_for_summary:
            if metric in aggregated_results:
                for model_name, values in aggregated_results[metric].items():
                    if isinstance(values, list) and values:
                        mean_val, std_val = np.mean(values), np.std(values)
                        summary_data[model_name][f'Mean {metric.replace("_", " ").title()}'] = f"{mean_val:.3f} \u00B1 {std_val:.3f}"
                    else:
                        summary_data[model_name][f'Mean {metric.replace("_", " ").title()}'] = "No Data"

        summary_df = pd.DataFrame.from_dict(summary_data, orient='index')
        summary_df.index.name = "Model"
        
        desired_order = ['Validity', 'Uniqueness', 'Novelty', 'FCD'] + [f'Mean {m.replace("_", " ").title()}' for m in metrics_for_summary]
        existing_cols = [col for col in desired_order if col in summary_df.columns]
        summary_df = summary_df[existing_cols]

        print("\n--- Statistical Summary of Molecular Metrics ---")
        formatters = {col: lambda x: f"{x:.3f}" if pd.notna(x) else "N/A" for col in ['Validity', 'Uniqueness', 'Novelty', 'FCD'] if col in summary_df.columns}
        print(summary_df.to_string(formatters=formatters))

        summary_csv_path = self.metrics_dir / "statistical_summary.csv"
        summary_df.to_csv(summary_csv_path)
        print(f"\nStatistical summary saved to {summary_csv_path}")