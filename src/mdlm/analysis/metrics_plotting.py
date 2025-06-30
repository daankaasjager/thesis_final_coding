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
        self.plot_and_metrics_dir = Path(config.paths.metrics_dir)
        self.global_metric_bins = {}

    def _save(self, filename: str):
        path = self.plot_and_metrics_dir / filename
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

    def plot_token_frequency(self, counts: Dict[str, Counter], reference_name: str, run_type: str, top_n: int = 20):
        """
        Plots token frequency as grouped bar plots for reference vs each generated dataset.
        """
        ref_counts = counts.get(reference_name)
        if not ref_counts:
            logger.warning(f"No reference token counts found for {reference_name}.")
            return

        # Global top N tokens across all datasets
        all_tokens_combined = Counter()
        for counter in counts.values():
            all_tokens_combined.update(counter)
        global_top_tokens = [item[0] for item in all_tokens_combined.most_common(top_n)]

        model_keys = [k for k in counts.keys() if k != reference_name]
        if not model_keys:
            logger.warning("No generated datasets found for token frequency plotting.")
            return

        for model_key in model_keys:
            model_counts = counts[model_key]

            ref_freqs = [ref_counts.get(tok, 0) for tok in global_top_tokens]
            model_freqs = [model_counts.get(tok, 0) for tok in global_top_tokens]

            x = np.arange(len(global_top_tokens))
            width = 0.35

            plt.figure(figsize=(12, 6))
            plt.bar(x - width/2, ref_freqs, width, label=reference_name)
            plt.bar(x + width/2, model_freqs, width, label=model_key)

            plt.xticks(ticks=x, labels=global_top_tokens, rotation=90)
            plt.ylabel('Frequency')
            plt.title("Token Frequency (%)", fontsize=14)
            plt.legend(title="Dataset")

            plt.tight_layout()

            filename_safe_model = model_key.replace(" ", "_").lower()
            filename_safe_ref = reference_name.replace(" ", "_").lower()
            self._save(f"token_frequency_{run_type}_{filename_safe_ref}_vs_{filename_safe_model}.png")


    def plot_length_violin(self, lengths: Dict[str, List[int]], reference_name: str, run_type: str):
        """
        Plots sequence length distributions as side-by-side violin plots.
        """
        plt.figure(figsize=(12, 6))

        data = []
        labels = []
        for name, vals in lengths.items():
            if vals:
                data.append(vals)
                labels.append(name)

        if not data:
            logger.warning("No length data to plot.")
            return

        parts = plt.violinplot(data, showmeans=False, showextrema=True, showmedians=True)
        
        # Set colors
        colors = plt.cm.get_cmap('tab10').colors
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[i % len(colors)])
            pc.set_alpha(0.7)
            pc.set_edgecolor('black')
            pc.set_linewidth(1)

        plt.xticks(range(1, len(labels) + 1), labels, rotation=45, ha='right')
        plt.ylabel("Sequence Length")
        plt.title("Sequence Length Distribution", fontsize=14)

        plt.tight_layout()
        colors = plt.cm.get_cmap('tab10').colors
        handles = [plt.Rectangle((0,0),1,1,color=colors[i % len(colors)],alpha=0.7) for i in range(len(labels))]
        self._save(f"length_violin_{run_type}.png")

    def plot_property_violin(self, metric: str, data: Dict[str, List[float]], reference_name: str, run_type: str):
        """
        Plots chemical property distributions as side-by-side violin plots.
        """
        reference_data = data.get(reference_name, [])
        if not reference_data:
            logger.warning(f"Skipping {metric}: missing reference data ({reference_name}).")
            return

        model_keys = [k for k in data.keys() if k != reference_name]
        if not model_keys:
            logger.warning(f"No generated models found for {metric}.")
            return

        for model_key in model_keys:
            comparison_data = data[model_key]
            if not comparison_data:
                continue

            fig, ax = plt.subplots(figsize=(6,6))

            ax.set_title(metric.replace('_', ' ').title(), fontsize=14)

            parts = ax.violinplot([reference_data, comparison_data],
                                showmeans=False, showmedians=True, showextrema=True)

            colors = plt.cm.get_cmap('tab10').colors
            handles = [plt.Rectangle((0,0),1,1,color=colors[i % len(colors)],alpha=0.7) for i in range(2)]
            ax.legend(handles, [reference_name, model_key], title="Dataset", loc="best")

            ax.set_xticks([])
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.grid(True, axis='y', linestyle=':', alpha=0.6)

            fig.tight_layout()



            filename_safe_model = model_key.replace(" ", "_").lower()
            filename_safe_ref = reference_name.replace(" ", "_").lower()
            self._save(f"{metric}_{run_type}_{filename_safe_ref}_vs_{filename_safe_model}.png")

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

    def plot_split_violin(self, metric: str, data: Dict[str, List[float]], reference_name: str, comparison_prefix: str, output_suffix: str):
        """
        Generates split violin plots comparing reference data vs each comparison model.
        
        Args:
            metric: The metric name (e.g. 'logp').
            data: Dict mapping dataset/model names to lists of metric values.
            reference_name: The name of the reference dataset (e.g. 'Original data' or 'Baseline model').
            comparison_prefix: Only models with this prefix will be plotted (e.g. 'Tiny', 'Prepend conditioning').
            output_suffix: Suffix to append in output filenames (e.g. 'prelim' or 'conditioning').
        """
        reference_data = data.get(reference_name, [])
        if not reference_data:
            logger.warning(f"Skipping split violin for {metric}: missing reference data ({reference_name}).")
            return

        model_keys = [k for k in data.keys() if k != reference_name and k.startswith(comparison_prefix)]
        if not model_keys:
            logger.warning(f"No comparison models found for {metric} with prefix {comparison_prefix}.")
            return

        for model_key in model_keys:
            comparison_data = data[model_key]
            if not comparison_data:
                logger.warning(f"No data for model {model_key}, skipping.")
                continue

            fig, ax = plt.subplots(figsize=(6, 5))
            
            # --- CHANGE 1: Use ax.set_title for better placement ---
            ax.set_title(metric.replace('_', ' ').title(), fontsize=14, pad=15) # `pad` adds some space
            ax.set_xticks([]) 
            colors = plt.cm.get_cmap('tab10').colors

            self._draw_split_violin(ax, reference_data, comparison_data,
                                    reference_name, model_key,
                                    colors[0], colors[1], position=1)

            # To avoid the x-tick label from becoming too crowded, let's use a more concise format.
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.grid(True, axis='y', linestyle=':', alpha=0.6)
            
            handles = [plt.Rectangle((0,0),1,1,color=colors[0],alpha=0.7),
                       plt.Rectangle((0,0),1,1,color=colors[1],alpha=0.7)]
            ax.legend(handles, [reference_name, model_key], title="Dataset")

            # --- CHANGE 2: Simplify tight_layout call ---
            plt.tight_layout() 
            
            filename_safe_model = model_key.replace(" ", "_").lower()
            filename_safe_ref = reference_name.replace(" ", "_").lower()
            self._save(f"{metric}_{output_suffix}_{filename_safe_ref}_vs_{filename_safe_model}.png")

            
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

        summary_csv_path = self.plot_and_metrics_dir / "statistical_summary.csv"
        summary_df.to_csv(summary_csv_path)
        print(f"\nStatistical summary saved to {summary_csv_path}")