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
        logger.info(f"Saved plot to {path}")
        plt.close()

    @staticmethod
    def _visible_names(all_names, reference_name):
        """
        Return names in the order [reference, *others*] but hide 'Original data'
        whenever it is **not** the reference.
        """
        return ([reference_name] +
                [n for n in all_names
                 if n != reference_name and
                    (reference_name == "Original data" or n != "Original data")])

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
        global_top_tokens = [tok for tok, _ in all_tokens_combined.most_common(top_n) if tok != '[EOS]']

        if '[EOS]' in all_tokens_combined:
            if '[EOS]' not in global_top_tokens:
                global_top_tokens.append('[EOS]')

        model_keys = [
            k for k in counts.keys()
            if k not in (reference_name, "Original data")
        ]
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

            eos_token = "[EOS]"
            labels = global_top_tokens.copy()
            if eos_token in labels and labels[-1] != eos_token:
                eos_rank = labels.index(eos_token) + 1  # 1-based
                # e.g., change label to "[EOS] (rank 5)" or "[EOS]⁵"
                labels[labels.index(eos_token)] = f"{eos_token}$^{{{eos_rank}}}$"

            plt.xticks(ticks=x, labels=labels, rotation=90)
            plt.ylabel('Frequency')
            plt.title("Token Frequency (%)", fontsize=14)
            plt.legend(title="Dataset")

            plt.tight_layout()

            filename_safe_model = model_key.replace(" ", "_").lower()
            filename_safe_ref = reference_name.replace(" ", "_").lower()
            self._save(f"{run_type}_token_frequency_{filename_safe_ref}_vs_{filename_safe_model}.png")


    def plot_length_violin(
            self,
            lengths: Dict[str, List[int]],
            reference_name: str,
            run_type: str
    ):
        """
        Sequence-length distributions as side-by-side violins.
        Reference colour = first entry; all other datasets share the comparison colour.
        """
        names  = self._visible_names(lengths.keys(), reference_name)
        if len(names) < 2:
            logger.warning("Not enough datasets for length violin"); return

        data   = [lengths[n] for n in names]
        plt.figure(figsize=(12, 6))

        parts = plt.violinplot(
            data,
            positions=np.arange(1, len(data) + 1),
            widths=0.9,
            showmeans=False,
            showmedians=True,
            showextrema=True
        )

        ref_colour  = plt.cm.get_cmap('tab10')(0)
        comp_colour = plt.cm.get_cmap('tab10')(1)
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(ref_colour if i == 0 else comp_colour)
            pc.set_alpha(0.7)
            pc.set_edgecolor('black')
            pc.set_linewidth(1)

        plt.xticks(range(1, len(names) + 1), names, rotation=45, ha='right')
        plt.ylabel("Sequence Length")
        plt.title("Sequence Length Distribution", fontsize=14)
        plt.grid(True, axis='y', linestyle=':', alpha=0.6)

        plt.tight_layout()
        self._save(f"{run_type}_length_violin.png")


    def plot_property_violin(
            self,
            metric: str,
            data: Dict[str, List[float]],
            reference_name: str,
            run_type: str
    ):
        """
        Make one side-by-side violin figure per metric.

        • x-axis = dataset / model  
        • y-axis = metric values  
        • first violin (reference) gets its own colour; all generated sets share one colour.
        """
        names = [reference_name] + [
            n for n in data
            if n != reference_name and n != "Original data" and data.get(n)
        ]
        if len(names) < 2:
            logger.warning(f"Not enough data to plot {metric}")
            return

        values = [data[n] for n in names]
        fig, ax = plt.subplots(figsize=(1.3 * len(values) + 3, 6))

        parts = ax.violinplot(
            values,
            positions=np.arange(1, len(values) + 1),
            widths=0.9,
            showmeans=False,
            showmedians=True,
            showextrema=True
        )

        ref_colour  = plt.cm.get_cmap('tab10')(0)   # reference colour
        comp_colour = plt.cm.get_cmap('tab10')(1)   # colour shared by all comparisons
        for i, body in enumerate(parts['bodies']):
            body.set_facecolor(ref_colour if i == 0 else comp_colour)
            body.set_alpha(0.7)
            body.set_edgecolor('black')
            body.set_linewidth(1)

        ax.set_xticks(np.arange(1, len(values) + 1))
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(metric.replace('_', ' ').title(), fontsize=14, pad=12)
        ax.grid(True, axis='y', linestyle=':', alpha=0.6)
        try:
            # Read the CSV with median percentiles
            percentile_path = Path(self.config.paths.median_percentile)
            if percentile_path.exists():
                df_medians = pd.read_csv(percentile_path)
                
                # Find the bottom_33 median for this specific metric
                metric_row = df_medians[df_medians['properties'] == metric]
                if not metric_row.empty:
                    bottom33_val = metric_row['bottom33_median'].values[0]
                    
                    # Add horizontal dashed line
                    ax.axhline(
                        y=bottom33_val,
                        color='red',
                        linestyle='--',
                        linewidth=1.5,
                        alpha=0.7,
                        label='Bottom 33% Median'
                    )
                    
                    # Add to existing legend
                    handles, labels = ax.get_legend_handles_labels()
                    ax.legend(handles=handles)
        except Exception as e:
            logger.warning(f"Couldn't add bottom 33% line for {metric}: {str(e)}")

        handles = [plt.Rectangle((0, 0), 1, 1, color=ref_colour,  alpha=0.7),
                   plt.Rectangle((0, 0), 1, 1, color=comp_colour, alpha=0.7)]
        ax.legend(handles, [reference_name, "Generated models"], loc="best")

        fig.tight_layout()
        self._save(f"{run_type}_{metric}_side_by_side.png")

            
    def display_statistical_summary(self, aggregated_results: Dict[str, Dict[str, any]], fcd_scores: Dict[str, float], run_type: str):
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

        logger.info("\n--- Statistical Summary of Molecular Metrics ---")
        formatters = {
            col: (lambda x: f"{float(x):.3f}" if pd.notna(x) and isinstance(x, (int, float, complex, np.number)) else str(x))
            for col in ['Validity', 'Uniqueness', 'Novelty', 'FCD']
            if col in summary_df.columns
        }
        logger.info(summary_df.to_string(formatters=formatters))

        summary_csv_path = self.plot_and_metrics_dir / f"{run_type}_statistical_summary.csv"
        summary_df.to_csv(summary_csv_path)
        logger.info(f"\nStatistical summary saved to {summary_csv_path}")