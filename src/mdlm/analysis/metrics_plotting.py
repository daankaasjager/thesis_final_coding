import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Counter, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

logger = logging.getLogger(__name__)

plt.style.use(["science", "no-latex"])


class MetricPlotter:
    def __init__(self, config):
        self.config = config
        self.plot_and_metrics_dir = Path(config.paths.metrics_dir)
        self.global_metric_bins = {}

    def _save(self, filename: str):
        path = self.plot_and_metrics_dir / filename
        os.makedirs(path.parent, exist_ok=True)
        plt.savefig(path, bbox_inches="tight")
        logger.info(f"Saved plot to {path}")
        plt.close()

    @staticmethod
    def _visible_names(all_names, reference_name):
        """
        Return names in the order [reference, *others*] but hide 'Original data'
        whenever it is **not** the reference. (Used for 'preliminaries' run type).
        """
        return [reference_name] + [
            n
            for n in all_names
            if n != reference_name
            and (reference_name == "Original data" or n != "Original data")
        ]

    def set_global_bins(self, all_metric_values: Dict[str, List[float]]):
        """Calculates and stores global min/max for each chemical metric."""
        for metric, values in all_metric_values.items():
            if values:
                min_val = min(values)
                max_val = max(values)
                self.global_metric_bins[metric] = np.linspace(min_val, max_val, 50)
            else:
                self.global_metric_bins[metric] = np.linspace(0, 1, 50)

    def plot_token_frequency(
        self,
        counts: Dict[str, Counter],
        reference_name: str,
        run_type: str,
        top_n: int = 20,
    ):
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
        global_top_tokens = [
            tok for tok, _ in all_tokens_combined.most_common(top_n) if tok != "[EOS]"
        ]

        if "[EOS]" in all_tokens_combined:
            if "[EOS]" not in global_top_tokens:
                global_top_tokens.append("[EOS]")

        model_keys = [
            k for k in counts.keys() if k not in (reference_name, "Original data")
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
            plt.bar(x - width / 2, ref_freqs, width, label=reference_name)
            plt.bar(x + width / 2, model_freqs, width, label=model_key)

            eos_token = "[EOS]"
            labels = global_top_tokens.copy()
            if eos_token in labels and labels[-1] != eos_token:
                eos_rank = labels.index(eos_token) + 1  # 1-based
                labels[labels.index(eos_token)] = f"{eos_token}$^{{{eos_rank}}}$"

            plt.xticks(ticks=x, labels=labels, rotation=90)
            plt.ylabel("Frequency")
            plt.title("Token Frequency (%)", fontsize=14)
            plt.legend(title="Dataset")
            plt.tight_layout()

            filename_safe_model = model_key.replace(" ", "_").lower()
            filename_safe_ref = reference_name.replace(" ", "_").lower()
            self._save(
                f"{run_type}_token_frequency_{filename_safe_ref}_vs_{filename_safe_model}.png"
            )

    def plot_length_violin(
        self, lengths: Dict[str, List[int]], reference_name: str, run_type: str
    ):
        """
        Plots sequence-length distributions as side-by-side violins.
        The order of violins now matches the order in the source dictionary.
        """
        # MODIFICATION: Use the dictionary's key order directly.
        names = list(lengths.keys())

        if len(names) < 2:
            logger.warning("Not enough datasets for length violin")
            return

        data = [lengths[n] for n in names]
        plt.figure(figsize=(1.3 * len(data) + 3, 6))

        parts = plt.violinplot(
            data,
            positions=np.arange(1, len(data) + 1),
            widths=0.9,
            showmeans=False,
            showmedians=True,
            showextrema=True,
        )

        if run_type.startswith("conditioning"):
            # MODIFICATION: Changed comparison color to purple.
            baseline_colour = plt.cm.get_cmap("tab10")(1)
            original_data_colour = plt.cm.get_cmap("tab10")(0)
            comp_colour = "#B266FF"  # New purple color

            color_map = defaultdict(lambda: comp_colour)
            color_map[reference_name] = baseline_colour
            color_map["Original data"] = original_data_colour

            for i, pc in enumerate(parts["bodies"]):
                pc.set_facecolor(color_map[names[i]])
                pc.set_alpha(0.7)
                pc.set_edgecolor("black")
                pc.set_linewidth(1)

            handles = [plt.Rectangle((0, 0), 1, 1, color=baseline_colour, alpha=0.7)]
            labels = [reference_name]
            if "Original data" in names:
                handles.append(
                    plt.Rectangle((0, 0), 1, 1, color=original_data_colour, alpha=0.7)
                )
                labels.append("Original data")
            if any(n not in [reference_name, "Original data"] for n in names):
                handles.append(
                    plt.Rectangle((0, 0), 1, 1, color=comp_colour, alpha=0.7)
                )
                labels.append("Generated models")
            plt.legend(handles, labels, loc="best")

        else:  # 'preliminaries'
            ref_colour = plt.cm.get_cmap("tab10")(0)
            comp_colour = plt.cm.get_cmap("tab10")(1)
            # This logic assumes the reference is the first item in the dict, which it is for prelims.
            for i, pc in enumerate(parts["bodies"]):
                pc.set_facecolor(ref_colour if i == 0 else comp_colour)
                pc.set_alpha(0.7)
                pc.set_edgecolor("black")
                pc.set_linewidth(1)

        plt.xticks(range(1, len(names) + 1), names, rotation=45, ha="right")
        plt.ylabel("Sequence Length")
        plt.title("Sequence Length Distribution", fontsize=14)
        plt.grid(True, axis="y", linestyle=":", alpha=0.6)

        plt.tight_layout()
        self._save(f"{run_type}_length_violin.png")

    def plot_property_violin(
        self,
        metric: str,
        data: Dict[str, List[float]],
        reference_name: str,
        run_type: str,
    ):
        """
        Makes one side-by-side violin figure per metric.
        The order of violins now matches the order in the source dictionary.
        The red line for 'Bottom 33% Median' is ONLY shown for 'conditioning'.
        """
        # MODIFICATION: Use the dictionary's key order directly.
        names = [n for n in data.keys() if data.get(n)]

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
            showextrema=True,
        )

        handles, legend_labels = [], []
        if run_type.startswith("conditioning"):
            # MODIFICATION: Changed comparison color to purple.
            baseline_colour = plt.cm.get_cmap("tab10")(1)
            original_data_colour = plt.cm.get_cmap("tab10")(0)
            comp_colour = "#B266FF"  # New purple color

            color_map = defaultdict(lambda: comp_colour)
            color_map[reference_name] = baseline_colour
            color_map["Original data"] = original_data_colour

            for i, body in enumerate(parts["bodies"]):
                body.set_facecolor(color_map[names[i]])
                body.set_alpha(0.7)
                body.set_edgecolor("black")
                body.set_linewidth(1)

            handles.append(
                plt.Rectangle((0, 0), 1, 1, color=baseline_colour, alpha=0.7)
            )
            legend_labels.append(reference_name)
            if "Original data" in names:
                handles.append(
                    plt.Rectangle((0, 0), 1, 1, color=original_data_colour, alpha=0.7)
                )
                legend_labels.append("Original data")
            if any(n not in [reference_name, "Original data"] for n in names):
                handles.append(
                    plt.Rectangle((0, 0), 1, 1, color=comp_colour, alpha=0.7)
                )
                legend_labels.append("Generated models")

        else:  # 'preliminaries'
            ref_colour = plt.cm.get_cmap("tab10")(0)
            comp_colour = plt.cm.get_cmap("tab10")(1)
            for i, body in enumerate(parts["bodies"]):
                body.set_facecolor(ref_colour if i == 0 else comp_colour)
                body.set_alpha(0.7)
                body.set_edgecolor("black")
                body.set_linewidth(1)

            handles = [
                plt.Rectangle((0, 0), 1, 1, color=ref_colour, alpha=0.7),
                plt.Rectangle((0, 0), 1, 1, color=comp_colour, alpha=0.7),
            ]
            legend_labels = [reference_name, "Generated models"]

        ax.set_xticks(np.arange(1, len(values) + 1))
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(metric.replace("_", " ").title(), fontsize=14, pad=12)
        ax.grid(True, axis="y", linestyle=":", alpha=0.6)

        if run_type == "conditioning":
            try:
                percentile_path = Path(self.config.paths.median_percentile)
                if percentile_path.exists():
                    df_medians = pd.read_csv(percentile_path)
                    metric_row = df_medians[df_medians["properties"] == metric]
                    if not metric_row.empty:
                        bottom33_val = metric_row["bottom33_median"].values[0]
                        ax.axhline(
                            y=bottom33_val,
                            color="red",
                            linestyle="--",
                            linewidth=1.5,
                            alpha=0.7,
                            label="Bottom 33% Median",
                        )
            except Exception as e:
                logger.warning(f"Couldn't add bottom 33% line for {metric}: {str(e)}")

        line_handles, line_labels = ax.get_legend_handles_labels()
        ax.legend(
            handles=handles + line_handles,
            labels=legend_labels + line_labels,
            loc="best",
        )

        fig.tight_layout()
        self._save(f"{run_type}_{metric}_side_by_side.png")

    def display_statistical_summary(
        self, aggregated_results, fcd_scores, reference_name, run_type, mae_distances
    ):
        """
        Generates and saves three separate summary tables:
        1. Validity, Uniqueness, Novelty, and FCD scores.
        2. Mean and Standard Deviation for all property distributions.
        3. Property Distances (Wasserstein for unconditioned, MAE for conditioned).
        """
        if not aggregated_results:
            logger.error("Aggregated results are empty. Cannot generate summaries.")
            return

        ordered_model_names = list(next(iter(aggregated_results.values())).keys())

        # --- 1. VUNF + FCD Summary ---
        vunf_data = defaultdict(dict)
        for model_name in ordered_model_names:
            vunf_data[model_name]["Validity"] = aggregated_results.get(
                "validity", {}
            ).get(model_name, np.nan)
            vunf_data[model_name]["Uniqueness"] = aggregated_results.get(
                "uniqueness", {}
            ).get(model_name, np.nan)

            if model_name != reference_name and model_name != "Original data":
                vunf_data[model_name]["Novelty"] = aggregated_results.get(
                    "novelty", {}
                ).get(model_name, np.nan)
                vunf_data[model_name]["FCD"] = fcd_scores.get(model_name, np.nan)
            else:
                vunf_data[model_name]["Novelty"] = np.nan
                vunf_data[model_name]["FCD"] = np.nan

        vunf_df = pd.DataFrame.from_dict(vunf_data, orient="index")
        vunf_df.index.name = "Model"
        vunf_df = vunf_df.reindex(ordered_model_names)[
            ["Validity", "Uniqueness", "Novelty", "FCD"]
        ]
        vunf_path = self.plot_and_metrics_dir / f"{run_type}_vunf_fcd_summary.csv"
        vunf_df.to_csv(vunf_path, float_format="%.4f")
        logger.info(f"VUNF+FCD summary saved to {vunf_path}")

        distribution_metrics = sorted(
            [
                m
                for m, r in aggregated_results.items()
                if isinstance(next(iter(r.values()), None), list)
            ]
        )
        if not distribution_metrics:
            logger.warning(
                "No distribution metrics found. Skipping Mean/Std and Distance reports."
            )
            return

        # --- 2. Mean and Std Summary ---
        means_stds_data = defaultdict(lambda: defaultdict(float))
        for metric in distribution_metrics:
            for model_name in ordered_model_names:
                values = aggregated_results[metric].get(model_name, [])
                if values:
                    means_stds_data[model_name][f"{metric}_mean"] = np.mean(values)
                    means_stds_data[model_name][f"{metric}_std"] = np.std(values)
                else:
                    means_stds_data[model_name][f"{metric}_mean"] = np.nan
                    means_stds_data[model_name][f"{metric}_std"] = np.nan

        means_stds_df = pd.DataFrame.from_dict(means_stds_data, orient="index")
        means_stds_df.index.name = "Model"
        means_stds_df = means_stds_df.reindex(ordered_model_names)
        means_stds_path = (
            self.plot_and_metrics_dir / f"{run_type}_properties_summary.csv"
        )
        means_stds_df.to_csv(means_stds_path, float_format="%.4f")
        logger.info(f"Property means/stds summary saved to {means_stds_path}")

        # --- 3. Property Distance (Wasserstein or MAE) Summary ---
        logger.info(
            "Generating property distance summary (Wasserstein for unconditioned, MAE for conditioned)."
        )
        prop_distance_data = defaultdict(dict)
        original_data_dist = {
            m: aggregated_results[m].get("Original data", [])
            for m in distribution_metrics
        }

        for model_name in ordered_model_names:
            if model_name in mae_distances:  # Case 1: Conditioned model
                for metric in distribution_metrics:
                    prop_distance_data[model_name][metric] = mae_distances[
                        model_name
                    ].get(metric, np.nan)
            else:  # Case 2: Unconditioned model
                if model_name == "Original data":
                    for metric in distribution_metrics:
                        prop_distance_data[model_name][metric] = 0.0
                    continue

                for metric in distribution_metrics:
                    model_values = aggregated_results[metric].get(model_name, [])
                    ref_values = original_data_dist.get(metric, [])
                    if model_values and ref_values:
                        prop_distance_data[model_name][metric] = wasserstein_distance(
                            model_values, ref_values
                        )
                    else:
                        prop_distance_data[model_name][metric] = np.nan

        prop_distance_df = pd.DataFrame.from_dict(prop_distance_data, orient="index")
        prop_distance_df.index.name = "Model"
        prop_distance_df = prop_distance_df.reindex(ordered_model_names)[
            distribution_metrics
        ]
        wd_path = self.plot_and_metrics_dir / f"{run_type}_wasserstein_summary.csv"
        prop_distance_df.to_csv(wd_path, float_format="%.4f")
        logger.info(f"Property distance (WD/MAE) summary saved to {wd_path}")
