import logging
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sympy import plot

logger = logging.getLogger(__name__)


def plot_cutoff_summary(metrics_dir: Path, df_percent: pd.DataFrame):

    plt.figure(figsize=(10, len(df_percent)*0.5 + 2))
    sns.heatmap(df_percent, annot=True, fmt=".1f", cmap="Reds")
    plt.title("Filter % per Hard Bound and Model")
    plt.xlabel("Model")
    plt.ylabel("Property")
    plt.tight_layout()

    heatmap_path = metrics_dir/ "hard_bound_cutoff_heatmap.png"
    plt.savefig(heatmap_path)
    plt.close()
    logger.info(f"Saved cutoff heatmap to {heatmap_path}")


def aggregate_cutoff_summaries(metrics_dir: Path):
    """
    Loads all cutoff summary CSVs in a metrics directory and aggregates counts.
    """
    all_files = list(metrics_dir.glob("*hard_bound_cutoff_summary.csv"))
    total_df = None

    for file in all_files:
        df = pd.read_csv(file, index_col=0)
        if total_df is None:
            total_df = df
        else:
            total_df = total_df.add(df, fill_value=0)

    total_df['total_cutoffs'] = total_df.sum(axis=1)
    df_percent = total_df.div(total_df['total_cutoffs'], axis=0) * 100
    plot_cutoff_summary(metrics_dir, df_percent)
    agg_path = metrics_dir / "aggregate_hard_bound_cutoff_summary.csv"
    total_df.to_csv(agg_path)
    logger.info(f"Saved aggregated hard bound cutoff counts to {agg_path}")

    agg_path_pct = metrics_dir / "aggregate_hard_bound_cutoff_summary_percent.csv"
    df_percent.to_csv(agg_path_pct)
    logger.info(f"Saved aggregated hard bound cutoff percentages to {agg_path_pct}")

    return total_df, df_percent
