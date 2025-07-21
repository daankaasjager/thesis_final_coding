import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def plot_cutoff_summary(metrics_dir: Path, df_percent: pd.DataFrame):
    """
    Plots a heatmap from a percent_filtered pivot table and saves it.
    """
    plt.figure(figsize=(10, len(df_percent) * 0.5 + 2))
    sns.heatmap(df_percent, annot=True, fmt=".2f", cmap="Reds")
    title = "Filter % per Property and Model"
    plt.title(title)
    plt.xlabel("Model")
    plt.ylabel("Property")
    plt.tight_layout()

    heatmap_path = metrics_dir / "hard_bound_cutoff_heatmap.png"
    plt.savefig(heatmap_path)
    plt.close()
    logger.info(f"Saved cutoff heatmap to {heatmap_path}")


def aggregate_model_summaries(base_dir: Path):
    """
    Aggregates all model summary CSVs from subdirectories and plots a single heatmap.
    """
    all_dfs = []

    for model_dir in base_dir.iterdir():
        if not model_dir.is_dir():
            continue

        csv_files = list(model_dir.glob("*hard_bound_cutoff_summary.csv"))
        if not csv_files:
            logger.warning(f"No summary file found in {model_dir}")
            continue

        df = pd.read_csv(csv_files[0])
        if df.empty:
            logger.warning(f"Empty summary file in {model_dir}")
            continue

        # Replace model name with directory name (and make it prettier for X-axis)
        model_name = model_dir.name.replace("_", " ")
        df["model"] = model_name
        all_dfs.append(df[["model", "property", "percent_filtered"]])

    if not all_dfs:
        logger.error("No valid summary data found.")
        return

    combined_df = pd.concat(all_dfs, ignore_index=True)
    df_pivot = combined_df.pivot(
        index="property", columns="model", values="percent_filtered"
    )

    # Desired order of models (after replacing _ with space)
    desired_order = [
        "model size tiny",
        "model size small",
        "ape 70",
        "ape 80",
        "ape 110",
        "prepend 1",
        "prepend 3",
        "prepend 8",
        "prepend all",
        "embedding 1",
        "embedding 3",
        "embedding 8",
        "embedding all",
        "0.3 cfg",
        "1.0 cfg",
        "4.0 cfg",
    ]

    # Keep only columns that exist in the pivot
    existing_order = [col for col in desired_order if col in df_pivot.columns]
    df_pivot = df_pivot[existing_order]

    plot_cutoff_summary(base_dir, df_pivot)
