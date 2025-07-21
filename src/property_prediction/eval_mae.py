"""
Evaluate a pretrained MolPropModule on molecular property data and plot per-property MAE.

This script:
1. Loads a checkpointed model.
2. Prepares graph data from a CSV of SELFIES strings.
3. Splits into training and validation sets.
4. Computes predictions and true values.
5. Calculates mean absolute error per property, denormalising if desired.
6. Plots and saves a bar chart of per-property MAE.
7. Writes results to CSV.
"""

import json
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import logging
from torch_geometric.loader import DataLoader

from .gcnn import MolPropModule
from .graph_utils import prepare_graph_dataset, split_and_load

logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _denormalise(arr: np.ndarray, mean: pd.Series, std: pd.Series) -> np.ndarray:
    """Undo column-wise z-score normalization."""
    return arr * std.values + mean.values


def _collect_preds(
    model: torch.nn.Module,
    loader: DataLoader,
    n_props: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Gather true and predicted values from the model over a DataLoader.

    Args:
        model: the PyTorch model to evaluate
        loader: DataLoader yielding batches with `.y` and graph attributes
        n_props: number of property columns in each target vector

    Returns:
        y_true: stacked true values, shape (N, n_props)
        y_pred: stacked model predictions, shape (N, n_props)
    """
    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            y_true.append(batch.y.view(-1, n_props).cpu().numpy())
            y_pred.append(model(batch).cpu().numpy())
    return np.vstack(y_true), np.vstack(y_pred)


def _evaluate(
    checkpoint: str,
    csv_path: str,
    prop_cols: Sequence[str],
    batch_size: int = 256,
    denorm: bool = True,
) -> pd.DataFrame:
    """
    Run evaluation on validation split and save a bar plot of per-property MAE.

    Args:
        checkpoint: path to the .ckpt file for the LightningModule
        csv_path: path to CSV with columns 'smiles' and `prop_cols`
        prop_cols: list of property column names to predict
        batch_size: batch size for DataLoader
        denorm: whether to undo z-score normalization on outputs

    Returns:
        DataFrame with columns ['property', 'MAE'] sorted by MAE ascending
    """
    df = pd.read_csv(csv_path)
    data_list, mean, std = prepare_graph_dataset(df, prop_cols, normalize=True)
    _, val_loader = split_and_load(data_list, batch_size=batch_size, val_ratio=0.2)
    lit = MolPropModule.load_from_checkpoint(checkpoint)
    lit.model.to(DEVICE)
    y_true, y_pred = _collect_preds(lit.model, val_loader, len(prop_cols))

    if denorm:
        y_true = _denormalise(y_true, mean, std)
        y_pred = _denormalise(y_pred, mean, std)

    mae = np.mean(np.abs(y_true - y_pred), axis=0)
    df_mae = pd.DataFrame({"property": prop_cols, "MAE": mae}).sort_values("MAE")

    fig, ax = plt.subplots(figsize=(10, 4))
    df_mae.plot.bar(x="property", y="MAE", ax=ax, legend=False)
    ax.set_ylabel("MAE on validation set")
    ax.set_xlabel("")
    ax.set_title("Per-property MAE")
    fig.tight_layout()

    output_dir = Path(csv_path).parent.parent / "metric_plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "val_mae_per_property.png"
    fig.savefig(plot_path, dpi=150)
    logger.info(f"Saved MAE plot to {plot_path}")

    return df_mae


if __name__ == "__main__":
    CHECKPOINT = "/scratch/.../property_prediction.ckpt"
    CSV = "/scratch/.../filtered_selfies.csv"
    PROP_COLS = [
        "nbo_P", "nmr_P", "pyr_P", "fmo_mu", "vmin_r", "volume", "fmo_eta",
        "fukui_m", "fukui_p", "nuesp_P", "somo_rc", "nbo_P_rc", "pyr_alpha",
        "qpole_amp", "vbur_vbur", "Pint_P_min", "sterimol_L", "sterimol_B1",
        "sterimol_B5", "dipolemoment", "efgtens_xx_P", "efgtens_yy_P",
        "nbo_bd_e_max", "nbo_lp_P_occ", "qpoletens_yy", "E_solv_elstat",
        "nbo_bds_e_avg", "sterimol_burL", "nbo_bd_occ_avg", "sterimol_burB5",
        "vbur_ovbur_min", "vbur_qvbur_min", "nbo_bds_occ_max",
        "vbur_ratio_vbur_vtot", "mol_wt", "sa_score",
    ]

    table = _evaluate(CHECKPOINT, CSV, PROP_COLS)
    out_csv = Path(CSV).parent.parent / "val_mae_per_property.csv"
    table.to_csv(out_csv, index=False)
    logger.info(f"Saved MAE table to {out_csv}")
