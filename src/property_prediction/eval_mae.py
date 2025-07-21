# eval_mae.py  delet this file

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from gcnn import MolPropModule  # your LightningModule

from property_prediction.graph_utils import (prepare_graph_dataset,
                                             split_and_load)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def denormalise(arr, mean, std):
    """Undo z-score normalisation column-wise."""
    return arr * std.values + mean.values


def collect_preds(model, loader, n_props):
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            y_true.append(batch.y.view(-1, n_props).cpu().numpy())
            y_pred.append(model(batch).cpu().numpy())
    return np.vstack(y_true), np.vstack(y_pred)


def evaluate(checkpoint, csv_path, prop_cols, batch_size=256, denorm=True):
    # 1) reload data exactly as in training -------------------------------
    df = pd.read_csv(csv_path)
    data_list, mean, std = prepare_graph_dataset(df, prop_cols, normalize=True)
    _, val_loader = split_and_load(data_list, batch_size=batch_size, val_ratio=0.2)

    # 2) load model -------------------------------------------------------
    lit = MolPropModule.load_from_checkpoint(checkpoint)
    lit.model.to(DEVICE).eval()

    # 3) predict ----------------------------------------------------------
    y, yhat = collect_preds(lit.model, val_loader, len(prop_cols))

    # 4) denormalise if you trained on z-scores ---------------------------
    if denorm:
        y = denormalise(y, mean, std)
        yhat = denormalise(yhat, mean, std)

    mae = np.mean(np.abs(y - yhat), axis=0)
    df_mae = pd.DataFrame({"property": prop_cols, "MAE": mae})
    fig, ax = plt.subplots(figsize=(10, 4))
    df_mae.sort_values("MAE").plot.bar(x="property", y="MAE", ax=ax, legend=False)
    ax.set_ylabel("MAE on validation set")
    ax.set_xlabel("")
    ax.set_title("Per-property MAE")
    plt.tight_layout()
    fig.savefig(
        "/scratch/s3905845/thesis_final_coding/data/kraken/metric_plots/val_mae_per_property.png",
        dpi=150,
    )
    print("finished evaluation")
    return df_mae


if __name__ == "__main__":
    CHECKPOINT = (
        "/scratch/s3905845/thesis_final_coding/checkpoints/property_prediction.ckpt"
    )
    CSV = "/scratch/s3905845/thesis_final_coding/data/kraken/training_data/filtered_selfies.csv"
    PROP_COLS = [
        "nbo_P",
        "nmr_P",
        "pyr_P",
        "fmo_mu",
        "vmin_r",
        "volume",
        "fmo_eta",
        "fukui_m",
        "fukui_p",
        "nuesp_P",
        "somo_rc",
        "nbo_P_rc",
        "pyr_alpha",
        "qpole_amp",
        "vbur_vbur",
        "Pint_P_min",
        "sterimol_L",
        "sterimol_B1",
        "sterimol_B5",
        "dipolemoment",
        "efgtens_xx_P",
        "efgtens_yy_P",
        "nbo_bd_e_max",
        "nbo_lp_P_occ",
        "qpoletens_yy",
        "E_solv_elstat",
        "nbo_bds_e_avg",
        "sterimol_burL",
        "nbo_bd_occ_avg",
        "sterimol_burB5",
        "vbur_ovbur_min",
        "vbur_qvbur_min",
        "nbo_bds_occ_max",
        "vbur_ratio_vbur_vtot",
        "mol_wt",
        "sa_score",
    ]

    table = evaluate(CHECKPOINT, CSV, PROP_COLS)
    table.to_csv(
        "/scratch/s3905845/thesis_final_coding/val_mae_per_property.csv", index=False
    )
    print(table)
