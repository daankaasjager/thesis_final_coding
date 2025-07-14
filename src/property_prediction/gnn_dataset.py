"""This function is only used to convert SMILES strings
 into PyTorch Geometric Data objects, for molecular property prediction outside
 of the diffusion model."""

import torch
from rdkit import Chem
from torch_geometric.data import Data
from torch_geometric.utils import from_smiles
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
import pandas as pd
import numpy as np
import json
from pathlib import Path


def _smiles_to_graph(smiles: str, props: np.ndarray) -> Data | None:
    """Convert a SMILES string to a PyG graph with scalar property target."""
    data = from_smiles(smiles)
    data.y = torch.tensor(props, dtype=torch.float32)
    if data.edge_attr is not None and data.edge_attr.dtype != torch.float32:
        data.edge_attr = data.edge_attr.float()
    return data

def to_serializable(x):
    """
    Convert NumPy / Torch / pandas objects to JSON-friendly types.
    """
    if isinstance(x, (np.generic,)):        # numpy scalar (float32, int64, …)
        return x.item()
    if hasattr(x, "tolist"):                # ndarray, pandas Series, Torch tensor
        return x.tolist()
    if isinstance(x, (dict, list, int, float, str, bool)) or x is None:
        return x
    raise TypeError(f"{type(x)} is not JSON serializable")

def prepare_graph_dataset(
    df: pd.DataFrame,
    prop_columns: list[str],
    normalize: bool = True,
    stats_path: str | None = None,
):
    """
    Calculates normalization stats, saves them, and converts a DataFrame to a list of graph objects.
    """
    props = df[prop_columns].astype(np.float32)

    # --- Normalization ---
    if normalize:
        mean = props.mean()
        std = props.std()
        normed_props = (props - mean) / std
    else:
        mean = pd.Series(0.0, index=prop_columns)  # dummy values
        std = pd.Series(1.0, index=prop_columns)
        normed_props = props

    # --- Save stats ---
    if stats_path:
        # This saves the mean/std as a dictionary keyed by property name, which is more robust.
        stats_to_save = {"mean": mean.to_dict(), "std": std.to_dict()}
        path = Path(stats_path)
        path.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
        path.write_text(json.dumps(stats_to_save, indent=4))
        print(f"Normalization stats saved to {path}")

    data_list = []
    for i, row in df.iterrows():
        graph = _smiles_to_graph(row["smiles"], normed_props.iloc[i].values)
        if graph is not None:
            data_list.append(graph)

    return data_list, mean, std


def split_and_load(data_list, batch_size=32, val_ratio=0.2, num_workers=0):
    """Splits PyG Data objects into train/val loaders using torch random_split."""
    n = len(data_list)
    n_val = int(val_ratio * n)
    n_train = n - n_val

    train_data, val_data = random_split(data_list, [n_train, n_val])

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader
