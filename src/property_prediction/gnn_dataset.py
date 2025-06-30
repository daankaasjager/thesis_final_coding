"""This function is only used to convert SMILES strings
 into PyTorch Geometric Data objects, for molecular property prediction outside
 of the diffusion model."""

import torch
from rdkit import Chem
from torch_geometric.data import Data
from torch_geometric.utils import from_rdkit_mol
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


def _mol_to_graph(smiles: str, props: np.ndarray) -> Data | None:
    """Convert a SMILES string to a PyG graph with scalar property target."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    data = from_rdkit_mol(mol, use_explicit_H=True, use_chirality=True)
    data.y = torch.tensor(props, dtype=torch.float32)
    return data



def prepare_graph_dataset(df: pd.DataFrame, prop_columns: list[str], normalize=True):
    """Converts a DataFrame with SMILES and property columns into PyG Data objects."""
    props = df[prop_columns].astype(np.float32)

    if normalize:
        # Normalize properties
        mean = props.mean()
        std = props.std()
        normed_props = (props - mean) / std
        

    data_list = []
    for i, row in df.iterrows():
        graph = _mol_to_graph(row["smiles"], normed_props.iloc[i].values)
        if graph is not None:mo
            data_list.append(graph)

    return data_list, mean, std


def split_and_load(data_list, batch_size=32, val_ratio=0.2, num_workers=0):
    """Splits PyG Data objects into train/val loaders."""
    train_data, val_data = train_test_split(data_list, test_size=val_ratio, shuffle=True, random_state=42)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader
