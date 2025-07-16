"""This function is only used to convert SMILES strings
 into PyTorch Geometric Data objects, for molecular property prediction outside
 of the diffusion model."""

from venv import logger
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

# This part contains modified code written
# by Akshat Nigam for the KRAKEN project found
# at https://github.com/aspuru-guzik-group/kraken

from rdkit import Chem
from torch_geometric.data import Data as GraphData
import torch
import logging

logger  = logging.getLogger(__name__)


def one_hot_encode(x, allowable_set, check_validity=False):
    """
    Maps inputs to a one-hot encoding, according to a given list.
    If `check_validity` is set to true, the element not in the
    allowable set is set to the last element.
    """
    if x not in allowable_set:
        if check_validity:
            raise ValueError(f"{x} is not in the allowable set: {allowable_set}")
        x = allowable_set[-1]

    return list(map(lambda s: x == s, allowable_set))


def atom_features(
    atom,
    available_atoms: list[str],
    atomdata=[],
    atomidx=None,
    explicit_H=True,
    use_chirality=True,
):
    possible_hybridizations = [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ]
    results = []
    results += one_hot_encode(atom.GetSymbol(), available_atoms)
    results += one_hot_encode(
        atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], check_validity=True
    )
    results += one_hot_encode(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6])
    results += [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()]
    results += one_hot_encode(atom.GetHybridization(), possible_hybridizations)
    results += [atom.GetIsAromatic()]

    if len(atomdata) != 0:
        print("Add additional atomdata")
        results += atomdata[atomidx]

    # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
    if not explicit_H:
        results += one_hot_encode(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])

    if use_chirality:
        try:
            results += one_hot_encode(atom.GetProp("_CIPCode"), ["R", "S"])
        except KeyError:
            results += [False, False]
        results += [atom.HasProp("_ChiralityPossible")]

    return results


def bond_features(bond, bonddata=[], bondidx=None, use_chirality=False):
    bt = bond.GetBondType()

    bond_feats = [
        bt == Chem.rdchem.BondType.SINGLE,
        bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE,
        bt == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing(),
    ]
    if len(bonddata) != 0:
        logger.info("Adding additional bond data")
        bond_feats += bonddata[bondidx]
    if use_chirality:
        logger.info("Adding chirality bond data")
        bond_feats = bond_feats + one_hot_encode(
            str(bond.GetStereo()), ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"]
        )

    return bond_feats


def get_bond_pair(mol):
    bonds = mol.GetBonds()
    res = [[], []]
    for bond in bonds:
        res[0] += [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]
        res[1] += [bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()]
    return res


def n_atom_features(available_atoms: list[str], additional_atom_features: int = 0):
    atom = Chem.MolFromSmiles("C").GetAtomWithIdx(0)
    return len(atom_features(atom, available_atoms)) + additional_atom_features


def n_bond_features(additional_bond_features: int = 0):
    bond = Chem.MolFromSmiles("CC").GetBondWithIdx(0)
    return len(bond_features(bond)) + additional_bond_features


def get_molecule_info(
    mol, available_atoms
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sorted_atoms = sorted(available_atoms)

    atoms = mol.GetAtoms()
    bonds = mol.GetBonds()
    node_f = [atom_features(atom, sorted_atoms) for atom in atoms]

    edge_index = get_bond_pair(mol)
    edge_attr = [bond_features(bond, use_chirality=False) for bond in bonds]

    edge_attr = edge_attr + edge_attr

    return node_f, edge_index, edge_attr  # float, long, float


def get_molecule_graph(mol, available_atoms, dtype=torch.float):
    if isinstance(mol, str):
        mol = Chem.MolFromSmiles(mol)

    node_f, edge_index, edge_attr = get_molecule_info(mol, available_atoms)
    node_f = torch.tensor(node_f, dtype=dtype)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_attr = torch.tensor(edge_attr, dtype=dtype)

    return GraphData(x=node_f, edge_index=edge_index, edge_attr=edge_attr)


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
        graph = get_molecule_graph(row["smiles"], normed_props.iloc[i].values)
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
