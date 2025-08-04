import json
import logging
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from torch.utils.data import random_split
from torch_geometric.data import Data as GraphData
from torch_geometric.loader import DataLoader

logger = logging.getLogger(__name__)


def _one_hot_encode(
    value: Any, allowable_set: List[Any], check_validity: bool = False
) -> List[bool]:
    """
    Return a one-hot encoding for `value` against `allowable_set`.

    Args:
        value: Element to encode.
        allowable_set: Sequence of allowed values.
        check_validity: If True, invalid value raises ValueError.
    """
    if value not in allowable_set:
        if check_validity:
            raise ValueError(f"{value} not in allowable set {allowable_set}")
        value = allowable_set[-1]
    return [value == item for item in allowable_set]


def _atom_features(
    atom: Chem.rdchem.Atom,
    available_atoms: List[str],
    extra_features: Optional[List[List[float]]] = None,
    atom_idx: Optional[int] = None,
    explicit_h: bool = True,
    use_chirality: bool = True,
) -> List[Union[bool, float]]:
    """
    Generate feature vector for an atom.

    Args:
        atom: RDKit Atom instance.
        available_atoms: List of element symbols.
        extra_features: Optional additional feature lists.
        atom_idx: Index for extra_features lookup.
        explicit_h: Include explicit hydrogen count features.
        use_chirality: Include chirality features.
    """
    hybridizations = [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ]
    features: List[Union[bool, float]] = []
    features += _one_hot_encode(atom.GetSymbol(), available_atoms)
    features += _one_hot_encode(atom.GetDegree(), list(range(11)), check_validity=True)
    features += _one_hot_encode(atom.GetImplicitValence(), list(range(7)))
    features += [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()]
    features += _one_hot_encode(atom.GetHybridization(), hybridizations)
    features += [atom.GetIsAromatic()]
    if extra_features and atom_idx is not None:
        logger.info("Adding extra atom features")
        features += extra_features[atom_idx]
    if not explicit_h:
        features += _one_hot_encode(atom.GetTotalNumHs(), list(range(5)))
    if use_chirality:
        cip = atom.GetProp("_CIPCode") if atom.HasProp("_CIPCode") else None
        features += _one_hot_encode(cip, ["R", "S"])
        features += [atom.HasProp("_ChiralityPossible")]
    return features


def _bond_features(
    bond: Chem.rdchem.Bond,
    extra_features: Optional[List[List[float]]] = None,
    bond_idx: Optional[int] = None,
    use_chirality: bool = False,
) -> List[Union[bool, float]]:
    """
    Generate feature vector for a bond.

    Args:
        bond: RDKit Bond instance.
        extra_features: Optional additional feature lists.
        bond_idx: Index for extra_features lookup.
        use_chirality: Include stereochemistry features.
    """
    bt = bond.GetBondType()
    feats: List[Union[bool, float]] = [
        bt == Chem.rdchem.BondType.SINGLE,
        bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE,
        bt == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing(),
    ]
    if extra_features and bond_idx is not None:
        logger.info("Adding extra bond features")
        feats += extra_features[bond_idx]
    if use_chirality:
        stereo = str(bond.GetStereo())
        feats += _one_hot_encode(
            stereo, ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"]
        )
    return feats


def _get_bond_pair(mol: Chem.rdchem.Mol) -> Tuple[List[int], List[int]]:
    """
    Return paired bond indices for graph edges.
    """
    begin, end = ([], [])
    for bond in mol.GetBonds():
        i, j = (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        begin += [i, j]
        end += [j, i]
    return (begin, end)


def n_atom_features(available_atoms: List[str], extra: int = 0) -> int:
    """Return dimension of atom feature vector."""
    atom = Chem.MolFromSmiles("C").GetAtomWithIdx(0)
    return len(_atom_features(atom, available_atoms)) + extra


def n_bond_features(extra: int = 0) -> int:
    """Return dimension of bond feature vector."""
    bond = Chem.MolFromSmiles("CC").GetBondWithIdx(0)
    return len(_bond_features(bond)) + extra


def _get_molecule_info(
    mol: Union[str, Chem.rdchem.Mol], available_atoms: List[str]
) -> Tuple[
    List[List[Union[bool, float]]],
    Tuple[List[int], List[int]],
    List[List[Union[bool, float]]],
]:
    """
    Extract node and edge features from a molecule.

    Args:
        mol: SMILES string or RDKit Mol.
        available_atoms: List of element symbols.
    Returns:
        node features, edge indices, edge features
    """
    if isinstance(mol, str):
        mol = Chem.MolFromSmiles(mol)
    atoms = mol.GetAtoms()
    bonds = mol.GetBonds()
    node_feats = [_atom_features(a, available_atoms) for a in atoms]
    edge_idx = _get_bond_pair(mol)
    bond_feats = [_bond_features(b) for b in bonds]
    return (node_feats, edge_idx, bond_feats + bond_feats)


def get_molecule_graph(
    mol: Union[str, Chem.rdchem.Mol],
    available_atoms: List[str],
    dtype: torch.dtype = torch.float,
) -> GraphData:
    """
    Build a PyG GraphData object from a molecule.

    Args:
        mol: SMILES string or RDKit Mol.
        available_atoms: List of element symbols.
        dtype: Torch tensor data type.
    """
    node_feats, edge_idx, edge_attr = _get_molecule_info(mol, available_atoms)
    x = torch.tensor(node_feats, dtype=dtype)
    edge_index = torch.tensor(edge_idx, dtype=torch.long)
    edge_attr = torch.tensor(edge_attr, dtype=dtype)
    return GraphData(x=x, edge_index=edge_index, edge_attr=edge_attr)


def to_serializable(obj: Any) -> Any:
    """
    Convert numpy, pandas, or torch objects to JSON-serializable types.
    """
    if isinstance(obj, np.generic):
        return obj.item()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, (dict, list, int, float, str, bool)) or obj is None:
        return obj
    raise TypeError(f"Type {type(obj)} not JSON serializable")


def prepare_graph_dataset(
    df: pd.DataFrame,
    prop_columns: List[str],
    normalize: bool = True,
    stats_path: Optional[str] = None,
) -> Tuple[List[GraphData], pd.Series, pd.Series]:
    """
    Convert DataFrame to graph dataset with optional normalization.

    Args:
        df: Input DataFrame with a 'smiles' column.
        prop_columns: Columns to normalize and attach as features.
        normalize: Whether to z-score normalize properties.
        stats_path: Path to save normalization stats JSON.
    Returns:
        list of GraphData, mean series, std series
    """
    props = df[prop_columns].astype(np.float32)
    if normalize:
        mean, std = (props.mean(), props.std())
        props = (props - mean) / std
    else:
        mean = pd.Series(0.0, index=prop_columns)
        std = pd.Series(1.0, index=prop_columns)
    if stats_path:
        stats = {"mean": mean.to_dict(), "std": std.to_dict()}
        p = Path(stats_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(stats, indent=4))
        logger.info(f"Saved normalization stats to {p}")
    dataset = []
    for idx, row in df.iterrows():
        graph = get_molecule_graph(row["smiles"], row[prop_columns].tolist())
        dataset.append(graph)
    return (dataset, mean, std)


def split_and_load(
    data_list: List[GraphData],
    batch_size: int = 32,
    val_ratio: float = 0.2,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """
    Split dataset into train/validation loaders.

    Args:
        data_list: List of GraphData objects.
        batch_size: Batch size for loaders.
        val_ratio: Fraction of data for validation.
        num_workers: DataLoader worker count.
    Returns:
        train and validation DataLoaders.
    """
    total = len(data_list)
    val_count = int(val_ratio * total)
    train_count = total - val_count
    train_ds, val_ds = random_split(data_list, [train_count, val_count])
    return (
        DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
        ),
        DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
        ),
    )
