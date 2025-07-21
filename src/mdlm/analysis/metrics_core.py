import os
import re
import sys
from collections import Counter
from typing import Dict, List, Tuple

import selfies
from rdkit import Chem
from rdkit.Chem import Crippen, RDConfig, rdMolDescriptors

sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer

TOKEN_PATTERN = re.compile(r"\[[^\]]+\]")


def compute_token_stats(samples: List[str]) -> Tuple[Counter, List[int]]:
    token_counts = Counter()
    lengths = []

    for s in samples:
        tokens = TOKEN_PATTERN.findall(s)
        token_counts.update(tokens)
        lengths.append(len(tokens))

    total_tokens = sum(token_counts.values())
    if total_tokens == 0:
        token_freqs = {tok: 0.0 for tok in token_counts}
    else:
        token_freqs = {
            tok: count / total_tokens * 100 for tok, count in token_counts.items()
        }

    return token_freqs, lengths


def clean_generated_data(sample: str, alphabet: List[str]) -> str:
    return "".join(
        tok
        for tok in TOKEN_PATTERN.findall(sample)
        if tok not in ("[BOS]", "[EOS]", "[PAD]", "[UNK]") and tok in alphabet
    )


def get_valid_molecules(samples: List[str], config) -> List[Chem.Mol]:
    mols = []
    try:
        with open(config.paths.selfies_alphabet, "r", encoding="utf-8") as f:
            raw_tokens = {line.strip() for line in f if line.strip()}
    except FileNotFoundError:
        raise ValueError(f"Alphabet file not found at {config.paths.selfies_alphabet}")

    for s in samples:
        cleaned_selfies = clean_generated_data(s, list(raw_tokens))
        try:
            smiles = selfies.decoder(cleaned_selfies)
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                mols.append(mol)
        except Exception:
            continue
    return mols


def compute_chemical_metrics(
    mols: List[Chem.Mol], metrics: List[str]
) -> Dict[str, List[float]]:
    results = {}
    for metric in metrics:
        values = []
        if metric in METRIC_REGISTRY:
            for mol in mols:
                try:
                    val = METRIC_REGISTRY[metric](mol)
                    if val is not None:
                        values.append(val)
                except Exception:
                    pass
        results[metric] = values
    return results


def compute_validity(
    original_samples: List[str], valid_molecules: List[Chem.Mol]
) -> float:
    if not original_samples:
        return 0.0
    return len(valid_molecules) / len(original_samples) * 100.0


def compute_uniqueness(canonical_smiles: List[str]) -> float:
    if not canonical_smiles:
        return 0.0
    unique_smiles = set(canonical_smiles)
    return len(unique_smiles) / len(canonical_smiles) * 100.0


def compute_novelty(
    generated_canonical_smiles: List[str], original_canonical_smiles: List[str]
) -> float:
    if not generated_canonical_smiles:
        return 0.0

    unique_generated_smiles = set(generated_canonical_smiles)
    original_smiles_set = set(original_canonical_smiles)

    novel_smiles = unique_generated_smiles - original_smiles_set

    return len(novel_smiles) / len(unique_generated_smiles) * 100.0


METRIC_REGISTRY = {
    "sascore": sascorer.calculateScore,
    "logp": Crippen.MolLogP,
    "molweight": rdMolDescriptors.CalcExactMolWt,
    "num_rings": rdMolDescriptors.CalcNumRings,
    "tpsa": rdMolDescriptors.CalcTPSA,
    "tetrahedral_carbons": rdMolDescriptors.CalcFractionCSP3,
}
