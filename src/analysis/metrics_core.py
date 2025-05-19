import os
import re
import sys

import selfies
from rdkit import Chem
from rdkit.Chem import Crippen, RDConfig, rdMolDescriptors

sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer


def remove_bos_eos(sample: str):
    return "".join(
        tok
        for tok in re.findall(r"\[[^\]]+\]", sample)
        if tok not in ("[BOS]", "[EOS]")
    )


def get_valid_molecules(samples, name):
    mols = []
    for s in samples:
        smiles = selfies.decoder(remove_bos_eos(s))
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            mols.append(mol)
    return mols


def compute_standard_metric(mol, metric):
    return METRIC_REGISTRY.get(metric, lambda m: None)(mol)


METRIC_REGISTRY = {
    "sascore": sascorer.calculateScore,
    "logp": Crippen.MolLogP,
    "molweight": rdMolDescriptors.CalcExactMolWt,
    "num_rings": rdMolDescriptors.CalcNumRings,
}
