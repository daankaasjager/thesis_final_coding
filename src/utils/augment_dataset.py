import os
import pandas as pd
from rdkit import Chem
import selfies as sf
from tqdm import tqdm


def augment_dataset(config):
    # Load the original dataset
    df = pd.read_csv(config.local_paths.raw_data)

    augmented_rows = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        smiles = row['smiles']
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            continue  

        augmented_rows.append(row.to_dict())

        seen = set()
        attempts = 0
        while len(seen) < 10 and attempts < 50:
            rand_smiles = Chem.MolToSmiles(
                mol,
                canonical=False,
                doRandom=True,
                isomericSmiles=False,
                kekuleSmiles=True,
            )
            if rand_smiles in seen:
                attempts += 1
                continue

            try:
                rand_selfies = sf.encoder(rand_smiles)
            except Exception:
                attempts += 1
                continue

            seen.add(rand_smiles)

            new_row = row.copy()
            new_row['smiles'] = rand_smiles
            new_row['selfies'] = rand_selfies
            augmented_rows.append(new_row.to_dict())


    augmented_df = pd.DataFrame(augmented_rows)
    augmented_df.to_csv(config.local_paths.augmented_data, index=False)

