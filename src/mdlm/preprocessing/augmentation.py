import logging

import pandas as pd
import selfies as sf
from omegaconf import DictConfig
from rdkit import Chem
from tqdm import tqdm

logger = logging.getLogger(__name__)


def apply_augmentation(config: DictConfig, df: pd.DataFrame) -> pd.DataFrame:
    """
    Augments DataFrame with rearranged SMILES/SELFIES based on config settings.
    Operates in memory.

    Args:
        df (pd.DataFrame): Input DataFrame, must contain a 'smiles' column.
        config (DictConfig): Configuration object with preprocessing settings
                              (e.g., preprocessing.augment, preprocessing.augmentation_n).

    Returns:
        pd.DataFrame: DataFrame with original and augmented rows. If augmentation
                      is disabled in config or fails, returns the original DataFrame.
    """

    n_augmentations = config.preprocessing.augmentations_per_molecule
    max_attempts_per_mol = config.preprocessing.augmentations_max_attempts
    logger.info(
        f"Starting data augmentation (target: {n_augmentations} versions per molecule)..."
    )

    augmented_rows = []
    skipped_mols = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Augmenting molecules"):
        original_smiles = row["smiles"]
        try:
            mol = Chem.MolFromSmiles(original_smiles)
        except Exception as e:
            logger.warning(
                f"Could not parse SMILES '{original_smiles}': {e}. Skipping."
            )
            skipped_mols += 1
            continue
        if mol is None:
            logger.warning(
                f"Skipping invalid or unparseable SMILES for augmentation: {original_smiles}"
            )
            skipped_mols += 1
            continue
        augmented_rows.append(row.to_dict())
        seen_smiles = {original_smiles}
        attempts = 0

        while (
            len(seen_smiles) < (n_augmentations + 1) and attempts < max_attempts_per_mol
        ):
            attempts += 1
            try:
                rand_smiles = Chem.MolToSmiles(
                    mol,
                    canonical=False,
                    doRandom=True,
                    isomericSmiles=False,
                    kekuleSmiles=True,
                )

                if rand_smiles in seen_smiles:
                    continue

                rand_selfies = sf.encoder(rand_smiles)
                if rand_selfies is None:
                    logger.debug(
                        f"SELFIES encoding failed for augmented SMILES: {rand_smiles}"
                    )
                    raise ValueError("SELFIES encoding failed")

                seen_smiles.add(rand_smiles)
                new_row = row.copy()
                new_row["smiles"] = rand_smiles
                new_row["selfies"] = rand_selfies
                augmented_rows.append(new_row.to_dict())
                attempts = 0

            except Exception as e:
                logger.debug(
                    f"Skipping one augmentation attempt for {original_smiles} due to error: {e}"
                )
                continue

        if len(seen_smiles) < (n_augmentations + 1):
            logger.debug(
                f"Could only generate {len(seen_smiles)-1}/{n_augmentations} unique augmentations for {original_smiles} after {max_attempts_per_mol} attempts."
            )

    if not augmented_rows:
        logger.warning(
            "Augmentation resulted in zero rows. Returning original DataFrame."
        )
        return df

    augmented_df = pd.DataFrame(augmented_rows)
    logger.info(
        f"Data augmentation complete. Initial rows: {len(df)}, Augmented rows: {len(augmented_df)}. Skipped/invalid molecules: {skipped_mols}"
    )

    if "selfies" not in augmented_df.columns and "selfies" in df.columns:
        augmented_df["selfies"] = df["selfies"]
    elif "selfies" not in augmented_df.columns:
        logger.warning(
            "Augmented DataFrame is missing 'selfies' column after augmentation process."
        )

    return augmented_df
