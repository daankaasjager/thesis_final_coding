# predict.py

import torch
import pandas as pd
import json
from pathlib import Path
from rdkit import Chem
from torch_geometric.data import DataLoader

from gcnn import MPNNet # We only need the nn.Module for inference
from gnn_dataset import _smiles_to_graph # Re-use the graph conversion utility

def predict_properties(smiles_list: list[str], model_dir: str) -> pd.DataFrame:
    """
    Predicts properties for a list of SMILES strings using a trained model.

    Args:
        smiles_list: A list of molecular SMILES strings.
        model_dir: The path to the directory containing the trained model artifacts
                   (e.g., './trained_model').

    Returns:
        A pandas DataFrame with the SMILES and their predicted properties.
    """
    model_path = Path(model_dir) / "pytorch"
    
    # 1. Load the trained model
    print("--> Loading pretrained model...")
    model = MPNNet.from_pretrained(str(model_path))
    model.eval() # Set model to evaluation mode

    # 2. Load the normalization statistics
    print("--> Loading normalization statistics...")
    with open(Path(model_dir) / "normalization.json", "r") as f:
        norm_stats = json.load(f)
    
    props_mean = pd.Series(norm_stats['mean'])
    props_std = pd.Series(norm_stats['std'])
    prop_names = list(props_mean.index)

    # 3. Prepare graph data for the new SMILES
    # Note: For properties, we pass a dummy array since it's not used in prediction.
    dummy_props = [0.0] * len(prop_names)
    data_list = [_smiles_to_graph(smi, dummy_props) for smi in smiles_list]
    data_list = [d for d in data_list if d is not None] # Filter out invalid SMILES

    if not data_list:
        print("No valid SMILES strings to predict.")
        return pd.DataFrame()

    loader = DataLoader(data_list, batch_size=len(data_list), shuffle=False)
    
    # 4. Perform prediction
    print(f"--> Predicting properties for {len(data_list)} molecules...")
    predictions_normalized = []
    with torch.no_grad():
        for batch in loader:
            output = model(batch)
            predictions_normalized.append(output.cpu())
    
    predictions_normalized = torch.cat(predictions_normalized).numpy()

    # 5. De-normalize the predictions to get the actual scale
    predictions = (predictions_normalized * props_std.values) + props_mean.values

    # 6. Format the results into a DataFrame
    results_df = pd.DataFrame(predictions, columns=prop_names)
    # Get original smiles for the valid graphs
    valid_smiles = [smi for smi, d in zip(smiles_list, data_list) if d is not None]
    results_df.insert(0, 'smiles', valid_smiles)

    return results_df


if __name__ == '__main__':
    # --- Example Usage ---
    
    # List of new molecules for which you want to predict properties
    new_smiles = [
        'CCO',          # Ethanol
        'c1ccccc1',     # Benzene
        'invalid-smiles', # This will be filtered out
        'CN1C=NC2=C1C(=O)N(C)C(=O)N2C' # Caffeine
    ]

    # Path to the directory where you saved your trained model
    trained_model_directory = './trained_model'

    # Get predictions
    predicted_properties_df = predict_properties(new_smiles, trained_model_directory)

    print("\n--> Prediction Results:")
    print(predicted_properties_df)