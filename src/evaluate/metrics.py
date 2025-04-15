from rdkit import Chem
from rdkit.Chem import RDConfig
from rdkit.Chem import Crippen
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Draw
import selfies
import sys
import os
import re
import matplotlib.pyplot as plt

sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

# ------------------------------------------------------------------------------
# Utility Function: Remove BOS/EOS tokens
# ------------------------------------------------------------------------------
def remove_bos_eos_tokens(sample: str) -> str:
    """
    Removes [BOS] and [EOS] tokens from a SELFIES string.
    
    Args:
        sample (str): A SELFIES string possibly containing [BOS] or [EOS] tokens.
        
    Returns:
        str: The cleaned SELFIES string with [BOS] and [EOS] removed.
    """
    tokens = re.findall(r'\[[^\]]*\]', sample)
    cleaned_tokens = [tok for tok in tokens if tok not in ("[BOS]", "[EOS]")]
    return ''.join(cleaned_tokens)

# ------------------------------------------------------------------------------
# Helper Function: Compute a chosen metric for an RDKit Mol
# ------------------------------------------------------------------------------
def compute_metric(mol, metric: str) -> float:
    """
    Given an RDKit Mol object and a specified metric, compute the metric value.
    
    Args:
        mol (rdkit.Chem.Mol): A valid RDKit Mol object
        metric (str): The metric to calculate ('sascore', 'molweight', 'logp')
        
    Returns:
        float: The computed metric value.
    """
    if metric == 'sascore':
        return sascorer.calculateScore(mol)
    elif metric == 'molweight':
        return rdMolDescriptors.CalcExactMolWt(mol)
    elif metric == 'logp':
        return Crippen.MolLogP(mol)
    else:
        raise ValueError(f"Unsupported metric: {metric}")

# ------------------------------------------------------------------------------
# Single Plotting Function for Any Metric
# ------------------------------------------------------------------------------
def plot_distribution(config, data_values, metric_name: str, name: str = "default"):
    """
    Plot a histogram of the given data values and save it.
    
    Args:
        config: A configuration object that holds directory path information.
        data_values (list): List of numeric values to plot.
        metric_name (str): The name of the metric (e.g., 'sascore', 'molweight', 'logp').
        name (str): Identifier (e.g., 'original', 'generated') for labeling/saving the plot.
    """
    if not data_values:
        print(f"No data to plot for metric '{metric_name}'.")
        return

    mean_value = sum(data_values) / len(data_values)
    
    plt.figure(figsize=(10, 5))
    plt.hist(data_values, bins=50, edgecolor='black', alpha=0.75, density=True)
    plt.axvline(mean_value, color='red', linestyle='dotted', linewidth=2,
                label=f"Mean = {mean_value:.2f}")
    plt.title(f"Distribution of {metric_name.title()} ({name})")
    plt.xlabel(metric_name.title())
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    
    save_path = f"{config.directory_paths.images_dir}{metric_name}_distribution_{name}.png"
    plt.savefig(save_path)
    plt.show()
    plt.close()

def synthesize_molecule(mol, filename):
    """
    Converts an RDKit Mol object to an image and saves it.

    Args:
        mol (rdkit.Chem.rdchem.Mol): The molecule to be visualized.
        filename (str): The base path where the image will be saved (without extension).
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Generate an image of the molecule
    img = Draw.MolToImage(mol, size=(300, 300))

    # Append '.png' extension
    filename_with_extension = f"{filename}.png"

    # Save the image
    img.save(filename_with_extension)

def calculate_and_plot_metrics(config, selfies_strings, metrics, name: str = "default", use_moses: bool = False):
    """
    Given a list of SELFIES strings, compute each metric (SAscore, MolWeight, LogP, etc.)
    for every valid molecule, then plot the distributions of all requested metrics.

    Args:
        config: A configuration object that holds directory path information.
        selfies_strings (list): List of SELFIES strings.
        metrics (list): A list of metrics to compute (e.g., ['sascore', 'molweight', 'logp']).
        name (str): Identifier (e.g., 'original', 'generated') for labeling.
        
    Returns:
        dict: A dictionary where each key is a metric and each value is a list of computed values.
    """
    # Prepare a dictionary to hold metric values: { 'sascore': [], 'molweight': [], 'logp': [] }
    metric_values_dict = {m: [] for m in metrics}
    
    failed_smiles_count = 0
    valid_mol_count = 0

    # First pass: convert SELFIES -> SMILES -> RDKit Mol, count failures
    valid_mols = []  # store RDKit Mol objects for further processing
    for selfies_str in selfies_strings:
        # 1) Clean out BOS/EOS tokens
        cleaned_selfies = remove_bos_eos_tokens(selfies_str)
        
        # 2) Convert SELFIES to SMILES
        smiles = selfies.decoder(cleaned_selfies)
        mol = Chem.MolFromSmiles(smiles)
        
        # 3) If invalid, increment fail count
        if mol is None:
            failed_smiles_count += 1
        else:
            valid_mols.append(mol)
    if use_moses:
        import moses
        metrics = moses.get_all_metrics(valid_mols)
        with open(os.path.join(config.directory_paths.images_dir, "moses_metrics.txt"), "w") as f:
            for metric in metrics:
                f.write(f"{metric}\n")

    valid_mol_count = len(valid_mols)

    # For each valid Mol, compute each requested metric
    counter= 0
    for mol in valid_mols:
        for metric in metrics:
            value = compute_metric(mol, metric)
            if (metric == 'sascore' and value <4):
                synthesize_molecule(mol, os.path.join(config.directory_paths.images_dir, str(counter)))
                counter +=1
            if counter > 100:
                return
            metric_values_dict[metric].append(value)

    # Print overarching success/failure
    print(f"Number of failed SMILES: {failed_smiles_count}")
    print(f"Number of valid SMILES: {valid_mol_count}\n")

    # Plot distributions and print stats for each metric
    for metric in metrics:
        values = metric_values_dict[metric]
        if values:
            avg_val = sum(values) / len(values)
            print(f"Metric: {metric}")
            print(f"  Average {metric}: {avg_val:.3f}")
            print(f"  Values count: {len(values)}")
        else:
            print(f"Metric: {metric}")
            print(f"  No valid values computed.")

        # Plot distribution for each metric
        plot_distribution(config, values, metric, name)
    
    return metric_values_dict
