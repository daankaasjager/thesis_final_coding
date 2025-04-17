import os
import re
from collections import Counter
import logging

logger = logging.getLogger(__name__)

import matplotlib.pyplot as plt
import selfies

from rdkit import Chem
from rdkit.Chem import RDConfig
from rdkit.Chem import Crippen
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Draw

import sys
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer



def remove_bos_eos_tokens(sample: str) -> str:
    tokens = re.findall(r'\[[^\]]*\]', sample)
    return "".join(tok for tok in tokens if tok not in ("[BOS]", "[EOS]"))


def compute_standard_metric(mol, metric: str) -> float:
    if metric == 'sascore':
        return sascorer.calculateScore(mol)
    elif metric == 'molweight':
        return rdMolDescriptors.CalcExactMolWt(mol)
    elif metric == 'logp':
        return Crippen.MolLogP(mol)
    elif metric == 'num_rings':
        return rdMolDescriptors.CalcNumRings(mol)
    else:
        raise ValueError(f"Unsupported metric: {metric}")


def compute_token_frequency(config, samples, name):
    """
    Creates a bar chart of token frequency for the given 'samples'.
    Saves as 'token_frequency_histogram_{name}.png'.
    Returns None (since there's no numeric "per-molecule" result).
    """
    if not config.plot_dist:
        return

    token_pattern = re.compile(r'\[[^\]]*\]')
    token_counts = Counter()
    
    for sample in samples:
        tokens = token_pattern.findall(sample)
        token_counts.update(tokens)

    if not token_counts:
        return

    tokens, counts = zip(*token_counts.most_common())
    total = sum(counts)
    norm_counts = [count / total for count in counts]

    plt.figure(figsize=(12, 6))
    plt.bar(tokens, norm_counts)
    plt.xticks(rotation=90)
    plt.title(f"Normalized Token Frequency Distribution ({name})")
    plt.ylabel("Normalized Frequency")
    plt.tight_layout()

    save_path = os.path.join(config.directory_paths.metrics_dir, f"token_frequency_histogram_{name}.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def compute_length_distribution(config, samples, name):
    """
    Creates a histogram of the number of tokens for each molecule in 'samples'.
    Saves as 'molecule_length_histogram_{name}.png'.
    Returns None.
    """
    if not config.plot_dist:
        return

    token_pattern = re.compile(r'\[[^\]]*\]')
    lengths = []
    for sample in samples:
        tokens = token_pattern.findall(sample)
        lengths.append(len(tokens))

    if not lengths:
        return

    plt.figure(figsize=(10, 5))
    bins = range(min(lengths), max(lengths) + 2)
    plt.hist(lengths, bins=bins, align='left', edgecolor='black')
    plt.title(f"Histogram of Molecule Lengths ({name})")
    plt.xlabel("Number of Tokens")
    plt.ylabel("Frequency")
    plt.tight_layout()

    save_path = os.path.join(config.directory_paths.metrics_dir, f"molecule_length_histogram_{name}.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def synthesize_molecule(mol, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    img = Draw.MolToImage(mol, size=(300, 300))
    img.save(f"{filename}.png")


def calculate_and_plot_metrics(config,
    samples,
    metrics,
    name: str = "default",
    use_moses: bool = False
):
    """
    For each item in `metrics`, we check:
      - If it's 'token_frequency' or 'length_distribution', we plot the distribution & return None.
      - Otherwise, we treat it as a standard RDKit metric ('sascore','num_rings', etc.).
    
    Returns a dict with numeric results for all standard metrics:
      { 'sascore': [...], 'num_rings': [...], ... }
    (Pseudo-metrics like 'token_frequency' do not appear here.)
    """

    # A dict to hold numeric metrics
    numeric_results = {}

    # Precompute RDKit Mols only once if we have any "standard" metrics
    standard_metrics = [m for m in metrics if m not in ('token_frequency', 'length_distribution')]
    valid_mols = []
    failed_smiles = 0

    # Only if we have standard metrics, convert samples -> Mols
    if standard_metrics:
        for selfies_str in samples:
            cleaned = remove_bos_eos_tokens(selfies_str)
            smiles = selfies.decoder(cleaned)
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                failed_smiles += 1
            else:
                valid_mols.append(mol)

        if use_moses:
            import moses
            moses_metrics = moses.get_all_metrics(valid_mols)
            # to do

        logger.info(f"[{name.upper()}] - Standard metric Mols: {len(valid_mols)} valid, {failed_smiles} failed.\n")

    # Now iterate over requested metrics
    for metric in metrics:
        if metric == 'token_frequency':
            compute_token_frequency(config, samples, name)
            continue

        if metric == 'length_distribution':
            compute_length_distribution(config, samples, name)
            continue

        # Standard RDKit metrics
        metric_values = []
        synth_counter = 0
        for mol in valid_mols:
            value = compute_standard_metric(mol, metric)
            if value is None:
                # print the failed smiles molecule
                logger.info(f"[{name.upper()}] Invalid {metric} value for molecule: {Chem.MolToSmiles(mol)}")
            else:
                metric_values.append(value)

            # e.g. if metric is sascore < 4 => save image
            if metric == 'sascore' and value != None and value < 4:
                outpath = os.path.join(
                    config.directory_paths.synthesize_dir,
                    f"synthesized_{name}_{synth_counter}"
                )
                synthesize_molecule(mol, outpath)
                synth_counter += 1
                if synth_counter >= 100:
                    break

        # Store the results in the dictionary
        numeric_results[metric] = metric_values
        logger.info(f"[{name.upper()}] Metric: {metric} -> {len(metric_values)} values.")   
        if len(metric_values) > 0:
            avg_val = sum(metric_values) / len(metric_values)
            logger.info(f"[{name.upper()}] Metric: {metric}")
            logger.info(f"  Average {metric}: {avg_val:.3f}")
            logger.info(f"  Count: {len(metric_values)}")

            if config.plot_dist:
                plt.figure(figsize=(10, 5))
                plt.hist(metric_values, bins=50, edgecolor='black', alpha=0.75, density=True)
                plt.axvline(avg_val, color='red', linestyle='dotted', linewidth=2,
                            label=f"Mean={avg_val:.2f}")
                plt.title(f"Distribution of {metric.title()} ({name})")
                plt.xlabel(metric.title())
                plt.ylabel("Frequency")
                plt.legend()
                plt.tight_layout()

                save_path = os.path.join(
                    config.directory_paths.metrics_dir,
                    f"{metric}_distribution_{name}.png"
                )
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path)
                plt.close()
        else:
            print(f"[{name.upper()}] Metric: {metric} -> No valid values.")

    return numeric_results
