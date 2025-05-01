import os
import re
import logging
from collections import Counter
from typing import Dict
import matplotlib.pyplot as plt
import selfies

from rdkit import Chem
from rdkit.Chem import RDConfig, Crippen, rdMolDescriptors, Draw

import sys
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

logger = logging.getLogger(__name__)


def _remove_bos_eos_tokens(sample: str) -> str:
    tokens = re.findall(r'\[[^\]]*\]', sample)
    return "".join(tok for tok in tokens if tok not in ("[BOS]", "[EOS]"))


def _compute_standard_metric(mol, metric: str) -> float:
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


def _compute_token_frequency(config, samples, name):
    if not config.eval.plot_dist:
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

    save_path = os.path.join(config.local_paths.metrics_dir, f"token_frequency_histogram_{name}.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def _compute_length_distribution(config, samples, name):
    if not config.eval.plot_dist:
        return

    token_pattern = re.compile(r'\[[^\]]*\]')
    lengths = [len(token_pattern.findall(s)) for s in samples]
    if not lengths:
        return

    plt.figure(figsize=(10, 5))
    bins = range(min(lengths), max(lengths) + 2)
    plt.hist(lengths, bins=bins, align='left', edgecolor='black')
    plt.title(f"Histogram of Molecule Lengths ({name})")
    plt.xlabel("Number of Tokens")
    plt.ylabel("Frequency")
    plt.tight_layout()

    save_path = os.path.join(config.local_paths.metrics_dir, f"molecule_length_histogram_{name}.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def _synthesize_molecule(mol, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    img = Draw.MolToImage(mol, size=(300, 300))
    img.save(f"{filename}.png")


def _process_molecules(samples, metrics, use_moses, name):
    standard_metrics = [m for m in metrics if m not in ('token_frequency', 'length_distribution')]
    valid_mols = []
    failed = 0

    if not standard_metrics:
        return [], []

    for selfies_str in samples:
        cleaned = _remove_bos_eos_tokens(selfies_str)
        smiles = selfies.decoder(cleaned)
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            valid_mols.append(mol)
        else:
            failed += 1

    if use_moses:
        import moses
        _ = moses.get_all_metrics(valid_mols)  # Placeholder

    logger.info(f"[{name.upper()}] - Standard metric Mols: {len(valid_mols)} valid, {failed} failed.\n")
    return valid_mols, standard_metrics


def _plot_metric_distribution(config, values, metric, name):
    if not config.eval.plot_dist or not values:
        return

    avg_val = sum(values) / len(values)
    plt.figure(figsize=(10, 5))
    plt.hist(values, bins=50, edgecolor='black', alpha=0.75, density=True)
    plt.axvline(avg_val, color='red', linestyle='dotted', linewidth=2, label=f"Mean={avg_val:.2f}")
    plt.title(f"Distribution of {metric.title()} ({name})")
    plt.xlabel(metric.title())
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(config.local_paths.metrics_dir, f"{metric}_distribution_{name}.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def _evaluate_standard_metrics(config, mols, metrics, name):
    results = {}
    for metric in metrics:
        values = []
        synth_count = 0

        for mol in mols:
            value = _compute_standard_metric(mol, metric)
            if value is not None:
                values.append(value)

                if metric == 'sascore' and value < 4:
                    outpath = os.path.join(
                        config.local_paths.synthesize_dir,
                        f"synthesized_{name}_{synth_count}"
                    )
                    _synthesize_molecule(mol, outpath)
                    synth_count += 1
                    if synth_count >= 100:
                        break

        results[metric] = values
        logger.info(f"[{name.upper()}] Metric: {metric} -> {len(values)} values.")

        if values:
            avg = sum(values) / len(values)
            logger.info(f"  Average {metric}: {avg:.3f}")
            logger.info(f"  Count: {len(values)}")
        else:
            logger.info(f"  No valid values for {metric}.")

        _plot_metric_distribution(config, values, metric, name)

    return results


def _plot_overlay_metric_distribution(config, metric, results_by_name):
    if not config.plot_dist:
        return

    plt.figure(figsize=(10, 5))

    for name, values in results_by_name.items():
        if not values:
            continue
        plt.hist(values, bins=50, alpha=0.5, density=True, label=name.title())

    plt.title(f"Overlay Distribution: {metric.title()}")
    plt.xlabel(metric.title())
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(config.local_paths.metrics_dir, f"{metric}_overlay_distribution.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def calculate_and_plot_metrics_multi(
    config,
    sample_dict: Dict[str, list],
    metrics: list,
    use_moses: bool = False
) -> Dict[str, Dict[str, list]]:
    """
    Given a dictionary {name: samples}, compute and optionally overlay metrics
    for all names in the dictionary (e.g. 'original', 'generated').

    Returns: nested dict of {metric: {name: values}}
    """
    all_results = {metric: {} for metric in metrics}

    for name, samples in sample_dict.items():
        result = calculate_and_plot_metrics(
            config=config,
            samples=samples,
            metrics=metrics,
            name=name,
            use_moses=use_moses
        )
        for metric in result:
            all_results[metric][name] = result[metric]

    for metric in all_results:
        _plot_overlay_metric_distribution(config, metric, all_results[metric])

    return all_results



def calculate_and_plot_metrics(config, samples, metrics, name: str = "default", use_moses: bool = False):
    numeric_results = {}

    if "token_frequency" in metrics:
        _compute_token_frequency(config, samples, name)

    if "length_distribution" in metrics:
        _compute_length_distribution(config, samples, name)

    mols, standard_metrics = _process_molecules(samples, metrics, use_moses, name)
    if standard_metrics and mols:
        numeric_results = _evaluate_standard_metrics(config, mols, standard_metrics, name)

    return numeric_results
