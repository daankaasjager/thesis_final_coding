import logging
from collections import defaultdict

import numpy as np
import rdkit.Chem as Chem
from fcd import get_fcd

from .metrics_core import (
    compute_chemical_metrics,
    compute_novelty,
    compute_token_stats,
    compute_uniqueness,
    compute_validity,
    get_valid_molecules,
)
from .metrics_plotting import MetricPlotter

logger = logging.getLogger(__name__)


def _split_samples(sample_objects):
    """Helper to split sample objects into SELFIES and property dictionaries."""
    selfies, props = ([], [])
    for x in sample_objects:
        if isinstance(x, str):
            selfies.append(x)
            props.append(None)
        elif isinstance(x, dict):
            selfies.append(x.get("selfies", ""))
            props.append(x.get("predicted_properties"))
        else:
            continue
    return (selfies, props)


class MetricRunner:

    def __init__(self, config):
        self.plotter = MetricPlotter(config)
        self.config = config
        self.reference_name = ""
        self.original_smiles = []

    def _calculate_mae_for_dataset(self, samples, properties):
        """
        Calculates the Mean Absolute Error between generated properties and conditioning targets.
        """
        mae_results = {}
        all_prop_pairs = defaultdict(lambda: {"predicted": [], "target": []})
        for sample in samples:
            if not isinstance(sample, dict):
                continue
            predicted_props = sample.get("predicted_properties")
            target_props = sample.get("conditioning_properties")
            if not isinstance(predicted_props, dict) or not isinstance(
                target_props, dict
            ):
                continue
            for prop_key in properties:
                predicted_val = predicted_props.get(prop_key)
                target_val = target_props.get(prop_key)
                if predicted_val is not None and target_val is not None:
                    try:
                        all_prop_pairs[prop_key]["predicted"].append(
                            float(predicted_val)
                        )
                        all_prop_pairs[prop_key]["target"].append(float(target_val))
                    except (ValueError, TypeError):
                        continue
        for prop_key, values in all_prop_pairs.items():
            if values["predicted"]:
                pred_arr = np.array(values["predicted"])
                targ_arr = np.array(values["target"])
                mae = np.mean(np.abs(pred_arr - targ_arr))
                mae_results[prop_key] = mae
            else:
                mae_results[prop_key] = np.nan
        return mae_results

    def _process_dataset(self, samples, name, metrics, property_list):
        results = {}
        selfies, props_dicts = _split_samples(samples)
        if any(props_dicts):
            prop_values = defaultdict(list)
            for pd in props_dicts:
                if pd is None:
                    continue
                for prop in property_list:
                    val = pd.get(prop)
                    if val is not None:
                        prop_values[prop].append(float(val))
            for prop, vals in prop_values.items():
                results[prop] = vals
        token_based_metrics = [
            m for m in metrics if m in ("token_frequency", "length_distribution")
        ]
        if token_based_metrics:
            token_counts, lengths = compute_token_stats(selfies)
            if "token_frequency" in metrics:
                results["token_frequency"] = token_counts
            if "length_distribution" in metrics:
                results["length_distribution"] = lengths
        mols = get_valid_molecules(selfies, self.config)
        valid_smiles = [Chem.MolToSmiles(mol, canonical=True) for mol in mols if mol]
        if "validity" in metrics:
            results["validity"] = compute_validity(selfies, mols)
            logger.info(f"[{name}] Validity: {results['validity']:.3f}")
        if "uniqueness" in metrics:
            results["uniqueness"] = compute_uniqueness(valid_smiles)
            logger.info(f"[{name}] Uniqueness: {results['uniqueness']:.3f}")
        if "novelty" in metrics and name != "Original data":
            if not self.original_smiles:
                logger.warning("Original SMILES not available for novelty calculation.")
            else:
                results["novelty"] = compute_novelty(valid_smiles, self.original_smiles)
                logger.info(f"[{name}] Novelty: {results['novelty']:.3f}")
        chemical_metrics_list = [
            m
            for m in metrics
            if m
            not in (
                "validity",
                "uniqueness",
                "novelty",
                "token_frequency",
                "length_distribution",
            )
            and m not in property_list
        ]
        if chemical_metrics_list and mols:
            chem_results = compute_chemical_metrics(mols, chemical_metrics_list)
            for metric, values in chem_results.items():
                results[metric] = values
        return (results, valid_smiles)

    def run_multi(self, sample_dict, metrics, properties, reference_name, run_type):
        logger.info(
            f"Evaluating datasets (run_type='{run_type}', reference='{reference_name}')"
        )
        self.reference_name = reference_name
        aggregated = defaultdict(lambda: defaultdict(list))
        canonicalized = {}
        mae_distances = defaultdict(dict)
        original_samples = sample_dict.get("Original data", [])
        if not original_samples:
            logger.error(
                "'Original data' not found in samples. Cannot compute novelty."
            )
            self.original_smiles = []
        else:
            orig_selfies, _ = _split_samples(original_samples)
            orig_mols = get_valid_molecules(orig_selfies, self.config)
            self.original_smiles = [
                Chem.MolToSmiles(m, canonical=True) for m in orig_mols if m
            ]
        all_distribution_metrics = [
            m
            for m in metrics + properties
            if m
            not in (
                "validity",
                "uniqueness",
                "novelty",
                "token_frequency",
                "length_distribution",
            )
        ]
        for name, samples in sample_dict.items():
            if not samples:
                continue
            ds_results, canon_smiles = self._process_dataset(
                samples, name, metrics, properties
            )
            canonicalized[name] = canon_smiles
            for metric_name, value in ds_results.items():
                aggregated[metric_name][name] = value
            is_conditioned_model = (
                run_type.startswith("conditioning")
                and name != self.reference_name
                and (name != "Original data")
                and any(
                    (
                        s.get("conditioning_properties")
                        for s in samples
                        if isinstance(s, dict)
                    )
                )
            )
            if is_conditioned_model:
                logger.info(f"Calculating MAE for conditioned model: {name}")
                mae_distances[name] = self._calculate_mae_for_dataset(
                    samples, all_distribution_metrics
                )
        fcd_scores = {}
        ref_smiles_for_fcd = canonicalized.get(reference_name, [])
        if ref_smiles_for_fcd:
            for name, gen_smiles in canonicalized.items():
                if name != reference_name and gen_smiles:
                    fcd_scores[name] = get_fcd(gen_smiles, ref_smiles_for_fcd)
        self.plotter.display_statistical_summary(
            aggregated, fcd_scores, self.reference_name, run_type, mae_distances
        )
        return
