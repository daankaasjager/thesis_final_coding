# metrics_runner.py
from ast import Str
import logging
from re import L
from typing import Dict, List, Tuple, Union
import pandas as pd

from collections import defaultdict
from fcd import canonical_smiles, get_fcd, load_ref_model
import rdkit.Chem as Chem

from .metrics_core import get_valid_molecules, compute_chemical_metrics, compute_token_stats, \
                          compute_validity, compute_uniqueness, compute_novelty
from .metrics_plotting import MetricPlotter

logger = logging.getLogger(__name__)

def _split_samples(sample_objects):
    selfies, props = [], []
    for x in sample_objects:
        if isinstance(x, str):
            selfies.append(x)
            props.append(None)
        elif isinstance(x, dict):
            selfies.append(x.get("selfies", ""))
            props.append(
                x.get("predicted_properties")        # generated data
                or x.get("properties")               # (optional) alternative key
            )
        else:
            continue
    return selfies, props

class MetricRunner:
    def __init__(self, config):
        self.plotter = MetricPlotter(config)
        self.config = config

    def _process_dataset(self,
                     samples: List[Union[str, Dict]],
                     name: str,
                     metrics: List[str],
                     property_list: List[str]): 
        results = {}
        selfies, props_dicts = _split_samples(samples)
        if any(props_dicts):
            from collections import defaultdict
            prop_values = defaultdict(list)
            for pd in props_dicts:
                if pd is None:
                    continue
                for prop in property_list:
                    val = pd.get(prop, None)
                    if val is not None:
                        prop_values[prop].append(float(val))
            for prop, vals in prop_values.items():
                results[prop] = vals
        token_based_metrics = [m for m in metrics if m in ('token_frequency', 'length_distribution')]
        if token_based_metrics:
            token_counts, lengths = compute_token_stats(selfies)
            if 'token_frequency' in metrics:
                results['token_frequency'] = token_counts
            if 'length_distribution' in metrics:
                results['length_distribution'] = lengths

        # --- Chemical metrics (requires valid molecules) ---
        chemical_metrics_list = [m for m in metrics if m not in ('token_frequency', 'length_distribution', 'validity', 'uniqueness', 'novelty')
                                    and m not in property_list]
        
        mols = get_valid_molecules(selfies, self.config)
        valid_smiles = [Chem.MolToSmiles(mol, canonical=True) for mol in mols if mol is not None]

        # --- Validity, Uniqueness, Novelty ---
        if 'validity' in metrics:
            validity_score = compute_validity(selfies, mols)
            results['validity'] = validity_score
            logger.info(f"[{name}] Validity: {validity_score:.3f}")

        if 'uniqueness' in metrics:
            uniqueness_score = compute_uniqueness(valid_smiles)
            results['uniqueness'] = uniqueness_score
            logger.info(f"[{name}] Uniqueness: {uniqueness_score:.3f}")
        
        # Novelty can only be computed for generated samples against original
        if 'novelty' in metrics and name != "Original data":
            if not self.original_smiles:
                logger.warning("Original SMILES not loaded for novelty calculation. Skipping.")
            else:
                novelty_score = compute_novelty(valid_smiles, self.original_smiles)
                results['novelty'] = novelty_score
                logger.info(f"[{name}] Novelty: {novelty_score:.3f}")
        
        # --- Other chemical properties ---
        if chemical_metrics_list:
            chem_results = compute_chemical_metrics(mols, chemical_metrics_list)
            for metric, values in chem_results.items():
                results[metric] = values # Store values, not just average
                if values: # Avoid division by zero for empty lists
                    logger.info(f"[{name}] {metric}: avg={sum(values)/len(values):.3f}, n={len(values)}")
                else:
                    logger.info(f"[{name}] {metric}: No valid molecules for calculation.")
        
        return results, valid_smiles # Return canonical smiles for FCD and uniqueness/novelty


    def run_multi(
            self,
            sample_dict: Dict[str, List[str]],
            metrics: List[str],
            properties: List[str],
            reference_name: str,
            run_type: str
    ):
        """
        Evaluate every dataset in sample_dict, taking `reference_name`
        as the dataset to compare against (could be 'Original data' or
        any baseline model).  `run_type` is just the tag used in filenames.
        """
        logger.info(f"Evaluating datasets (run_type='{run_type}', reference='{reference_name}')")

        # Store reference so other helpers (e.g. novelty) can see it
        self.reference_name = reference_name

        aggregated = defaultdict(lambda: defaultdict(list))
        canonicalized = {}                     # For FCD

        # --- reference smiles for novelty/FCD ---
        if reference_name in sample_dict:
            ref_objs, _ = _split_samples(sample_dict[reference_name])
            ref_mols = get_valid_molecules(ref_objs, self.config)
            self.original_smiles = [
                Chem.MolToSmiles(mol, canonical=True) for mol in ref_mols if mol is not None
            ]
            canonicalized[reference_name] = self.original_smiles
        else:
            logger.error(f"Reference dataset '{reference_name}' not found – novelty will be skipped.")
            self.original_smiles = []

        # --- collect all metric values ---
        global_bins = defaultdict(list)
        for name, samples in sample_dict.items():
            ds_results, canon_smiles = self._process_dataset(samples, name, metrics, properties)
            canonicalized[name] = canon_smiles

            for metric, values in ds_results.items():
                aggregated[metric][name] = values
                if metric not in ('token_frequency', 'length_distribution',
                                  'validity', 'uniqueness', 'novelty'):
                    global_bins[metric].extend(values)

        # global histogram bins
        self.plotter.set_global_bins(global_bins)

        # --- plotting ---
        for metric, data in aggregated.items():
            if metric == 'token_frequency':
                self.plotter.plot_token_frequency(data, reference_name, run_type)
            elif metric == 'length_distribution':
                self.plotter.plot_length_violin(data, reference_name, run_type)
            elif metric in ('validity', 'uniqueness', 'novelty'):
                continue      # single values – handled in summary
            else:
                self.plotter.plot_property_violin(metric, data, reference_name, run_type)

        # --- FCD vs. reference ---
        fcd_scores = {}
        fcd_model = load_ref_model()
        ref_smi = canonicalized.get(reference_name, [])
        for name, smi in canonicalized.items():
            if name == reference_name or not smi or not ref_smi:
                fcd_scores[name] = "N/A"
                continue
            fcd_scores[name] = get_fcd(smi, ref_smi, fcd_model)
            logger.info(f"[{name}] FCD vs {reference_name}: {fcd_scores[name]:.3f}")

        return aggregated, fcd_scores