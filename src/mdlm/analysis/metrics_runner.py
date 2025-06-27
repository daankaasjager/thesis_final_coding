# metrics_runner.py
import logging
from re import L
from typing import Dict, List, Tuple

from collections import defaultdict
from fcd import canonical_smiles, get_fcd, load_ref_model
import rdkit.Chem as Chem

from .metrics_core import get_valid_molecules, compute_chemical_metrics, compute_token_stats, \
                          compute_validity, compute_uniqueness, compute_novelty
from .metrics_plotting import MetricPlotter

logger = logging.getLogger(__name__)

class MetricRunner:
    def __init__(self, config):
        self.plotter = MetricPlotter(config)
        self.config = config

    def _process_dataset(self, samples: List[str], name: str, metrics: List[str]):
        results = {}
        
        # --- Token-based metrics (token_frequency, length_distribution) ---
        token_based_metrics = [m for m in metrics if m in ('token_frequency', 'length_distribution')]
        if token_based_metrics:
            token_counts, lengths = compute_token_stats(samples)
            if 'token_frequency' in metrics:
                results['token_frequency'] = token_counts
            if 'length_distribution' in metrics:
                results['length_distribution'] = lengths

        # --- Chemical metrics (requires valid molecules) ---
        chemical_metrics_list = [m for m in metrics if m not in ('token_frequency', 'length_distribution', 'validity', 'uniqueness', 'novelty')]
        
        mols = get_valid_molecules(samples, self.config)
        valid_smiles = [Chem.MolToSmiles(mol, canonical=True) for mol in mols if mol is not None]

        # --- Validity, Uniqueness, Novelty ---
        if 'validity' in metrics:
            validity_score = compute_validity(samples, mols)
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


    def run_multi(self, sample_dict: Dict[str, List[str]], metrics: List[str]):
        logger.info("Evaluating original and generated samples with and without overlay...")

        aggregated = defaultdict(lambda: defaultdict(list))
        canonicalized_samples_for_fcd = {} # Store canonical SMILES for FCD calculation

        # Pre-process original data to get canonical SMILES for novelty
        if "Original data" in sample_dict:
            # Need to re-process original data to get canonical SMILES for novelty
            original_samples = sample_dict["Original data"]
            original_mols = get_valid_molecules(original_samples, self.config)
            self.original_smiles = [Chem.MolToSmiles(mol, canonical=True) for mol in original_mols if mol is not None]
            canonicalized_samples_for_fcd["Original data"] = self.original_smiles
        else:
            logger.error("Original data not found in sample_dict. Cannot compute novelty.")
            self.original_smiles = [] # Ensure it's empty if original data is missing

        # First pass: Process all datasets to gather raw data and canonical smiles
        all_metric_values_for_global_bins = defaultdict(list)
        fcd_scores = {}

        for name, samples in sample_dict.items():
            dataset_results, canonical_smiles_list = self._process_dataset(samples, name, metrics)
            canonicalized_samples_for_fcd[name] = canonical_smiles_list # Store for FCD
            
            for metric, data in dataset_results.items():
                if metric in ['token_frequency', 'length_distribution', 'validity', 'uniqueness', 'novelty']:
                    aggregated[metric][name] = data
                else: 
                    aggregated[metric][name] = data
                    all_metric_values_for_global_bins[metric].extend(data)
        
        # Set global bins for chemical metrics before plotting
        self.plotter.set_global_bins(all_metric_values_for_global_bins)

        # Second pass: Plotting using aggregated data and global bins
        for metric, data_dict in aggregated.items():
            if metric == 'token_frequency':
                self.plotter.plot_token_frequency(data_dict)
            elif metric == 'length_distribution':
                self.plotter.plot_length_distribution(data_dict)
            elif metric in ['validity', 'uniqueness', 'novelty']:
                # These are single values, can be displayed in a table
                pass
            else: 
                self.plotter.plot_conditioning_violin(metric, data_dict)
                self.plotter.plot_baseline_violin(metric, data_dict)
        
        # Calculate FCD scores
        fcd_model = load_ref_model()
        for name, generated_smiles in canonicalized_samples_for_fcd.items():
            if name != "Original data" and "Original data" in canonicalized_samples_for_fcd:
                # Ensure both lists are not empty
                if generated_smiles and canonicalized_samples_for_fcd["Original data"]:
                    fcd_score = get_fcd(generated_smiles, canonicalized_samples_for_fcd["Original data"], fcd_model)
                    fcd_scores[name] = fcd_score
                    logger.info(f"[{name}] FCD score: {fcd_score:.3f}")
                else:
                    logger.warning(f"Skipping FCD for {name} due to empty sample list.")
            elif name == "Original data":
                fcd_scores[name] = "N/A" # FCD is typically for generated vs. original

        return aggregated, fcd_scores # Return FCD scores as well