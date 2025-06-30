from property_prediction.gcnn import MolPropModule
from property_prediction.data.gnn_dataset import prepare_graph_dataset, split_and_load

from hydra.utils import instantiate
import logging
from omegaconf import DictConfig

def train_property_predictor(config):
    """
    Train the lightning model for property prediction.
    Args:
        config: Hydra config.
    """

