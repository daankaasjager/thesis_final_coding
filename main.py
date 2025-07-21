import sys
from pathlib import Path

import logging
import os

import hydra
import lightning as L
import torch
from omegaconf import DictConfig, OmegaConf

from common.utils import configure_logging, resolve_paths

configure_logging()
logger = logging.getLogger(__name__)

"""The coding framework of this project is based on the original implementation
of Masked Diffusion Language Models (MDLM). 
doi: 10.48550/arXiv.2406.07524
repository: https://github.com/kuleshov-group/mdlm/tree/bbc8fb61077a3ca38eab2423d0f81a5484754f51"""


OmegaConf.register_new_resolver("cwd", os.getcwd)
OmegaConf.register_new_resolver("device_count", torch.cuda.device_count)
OmegaConf.register_new_resolver("eval", eval)
OmegaConf.register_new_resolver("div_up", lambda x, y: (x + y - 1) // y)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(config: DictConfig):
    """Main function to run the script.
    Args:
        config (DictConfig): Configuration object.
    """

    """sets up seeding, logging, and the config file"""
    rank = int(os.environ.get("LOCAL_RANK", 0))
    if rank == 0:
        logger.info(OmegaConf.to_yaml(config))
    L.seed_everything(config.seed, verbose=False)
    config = resolve_paths(config)

    if config.mode == "train":
        from src.mdlm import train

        train(config)
    elif config.mode == "generate":
        from src.mdlm import generate_samples

        generate_samples(config)
    elif config.mode == "evaluate":
        from src.mdlm import evaluate_conditioning, evaluate_preliminaries

        evaluate_conditioning(config, "Small WordLevel")
        evaluate_preliminaries(config)
    elif config.mode == "train_property_prediction":
        from src.property_prediction.training import train_property_predictor

        train_property_predictor(config.property_prediction)
    elif config.mode == "predict_properties":
        from src.property_prediction.inference import predict_properties

        predict_properties(config.property_prediction)
    elif config.mode == "visualize_cutoffs":
        from src.property_prediction.visualize_cutoffs import \
            aggregate_model_summaries

        sample_dir = Path(config.property_prediction.sample_dir)
        aggregate_model_summaries(sample_dir)


if __name__ == "__main__":
    print("Program initiated")
    print("PyTorch version:", torch.__version__)
    print("CUDA version (compiled):", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())
    run()
