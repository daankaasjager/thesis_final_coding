import hydra
from omegaconf import OmegaConf, DictConfig
import lightning as L
import logging
import torch
import os

from src.utils.setup import resolve_paths

from src.utils.logging_config import configure_logging
configure_logging()
logger = logging.getLogger(__name__)

"""The coding framework of this project is based on the original implementation
of Masked Diffusion Language Models (MDLM). 
doi: 10.48550/arXiv.2406.07524
repository: https://github.com/kuleshov-group/mdlm/tree/bbc8fb61077a3ca38eab2423d0f81a5484754f51"""


OmegaConf.register_new_resolver(
  'cwd', os.getcwd)
OmegaConf.register_new_resolver(
  'device_count', torch.cuda.device_count)
OmegaConf.register_new_resolver(
  'eval', eval)
OmegaConf.register_new_resolver(
  'div_up', lambda x, y: (x + y - 1) // y)

@hydra.main(version_base=None, config_path='configs',
            config_name='config')
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
    if config.mode == 'augment':
        from src.utils.augment_dataset import augment_dataset
        augment_dataset(config)
    if config.mode == 'train':
        from src.train import train
        train(config)
    elif config.mode == "sample":
        from src.generate_samples import generate_samples
        generate_samples(config)
    elif config.mode == "evaluate_samples":
        from src.evaluate_samples import evaluate_samples
        evaluate_samples(config)


if __name__ == "__main__":
    print("Program initiated")
    print("PyTorch version:", torch.__version__)
    print("CUDA version (compiled):", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())
    run()

    # TO DO: Check even fast GPUs, check configuration for different loader sizes, do more epochs and faster training.
    # Perhaps even try different models and then just run it on a job