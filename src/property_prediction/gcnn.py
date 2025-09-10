"""
Inspired by Akshat Nigam’s code for the KRAKEN project (https://github.com/aspuru-guzik-group/kraken).
Defines MPNNet for message-passing on molecular graphs and MolPropModule for training and inference.
"""

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from lightning.pytorch.cli import instantiate_class
from omegaconf import OmegaConf
from torch_geometric.data import Data as GraphData

if TYPE_CHECKING:
    from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class MPNNet(nn.Module):
    """
    Message-passing neural network for molecular graphs.

    Args:
        config (DictConfig): model and training configuration, must include:
            - node_dim: input atom feature dimension
            - edge_dim: input bond feature dimension
            - atom_dim: hidden atom embedding size
            - conv.dim: hidden size for bond MLP
            - conv.aggr: aggregation method for NNConv
            - emb_steps: number of message-passing rounds
            - mlp.layers: list of hidden sizes for final MLP
            - mlp.batch_norm: whether to use BatchNorm in MLP
            - mlp.dropout: dropout probability in MLP
        out_dim (int): dimensionality of final prediction (default: 34 for KRAKEN properties).
    """

    def __init__(self, config: "DictConfig", out_dim: int = 34):
        super().__init__()
        self.config = config

        if not config.node_dim or not config.edge_dim:
            raise ValueError(
                "node_dim and edge_dim must be set (from dataset preprocessing)."
            )
        if not config.mlp.layers:
            raise ValueError("MLP layers must be specified in config.")

        self.num_embeds = config.num_embeds
        self.linatoms = nn.Linear(config.node_dim, config.atom_dim)

        bond_mlp = nn.Sequential(
            nn.Linear(config.edge_dim, config.conv.dim),
            nn.ReLU(),
            nn.Linear(config.conv.dim, config.atom_dim**2),
        )
        self.conv = gnn.NNConv(
            in_channels=config.atom_dim,
            out_channels=config.atom_dim,
            nn=bond_mlp,
            aggr=config.conv.aggr,
            root_weight=False,
        )
        self.gru = nn.GRU(config.atom_dim, config.atom_dim)
        self.set2set = gnn.Set2Set(config.atom_dim, processing_steps=config.emb_steps)
        self.batch_norm = nn.BatchNorm1d(config.atom_dim * 2)

        mlp_layers: list[nn.Module] = []
        dims = [config.atom_dim * 2] + config.mlp.layers
        for in_dim, out_dim in zip(dims, dims[1:]):
            mlp_layers.append(nn.Linear(in_dim, out_dim))
            mlp_layers.append(nn.ReLU())
            if config.mlp.batch_norm:
                mlp_layers.append(nn.BatchNorm1d(out_dim))
            if config.mlp.dropout > 0:
                mlp_layers.append(nn.Dropout(config.mlp.dropout))
        self.mlp = nn.Sequential(*mlp_layers)

        self.pred = nn.Linear(config.mlp.layers[-1], out_dim)

    @classmethod
    def from_pretrained(cls, path: str) -> "MPNNet":
        """
        Load a pretrained MPNNet from disk.

        Args:
            path (str): directory containing 'config.json' and 'model.pt'

        Returns:
            MPNNet: model loaded with pretrained weights
        """
        config_dict = json.loads(Path(path, "config.json").read_text())
        config = OmegaConf.create(config_dict)

        model = cls(config, config.out_dim)
        state = torch.load(Path(path, "model.pt"), weights_only=True)
        model.load_state_dict(state)
        return model

    def forward(self, graph_data: GraphData) -> torch.Tensor:
        """
        Compute property predictions for a batch of molecular graphs.

        Args:
            graph_data: PyG Data batch with attributes x, edge_index, edge_attr, batch

        Returns:
            Tensor: shape (batch_size, out_dim) of predictions
        """
        if graph_data.edge_attr.size(1) == 0:
            raise ValueError("Edge attributes are empty; cannot perform convolution.")

        x = F.relu(self.linatoms(graph_data.x))
        h = x.unsqueeze(0)

        for _ in range(self.num_embeds):
            m = F.relu(self.conv(x, graph_data.edge_index, graph_data.edge_attr))
            x, h = self.gru(m.unsqueeze(0), h)
            x = x.squeeze(0)

        x = self.set2set(x, graph_data.batch)
        x = self.batch_norm(x)
        x = self.mlp(x)
        return self.pred(x)


class MolPropModule(L.LightningModule):
    """
    LightningModule wrapper for training and evaluating MPNNet models.

    Args:
        config (DictConfig): training and model configuration
    """

    def __init__(self, config: "DictConfig"):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config

    def configure_model(self) -> None:
        """Instantiate and optionally compile the MPNNet model."""
        self.model = MPNNet(self.config, 34)
        logger.info(
            f"Number of parameters: {sum(p.numel() for p in self.model.parameters())}"
        )
        try:
            torch.compile(self.model)
        except Exception as e:
            logger.error(f"Error compiling model: {e}; skipping compilation.")

    def configure_optimizers(self) -> dict[str, object]:
        """
        Create optimizer and LR scheduler from config.

        Returns:
            dict: with keys 'optimizer', 'lr_scheduler', and 'monitor'
        """
        optimizer = instantiate_class(self.model.parameters(), self.config.optimizer)
        lr_scheduler = instantiate_class(optimizer, self.config.lr_scheduler)
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": "val/loss",
        }

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        """
        Save model config and weights alongside Lightning checkpoints.

        Args:
            checkpoint: checkpoint dict provided by Lightning
        """
        ckpt_key = next(
            (k for k in checkpoint["callbacks"] if "ModelCheckpoint" in k), None
        )
        if ckpt_key:
            ckpt_dir = Path(checkpoint["callbacks"][ckpt_key]["dirpath"])
            save_dir = ckpt_dir / "pytorch"
            save_dir.mkdir(parents=True, exist_ok=True)

            cfg = OmegaConf.to_container(self.config, resolve=True)
            cfg = {
                k: v for k, v in cfg.items() if k not in ("optimizer", "lr_scheduler")
            }
            (save_dir / "config.json").write_text(json.dumps(cfg, indent=4))
            torch.save(self.model.state_dict(), save_dir / "model.pt")

    def training_step(self, batch: GraphData, batch_idx: int) -> torch.Tensor:
        """
        Run a training step and log the loss.

        Args:
            batch: batch of graph data
            batch_idx: index of the batch

        Returns:
            Tensor: computed MSE loss
        """
        preds = self.model(batch)
        target = batch.y.view(-1, self.config.out_dim)
        loss = F.mse_loss(preds, target)
        self.log(
            "train/loss",
            loss,
            prog_bar=True,
            logger=True,
            batch_size=getattr(self, "batch_size", batch.num_graphs),
        )
        if "ReduceLROnPlateau" not in self.config.lr_scheduler.class_path:
            self.lr_schedulers().step()
        return loss

    def validation_step(self, batch: GraphData, batch_idx: int) -> None:
        """
        Run a validation step and log the loss.

        Args:
            batch: batch of graph data
            batch_idx: index of the batch
        """
        preds = self.model(batch)
        loss = F.mse_loss(preds, batch.y.view(-1, self.config.out_dim))
        self.batch_size = getattr(self, "batch_size", batch.num_graphs)
        self.log(
            "val/loss", loss, prog_bar=True, logger=True, batch_size=self.batch_size
        )

    def test_step(self, batch: GraphData, batch_idx: int) -> None:
        """
        Run a test step and log the loss.

        Args:
            batch: batch of graph data
            batch_idx: index of the batch
        """
        preds = self.model(batch)
        loss = F.mse_loss(preds, batch.y.view(-1, self.config.out_dim))
        self.batch_size = getattr(self, "batch_size", batch.num_graphs)
        self.log(
            "test/loss", loss, prog_bar=True, logger=True, batch_size=self.batch_size
        )

    def forward(self, batch: GraphData) -> torch.Tensor:
        """Direct forward pass to the underlying MPNNet model."""
        return self.model(batch)
