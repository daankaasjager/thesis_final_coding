"""Graph Convolutional Neural Network (GCN) for Molecular Property Prediction
This module defines a GCN model for predicting molecular properties using PyTorch Geometric.
It is taken the GitLab repository of Andrei Voinea"""

from pathlib import Path
import lightning as L
from lightning.pytorch.cli import instantiate_class
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.nn as gnn

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omegaconf import DictConfig

from omegaconf import OmegaConf
import json

from mol2mol.utils import get_pylogger

logger = get_pylogger(__name__)


class MPNNet(nn.Module):
    def __init__(self, config: "DictConfig", out_dim: int = 1):
        super(MPNNet, self).__init__()
        self.config = config
        if not config.node_dim or not config.edge_dim:
            raise ValueError(
                "node_dim and edge_dim must be set. "
                + "These values should be obtained from the dataset. "
                + "If you are using a custom script, you must calculate "
                + "these values and pass them to the model."
            )

        if not (config.mlp.layers and len(config.mlp.layers) > 0):
            raise ValueError("MLP layers must be set")

        self.num_embeds = config.num_embeds

        self.linatoms = nn.Linear(config.node_dim, config.atom_dim)

        nnet = nn.Sequential(
            nn.Linear(config.edge_dim, config.conv.dim),
            nn.ReLU(),
            nn.Linear(config.conv.dim, config.atom_dim**2),
        )

        self.conv = gnn.NNConv(
            config.atom_dim,
            config.atom_dim,
            nnet,
            aggr=config.conv.aggr,
            root_weight=False,
        )

        self.gru = nn.GRU(config.atom_dim, config.atom_dim)

        self.set2set = gnn.Set2Set(config.atom_dim, processing_steps=config.emb_steps)

        self.batch_norm = nn.BatchNorm1d(config.atom_dim * 2)

        layers = []
        net_dims = [config.atom_dim * 2] + config.mlp.layers
        for i in range(len(net_dims) - 1):
            layers.append(nn.Linear(net_dims[i], net_dims[i + 1]))
            layers.append(nn.ReLU())
            if config.mlp.batch_norm:
                layers.append(nn.BatchNorm1d(net_dims[i + 1]))
            if config.mlp.dropout > 0:
                layers.append(nn.Dropout(config.mlp.dropout))

        self.mlp = nn.Sequential(*layers)

        self.pred = nn.Linear(config.mlp.layers[-1], out_dim)

    @classmethod
    def from_pretrained(self, path: str):
        config_path = Path(path) / "config.json"
        with open(config_path, "r") as f:
            config = json.load(f)
        config = OmegaConf.create(config)

        model = MPNNet(config, config.out_dim)
        model.load_state_dict(torch.load(Path(path) / "model.pt", weights_only=True))

        return model

    def forward(self, graph_data):
        if graph_data.edge_attr.shape[1] == 0:
            raise ValueError("Edge attributes are empty")

        x = F.relu(self.linatoms(graph_data.x))

        h = x.unsqueeze(0)
        # Embed graph
        for _ in range(self.num_embeds):
            m = F.relu(self.conv(x, graph_data.edge_index, graph_data.edge_attr))
            x, h = self.gru(m.unsqueeze(0), h)
            x = x.squeeze(0)

        x = self.set2set(x, graph_data.batch)
        x = self.batch_norm(x)
        x = self.mlp(x)
        x = self.pred(x)

        return x


class MolPropModule(L.LightningModule):
    def __init__(self, config: "DictConfig"):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config

    def configure_model(self):
        self.model = MPNNet(self.config, self.config.out_dim)

        logger.info(f"Number of parameters: {sum(p.numel() for p in self.model.parameters())}")
        try:
            torch.compile(self.model)
        except Exception as e:
            logger.error(f"Error compiling model: {e}\nSkipping compilation...")

    def configure_optimizers(self):
        optimizer = instantiate_class(self.model.parameters(), self.config.optimizer)
        lr_scheduler = instantiate_class(optimizer, self.config.lr_scheduler)

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": "val/loss",
        }

    def on_save_checkpoint(self, checkpoint: torch.Dict[str, torch.Any]) -> None:
        ckpt_key = None
        for key in checkpoint["callbacks"].keys():
            if "ModelCheckpoint" in key:
                ckpt_key = key
                break

        if ckpt_key:
            ckpt_dir = checkpoint["callbacks"][ckpt_key]["dirpath"]
            Path(ckpt_dir / "pytorch").mkdir(parents=True, exist_ok=True)

            with open(Path(ckpt_dir) / "pytorch" / "config.json", "w") as f:
                cfg = OmegaConf.to_container(self.config, resolve=True)
                cfg = {
                    k: v
                    for k, v in cfg.items()
                    if k not in ["optimizer", "lr_scheduler"]
                }
                json.dump(cfg, f, indent=4)

            torch.save(self.model.state_dict(), Path(ckpt_dir) / "pytorch" / "model.pt")

    def training_step(self, batch, batch_idx):
        output = self.model(batch)

        loss = F.mse_loss(output, batch.y.reshape(-1, self.config.out_dim))

        self.log(
            "train/loss", loss, prog_bar=True, logger=True, batch_size=self.batch_size
        )

        if "ReduceLROnPlateau" not in self.config.lr_scheduler.class_path:
            sch = self.lr_schedulers()
            sch.step()

        return loss

    def validation_step(self, batch, batch_idx):
        output = self.model(batch)
        loss = F.mse_loss(output, batch.y.reshape(-1, self.config.out_dim))

        if not hasattr(self, "batch_size"):
            self.batch_size = int(max(batch.batch) + 1)
        self.log(
            "val/loss", loss, prog_bar=True, logger=True, batch_size=self.batch_size
        )

    def test_step(self, batch, batch_idx):
        output = self.model(batch)
        loss = F.mse_loss(output, batch.y.reshape(-1, self.config.out_dim))

        if not hasattr(self, "batch_size"):
            self.batch_size = int(max(batch.batch) + 1)
        self.log(
            "test/loss", loss, prog_bar=True, logger=True, batch_size=self.batch_size
        )

    def forward(self, batch):
        return self.model(batch)
