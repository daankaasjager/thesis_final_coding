"""Graph Convolutional Neural Network (GCN) for Molecular Property Prediction
This module defines a GCN model for predicting molecular properties using PyTorch Geometric.
It is taken the GitLab repository of Andrei Voinea."""

from pathlib import Path
import lightning as L
from lightning.pytorch.cli import instantiate_class
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any
import torch_geometric.nn as gnn
from torch_geometric.utils import dropout_adj  

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omegaconf import DictConfig

from omegaconf import OmegaConf
import json
import logging

logger = logging.getLogger(__name__)


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
        """self.edge_dropout_p = config.reg.edge_dropout    # just store the prob
        self.node_dropout_p = config.reg.node_dropout"""
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

        
        x = F.relu(self.linatoms(graph_data.x.float()))

        h = x.unsqueeze(0)
        # Embed graph
        for _ in range(self.num_embeds):
            edge_index, edge_attr = dropout_adj(
                graph_data.edge_index,
                graph_data.edge_attr,
                p=self.edge_dropout_p,
                force_undirected=False,
                training=self.training,
            )
            if edge_attr is not None and edge_attr.dtype != torch.float32:
                edge_attr = edge_attr.float()
            m = F.relu(self.conv(x, edge_index, edge_attr))
            m = F.dropout(m, p=self.node_dropout_p, training=self.training)
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
        # The model now correctly uses self.hparams, which is the saved config
        self.model = MPNNet(self.hparams, self.hparams.out_dim)
        logger.info(f"Number of parameters: {sum(p.numel() for p in self.model.parameters())}")
        try:
            torch.compile(self.model)
        except Exception as e:
            logger.error(f"Error compiling model: {e}\nSkipping compilation...")

    def configure_optimizers(self):
        optimizer = instantiate_class(self.model.parameters(), self.hparams.optimizer)
        lr_scheduler = instantiate_class(optimizer, self.hparams.lr_scheduler)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": lr_scheduler, "monitor": "val/loss"},
        }

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """
        Saves a clean, inference-ready version of the model and its config.
        This version is more robust than iterating through callback keys.
        """
        model_checkpoint_callback = None
        for callback in self.trainer.callbacks:
            if isinstance(callback, L.pytorch.callbacks.ModelCheckpoint):
                model_checkpoint_callback = callback
                break

        if model_checkpoint_callback is None:
            logger.warning("Could not find ModelCheckpoint callback, skipping model export.")
            return
            
        output_dir = Path(model_checkpoint_callback.dirpath)
        export_path = output_dir / "pytorch"
        export_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Exporting inference-ready model to {export_path}")
        
        config_to_save = OmegaConf.to_container(self.hparams, resolve=True)
        
        keys_to_remove = ["optimizer", "lr_scheduler"]
        inference_config = {
            k: v for k, v in config_to_save.items() if k not in keys_to_remove
        }

        with open(export_path / "config.json", "w") as f:
            json.dump(inference_config, f, indent=4)

        torch.save(self.model.state_dict(), export_path / "model.pt")


    def training_step(self, batch, batch_idx):
        output = self.model(batch)
        loss = F.mse_loss(output, batch.y.reshape(-1, self.hparams.out_dim))
        
        batch_size = self.trainer.datamodule.batch_size if self.trainer.datamodule else self.trainer.train_dataloader.batch_size
        self.log("train/loss", loss, prog_bar=True, logger=True, batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        output = self.model(batch)
        loss = F.mse_loss(output, batch.y.reshape(-1, self.hparams.out_dim))
        
        val_loader = self.trainer.val_dataloaders
        if isinstance(val_loader, list):
            val_loader = val_loader[0]
        batch_size = getattr(val_loader, 'batch_size', None)
        self.log("val/loss", loss, prog_bar=True, logger=True, batch_size=batch_size)
    

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
