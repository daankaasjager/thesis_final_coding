import itertools
import logging
import math
from dataclasses import dataclass

import hydra.utils
import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
import torchmetrics
import transformers
from torch import Tensor
from tqdm import tqdm

from .utils.length_sampler import LengthSampler
from .preprocessing import map_target_properties_to_bins, normalize_scalar_target_properties

from .modeling import (ExponentialMovingAverage,
                      FaultTolerantDistributedSampler,
                      RandomFaultTolerantSampler, get_noise)
from .models import DIT

logger = logging.getLogger(__name__)

LOG2 = math.log(2)


def _sample_categorical(categorical_probs):
    gumbel_norm = 1e-10 - (torch.rand_like(categorical_probs) + 1e-10).log()
    return (categorical_probs / gumbel_norm).argmax(dim=-1)


def _unsqueeze(x, reference):
    return x.view(*x.shape, *((1,) * (len(reference.shape) - len(x.shape))))


@dataclass
class Loss:
    loss: torch.FloatTensor
    nlls: torch.FloatTensor
    token_mask: torch.FloatTensor


class NLL(torchmetrics.aggregation.MeanMetric):
    pass


class BPD(NLL):
    def compute(self) -> Tensor:
        return self.mean_value / self.weight / LOG2


class Perplexity(NLL):
    def compute(self) -> Tensor:
        return torch.exp(self.mean_value / self.weight)


class Diffusion(L.LightningModule):
    def __init__(self, config, tokenizer: transformers.PreTrainedTokenizer):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        self.tokenizer = tokenizer
        self.vocab_size = self.tokenizer.vocab_size
        self.sampler = self.config.sampling.predictor
        self.antithetic_sampling = self.config.training.antithetic_sampling
        self.importance_sampling = self.config.training.importance_sampling
        self.change_of_variables = self.config.training.change_of_variables
        if (
            not hasattr(self.tokenizer, "mask_token")
            or self.tokenizer.mask_token is None
        ):
            self.mask_index = self.vocab_size
            self.vocab_size += 1
        else:
            self.mask_index = self.tokenizer.mask_token_id
        self.parameterization = self.config.parameterization
        self.backbone = DIT(self.config, vocab_size=self.vocab_size)

        self.T = self.config.sampling.T

        self.softplus = torch.nn.Softplus()
        # metrics are automatically reset at end of epoch
        metrics = torchmetrics.MetricCollection(
            {
                "nll": NLL(),
                "bpd": BPD(),
                "ppl": Perplexity(),
            }
        )
        metrics.set_dtype(torch.float64)
        self.train_metrics = metrics.clone(prefix="train/")
        self.valid_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

        # generative perplexity
        self.gen_ppl_metric = Perplexity()

        self.noise = get_noise(self.config, dtype=self.dtype)
        if self.config.training.ema > 0:
            self.ema = ExponentialMovingAverage(
                itertools.chain(self.backbone.parameters(), self.noise.parameters()),
                decay=self.config.training.ema,
            )
        else:
            self.ema = None

        self.lr = self.config.optim.lr
        self.sampling_eps = self.config.training.sampling_eps
        self.time_conditioning = self.config.sampling.time_conditioning
        self.sample_output_length = self.config.model.length
        self.neg_infinity = -1000000.0
        self.fast_forward_epochs = None
        self.fast_forward_batches = None

        if self.config.model.sample_length_mode == "histogram":
            self.length_sampler = LengthSampler(
                hist_file=self.config.paths.length_histogram,
                max_len=self.config.model.length,
                device=self.device,
            )
        else:
            self.fixed_length = self.config.model.length

        self._validate_configuration()


    def _validate_configuration(self):
        assert not (self.change_of_variables and self.importance_sampling)
        if self.T > 0:
            assert self.parameterization in {"subs"}

    def on_load_checkpoint(self, checkpoint):
        if self.ema:
            self.ema.load_state_dict(checkpoint["ema"])
        # Copied from:
        # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py#L41
        self.fast_forward_epochs = checkpoint["loops"]["fit_loop"]["epoch_progress"][
            "current"
        ]["completed"]
        self.fast_forward_batches = checkpoint["loops"]["fit_loop"][
            "epoch_loop.batch_progress"
        ]["current"]["completed"]

    def on_save_checkpoint(self, checkpoint):
        if self.ema:
            checkpoint["ema"] = self.ema.state_dict()
        # Copied from:
        # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/tasks/seq.py
        # ['epoch_loop.batch_progress']['total']['completed'] is 1 iteration
        # behind, so we're using the optimizer's progress.
        checkpoint["loops"]["fit_loop"]["epoch_loop.batch_progress"]["total"][
            "completed"
        ] = (
            checkpoint["loops"]["fit_loop"][
                "epoch_loop.automatic_optimization.optim_progress"
            ]["optimizer"]["step"]["total"]["completed"]
            * self.trainer.accumulate_grad_batches
        )
        checkpoint["loops"]["fit_loop"]["epoch_loop.batch_progress"]["current"][
            "completed"
        ] = (
            checkpoint["loops"]["fit_loop"][
                "epoch_loop.automatic_optimization.optim_progress"
            ]["optimizer"]["step"]["current"]["completed"]
            * self.trainer.accumulate_grad_batches
        )
        # _batches_that_stepped tracks the number of global steps, not the number
        # of local steps, so we don't multiply with self.trainer.accumulate_grad_batches here.
        checkpoint["loops"]["fit_loop"]["epoch_loop.state_dict"][
            "_batches_that_stepped"
        ] = checkpoint["loops"]["fit_loop"][
            "epoch_loop.automatic_optimization.optim_progress"
        ][
            "optimizer"
        ][
            "step"
        ][
            "total"
        ][
            "completed"
        ]
        if "sampler" not in checkpoint.keys():
            checkpoint["sampler"] = {}
        if hasattr(self.trainer.train_dataloader.sampler, "state_dict"):
            sampler_state_dict = self.trainer.train_dataloader.sampler.state_dict()
            checkpoint["sampler"]["random_state"] = sampler_state_dict.get(
                "random_state", None
            )
        else:
            checkpoint["sampler"]["random_state"] = None

    def on_train_start(self):
        if self.ema:
            self.ema.move_shadow_params_to_device(self.device)

        distributed = (
            self.trainer._accelerator_connector.use_distributed_sampler
            and self.trainer._accelerator_connector.is_distributed
        )

        sampler_cls = (
            FaultTolerantDistributedSampler
            if distributed
            else RandomFaultTolerantSampler
        )

        updated_dls = []
        for dl in self.trainer.fit_loop._combined_loader.flattened:
            if hasattr(dl.sampler, "shuffle"):
                dl_sampler = sampler_cls(dl.dataset, shuffle=dl.sampler.shuffle)
            else:
                dl_sampler = sampler_cls(dl.dataset)

            if (
                distributed
                and self.fast_forward_epochs is not None
                and self.fast_forward_batches is not None
            ):
                dl_sampler.load_state_dict(
                    {
                        "epoch": self.fast_forward_epochs,
                        "counter": self.fast_forward_batches
                        * self.config.loader.batch_size,
                    }
                )

            # FIX: Preserve the original collate_fn to avoid list conversion issue
            updated_dls.append(
                torch.utils.data.DataLoader(
                    dl.dataset,
                    batch_size=self.config.loader.batch_size,
                    num_workers=self.config.loader.num_workers,
                    pin_memory=self.config.loader.pin_memory,
                    sampler=dl_sampler,
                    shuffle=False,
                    persistent_workers=True,
                    collate_fn=dl.collate_fn, 
                )
            )

        self.trainer.fit_loop._combined_loader.flattened = updated_dls

    def optimizer_step(self, *args, **kwargs):
        grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1e9)
        self.log("trainer/grad_norm", grad_norm, on_step=True, prog_bar=False) #CHECK THIS LATER
        super().optimizer_step(*args, **kwargs)
        if self.ema:
            self.ema.update(
                itertools.chain(self.backbone.parameters(), self.noise.parameters())
            )

    def _subs_parameterization(self, logits, xt):
        # log prob at the mask index = - infinity
        logits[:, :, self.mask_index] += self.neg_infinity
        logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        # NOTE: Interesting to see that this was the same as d3pm up to here.
        # Apply updates directly in the logits matrix.
        # For the logits of the unmasked tokens, set all values
        # to -infinity except for the indices corresponding to
        # the unmasked tokens.
        unmasked_indices = xt != self.mask_index
        logits[unmasked_indices] = self.neg_infinity
        logits[unmasked_indices, xt[unmasked_indices]] = 0
        return logits

    def _process_sigma(self, sigma):
        if sigma is None:
            assert self.parameterization == "ar"
            return sigma
        if sigma.ndim > 1:
            sigma = sigma.squeeze(-1)
        if not self.time_conditioning:
            sigma = torch.zeros_like(sigma)
        assert sigma.ndim == 1, sigma.shape
        return sigma

    def forward(self, x, sigma, property_conditioning_vector=None, force_unconditional_pass=False):
        """Ensure all inputs are on the correct device before computation."""

        # Ensure device consistency
        device = next(self.backbone.parameters()).device  # Get backbone device
        x = x.to(device)  # Move input to same device as backbone
        sigma = self._process_sigma(sigma)
        if property_conditioning_vector is not None:
            property_conditioning_vector = property_conditioning_vector.to(
                device, dtype=self.dtype)
        logits = self.backbone(
            x, sigma, property_conditioning_vector, force_unconditional_pass
        )

        return self._subs_parameterization(logits=logits, xt=x)

    def _d3pm_loss(self, model_output, xt, x0, t):
        dt = 1 / self.T

        if torch.is_tensor(t):
            t = t[:, None]
            assert t.ndim == 2
            t = t.clamp(0.0, 1.0 - 1e-4)
        alpha_t = 1 - t + torch.zeros_like(xt)
        alpha_s = 1 - (t - dt) + torch.zeros_like(xt)

        log_x_theta_at_x0 = torch.gather(model_output, -1, x0[:, :, None]).squeeze(-1)
        log_x_theta_at_m = model_output[:, :, self.mask_index]
        x_theta_at_m = log_x_theta_at_m.exp()

        term_1_coef = dt / t
        term_1_log_nr = torch.log(alpha_t * x_theta_at_m / t + 1)
        term_1_log_dr = log_x_theta_at_x0

        term_2_coef = 1 - dt / t
        term_2_log_nr = term_1_log_nr
        term_2_log_dr = torch.log(alpha_s * x_theta_at_m / (t - dt) + 1)

        L_vb_masked = term_1_coef * (term_1_log_nr - term_1_log_dr) + term_2_coef * (
            term_2_log_nr - term_2_log_dr
        )

        L_vb = L_vb_masked * (xt == self.mask_index)

        return self.T * L_vb

    def _compute_loss(self, batch, prefix):
        if "attention_mask" in batch:
            attention_mask = batch["attention_mask"]
        else:
            attention_mask = None
        losses = self._loss(batch["input_ids"], attention_mask, batch["cond_props"])
        loss = losses.loss

        if prefix == "train":
            self.train_metrics.update(losses.nlls, losses.token_mask)
            metrics = self.train_metrics
        elif prefix == "val":
            self.valid_metrics.update(losses.nlls, losses.token_mask)
            metrics = self.valid_metrics
        elif prefix == "test":
            self.test_metrics.update(losses.nlls, losses.token_mask)
            metrics = self.test_metrics
        else:
            raise ValueError(f"Invalid prefix: {prefix}")

        self.log_dict(metrics, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def on_train_epoch_start(self):
        self.backbone.train()
        self.noise.train()

    def training_step(self, batch, batch_idx):
        loss = self._compute_loss(batch, prefix="train")
        if batch_idx % 100 == 0:  # Check every 100 batches
            for name, param in self.backbone.named_parameters():
                if "property_map" in name and param.grad is not None:
                    self.log(f"grads/{name}", param.grad.norm(), prog_bar=True)
        self.log(
            name="trainer/loss",
            value=loss.item(),
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )
        return loss

    def on_validation_epoch_start(self):
        if self.ema:
            self.ema.store(
                itertools.chain(self.backbone.parameters(), self.noise.parameters())
            )
            self.ema.copy_to(
                itertools.chain(self.backbone.parameters(), self.noise.parameters())
            )
        self.backbone.eval()
        self.noise.eval()

    def validation_step(self, batch):
        loss = self._compute_loss(batch, prefix="val")
        return loss

    def on_validation_epoch_end(self):
        if self.ema:
            self.ema.restore(
                itertools.chain(self.backbone.parameters(), self.noise.parameters())
            )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            itertools.chain(self.backbone.parameters(), self.noise.parameters()),
            lr=self.config.optim.lr,
            betas=(self.config.optim.beta1, self.config.optim.beta2),
            eps=self.config.optim.eps,
            weight_decay=self.config.optim.weight_decay,
        )

        scheduler = hydra.utils.instantiate(
            self.config.lr_scheduler, optimizer=optimizer
        )
        scheduler_dict = {
            "scheduler": scheduler,
            "interval": "step",
            "monitor": "val/loss",
            "name": "trainer/lr",
        }
        return [optimizer], [scheduler_dict]

    def q_xt(self, x, move_chance, attention_mask):  # <-- Add attention_mask here
        """Computes the noisy sample xt, ensuring BOS/EOS are not masked."""
        move_probs = torch.rand(*x.shape, device=x.device)
        eligible_mask = torch.ones_like(x, dtype=torch.bool)

        # the first token is always the <BOS> token, so we don't want to mask it hence, 1:n_props+1
        n_props = len(getattr(self.config.conditioning, "properties", []))
        eligible_mask[:, 1:n_props +1] = False

        """-- EOS masking code (don't want this because then it doesn't predict the EOS anymore for variable lengths---
        
        lengths = attention_mask.sum(dim=1).long()
        valid_lengths = torch.clamp(
            lengths, min=2
        )  # Assuming BOS/EOS always present
        eos_indices = valid_lengths - 1
        batch_indices = torch.arange(x.shape[0], device=x.device)
        try:
            eligible_mask[batch_indices, eos_indices] = False
        except IndexError:
            logger.error(
                f"IndexError in q_xt: EOS masking prevention failed. Indices: {eos_indices.tolist()}, Shape: {eligible_mask.shape}, Lengths: {lengths.tolist()}"
            )"""
        move_chance_expanded = move_chance.expand_as(x)
        move_indices = (move_probs < move_chance_expanded) & eligible_mask
        xt = torch.where(move_indices, self.mask_index, x)
        return xt
    

    def _sample_prior(self, *batch_dims, target_properties=None):
        """
        Creates a [MASK]-only sequence (plus optional conditioning prefix).

        Parameters
        ----------
        *batch_dims : (int,) or (int, int)
            batch_dims[0] = batch size  (required)
            batch_dims[1] = requested sequence length (ignored when
                        sample_length_mode == "histogram")
        target_properties : dict | None
            Used only when `conditioning.prepend` is enabled.

        Returns
        -------
        torch.LongTensor  shape == (batch_size, L)
        """

        # ------------------------------------------------------------------ #
        # 0. parse the incoming dimensions                                   #
        # ------------------------------------------------------------------ #
        if len(batch_dims) == 0:
            raise ValueError("_sample_prior needs at least the batch size")

        batch_size = int(batch_dims[0])

        if self.config.model.sample_length_mode == "histogram":
            # Draw ONE length and share it across the whole batch so that
            # downstream code can continue to treat the tensor as rectangular.
            L = int(self.length_sampler.sample(1))
        else:  # "fixed"
            if len(batch_dims) < 2:
                raise ValueError("fixed-length mode expects (batch, seq_len)")
            L = int(batch_dims[1])

        max_L = self.config.model.length   # positional-encoding cap
        if L > max_L:
            raise ValueError(
                f"Chosen length {L} exceeds model.length={max_L}. "
                "Increase the cap or trim the histogram."
            )

        # ------------------------------------------------------------------ #
        # 1. optional property-conditioning prefix                           #
        # ------------------------------------------------------------------ #
        if self.config.conditioning.prepend and target_properties is not None:
            bin_token_ids = map_target_properties_to_bins(
                self.config, target_properties, self.tokenizer
            )                                  # e.g. [145, 876, 32]
            prefix = torch.tensor(
                bin_token_ids, device=self.device, dtype=torch.long
            ).unsqueeze(0).expand(batch_size, -1)          # (B, P)

            # make sure prefix fits; stretch L if necessary
            if prefix.shape[1] > L:
                L = prefix.shape[1]

            suffix_len = L - prefix.shape[1]
            suffix = torch.full(
                (batch_size, suffix_len),
                self.mask_index,
                dtype=torch.long,
                device=self.device,
            )
            return torch.cat([prefix, suffix], dim=1)      # (B, L)

        # ------------------------------------------------------------------ #
        # 2. unconditional prior                                             #
        # ------------------------------------------------------------------ #
        return torch.full(
            (batch_size, L),
            self.mask_index,
            dtype=torch.long,
            device=self.device,
        )


    def _ddpm_caching_update(self, x, t, dt, p_x0=None, target_properties=None, current_guidance_scale=0.0):
        assert self.config.noise.type == "loglinear"
        sigma_t, _ = self.noise(t)
        if t.ndim > 1:
            t = t.squeeze(-1)
        assert t.ndim == 1

        move_chance_t = t[:, None, None]
        move_chance_s = (t - dt)[:, None, None]
        assert move_chance_t.ndim == 3, move_chance_t.shape
        if p_x0 is None:
            logits_cond = self.forward(
                x, sigma_t, target_properties, force_unconditional_pass=False
            )

            if current_guidance_scale == 0.0 or not self.config.conditioning.cfg:
                guided_logits = logits_cond
            else:
                # ---- unconditional branch for CFG
                logits_uncond = self.forward(
                    x, sigma_t, target_properties, force_unconditional_pass=True
                )
                guided_logits = ((1.0 + current_guidance_scale) * logits_cond
                                - current_guidance_scale * logits_uncond)

            log_p_x0 = self._subs_parameterization(logits=guided_logits, xt=x)
            p_x0 = log_p_x0.exp()

        assert move_chance_t.ndim == p_x0.ndim
        q_xs = p_x0 * (move_chance_t - move_chance_s)
        q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]
        _x = _sample_categorical(q_xs)

        copy_flag = (x != self.mask_index).to(x.dtype)
        return p_x0, copy_flag * x + (1 - copy_flag) * _x

    def _ddpm_update(self, x, t, dt, target_properties, current_guidance_scale):
        sigma_t, _ = self.noise(t)
        sigma_s, _ = self.noise(t - dt)
        if sigma_t.ndim > 1:
            sigma_t = sigma_t.squeeze(-1)
        if sigma_s.ndim > 1:
            sigma_s = sigma_s.squeeze(-1)
        assert sigma_t.ndim == 1, sigma_t.shape
        assert sigma_s.ndim == 1, sigma_s.shape
        move_chance_t = 1 - torch.exp(-sigma_t)
        move_chance_s = 1 - torch.exp(-sigma_s)
        move_chance_t = move_chance_t[:, None, None]
        move_chance_s = move_chance_s[:, None, None]
        
        raw_logits_cond = self.forward(x, sigma_t, target_properties, force_unconditional_pass=False)
        if current_guidance_scale == 0.0 or not self.config.conditioning.cfg:
            guided_raw_logits = raw_logits_cond
        else:
            raw_logits_uncond = self.forward(x, sigma_t, target_properties, force_unconditional_pass=True)
            guided_raw_logits = (1.0 + current_guidance_scale) * raw_logits_cond - current_guidance_scale * raw_logits_uncond
        log_p_x0_parameterized = self._subs_parameterization(logits=guided_raw_logits, xt=x)
        # Clamp to avoid log(0) or small number issues if any probability is exactly 0
        p_x0_probs = torch.exp(log_p_x0_parameterized).clamp(min=1e-20)


        q_xs_non_mask_term = p_x0_probs * (move_chance_t - move_chance_s)
        q_xs_non_mask_term = torch.clamp(q_xs_non_mask_term, min=0) # Ensure non-negative probabilities

        final_probs_for_sampling = q_xs_non_mask_term.clone()
        
        final_probs_for_sampling[:, :, self.mask_index] = final_probs_for_sampling[:, :, self.mask_index] + move_chance_s.squeeze(-1) 
        
        _x_sampled_token = _sample_categorical(final_probs_for_sampling)

        copy_flag = (x != self.mask_index).to(x.dtype)
        x_s = copy_flag * x + (1 - copy_flag) * _x_sampled_token
        return x_s

    def get_score(self, x, sigma):
        model_output = self.forward(x, sigma)
        if self.parameterization == "subs":
            log_k = -torch.log(torch.expm1(sigma)).squeeze(-1)
            assert log_k.ndim == 1

            masked_score = model_output + log_k[:, None, None]
            masked_score[:, :, self.mask_index] = 0

            unmasked_score = self.neg_infinity * torch.ones_like(model_output)
            unmasked_score = torch.scatter(
                unmasked_score,
                -1,
                x[..., None],
                torch.zeros_like(unmasked_score[..., :1]),
            )
            unmasked_score[:, :, self.mask_index] = -(
                log_k[:, None] * torch.ones_like(x)
            )

            masked_indices = (x == self.mask_index).to(model_output.dtype)[:, :, None]
            model_output = masked_score * masked_indices + unmasked_score * (
                1 - masked_indices
            )
        return model_output.exp()

    def _staggered_score(self, score, dsigma):
        score = score.clone()
        extra_const = (1 - dsigma.exp()) * score.sum(dim=-1)
        score *= dsigma.exp()[:, None]
        score[..., self.mask_index] += extra_const
        return score

    def _analytic_update(self, x, t, step_size):
        curr_sigma, _ = self.noise(t)
        next_sigma, _ = self.noise(t - step_size)
        dsigma = curr_sigma - next_sigma
        score = self.get_score(x, curr_sigma)
        stag_score = self._staggered_score(score, dsigma)
        probs = stag_score * self._transp_transition(x, dsigma)
        return _sample_categorical(probs)

    def _denoiser_update(self, x, t):
        sigma, _ = self.noise(t)
        score = self.get_score(x, sigma)
        stag_score = self._staggered_score(score, sigma)
        probs = stag_score * self._transp_transition(x, sigma)
        probs[..., self.mask_index] = 0
        samples = _sample_categorical(probs)
        return samples

    def _transp_transition(self, i, sigma):
        sigma = _unsqueeze(sigma, reference=i[..., None])
        edge = torch.exp(-sigma) * F.one_hot(i, num_classes=self.vocab_size)
        edge += torch.where(i == self.mask_index, 1 - torch.exp(-sigma).squeeze(-1), 0)[
            ..., None
        ]
        return edge

    def _sample_t(self, n, device):
        _eps_t = torch.rand(n, device=device)
        if self.antithetic_sampling:
            offset = torch.arange(n, device=device) / n
            _eps_t = (_eps_t / n + offset) % 1
        t = (1 - self.sampling_eps) * _eps_t + self.sampling_eps
        if self.importance_sampling:
            return self.noise.importance_sampling_transformation(t)
        return t

    def _reconstruction_loss(self, x0):
        t0 = torch.zeros(x0.shape[0], dtype=self.dtype, device=self.device)
        assert self.config.noise.type == "loglinear"
        # The above assert is for d3pm parameterization
        unet_conditioning = self.noise(t0)[0][:, None]
        model_output_t0 = self.forward(x0, unet_conditioning)
        return -torch.gather(
            input=model_output_t0, dim=-1, index=x0[:, :, None]
        ).squeeze(-1)

    def _forward_pass_diffusion(self, x0, attention_mask, property_conditioning_vector):
        t = self._sample_t(x0.shape[0], x0.device)
        if self.T > 0:
            t = (t * self.T).to(torch.int)
            t = t / self.T
            # t \in {1/T, 2/T, ..., 1}
            t += 1 / self.T

        if self.change_of_variables:
            unet_conditioning = t[:, None]
            f_T = torch.log1p(-torch.exp(-self.noise.sigma_max))
            f_0 = torch.log1p(-torch.exp(-self.noise.sigma_min))
            move_chance = torch.exp(f_0 + t * (f_T - f_0))
            move_chance = move_chance[:, None]
        else:
            sigma, dsigma = self.noise(t)
            unet_conditioning = sigma[:, None]
            move_chance = 1 - torch.exp(-sigma[:, None])

        xt = self.q_xt(x0, move_chance, attention_mask)

        model_output = self.forward(xt, unet_conditioning, property_conditioning_vector)
        if torch.isnan(model_output).any():
            logger.info("model output", model_output)

        if self.T > 0:
            diffusion_loss = self._d3pm_loss(
                model_output=model_output, xt=xt, x0=x0, t=t
            )
            if self.parameterization == "subs":
                reconstruction_loss = 0
            return reconstruction_loss + diffusion_loss

        # SUBS parameterization, continuous time.
        log_p_theta = torch.gather(
            input=model_output, dim=-1, index=x0[:, :, None]
        ).squeeze(-1)

        if self.change_of_variables or self.importance_sampling:
            return log_p_theta * torch.log1p(-torch.exp(-self.noise.sigma_min))

        return -log_p_theta * (dsigma / torch.expm1(sigma))[:, None]

    def _loss(self, x0, attention_mask, cond_props=None):
        loss = self._forward_pass_diffusion(x0, attention_mask, cond_props)

        nlls = loss * attention_mask
        count = attention_mask.sum()

        if count == 0:
            logger.warning("Attention mask sum is 0 in _loss, returning zero loss.")
            return Loss(loss=torch.tensor(0.0, device=x0.device, dtype=self.dtype), nlls=nlls, token_mask=attention_mask)

        batch_nll = nlls.sum()
        token_nll = batch_nll / count

        return Loss(loss=token_nll, nlls=nlls, token_mask=attention_mask)
    
    @torch.no_grad()
    def _sample(self, num_steps=None, eps=1e-5, target_properties=None, current_guidance_scale=None):
        """Generate samples from the model with progress tracking."""
        batch_size_per_gpu = self.config.loader.eval_batch_size
        if num_steps is None:
            num_steps = self.config.sampling.steps

        batch_size = self.config.loader.eval_batch_size
        if target_properties is None or self.config.conditioning.prepend:
            target_properties_tensor = None
        elif target_properties is not None \
            and (self.config.conditioning.embeddings or self.config.conditioning.cfg): # conditioning on properties with cfg or embeddings
            normalized_properties = normalize_scalar_target_properties(target_properties, self.config.paths.mean_std)
            target_properties_tensor = torch.tensor(list(normalized_properties.values())).unsqueeze(0).expand(batch_size, -1).to(self.device, dtype=self.dtype)
        x = self._sample_prior(batch_size_per_gpu, self.sample_output_length, target_properties=target_properties).to(
            self.device
        )
        timesteps = torch.linspace(1, eps, num_steps + 1, device=self.device)
        dt = (1 - eps) / num_steps
        p_x0_cache = None

        for i in tqdm(range(num_steps), desc="Diffusion steps", unit="step"):
            t = timesteps[i] * torch.ones(x.shape[0], 1, device=self.device)
            if self.sampler == "ddpm":
                x = self._ddpm_update(x, t, dt, target_properties_tensor, current_guidance_scale)
            elif self.sampler == "ddpm_cache":
                p_x0_cache, x_next = self._ddpm_caching_update(
                    x, t, dt, p_x0=p_x0_cache, target_properties=target_properties_tensor,
                    current_guidance_scale=current_guidance_scale
                )
                if not torch.allclose(x_next, x) or self.time_conditioning:
                    p_x0_cache = None
                x = x_next
            else:
                x = self._analytic_update(x, t, dt)

        if self.config.sampling.noise_removal:
            t = timesteps[-1] * torch.ones(x.shape[0], 1, device=self.device)
            if self.sampler == "analytic":
                x = self._denoiser_update(x, t)
            else:
                unet_conditioning = self.noise(t)[0]
                x = self.forward(x, unet_conditioning).argmax(dim=-1)
        return x

    def restore_model_and_sample(self, num_steps, eps=1e-5, target_properties=None, guidance_scale=None):
        """Generate samples from the model."""
        if self.ema:
            self.ema.store(
                itertools.chain(self.backbone.parameters(), self.noise.parameters())
            )
            self.ema.copy_to(
                itertools.chain(self.backbone.parameters(), self.noise.parameters())
            )
        self.backbone.eval()
        self.noise.eval()
    
        samples = self._sample(num_steps=num_steps, eps=eps, target_properties=target_properties, current_guidance_scale=guidance_scale)
        if self.ema:
            self.ema.restore(
                itertools.chain(self.backbone.parameters(), self.noise.parameters())
            )
        self.backbone.train()
        self.noise.train()
        return samples

    @torch.no_grad
    def sample_subs_guidance(self, n_samples, stride_length, num_strides, dt=0.001, target_properties=None):
        ones = torch.ones(n_samples, dtype=self.dtype, device=self.device)

        num_steps = int(1 / dt)
        sampling_steps = 0
        intermediate_tokens = []
        target = None
        for _ in range(num_strides + 1):
            p_x0_cache = None
            x = self._sample_prior(n_samples, self.sample_output_length, target_properties=target_properties).to(self.device)
            if target is not None:
                x[:, :-stride_length] = target
            for i in range(num_steps + 1):
                p_x0_cache, x_next = self._ddpm_caching_update(
                    x=x, t=(1 - i * dt) * ones, dt=dt, p_x0=p_x0_cache, 
                )
                if not torch.allclose(x_next, x) or self.time_conditioning:
                    p_x0_cache = None
                    sampling_steps += 1
                x = x_next
            x = self.forward(x, 0 * ones).argmax(dim=-1)
            intermediate_tokens.append(x[:, :stride_length].cpu().numpy())
            target = x[:, stride_length:]

        intermediate_tokens.append(target.cpu().numpy())
        intermediate_text_samples = []
        sequence_lengths = (
            (
                np.concatenate(intermediate_tokens, axis=1)[:, 1:]
                == self.tokenizer.eos_token_id
            ).cumsum(-1)
            == 0
        ).sum(-1)
        for i in range(2, len(intermediate_tokens) + 1):
            intermediate_text_samples.append(
                self.tokenizer.batch_decode(
                    np.concatenate(intermediate_tokens[:i], axis=1)
                )
            )
        return (sampling_steps, intermediate_text_samples, sequence_lengths)

    def restore_model_and_semi_ar_sample(self, stride_length, num_strides, dt=0.001, target_properties=None):
        """Generate samples from the model with progress tracking."""
        if self.ema:
            self.ema.store(
                itertools.chain(self.backbone.parameters(), self.noise.parameters())
            )
            self.ema.copy_to(
                itertools.chain(self.backbone.parameters(), self.noise.parameters())
            )

        self.backbone.eval()
        self.noise.eval()

        (sampling_steps, samples, sequence_lengths) = self.sample_subs_guidance(
            n_samples=self.config.loader.eval_batch_size,
            stride_length=stride_length,
            num_strides=num_strides,
            dt=dt,
            target_properties=target_properties
        )
        if self.ema:
            self.ema.restore(
                itertools.chain(self.backbone.parameters(), self.noise.parameters())
            )

        self.backbone.train()
        self.noise.train()
        return sampling_steps, samples, sequence_lengths
