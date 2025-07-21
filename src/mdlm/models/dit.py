"""
Core modules for the conditioned diffusion Transformer (DDiT) model, adapted
from Sahoo et al. (2024) to implement property conditioning strategies.
"""

import math
import sys
import logging
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import omegaconf
import huggingface_hub
from einops import rearrange

from mdlm.utils import get_torch_dtype

logger = logging.getLogger(__name__)
USE_FLASH_ATTN = False

# Attempt to enable FlashAttention on non-Windows platforms
if not sys.platform.startswith("win"):
    try:
        import flash_attn  # noqa: F401
        import flash_attn.layers.rotary  # noqa: F401
        USE_FLASH_ATTN = True
    except ImportError:
        USE_FLASH_ATTN = False

torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)


def _bias_dropout_add_scale(
    x: torch.Tensor,
    bias: Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: Optional[torch.Tensor],
    prob: float,
    training: bool,
) -> torch.Tensor:
    """
    Apply bias, dropout, scaling, and residual addition in one fused operation.

    Args:
        x: input tensor
        bias: optional bias tensor
        scale: element-wise scale tensor
        residual: optional residual tensor to add
        prob: dropout probability
        training: whether in training mode

    Returns:
        Transformed tensor of same shape as x
    """
    out = x + bias if bias is not None else x
    out = scale * F.dropout(out, p=prob, training=training)
    return residual + out if residual is not None else out


def get_bias_dropout_add_scale(training: bool):
    """
    Return a function that fuses bias, dropout, scale, and residual addition.

    Args:
        training: whether the function will run in training mode

    Returns:
        A callable with signature (x, bias, scale, residual, prob) -> Tensor
    """
    def fn(x, bias, scale, residual, prob):
        return _bias_dropout_add_scale(x, bias, scale, residual, prob, training)
    return fn


@torch.jit.script
def bias_dropout_add_scale_fused_train(
    x: torch.Tensor,
    bias: Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: Optional[torch.Tensor],
    prob: float,
) -> torch.Tensor:
    return _bias_dropout_add_scale(x, bias, scale, residual, prob, True)


@torch.jit.script
def bias_dropout_add_scale_fused_inference(
    x: torch.Tensor,
    bias: Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: Optional[torch.Tensor],
    prob: float,
) -> torch.Tensor:
    return _bias_dropout_add_scale(x, bias, scale, residual, prob, False)


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Apply adaptive layer-norm modulation: x * (1 + scale) + shift.

    Args:
        x: normalized input tensor
        shift: learned shift tensor
        scale: learned scale tensor

    Returns:
        Modulated tensor
    """
    return x * (1 + scale) + shift


@torch.jit.script
def _modulate_fused(
    x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    return _modulate(x, shift, scale)


class Rotary(nn.Module):
    """
    Precompute and apply rotary positional embeddings.

    Args:
        dim: feature dimension (must be even)
        base: base frequency for rotary embeddings
    """

    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = 0
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x: torch.Tensor, seq_dim: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute or retrieve cached cos/sin embeddings for sequence length.

        Args:
            x: input tensor of shape [B, S, ..., D]
            seq_dim: dimension index of sequence length (default: 1)

        Returns:
            Tuple of (cos, sin) tensors ready for rotary application
        """
        seq_len = x.size(seq_dim)
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()[None, :, None, None, :].repeat(1, 1, 3, 1, 1)
            sin = emb.sin()[None, :, None, None, :].repeat(1, 1, 3, 1, 1)
            cos[:, :, 2, :, :].fill_(1.0)
            sin[:, :, 2, :, :].fill_(0.0)
            self.cos_cached, self.sin_cached = cos, sin
        return self.cos_cached, self.sin_cached


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Helper to rotate half of the last dimension: (x1, x2) -> (-x2, x1).

    Args:
        x: tensor with last dimension even

    Returns:
        Rotated tensor
    """
    d = x.size(-1) // 2
    return torch.cat((-x[..., d:], x[..., :d]), dim=-1)


def _apply_rotary_pos_emb(
    qkv: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """
    Apply rotary positional embeddings to QKV tensor.

    Args:
        qkv: tensor of shape [B, S, 3, H, D]
        cos, sin: rotary embedding tensors

    Returns:
        Tensor of same shape with rotary applied
    """
    if USE_FLASH_ATTN:
        cos_ = cos[0, :, 0, 0, : cos.size(-1) // 2]
        sin_ = sin[0, :, 0, 0, : sin.size(-1) // 2]
        import flash_attn.layers.rotary as fa_rot  # noqa: F811
        return fa_rot.apply_rotary_emb_qkv_(qkv, cos_, sin_)
    d_rot = qkv.size(-1) // 2
    qkv_rot, qkv_pass = qkv[..., :d_rot], qkv[..., d_rot:]
    b, s, three, h, _ = qkv.shape
    cos_exp = cos[..., :d_rot].expand(b, s, three, h, d_rot)
    sin_exp = sin[..., :d_rot].expand(b, s, three, h, d_rot)
    rotated = qkv_rot * cos_exp + _rotate_half(qkv_rot) * sin_exp
    return torch.cat([rotated, qkv_pass], dim=-1)


class LayerNorm(nn.Module):
    """
    LayerNorm without half-precision instability by computing in float32.

    Args:
        dim: feature dimension
    """

    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply layer normalization in fp32 and cast back.

        Args:
            x: input tensor [..., D]

        Returns:
            Normalized tensor with learned weight applied
        """
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            normed = F.layer_norm(x.float(), [self.dim])
        return normed * self.weight


def residual_linear(
    x: torch.Tensor,
    W: torch.Tensor,
    x_skip: torch.Tensor,
    residual_scale: float,
) -> torch.Tensor:
    """
    Apply a linear transform with residual addition: x_skip + scale * W@x.

    Args:
        x: input tensor [..., Din]
        W: weight matrix [Dout, Din]
        x_skip: tensor [..., Dout] to add as residual
        residual_scale: scaling factor for the linear term

    Returns:
        Tensor [..., Dout]
    """
    dout, din = W.size(0), W.size(1)
    out = torch.addmm(
        x_skip.view(-1, dout), x.view(-1, din), W.T, alpha=residual_scale
    )
    return out.view(*x.shape[:-1], dout)


class TimestepEmbedder(nn.Module):
    """
    Embed scalar timesteps into a high-dimensional vector.

    Uses sinusoidal embeddings followed by an MLP.

    Args:
        hidden_size: output embedding size
        frequency_embedding_size: size of sinusoidal embedding (default: 256)
    """

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
        """
        Compute sinusoidal timestep embeddings.

        Args:
            t: integer tensor of shape [B]
            dim: embedding dimension
            max_period: base frequency for sinusoids

        Returns:
            Tensor [B, dim] of embeddings
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Embed timesteps via sinusoids + MLP.

        Args:
            t: tensor of shape [B]

        Returns:
            Tensor [B, hidden_size]
        """
        freq_emb = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(freq_emb)


class ContinuousPropertyEmbedder(nn.Module):
    """
    Embed a vector of continuous properties to conditioning space.

    Args:
        in_dim: number of properties
        out_dim: dimension of output embedding
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.p_emb = nn.Sequential(
            nn.Linear(in_dim, out_dim), nn.SiLU(), nn.Linear(out_dim, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor [B, in_dim] of property values

        Returns:
            Tensor [B, out_dim]
        """
        return self.p_emb(x.float())


class DDiTBlock(nn.Module):
    """
    Single Transformer block with adaptive layer-norm conditioning.

    Args:
        dim: model hidden size
        n_heads: number of attention heads
        cond_dim: dimension of conditioning embeddings
        mlp_ratio: expansion ratio in MLP (default: 4)
        dropout: dropout probability (default: 0.1)
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        cond_dim: int,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.norm1 = LayerNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.norm2 = LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_ratio * dim, dim),
        )
        self.dropout = dropout
        self.adaLN = nn.Linear(cond_dim, 6 * dim)
        nn.init.zeros_(self.adaLN.weight)
        nn.init.zeros_(self.adaLN.bias)

    def _get_bias_dropout_scale(self):
        return bias_dropout_add_scale_fused_train if self.training else bias_dropout_add_scale_fused_inference

    def forward(
        self,
        x: torch.Tensor,
        rotary_cos_sin: Tuple[torch.Tensor, torch.Tensor],
        cond: torch.Tensor,
        seqlens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply one attention + MLP block with conditioning.

        Args:
            x: input tensor [B, S, D]
            rotary_cos_sin: precomputed cos/sin for rotary embeddings
            cond: conditioning tensor [B, cond_dim]
            seqlens: optional cumulative lengths for FlashAttention

        Returns:
            Output tensor [B, S, D]
        """
        bias_dropout = self._get_bias_dropout_scale()
        shifts_scales = self.adaLN(cond)[:, None].chunk(6, dim=2)
        x_skip = x
        x = _modulate_fused(self.norm1(x), shifts_scales[0], shifts_scales[1])
        qkv = rearrange(self.attn_qkv(x), "b s (three h d) -> b s three h d", three=3, h=self.n_heads)
        cos, sin = rotary_cos_sin
        qkv = _apply_rotary_pos_emb(qkv, cos.to(qkv.dtype), sin.to(qkv.dtype))
        b, s, three, h, d = qkv.shape
        qkv = rearrange(qkv, "b s three h d -> (b s) three h d")
        if seqlens is None:
            cu_seqlens = torch.arange(0, (b + 1) * s, step=s, device=qkv.device, dtype=torch.int32)
        else:
            cu_seqlens = seqlens.cumsum(-1)
        if USE_FLASH_ATTN:
            import flash_attn.flash_attn_interface as fa_int  # noqa: F811
            attn_out = fa_int.flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens, s, 0.0, causal=False)
            x = rearrange(attn_out, "(b s) h d -> b s (h d)", b=b)
        else:
            q, k, v = rearrange(qkv, "(b s) three h d -> three b s h d", b=b)
            q = rearrange(q, "three b s h d -> (b h) s d")
            k = rearrange(k, "three b s h d -> (b h) s d")
            v = rearrange(v, "three b s h d -> (b h) s d")
            attn = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
            x = rearrange(attn, "(b h) s d -> b s (h d)", b=b)
        x = bias_dropout(self.attn_out(x), None, shifts_scales[2], x_skip, self.dropout)
        x = bias_dropout(
            self.mlp(_modulate_fused(self.norm2(x), shifts_scales[3], shifts_scales[4])),
            None,
            shifts_scales[5],
            x,
            self.dropout,
        )
        return x


class EmbeddingLayer(nn.Module):
    """
    Learnable embedding lookup via a parameter matrix.

    Args:
        dim: embedding dimension
        vocab_dim: size of the vocabulary
    """

    def __init__(self, dim: int, vocab_dim: int):
        super().__init__()
        self.embedding = nn.Parameter(torch.empty(vocab_dim, dim))
        nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

    def forward(self, x: torch.LongTensor) -> torch.Tensor:
        """
        Lookup embeddings for input indices.

        Args:
            x: tensor of indices [B, S]

        Returns:
            Tensor [B, S, dim] of embeddings
        """
        return self.embedding[x]


class DDitFinalLayer(nn.Module):
    """
    Final projection layer with adaptive layer-norm modulation.

    Args:
        hidden_size: feature dimension
        out_channels: output vocabulary size
        cond_dim: conditioning dimension
    """

    def __init__(self, hidden_size: int, out_channels: int, cond_dim: int):
        super().__init__()
        self.norm = LayerNorm(hidden_size)
        self.proj = nn.Linear(hidden_size, out_channels)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.adaLN = nn.Linear(cond_dim, 2 * hidden_size)
        nn.init.zeros_(self.adaLN.weight)
        nn.init.zeros_(self.adaLN.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Apply final layer-norm, modulation, and linear projection.

        Args:
            x: input tensor [B, S, hidden_size]
            cond: conditioning tensor [B, cond_dim]

        Returns:
            Output logits [B, S, out_channels]
        """
        shift, scale = self.adaLN(cond)[:, None].chunk(2, dim=2)
        x = _modulate_fused(self.norm(x), shift, scale)
        return self.proj(x)


class DIT(nn.Module, huggingface_hub.PyTorchModelHubMixin):
    """
    Full diffusion Transformer model for SELFIES generation.

    Args:
        config: OmegaConf or dict containing model hyperparameters
        vocab_size: size of the tokenizer vocabulary
    """

    def __init__(self, config: Any, vocab_size: int):
        super().__init__()
        if isinstance(config, dict):
            config = omegaconf.OmegaConf.create(config)
        self.config = config
        self.vocab_embed = EmbeddingLayer(config.model.hidden_size, vocab_size)
        self.time_embed = TimestepEmbedder(config.model.cond_dim)
        self.rotary = Rotary(config.model.hidden_size // config.model.n_heads)

        if (config.conditioning.embeddings or config.conditioning.cfg) and config.conditioning.properties:
            self.prop_embed = ContinuousPropertyEmbedder(
                len(config.conditioning.properties), config.model.cond_dim
            )
        else:
            self.prop_embed = None

        self.blocks = nn.ModuleList(
            DDiTBlock(
                config.model.hidden_size,
                config.model.n_heads,
                config.model.cond_dim,
                dropout=config.model.dropout,
            )
            for _ in range(config.model.n_blocks)
        )
        self.output_layer = DDitFinalLayer(
            config.model.hidden_size, vocab_size, config.model.cond_dim
        )

    def forward(
        self,
        indices: torch.LongTensor,
        sigma: torch.Tensor,
        property_conditioning_vector: Optional[torch.Tensor] = None,
        force_unconditional_pass: bool = False,
    ) -> torch.Tensor:
        """
        Generate logits for each token position given noise level and optional conditions.

        Args:
            indices: input token indices [B, S]
            sigma: noise levels [B]
            property_conditioning_vector: optional conditioning [B, P]
            force_unconditional_pass: mask out conditioning during inference

        Returns:
            Logits tensor [B, S, vocab_size]
        """
        x = self.vocab_embed(indices)
        t_emb = F.silu(self.time_embed(sigma))
        cond = t_emb

        if self.prop_embed is not None and property_conditioning_vector is not None:
            if self.training:
                mask = None
                if self.config.conditioning.cfg:
                    mask = torch.rand(indices.size(0), device=indices.device) > self.config.conditioning.cfg_prob
                cond = cond + self.prop_embed(
                    property_conditioning_vector.to(x.device, dtype=x.dtype)
                    if mask is None else property_conditioning_vector.masked_fill(~mask[:, None], 0)
                )
            elif not force_unconditional_pass:
                cond = cond + self.prop_embed(property_conditioning_vector.to(x.device, dtype=x.dtype))

        rotary_cos_sin = self.rotary(x)
        for block in self.blocks:
            x = block(x, rotary_cos_sin, cond)
        return self.output_layer(x, cond)
