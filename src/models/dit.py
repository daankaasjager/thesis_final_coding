import math
import typing
import huggingface_hub
import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import logging
import sys
from src.utils.get_torch_dtype import get_torch_dtype


logger = logging.getLogger(__name__)

# Detect OS and import appropriate efficient attention library.
if sys.platform.startswith('win'):
    USE_FLASH_ATTN = False
else:
    try:
        import flash_attn
        import flash_attn.layers.rotary
        USE_FLASH_ATTN = True
    except ImportError:
        USE_FLASH_ATTN = False

# Flags required to enable JIT fusion kernels
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)


def bias_dropout_add_scale(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float,
    training: bool) -> torch.Tensor:
    if bias is not None:
        out = scale * F.dropout(x + bias, p=prob, training=training)
    else:
        out = scale * F.dropout(x, p=prob, training=training)
    if residual is not None:
        out = residual + out
    return out


def get_bias_dropout_add_scale(training):
    def _bias_dropout_add(x, bias, scale, residual, prob):
        return bias_dropout_add_scale(x, bias, scale, residual, prob, training)
    return _bias_dropout_add


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


@torch.jit.script
def bias_dropout_add_scale_fused_train(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float) -> torch.Tensor:
    return bias_dropout_add_scale(x, bias, scale, residual, prob, True)


@torch.jit.script
def bias_dropout_add_scale_fused_inference(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float) -> torch.Tensor:
    return bias_dropout_add_scale(x, bias, scale, residual, prob, False)


@torch.jit.script
def modulate_fused(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return modulate(x, shift, scale)


class Rotary(nn.Module):
    def __init__(self, dim, base=10_000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, seq_dim=1):
        seq_len = x.shape[seq_dim]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq.clone())
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            # dims: batch, seq_len, qkv, head, dim
            self.cos_cached = emb.cos()[None, :, None, None, :].repeat(1, 1, 3, 1, 1)
            self.sin_cached = emb.sin()[None, :, None, None, :].repeat(1, 1, 3, 1, 1)
            # Make the transformation on v an identity.
            self.cos_cached[:, :, 2, :, :].fill_(1.)
            self.sin_cached[:, :, 2, :, :].fill_(0.)
        return self.cos_cached, self.sin_cached


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(qkv, cos, sin):
    if USE_FLASH_ATTN:
        # Use flash_attn's rotary function
        cos_ = cos[0, :, 0, 0, :cos.shape[-1] // 2]
        sin_ = sin[0, :, 0, 0, :sin.shape[-1] // 2]
        return flash_attn.layers.rotary.apply_rotary_emb_qkv_(qkv, cos_, sin_)
    else:
        # Custom implementation for Windows
        # Assume qkv shape: [b, s, three, h, d]
        d_rot = qkv.shape[-1] // 2
        qkv_rot = qkv[..., :d_rot]
        qkv_pass = qkv[..., d_rot:]
        # Expand cos and sin from shape [1, s, 1, 1, d_rot] to qkv's shape
        b, s, three, h, _ = qkv.shape
        cos_exp = cos[..., :d_rot].expand(b, s, three, h, d_rot)
        sin_exp = sin[..., :d_rot].expand(b, s, three, h, d_rot)
        rotated = qkv_rot * cos_exp + rotate_half(qkv_rot) * sin_exp
        return torch.cat([rotated, qkv_pass], dim=-1)


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#                                  Layers                                       #
#################################################################################
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones([dim]))
        self.dim = dim

    def forward(self, x):
        with torch.amp.autocast(device_type = x.device.type, enabled=False):
            x = F.layer_norm(x.float(), [self.dim])
        return x * self.weight[None, None, :]


def residual_linear(x, W, x_skip, residual_scale):
    """x_skip + residual_scale * W @ x"""
    dim_out, dim_in = W.shape[0], W.shape[1]
    return torch.addmm(x_skip.view(-1, dim_out),
                       x.view(-1, dim_in),
                       W.T,
                       alpha=residual_scale).view(*x.shape[:-1], dim_out)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True)
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(- math.log(max_period) * torch.arange(0, half, dtype=torch.float32) / half).to(t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """Embeds class labels into vector representations.
    Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, cond_size):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes + 1, cond_size)
        self.num_classes = num_classes

    def forward(self, labels):
        return self.embedding_table(labels)


#################################################################################
#                                 Core Model                                    #
#################################################################################
class DDiTBlock(nn.Module):
    def __init__(self, dim, n_heads, cond_dim, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.norm1 = LayerNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim, bias=True),
            nn.GELU(approximate='tanh'),
            nn.Linear(mlp_ratio * dim, dim, bias=True)
        )
        self.dropout2 = nn.Dropout(dropout)
        self.dropout = dropout
        self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

    def _get_bias_dropout_scale(self):
        if self.training:
            return bias_dropout_add_scale_fused_train
        else:
            return bias_dropout_add_scale_fused_inference

    def forward(self, x, rotary_cos_sin, c, seqlens=None):
        batch_size, seq_len = x.shape[0], x.shape[1]
        bias_dropout_scale_fn = self._get_bias_dropout_scale()

        (shift_msa, scale_msa, gate_msa,
         shift_mlp, scale_mlp, gate_mlp) = self.adaLN_modulation(c)[:, None].chunk(6, dim=2)

        # Attention operation
        x_skip = x
        x = modulate_fused(self.norm1(x), shift_msa, scale_msa)
        qkv = self.attn_qkv(x)
        qkv = rearrange(qkv, 'b s (three h d) -> b s three h d', three=3, h=self.n_heads)
        with torch.amp.autocast(device_type= qkv.device.type, enabled=False):
            cos, sin = rotary_cos_sin
            qkv = apply_rotary_pos_emb(qkv, cos.to(qkv.dtype), sin.to(qkv.dtype))
        # Flatten for attention
        qkv = rearrange(qkv, 'b s ... -> (b s) ...')
        if seqlens is None:
            cu_seqlens = torch.arange(0, (batch_size + 1) * seq_len, step=seq_len,
                                        dtype=torch.int32, device=qkv.device)
        else:
            cu_seqlens = seqlens.cumsum(-1)

        if USE_FLASH_ATTN:
            # Use flash_attn for Linux
            x = flash_attn.flash_attn_interface.flash_attn_varlen_qkvpacked_func(
                qkv, cu_seqlens, seq_len, 0., causal=False)
            x = rearrange(x, '(b s) h d -> b s (h d)', b=batch_size)
        else:
            # Fallback to standard scaled dot product attention
            qkv = rearrange(qkv, '(b s) three h d -> b s three h d', b=batch_size, three=3)
            q, k, v = qkv.unbind(dim=2)
            q = rearrange(q, 'b s h d -> (b h) s d')
            k = rearrange(k, 'b s h d -> (b h) s d')
            v = rearrange(v, 'b s h d -> (b h) s d')
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
            x = rearrange(out, '(b h) s d -> b s (h d)', b=batch_size)

        x = bias_dropout_scale_fn(self.attn_out(x), None, gate_msa, x_skip, self.dropout)
        # MLP operation
        x = bias_dropout_scale_fn(self.mlp(modulate_fused(self.norm2(x), shift_mlp, scale_mlp)),
                                    None, gate_mlp, x, self.dropout)
        return x


class EmbeddingLayer(nn.Module):
    def __init__(self, dim, vocab_dim):
        super().__init__()
        self.embedding = nn.Parameter(torch.empty((vocab_dim, dim)))
        torch.nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

    def forward(self, x):
        return self.embedding[x]


class DDitFinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels, cond_dim):
        super().__init__()
        self.norm_final = LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels)
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()
        self.adaLN_modulation = nn.Linear(cond_dim, 2 * hidden_size, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c)[:, None].chunk(2, dim=2)
        x = modulate_fused(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DIT(nn.Module, huggingface_hub.PyTorchModelHubMixin):
    def __init__(self, config, vocab_size: int):
        super().__init__()
        if isinstance(config, dict):
            config = omegaconf.OmegaConf.create(config)
        self.config = config
        self.vocab_size = vocab_size
        self.vocab_embed = EmbeddingLayer(config.model.hidden_size, vocab_size)
        self.sigma_map = TimestepEmbedder(config.model.cond_dim)
        self.rotary_emb = Rotary(config.model.hidden_size // config.model.n_heads)
        blocks = []
        for _ in range(config.model.n_blocks):
            blocks.append(DDiTBlock(config.model.hidden_size,
                                     config.model.n_heads,
                                     config.model.cond_dim,
                                     dropout=config.model.dropout))
        self.blocks = nn.ModuleList(blocks)
        self.output_layer = DDitFinalLayer(config.model.hidden_size, vocab_size, config.model.cond_dim)
        self.scale_by_sigma = config.model.scale_by_sigma

    def _get_bias_dropout_scale(self):
        if self.training:
            return bias_dropout_add_scale_fused_train
        else:
            return bias_dropout_add_scale_fused_inference

    def forward(self, indices, sigma):
        # --- START FIX ---
        # Ensure sigma is 1D before passing to TimestepEmbedder
        if sigma.ndim > 1:
            # Squeeze the last dimension if it exists and is 1
            if sigma.shape[-1] == 1:
                sigma = sigma.squeeze(-1)
            else:
                # Handle unexpected shape if necessary, or raise error
                logger.warning(f"Received sigma with unexpected shape {sigma.shape} in DIT.forward")
                # Attempt to squeeze anyway, or raise error if ambiguous
                sigma = sigma.squeeze() # Be cautious with general squeeze

        # Ensure it's 1D now, otherwise TimestepEmbedder will likely fail
        if sigma.ndim != 1:
             raise ValueError(f"Sigma must be 1D after processing, but got shape {sigma.shape}")
        # --- END FIX ---

        x = self.vocab_embed(indices)
        # Now pass the 1D sigma to sigma_map
        c = F.silu(self.sigma_map(sigma)) # sigma_map now receives (B,)

        rotary_cos_sin = self.rotary_emb(x)
        with torch.amp.autocast(device_type=x.device.type, dtype=get_torch_dtype(self.config.trainer.precision)):
            for i in range(len(self.blocks)):
                # c passed to blocks will now have the correct shape (B, cond_dim)
                x = self.blocks[i](x, rotary_cos_sin, c, seqlens=None)
            x = self.output_layer(x, c)
        return x