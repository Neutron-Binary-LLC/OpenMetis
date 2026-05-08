from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .workspace import MathWorkspace


@dataclass
class MathConfig:
    """
    Hyperparameter configuration for NeuroSymbolicReasoningCell.
    """
    # Core
    vocab_size: int = 32000
    dim: int = 2048
    n_heads: int = 16
    n_kv_heads: int = 4  # GQA: fewer KV heads than Q heads
    max_seq_len: int = 4096
    max_loop_iters: int = 16  # T — recurrent depth at inference
    prelude_layers: int = 2
    coda_layers: int = 2
    
    # Attention (attn_type selects between the two):
    attn_type: str = "mla"  # "gqa" | "mla"
    kv_lora_rank: int = 512  # [MLA] compressed KV latent cached instead of full K/V
    q_lora_rank: int = 1536  # [MLA] compressed Q latent dim
    qk_rope_head_dim: int = 64  # [MLA] per-head dims that receive RoPE
    qk_nope_head_dim: int = 128  # [MLA] per-head dims without positional encoding
    v_head_dim: int = 128  # [MLA] per-head value dimension
    
    # MoE FFN (used inside the recurrent block):
    n_experts: int = 64
    n_shared_experts: int = 2
    n_experts_per_tok: int = 4  # top-K routed
    expert_dim: int = 512  # fine-grained: dim // (n_experts // n_experts_per_tok)
    
    # Other:
    workspace_dim: int = 256
    act_threshold: float = 0.99  # ACT halting threshold
    rope_theta: float = 500000.0  # RoPE base frequency
    lora_rank: int = 16  # rank of the per-loop depth-wise LoRA adapter
    max_output_tokens: int = 4096
    dropout: float = 0.0
    device: str = "cpu"

class MathExpert(nn.Module):
    """
    A specialized expert for specific mathematical tasks.
    """
    def __init__(self, d_model: int, expert_type: str = "algebra"):
        super().__init__()
        self.expert_type = expert_type
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(0.1)
        )
        
    def forward(self, x: torch.Tensor, debug: bool = False) -> torch.Tensor:
        if debug:
            print(f"[MathExpert Trace] Type: {self.expert_type}, Input shape: {x.shape}, Mean: {x.mean().item():.4f}")
        return self.net(x)

class MoERouter(nn.Module):
    """
    Lightweight MoE router to select math experts.
    """
    def __init__(self, d_model: int, num_experts: int):
        super().__init__()
        self.router = nn.Linear(d_model, num_experts)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (batch, seq_len, d_model)
        logits = self.router(x)
        weights = F.softmax(logits, dim=-1)
        return weights, logits

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (no bias, no mean subtraction)."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        """Create an RMSNorm layer.

        Args:
            dim: Feature dimension to normalise over (the last axis of input).
            eps: Stability constant added before the reciprocal square-root to
                 prevent division by zero when the RMS is near zero.
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise *x* by its root-mean-square and apply a learnable scale.

        Args:
            x: Input tensor of arbitrary shape ``[..., dim]``.

        Returns:
            Normalised tensor, same shape as *x*.
        """
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight

class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) with lazy cache extension.

    Args:
        dim:         Per-head dimension (head_dim).
        max_seq_len: Initial cache size.
        base:        Frequency base (default 10 000).
    """

    def __init__(
            self, dim: int, max_seq_len: int = 8_192, base: float = 10_000.0
    ) -> None:
        """Initialise RoPE and pre-compute the cos/sin cache.

        Args:
            dim:         Per-head dimension.  Must be even.
            max_seq_len: Number of positions to cache on construction.  The
                         cache doubles automatically when a longer sequence
                         is encountered.
            base:        Frequency base θ.  Higher values slow the rotation
                         rate, extending effective context length.
        """
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        """Pre-compute and register ``_cos`` / ``_sin`` buffers up to *seq_len*.

        Called once at init and again (doubling capacity) whenever ``forward``
        is asked for a sequence longer than the current cache.

        Args:
            seq_len: Number of positions to pre-compute.
        """
        t = torch.arange(
            seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype
        )
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # [T, dim/2]
        emb = torch.cat([freqs, freqs], dim=-1)  # [T, dim]
        self.register_buffer("_cos", emb.cos()[None, None], persistent=False)
        self.register_buffer("_sin", emb.sin()[None, None], persistent=False)

    def forward(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return cached (cos, sin) tables for the first *seq_len* positions.

        Args:
            seq_len: Number of positions required.

        Returns:
            Tuple of ``(cos, sin)``, each shaped ``[1, 1, seq_len, dim]``,
            ready to broadcast with ``[B, H, T, dim]`` query / key tensors.
        """
        if seq_len > self._cos.shape[2]:
            self._build_cache(seq_len * 2)
        return self._cos[:, :, :seq_len], self._sin[:, :, :seq_len]

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Return *x* with its last dimension split and swapped with negation.

    Given ``x = [x₁, x₂]`` (each half of the last dim), returns
    ``[-x₂, x₁]``.  Combined with the cos/sin multiply in
    :func:`apply_rotary_emb` this implements the 2-D rotation matrix
    that defines RoPE.

    Args:
        x: Tensor with an even-sized last dimension.

    Returns:
        Rotated tensor with the same shape as *x*.
    """
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)

def apply_rotary_emb(
        x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Apply Rotary Position Embeddings in-place to query or key tensors.

    Implements ``x_rot = x * cos + rotate_half(x) * sin``, which is
    equivalent to multiplying each consecutive pair of dimensions by a
    2-D rotation matrix whose angle depends on the sequence position and
    the dimension's frequency.

    Args:
        x:   Query or key tensor, shape ``[B, H, T, d]``.
        cos: Pre-computed cosines, shape ``[1, 1, T, d]``.
        sin: Pre-computed sines,   shape ``[1, 1, T, d]``.

    Returns:
        Rotated tensor with the same shape and dtype as *x*.
    """
    return x * cos + _rotate_half(x) * sin


class NeuroSymbolicReasoningCell(nn.Module):
    def __init__(self, config: MathConfig):
        super().__init__()
        self.config = config
        self.d_model = config.dim
        self.workspace_dim = config.workspace_dim

        # === Core Components ===
        self.norm1 = RMSNorm(self.d_model)  # Use the good RMSNorm you wrote
        self.norm2 = RMSNorm(self.d_model)

        # Use your RotaryEmbedding
        self.rope = RotaryEmbedding(dim=config.qk_rope_head_dim, base=config.rope_theta)

        # Attention - simplified for now (can upgrade to MLA later)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=config.n_heads,
            dropout=config.dropout,
            batch_first=True
        )

        # Prelude / Coda
        self.prelude = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.d_model, nhead=config.n_heads,
                dim_feedforward=self.d_model * 4, dropout=config.dropout,
                batch_first=True, norm_first=True
            ) for _ in range(config.prelude_layers)
        ])

        self.coda = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.d_model, nhead=config.n_heads,
                dim_feedforward=self.d_model * 4, dropout=config.dropout,
                batch_first=True, norm_first=True
            ) for _ in range(config.coda_layers)
        ])

        # Workspace fusion
        self.workspace_cross_attn = nn.MultiheadAttention(
            embed_dim=self.d_model, num_heads=8, dropout=config.dropout, batch_first=True
        )
        self.workspace_projector = nn.Linear(self.workspace_dim, self.d_model)
        self.workspace_updater = nn.Linear(self.d_model, self.workspace_dim)

        # MoE (kept simple for now)
        self.experts = nn.ModuleList([MathExpert(self.d_model, t)
                                      for t in ["algebra", "calculus", "trig", "logic", "verification"]])
        self.router = MoERouter(self.d_model, len(self.experts))

        # Typed math proposal heads
        self.operation_heads = nn.ModuleDict({
            'derivative': nn.Linear(self.d_model, self.d_model),
            'integral': nn.Linear(self.d_model, self.d_model),
            'simplify': nn.Linear(self.d_model, self.d_model),
            'trig': nn.Linear(self.d_model, self.d_model),
            'exp_log': nn.Linear(self.d_model, self.d_model),
            'power': nn.Linear(self.d_model, self.d_model),
        })

        self.lora_a = nn.Parameter(torch.randn(config.max_loop_iters, self.d_model, config.lora_rank))
        self.lora_b = nn.Parameter(torch.zeros(config.max_loop_iters, config.lora_rank, self.d_model))

        self.dropout = nn.Dropout(config.dropout)

    def _propose_symbolic_update(self, h: torch.Tensor, workspace: MathWorkspace, step: int):
        """Typed, structured symbolic proposal."""
        global_feat = h.mean(dim=1)  # (B, D)

        proposals = {name: head(global_feat) for name, head in self.operation_heads.items()}

        # Combine with learned gating
        gate = torch.sigmoid(torch.stack(list(proposals.values()), dim=1))  # (B, num_ops, D)
        combined = torch.sum(gate * torch.stack(list(proposals.values()), dim=1), dim=1)

        delta_latent = self.workspace_updater(combined)

        # Optional: Apply concrete operations to numerical slots
        if workspace.numerical_values.requires_grad:
            # Example: numerical differentiation / integration on selected slots
            pass

        # Generate readable expression (for interpretability)
        delta_expr = [f"step{step}_update" for _ in range(len(global_feat))]

        return delta_latent, delta_expr

    def forward(self, x: torch.Tensor, workspace: Optional[MathWorkspace] = None,
                num_iterations: Optional[int] = None, debug: bool = False):

        batch_size, seq_len, _ = x.shape
        iters = num_iterations or self.config.max_loop_iters
        trace = [] if debug else None

        # Prelude
        for layer in self.prelude:
            x = layer(x)

        if workspace is None:
            workspace = MathWorkspace.new(batch_size, self.workspace_dim, x.device)

        h = x

        for i in range(iters):
            residual = h

            # 1. Self-Attention with LoRA
            h_norm = self.norm1(h)
            attn_out, _ = self.self_attn(h_norm, h_norm, h_norm)

            # Depth-specific LoRA
            if i < self.config.max_loop_iters:
                lora_update = (h_norm @ self.lora_a[i]) @ self.lora_b[i]
                attn_out = attn_out + lora_update

            h = residual + self.dropout(attn_out)
            h = self.norm1(h)  # Clean pre-norm style

            # 2. Workspace Cross-Attention (much cleaner!)
            ws_token = self.workspace_projector(workspace.latent_state).unsqueeze(1)  # (B, 1, D)
            h_aug, _ = self.workspace_cross_attn(
                query=h, key=ws_token, value=ws_token
            )
            h = h + self.dropout(h_aug)

            # 3. MoE FFN
            h_ffn = self.norm2(h)
            weights, _ = self.router(h_ffn)
            expert_outs = torch.stack([e(h_ffn) for e in self.experts], dim=-1)
            moe_out = torch.sum(expert_outs * weights.unsqueeze(2), dim=-1)

            h = h + self.dropout(moe_out)

            # 4. Symbolic Workspace Update
            delta_latent, delta_expr = self._propose_symbolic_update(h, workspace, i)
            workspace.update(delta_latent, delta_expr)

            # ACT Halting
            if torch.all(workspace.confidence > self.config.act_threshold):
                if debug:
                    print(f"Early stopping at iteration {i}")
                break

        # Coda
        for layer in self.coda:
            h = layer(h)

        return h, workspace, trace