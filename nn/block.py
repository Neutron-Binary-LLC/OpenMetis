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
    """
    A Hybrid Neuro-Symbolic Recurrent Block that combines neural processing
    with a persistent mathematical workspace.
    Supports advanced operations including differentiation, integration, and simplification.
    """
    def __init__(self, config: Optional[MathConfig] = None, **kwargs):
        super().__init__()
        if config is None:
            # Fallback for backward compatibility or direct param injection
            config = MathConfig(
                dim=kwargs.get("d_model", 512),
                n_heads=kwargs.get("num_heads", 8),
                max_loop_iters=kwargs.get("num_iterations", 6),
                workspace_dim=kwargs.get("workspace_dim", 256),
                n_experts=kwargs.get("num_experts", 4),
                dropout=kwargs.get("dropout", 0.1),
                device=str(kwargs.get("device", "cpu"))
            )
        
        self.config = config
        self.d_model = config.dim
        self.num_iterations = config.max_loop_iters
        self.workspace_dim = config.workspace_dim
        self.device = torch.device(config.device)
        
        # Neural components
        self.self_attn = nn.MultiheadAttention(self.d_model, config.n_heads, dropout=config.dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(self.d_model)
        self.norm2 = nn.LayerNorm(self.d_model)
        
        # Prelude layers
        self.prelude = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=self.d_model, nhead=config.n_heads, dim_feedforward=self.d_model*4, dropout=config.dropout, batch_first=True)
            for _ in range(config.prelude_layers)
        ])
        
        # Coda layers
        self.coda = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=self.d_model, nhead=config.n_heads, dim_feedforward=self.d_model*4, dropout=config.dropout, batch_first=True)
            for _ in range(config.coda_layers)
        ])
        
        self.ffn = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(self.d_model * 4, self.d_model),
            nn.Dropout(config.dropout)
        )
        
        # Workspace interaction
        self.workspace_projector = nn.Linear(self.workspace_dim, self.d_model)
        self.workspace_updater = nn.Linear(self.d_model, self.workspace_dim)
        
        # MoE Experts
        self.experts = nn.ModuleList([
            MathExpert(self.d_model, expert_type=t) 
            for t in ["algebra", "calculus", "numerical", "verification"][:config.n_experts]
        ])
        self.router = MoERouter(self.d_model, config.n_experts)
        
        # Math-specific heads for proposing operations
        self.deriv_head = nn.Linear(self.d_model, self.d_model)
        self.integral_head = nn.Linear(self.d_model, self.d_model)
        self.simplify_head = nn.Linear(self.d_model, self.d_model)
        
        # New foundational math heads
        self.trig_head = nn.Linear(self.d_model, self.d_model)      # sin, cos, tan
        self.exp_log_head = nn.Linear(self.d_model, self.d_model)   # exp, log
        self.pow_head = nn.Linear(self.d_model, self.d_model)       # power, sqrt
        
        # Stability mechanism (LTI-style transformation)
        self.stability_gate = nn.Parameter(torch.ones(1) * 0.9)
        self.dropout_layer = nn.Dropout(config.dropout)
        
        # LoRA adapters for depth adaptation (simplified version)
        self.lora_a = nn.Parameter(torch.randn(config.max_loop_iters, self.d_model, config.lora_rank))
        self.lora_b = nn.Parameter(torch.zeros(config.max_loop_iters, config.lora_rank, self.d_model))
        
        self.to(self.device)


    def forward(
        self, 
        x: torch.Tensor, 
        math_state_init: Optional[MathWorkspace] = None,
        num_iterations: Optional[int] = None,
        debug: bool = False
    ) -> Tuple[torch.Tensor, MathWorkspace, Optional[List[Dict[str, Any]]]]:
        """
        x: (batch, seq_len, d_model)
        num_iterations: Override default iterations at inference time.
        """
        batch_size, seq_len, _ = x.shape
        iters = num_iterations if num_iterations is not None else self.num_iterations
        
        trace = [] if debug else None

        # 0. Prelude (Standard Transformer layers)
        for layer in self.prelude:
            x = layer(x)

        # Initialize workspace if not provided
        workspace = math_state_init if math_state_init is not None else \
                    MathWorkspace(batch_size, self.workspace_dim, x.device)
        
        h = x # Current hidden state
        
        for i in range(iters):
            residual = h
            
            # 1. Neural Proposal (Self-Attention)
            h_norm = self.norm1(h)
            attn_out, _ = self.self_attn(h_norm, h_norm, h_norm)
            
            # LoRA depth adaptation
            if i < self.config.max_loop_iters:
                lora_update = (h_norm @ self.lora_a[i]) @ self.lora_b[i]
                attn_out = attn_out + lora_update

            h = self.norm1(h + self.dropout_layer(attn_out))
            
            # 2. Workspace Injection
            # Project workspace latent state into the hidden dimension
            ws_feat = self.workspace_projector(workspace.latent_state).unsqueeze(1) # (batch, 1, d_model)
            # Inject workspace into sequence (concatenation or addition)
            # Here we concatenate to allow the model to attend to it specifically
            h_augmented = torch.cat([h, ws_feat], dim=1) 
            
            # 3. MoE Expert Processing
            h_ffn = self.ffn(h_augmented)
            
            # Route based on the augmented state
            exp_weights, exp_logits = self.router(h_ffn) # (batch, seq_len+1, num_experts)
            
            expert_outputs = torch.stack([expert(h_ffn, debug=debug) for expert in self.experts], dim=-1) # (B, S+1, D, E)
            combined_expert_out = torch.sum(expert_outputs * exp_weights.unsqueeze(2), dim=-1)
            
            if debug:
                iter_trace = {
                    "iteration": i,
                    "expert_weights": exp_weights.detach().cpu(),
                    "expert_logits": exp_logits.detach().cpu(),
                    "input_mean": h_ffn.mean().item(),
                    "input_std": h_ffn.std().item()
                }
                trace.append(iter_trace)

            # Residual connection and stabilization
            # Trim back to original sequence length for the main hidden state
            h = self.norm2(residual + self.dropout_layer(combined_expert_out[:, :seq_len, :]))
            
            # 4. Symbolic Update (Propose changes to workspace)
            delta_ws = self._perform_math_operations(h, workspace)
            
            # Update workspace
            workspace.update(delta_ws)
            
            # ACT (Adaptive Computation Time) halting
            if self.config.act_threshold < 1.0:
                # Simplified ACT: if confidence is high enough across batch, we could stop
                # For simplicity in this demo, we just track it
                if torch.all(workspace.confidence > self.config.act_threshold):
                    break

        # 5. Coda (Standard Transformer layers)
        for layer in self.coda:
            h = layer(h)
            
        return h, workspace, trace

    def _perform_math_operations(self, h: torch.Tensor, workspace: MathWorkspace) -> torch.Tensor:
        """Core neuro-symbolic math engine."""
        # Mean pool for global decision
        global_feat = h.mean(dim=1)  # (B, D)

        # Propose operations
        deriv_feat = self.deriv_head(global_feat)
        integ_feat = self.integral_head(global_feat)
        simp_feat = self.simplify_head(global_feat)
        trig_feat = self.trig_head(global_feat)
        explog_feat = self.exp_log_head(global_feat)
        pow_feat = self.pow_head(global_feat)

        # Combine proposals
        math_update = deriv_feat + integ_feat + simp_feat + trig_feat + explog_feat + pow_feat

        # Differentiable operations via autograd
        if workspace.numerical_values.requires_grad:
            vars_tensor = workspace.numerical_values[:, :5]
            if vars_tensor.requires_grad:
                dummy_loss = vars_tensor.sum()
                grad = torch.autograd.grad(
                    dummy_loss, vars_tensor, create_graph=True, allow_unused=True
                )[0]
                if grad is not None:
                    workspace.numerical_values[:, 5:10] = grad

        # Simple symbolic-style integration approximation
        integral_approx = torch.tanh(integ_feat) * 0.1

        delta_latent = self.workspace_updater(math_update + integral_approx)

        # Update confidence based on "consistency" (magnitude of update)
        new_conf = torch.sigmoid(torch.mean(torch.abs(delta_latent), dim=1, keepdim=True))
        workspace.confidence = new_conf

        # Update numerical slots
        workspace.numerical_values[:, 10:12] = deriv_feat.mean(dim=1, keepdim=True)

        return delta_latent

    def compute_derivative(self, f_values: torch.Tensor, x_values: torch.Tensor) -> torch.Tensor:
        """Differentiable numerical differentiation."""
        dx = x_values[:, 1:] - x_values[:, :-1]
        df = f_values[:, 1:] - f_values[:, :-1]
        return df / (dx + 1e-8)

    def symbolic_integration_approx(self, expr_latent: torch.Tensor) -> torch.Tensor:
        """Approximate integration in latent space."""
        return torch.cumsum(expr_latent, dim=1) * 0.05

    def apply_trigonometric(self, x: torch.Tensor, func_type: str = "sin") -> torch.Tensor:
        """Apply foundational trigonometric functions."""
        if func_type == "sin":
            return torch.sin(x)
        elif func_type == "cos":
            return torch.cos(x)
        elif func_type == "tan":
            return torch.tan(x)
        return x

    def apply_exponential_log(self, x: torch.Tensor, func_type: str = "exp") -> torch.Tensor:
        """Apply exponential or logarithmic functions."""
        if func_type == "exp":
            return torch.exp(x)
        elif func_type == "log":
            # Add epsilon for numerical stability
            return torch.log(torch.abs(x) + 1e-8)
        return x

    def apply_power(self, x: torch.Tensor, exponent: float = 2.0) -> torch.Tensor:
        """Apply power functions."""
        if exponent == 0.5:
            return torch.sqrt(torch.abs(x) + 1e-8)
        return torch.pow(x, exponent)
