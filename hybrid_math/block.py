import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional, Any
from .workspace import MathWorkspace

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
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

class HybridRecurrentMathBlock(nn.Module):
    """
    A Hybrid Neuro-Symbolic Recurrent Block that combines neural processing
    with a persistent mathematical workspace.
    Supports advanced operations including differentiation, integration, and simplification.
    """
    def __init__(
        self, 
        d_model: int = 512, 
        num_heads: int = 8, 
        num_iterations: int = 6, 
        workspace_dim: int = 256, 
        num_experts: int = 4,
        dropout: float = 0.1,
        device: torch.device = torch.device('cpu')
    ):
        super().__init__()
        self.d_model = d_model
        self.num_iterations = num_iterations
        self.workspace_dim = workspace_dim
        self.device = device
        
        # Neural components
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        
        # Workspace interaction
        self.workspace_projector = nn.Linear(workspace_dim, d_model)
        self.workspace_updater = nn.Linear(d_model, workspace_dim)
        
        # MoE Experts
        self.experts = nn.ModuleList([
            MathExpert(d_model, expert_type=t) 
            for t in ["algebra", "calculus", "numerical", "verification"][:num_experts]
        ])
        self.router = MoERouter(d_model, num_experts)
        
        # Math-specific heads for proposing operations
        self.deriv_head = nn.Linear(d_model, d_model)
        self.integral_head = nn.Linear(d_model, d_model)
        self.simplify_head = nn.Linear(d_model, d_model)
        
        # Stability mechanism (LTI-style transformation)
        self.stability_gate = nn.Parameter(torch.ones(1) * 0.9)
        self.dropout_layer = nn.Dropout(dropout)
        self.to(device)
        
    def forward(
        self, 
        x: torch.Tensor, 
        math_state_init: Optional[MathWorkspace] = None,
        num_iterations: Optional[int] = None
    ) -> Tuple[torch.Tensor, MathWorkspace]:
        """
        x: (batch, seq_len, d_model)
        num_iterations: Override default iterations at inference time.
        """
        batch_size, seq_len, _ = x.shape
        iters = num_iterations if num_iterations is not None else self.num_iterations
        
        # Initialize workspace if not provided
        workspace = math_state_init if math_state_init is not None else \
                    MathWorkspace(batch_size, self.workspace_dim, x.device)
        
        h = x # Current hidden state
        
        for i in range(iters):
            residual = h
            
            # 1. Neural Proposal (Self-Attention)
            h_norm = self.norm1(h)
            attn_out, _ = self.self_attn(h_norm, h_norm, h_norm)
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
            exp_weights, _ = self.router(h_ffn) # (batch, seq_len+1, num_experts)
            
            expert_outputs = torch.stack([expert(h_ffn) for expert in self.experts], dim=-1) # (B, S+1, D, E)
            combined_expert_out = torch.sum(expert_outputs * exp_weights.unsqueeze(2), dim=-1)
            
            # Residual connection and stabilization
            # Trim back to original sequence length for the main hidden state
            h = self.norm2(residual + self.dropout_layer(combined_expert_out[:, :seq_len, :]))
            
            # 4. Symbolic Update (Propose changes to workspace)
            delta_ws = self._perform_math_operations(h, workspace)
            
            # Update workspace
            workspace.update(delta_ws)
            
        return h, workspace

    def _perform_math_operations(self, h: torch.Tensor, workspace: MathWorkspace) -> torch.Tensor:
        """Core neuro-symbolic math engine."""
        # Mean pool for global decision
        global_feat = h.mean(dim=1)  # (B, D)

        # Propose operations
        deriv_feat = self.deriv_head(global_feat)
        integ_feat = self.integral_head(global_feat)
        simp_feat = self.simplify_head(global_feat)

        # Combine proposals
        math_update = deriv_feat + integ_feat + simp_feat

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
