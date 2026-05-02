import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List


class MathWorkspace:
    """Persistent mathematical workspace across recurrent iterations."""

    def __init__(
            self,
            batch_size: int,
            workspace_dim: int,
            device: torch.device = torch.device('cpu'),
            num_num_slots: int = 16
    ):
        self.batch_size = batch_size
        self.workspace_dim = workspace_dim
        self.device = device
        self.num_num_slots = num_num_slots

        # Latent representation of current mathematical expression/context
        self.latent_state = torch.zeros(batch_size, workspace_dim, device=device)

        # Numerical values (variables, constants, evaluation points)
        self.numerical_values = torch.zeros(batch_size, num_num_slots, device=device)

        # Metadata
        self.iteration_count = 0
        self.confidence = torch.ones(batch_size, 1, device=device)

        # Store history of expressions (for debugging / backtracking)
        self.expression_history: List[str] = [""] * batch_size

    def update(self, delta_latent: torch.Tensor, new_confidence: Optional[torch.Tensor] = None):
        self.latent_state = self.latent_state + delta_latent
        if new_confidence is not None:
            self.confidence = new_confidence
        self.iteration_count += 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "latent_state": self.latent_state,
            "numerical_values": self.numerical_values,
            "iteration_count": self.iteration_count,
            "confidence": self.confidence,
        }

    def clone(self):
        new_ws = MathWorkspace(self.batch_size, self.workspace_dim, self.device, self.num_num_slots)
        new_ws.latent_state = self.latent_state.clone()
        new_ws.numerical_values = self.numerical_values.clone()
        new_ws.iteration_count = self.iteration_count
        new_ws.confidence = self.confidence.clone()
        new_ws.expression_history = self.expression_history.copy()
        return new_ws


class HybridRecurrentMathBlock(nn.Module):
    """
    Hybrid Neuro-Symbolic Recurrent Block with Mathematical Workspace.
    Supports advanced operations including differentiation and integration.
    """

    def __init__(
            self,
            d_model: int = 512,
            workspace_dim: int = 256,
            num_heads: int = 8,
            num_iterations: int = 6,
            num_experts: int = 4,
            dropout: float = 0.1,
            device: torch.device = torch.device('cpu')
    ):
        super().__init__()
        self.d_model = d_model
        self.workspace_dim = workspace_dim
        self.num_iterations = num_iterations
        self.device = device

        # Neural components
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )

        # Workspace projection
        self.workspace_proj = nn.Linear(workspace_dim, d_model)
        self.workspace_updater = nn.Linear(d_model, workspace_dim)

        # MoE Router (simple top-2)
        self.num_experts = num_experts
        self.expert_router = nn.Linear(d_model, num_experts)

        # Expert heads for different mathematical operations
        self.experts = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(num_experts)
        ])

        # Math-specific heads
        self.deriv_head = nn.Linear(d_model, d_model)  # Propose derivative
        self.integral_head = nn.Linear(d_model, d_model)  # Propose integral
        self.simplify_head = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.to(device)

    def forward(
            self,
            x: torch.Tensor,
            workspace: Optional[MathWorkspace] = None,
            num_iterations: Optional[int] = None
    ) -> Tuple[torch.Tensor, MathWorkspace]:
        """
        x: (batch, seq_len, d_model)
        """
        batch_size = x.shape[0]
        num_iters = num_iterations or self.num_iterations

        if workspace is None:
            workspace = MathWorkspace(
                batch_size=batch_size,
                workspace_dim=self.workspace_dim,
                device=self.device
            )

        h = x

        for it in range(num_iters):
            residual = h

            # 1. Neural Attention
            attn_out, _ = self.attention(h, h, h)
            h = self.norm1(h + self.dropout(attn_out))

            # 2. Project workspace into hidden space
            ws_embed = self.workspace_proj(workspace.latent_state).unsqueeze(1)  # (B, 1, D)
            h = torch.cat([h, ws_embed], dim=1)  # Inject workspace

            # 3. FFN + MoE
            ffn_out = self.ffn(h)

            # Simple MoE routing
            router_logits = self.expert_router(h.mean(dim=1))
            router_weights = F.softmax(router_logits, dim=-1)
            expert_out = torch.zeros_like(ffn_out)

            for e_idx, expert in enumerate(self.experts):
                mask = router_weights[:, e_idx].unsqueeze(1).unsqueeze(2)
                expert_out += mask * expert(ffn_out)

            h = self.norm2(residual + self.dropout(expert_out))

            # 4. Hybrid Math Operations
            delta_ws = self._perform_math_operations(h, workspace)

            # Update workspace
            workspace.update(delta_ws)

            # Trim back to original seq_len if needed
            h = h[:, :x.shape[1], :]

        return h, workspace

    def _perform_math_operations(self, h: torch.Tensor, workspace: MathWorkspace) -> torch.Tensor:
        """Core neuro-symbolic math engine."""
        batch_size = h.shape[0]

        # Mean pool for global decision
        global_feat = h.mean(dim=1)  # (B, D)

        # Propose operations
        deriv_feat = self.deriv_head(global_feat)
        integ_feat = self.integral_head(global_feat)
        simp_feat = self.simplify_head(global_feat)

        # Combine proposals
        math_update = deriv_feat + integ_feat + simp_feat

        # Differentiable operations via autograd
        # Example: If numerical_values[0] represents a function output,
        # we can compute gradients w.r.t. variables
        if workspace.numerical_values.requires_grad:
            # Example derivative computation
            vars_tensor = workspace.numerical_values[:, :5]  # Assume first 5 are variables
            if vars_tensor.requires_grad:
                # Simulate a loss for gradient-based differentiation
                dummy_loss = vars_tensor.sum()
                grad = torch.autograd.grad(
                    dummy_loss, vars_tensor, create_graph=True, allow_unused=True
                )[0]
                if grad is not None:
                    workspace.numerical_values[:, 5:10] = grad  # Store gradients

        # Simple symbolic-style integration approximation (learnable)
        # In production, you could call external SymPy in a differentiable wrapper
        # or use series expansion + neural correction
        integral_approx = torch.tanh(integ_feat) * 0.1  # Placeholder

        delta_latent = self.workspace_updater(math_update + integral_approx)

        # Update confidence based on "consistency"
        new_conf = torch.sigmoid(torch.mean(torch.abs(delta_latent), dim=1, keepdim=True))

        # Update workspace numerical slots with math results
        workspace.numerical_values[:, 10:12] = deriv_feat.mean(dim=1, keepdim=True)  # example

        return delta_latent

    def compute_derivative(self, f_values: torch.Tensor, x_values: torch.Tensor) -> torch.Tensor:
        """Differentiable numerical differentiation."""
        # Central difference
        dx = x_values[:, 1:] - x_values[:, :-1]
        df = f_values[:, 1:] - f_values[:, :-1]
        return df / (dx + 1e-8)

    def symbolic_integration_approx(self, expr_latent: torch.Tensor) -> torch.Tensor:
        """Placeholder for more advanced integration (can be extended)."""
        # Could integrate neural network approximation or series
        return torch.cumsum(expr_latent, dim=1) * 0.05

    # How to Use
    # Pythondevice = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #
    # block = HybridRecurrentMathBlock(
    #     d_model=512,
    #     workspace_dim=256,
    #     num_iterations=8,
    #     device=device
    # )
    #
    # x = torch.randn(4, 32, 512, device=device)  # batch=4, seq=32
    #
    # output, final_workspace = block(x, num_iterations=6)
    #
    # print("Final confidence:", final_workspace.confidence)
    # print("Iterations:", final_workspace.iteration_count)