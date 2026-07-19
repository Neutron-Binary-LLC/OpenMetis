import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List
from .expression import ExpressionTree, ExpressionNode


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

        self.symbolic_expr: List[str]  # Human-readable expressions (for debugging / verification)
        self.expression_trees: List[Optional[ExpressionTree]] = [None] * batch_size

        # Store history of expressions (for debugging / backtracking)
        self.expression_history: List[str] = [""] * batch_size

        self.step_history: List[Dict] = []  # Optional trace
        self.reasoning_traces: List[List[str]] = [[] for _ in range(batch_size)] # Chain of Thought traces

    @classmethod
    def new(cls, batch_size: int, workspace_dim: int, device: torch.device, num_slots: int = 16):
        ws = cls(
            batch_size=batch_size,
            workspace_dim=workspace_dim,
            device=device,
            num_num_slots=num_slots
        )
        ws.symbolic_expr = ["0"] * batch_size
        ws.confidence = torch.zeros(batch_size, 1, device=device)
        ws.step_history = []
        return ws

    def update(self, delta_latent: torch.Tensor, delta_expr: Optional[List[str]] = None, delta_trees: Optional[List[ExpressionTree]] = None, new_confidence: Optional[torch.Tensor] = None):
        self.latent_state = self.latent_state + delta_latent
        if delta_expr is not None:
            self.symbolic_expr = delta_expr
        if delta_trees is not None:
            self.expression_trees = delta_trees
        if new_confidence is not None:
            self.confidence = new_confidence
        else:
            self.confidence = torch.sigmoid(self.latent_state.norm(dim=-1, keepdim=True) * 0.1)
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
        new_ws.expression_trees = [t for t in self.expression_trees]
        return new_ws
