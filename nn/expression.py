import torch
import torch.nn as nn
from typing import List, Optional, Union, Dict, Any

class MathExpression:
    """
    A lightweight representation of a mathematical expression.
    Can be represented as tokens, a tree, or an embedding.
    """
    def __init__(
        self, 
        tokens: Optional[torch.Tensor] = None, 
        embedding: Optional[torch.Tensor] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.tokens = tokens  # (seq_len,) or (batch, seq_len)
        self.embedding = embedding  # (d_model,) or (batch, d_model)
        self.metadata = metadata or {}

    def __repr__(self):
        return f"MathExpression(metadata={self.metadata})"

class SymbolicOp:
    """Base class for differentiable symbolic operations."""
    @staticmethod
    def simplify(tensor: torch.Tensor) -> torch.Tensor:
        # Placeholder for symbolic simplification logic
        return tensor

    @staticmethod
    def differentiate(tensor: torch.Tensor, wrt: torch.Tensor) -> torch.Tensor:
        # Uses torch.autograd for differentiable 'symbolic' differentiation
        return torch.autograd.grad(tensor.sum(), wrt, create_graph=True)[0]

def apply_rewrite_rule(x: torch.Tensor, rule_weight: torch.Tensor) -> torch.Tensor:
    """
    Applies a learnable rewrite rule to the hidden state.
    x: (batch, d_model)
    rule_weight: (d_model, d_model)
    """
    return torch.matmul(x, rule_weight)
