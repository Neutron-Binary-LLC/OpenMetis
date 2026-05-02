import torch
import torch.nn as nn
from typing import Tuple, Optional, List, Dict, Any
from hybrid_math.block import HybridRecurrentMathBlock
from hybrid_math.workspace import MathWorkspace

class OpenMythosHybridModel(nn.Module):
    """
    OpenMythos-style model that stacks multiple HybridRecurrentMathBlocks
    to enable deep hierarchical mathematical reasoning.
    """
    def __init__(
        self,
        vocab_size: int = 1000,
        d_model: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        num_iterations: int = 4,
        workspace_dim: int = 256,
        num_experts: int = 4,
        dropout: float = 0.1,
        device: torch.device = torch.device('cpu')
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.device = device
        
        # Token embedding for symbolic input
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, 1024, d_model))
        
        # Stack of HybridRecurrentMathBlocks
        self.layers = nn.ModuleList([
            HybridRecurrentMathBlock(
                d_model=d_model,
                num_heads=num_heads,
                num_iterations=num_iterations,
                workspace_dim=workspace_dim,
                num_experts=num_experts,
                dropout=dropout,
                device=device
            ) for _ in range(num_layers)
        ])
        
        # Output head
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        
        self.to(device)

    def forward(
        self, 
        input_ids: torch.Tensor, 
        workspaces: Optional[List[MathWorkspace]] = None,
        num_iterations: Optional[int] = None
    ) -> Tuple[torch.Tensor, List[MathWorkspace]]:
        """
        Forward pass through the stacked hybrid blocks.
        Returns logits and a list of workspaces (one per layer).
        """
        b, t = input_ids.shape
        x = self.embedding(input_ids) + self.pos_encoding[:, :t, :]
        
        new_workspaces = []
        
        for i, layer in enumerate(self.layers):
            ws_init = workspaces[i] if workspaces is not None else None
            x, ws = layer(x, math_state_init=ws_init, num_iterations=num_iterations)
            new_workspaces.append(ws)
            
        x = self.ln_f(x)
        logits = self.head(x)
        
        return logits, new_workspaces

if __name__ == "__main__":
    # Quick sanity check
    model = OpenMythosHybridModel(vocab_size=100, d_model=128, num_layers=2)
    dummy_input = torch.randint(0, 100, (2, 10))
    logits, workspaces = model(dummy_input)
    print(f"Logits shape: {logits.shape}")
    print(f"Number of workspaces: {len(workspaces)}")
    print("Model initialized successfully.")
