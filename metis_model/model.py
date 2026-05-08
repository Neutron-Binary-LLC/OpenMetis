import torch
import torch.nn as nn
from typing import Tuple, Optional, List, Dict, Any
from nn.block import NeuroSymbolicReasoningCell, MathConfig
from nn.workspace import MathWorkspace

class OpenMetisHybridModel(nn.Module):
    """
    OpenMetis-style model that stacks multiple NeuroSymbolicReasoningCells
    to enable deep hierarchical mathematical reasoning.
    """
    def __init__(
        self,
        config: Optional[MathConfig] = None,
        num_layers: int = 4,
        **kwargs
    ):
        super().__init__()
        if config is None:
            # Check if variant is provided in kwargs
            variant_name = kwargs.get("variant", None)
            if variant_name:
                from nn.variants import VARIANTS
                config = VARIANTS.get(variant_name, VARIANTS["small"])
                # Update config with any explicit overrides
                config.vocab_size = kwargs.get("vocab_size", config.vocab_size)
                config.dim = kwargs.get("d_model", config.dim)
                config.max_seq_len = kwargs.get("max_seq_len", config.max_seq_len)
                config.device = str(kwargs.get("device", config.device))
            else:
                config = MathConfig(
                    vocab_size=kwargs.get("vocab_size", 1000),
                    dim=kwargs.get("d_model", 512),
                    n_heads=kwargs.get("num_heads", 8),
                    max_loop_iters=kwargs.get("num_iterations", 4),
                    workspace_dim=kwargs.get("workspace_dim", 256),
                    n_experts=kwargs.get("num_experts", 4),
                    dropout=kwargs.get("dropout", 0.1),
                    device=str(kwargs.get("device", "cpu"))
                )
        
        self.config = config
        self.d_model = config.dim
        self.num_layers = num_layers
        self.device = torch.device(config.device)
        
        # Token embedding for symbolic input
        self.embedding = nn.Embedding(config.vocab_size, self.d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, config.max_seq_len, self.d_model))
        
        # Stack of NeuroSymbolicReasoningCells
        self.layers = nn.ModuleList([
            NeuroSymbolicReasoningCell(config=config) for _ in range(num_layers)
        ])
        
        # Output head
        self.ln_f = nn.LayerNorm(self.d_model)
        self.head = nn.Linear(self.d_model, config.vocab_size)
        
        self.to(self.device)

    def forward(
        self, 
        input_ids: torch.Tensor, 
        workspaces: Optional[List[MathWorkspace]] = None,
        num_iterations: Optional[int] = None,
        debug: bool = False
    ) -> Tuple[torch.Tensor, List[MathWorkspace], List[Optional[List[Dict[str, Any]]]]]:
        """
        Forward pass through the stacked hybrid blocks.
        Returns logits, a list of workspaces (one per layer), and a list of traces (one per layer).
        """
        b, t = input_ids.shape
        x = self.embedding(input_ids) + self.pos_encoding[:, :t, :]
        
        new_workspaces = []
        traces = []
        
        for i, layer in enumerate(self.layers):
            ws_init = workspaces[i] if workspaces is not None else None
            x, ws, tr = layer(x, workspace=ws_init, num_iterations=num_iterations, debug=debug)
            new_workspaces.append(ws)
            traces.append(tr)
            
        x = self.ln_f(x)
        logits = self.head(x)
        
        return logits, new_workspaces, traces

if __name__ == "__main__":
    # Quick sanity check
    model = OpenMetisHybridModel(vocab_size=100, d_model=128, num_layers=2)
    dummy_input = torch.randint(0, 100, (2, 10))
    logits, workspaces, traces = model(dummy_input)
    print(f"Logits shape: {logits.shape}")
    print(f"Number of workspaces: {len(workspaces)}")
    print(f"Number of traces: {len(traces)}")
    print("Model initialized successfully.")
