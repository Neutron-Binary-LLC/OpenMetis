import torch
from typing import Dict, Any, Optional

class MathWorkspace:
    """
    A persistent state that carries across recurrent iterations.
    Supports storing expressions, numerical values, and metadata.
    """
    def __init__(
        self, 
        batch_size: int, 
        workspace_dim: int, 
        device: torch.device = torch.device('cpu')
    ):
        self.batch_size = batch_size
        self.workspace_dim = workspace_dim
        self.device = device
        
        # Latent state representing the current mathematical context
        self.latent_state = torch.zeros(batch_size, workspace_dim, device=device)
        
        # Numerical values for evaluation (e.g., constants, variables)
        self.numerical_values = torch.zeros(batch_size, 10, device=device) # Example: 10 slots
        
        # Metadata
        self.iteration_count = 0
        self.confidence = torch.ones(batch_size, 1, device=device)
        
    def update(self, delta_latent: torch.Tensor, new_confidence: Optional[torch.Tensor] = None):
        """Updates the workspace state."""
        self.latent_state = self.latent_state + delta_latent
        if new_confidence is not None:
            self.confidence = new_confidence
        self.iteration_count += 1

    def to_dict(self) -> Dict[str, Any]:
        """Returns a serializable dictionary of the workspace."""
        return {
            "latent_state": self.latent_state,
            "numerical_values": self.numerical_values,
            "iteration_count": self.iteration_count,
            "confidence": self.confidence
        }

    def clone(self):
        """Creates a copy of the workspace."""
        new_ws = MathWorkspace(self.batch_size, self.workspace_dim, self.device)
        new_ws.latent_state = self.latent_state.clone()
        new_ws.numerical_values = self.numerical_values.clone()
        new_ws.iteration_count = self.iteration_count
        new_ws.confidence = self.confidence.clone()
        return new_ws
