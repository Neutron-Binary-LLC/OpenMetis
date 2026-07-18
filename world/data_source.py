import torch
from typing import Dict, Any, Tuple

class FinancialDataSource:
    """
    Generates synthetic financial data for option pricing and risk management tasks.
    """
    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)

    def generate_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Generates a batch of option parameters.
        - S: Spot price (80 to 120)
        - K: Strike price (80 to 120)
        - T: Time to maturity (0.1 to 2.0 years)
        - r: Risk-free rate (0.01 to 0.1)
        - sigma: Volatility (0.1 to 0.5)
        """
        S = 80 + 40 * torch.rand(batch_size, device=self.device)
        K = 80 + 40 * torch.rand(batch_size, device=self.device)
        T = 0.1 + 1.9 * torch.rand(batch_size, device=self.device)
        r = 0.01 + 0.09 * torch.rand(batch_size, device=self.device)
        sigma = 0.1 + 0.4 * torch.rand(batch_size, device=self.device)
        
        # Randomly choose if it's a call (1) or put (0)
        is_call = torch.randint(0, 2, (batch_size,), device=self.device).float()
        
        return {
            "S": S,
            "K": K,
            "T": T,
            "r": r,
            "sigma": sigma,
            "is_call": is_call
        }

    def get_standardized_input(self, data: Dict[str, torch.Tensor], include_price: bool = False) -> torch.Tensor:
        """
        Converts the data dictionary into a tensor suitable for model input.
        Shape: (batch_size, 7)
        """
        feats = [
            data["S"],
            data["K"],
            data["T"],
            data["r"],
            data["sigma"],
            data["is_call"]
        ]
        # Always include a 7th feature (price or 0)
        if "price" in data:
            if include_price:
                feats.append(data["price"])
            else:
                feats.append(torch.zeros_like(data["price"]))
        else:
            feats.append(torch.zeros_like(data["S"]))
            
        return torch.stack(feats, dim=-1)
