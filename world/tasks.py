import torch
from typing import Dict, Any, List
from nn.fin_math import FinMathTools

class MathTask:
    """Base class for tasks in the Financial World."""
    def compute_ground_truth(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

class OptionPricingTask(MathTask):
    """Task: Predict the Black-Scholes price of an option."""
    def compute_ground_truth(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        # data['is_call'] is 1.0 or 0.0
        # FinMathTools._black_scholes_price expects 'is_call' in params
        # We need to loop or handle batching if FinMathTools supports it (it seems to support tensors)
        params = {
            "S": data["S"],
            "K": data["K"],
            "T": data["T"],
            "r": data["r"],
            "sigma": data["sigma"],
            "is_call": data["is_call"] > 0.5
        }
        return FinMathTools.price_option("black_scholes", params)

class GreekCalculationTask(MathTask):
    """Task: Predict the Delta of an option."""
    def compute_ground_truth(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        params = {
            "S": data["S"],
            "K": data["K"],
            "T": data["T"],
            "r": data["r"],
            "sigma": data["sigma"],
            "is_call": data["is_call"] > 0.5
        }
        greeks = FinMathTools.compute_greeks("black_scholes", params)
        return greeks["delta"]

class ImpliedVolTask(MathTask):
    """Task: Calibrate implied volatility from price. 
    Ground truth is the original sigma used to generate the price.
    """
    def compute_ground_truth(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        return data["sigma"]

TASKS = {
    "price": OptionPricingTask(),
    "delta": GreekCalculationTask(),
    "iv": ImpliedVolTask()
}
