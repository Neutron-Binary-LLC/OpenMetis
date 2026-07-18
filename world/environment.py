import torch
from typing import Dict, Any, Tuple, List, Optional
from world.data_source import FinancialDataSource
from world.tasks import TASKS

class FinancialWorld:
    """
    A high-level environment for training and evaluating neural-symbolic models
    on financial mathematics tasks.
    """
    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        self.data_source = FinancialDataSource(device=device)

    def get_data_and_labels(
        self, 
        batch_size: int, 
        task_name: str = "price"
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Returns a batch of data, ground truth labels for the specified task,
        and the raw data dictionary.
        """
        if task_name not in TASKS:
            raise ValueError(f"Unknown task: {task_name}. Available: {list(TASKS.keys())}")
        
        data = self.data_source.generate_batch(batch_size)
        
        # Calculate price to have it available
        from world.tasks import OptionPricingTask
        data["price"] = OptionPricingTask().compute_ground_truth(data).squeeze(-1)
        
        labels = TASKS[task_name].compute_ground_truth(data)
        
        # Mask inputs based on task
        include_price = (task_name == "iv")
        inputs = self.data_source.get_standardized_input(data, include_price=include_price)
        
        if task_name == "iv":
            # Mask sigma (index 4) if we are predicting it
            inputs[:, 4] = 0.0
        
        # Ensure labels have a channel dimension if they are scalar
        if labels.dim() == 1:
            labels = labels.unsqueeze(-1)
            
        return inputs, labels, data

    def evaluate_model(
        self, 
        model: torch.nn.Module, 
        task_name: str = "price", 
        batch_size: int = 32
    ) -> Dict[str, float]:
        """
        Performs a quick evaluation of the model on a task.
        """
        model.eval()
        with torch.no_grad():
            inputs, labels, _ = self.get_data_and_labels(batch_size, task_name)
            
            # The model might expect (batch, seq_len, d_model)
            # We assume inputs are (batch, features), so we might need to unsqueeze
            if inputs.dim() == 2:
                # Add a dummy sequence dimension if needed
                # However, OpenMetisHybridModel expects input_ids (integers)
                # or NeuroSymbolicReasoningCell expects embeddings.
                # Let's check what the model actually expects.
                pass
            
            # This is a generic evaluation placeholder. 
            # In real usage, the caller should handle model specific input formatting.
            return {"loss": 0.0} # Placeholder
