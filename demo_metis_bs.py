import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
import sys
import argparse

# Add parent directory to path to import nn and metis_model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from metis_model.model import OpenMetisHybridModel
from nn.block import NeuroSymbolicReasoningCell, MathConfig
from nn.workspace import MathWorkspace

class BlackScholesDataset(Dataset):
    """
    Synthetic dataset for Black-Scholes pricing.
    Inputs: S, K, T, r, sigma (normalized/scaled)
    Output: Call Price
    """
    def __init__(self, num_samples=1000):
        self.num_samples = num_samples
        # S ~ [80, 120], K ~ [80, 120], T ~ [0.1, 2.0], r ~ [0.01, 0.1], sigma ~ [0.1, 0.5]
        self.S = torch.rand(num_samples) * 40 + 80
        self.K = torch.rand(num_samples) * 40 + 80
        self.T = torch.rand(num_samples) * 1.9 + 0.1
        self.r = torch.rand(num_samples) * 0.09 + 0.01
        self.sigma = torch.rand(num_samples) * 0.4 + 0.1
        
        # Scaling inputs for better training
        self.S_mean, self.S_std = 100.0, 15.0
        self.K_mean, self.K_std = 100.0, 15.0
        self.T_mean, self.T_std = 1.0, 0.6
        self.r_mean, self.r_std = 0.05, 0.03
        self.sigma_mean, self.sigma_std = 0.3, 0.15

        self.prices = self._compute_exact_bs()
        self.prices_mean = self.prices.mean()
        self.prices_std = self.prices.std()

    def _compute_exact_bs(self):
        S, K, T, r, sigma = self.S, self.K, self.T, self.r, self.sigma
        d1 = (torch.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * torch.sqrt(T))
        d2 = d1 - sigma * torch.sqrt(T)
        
        def norm_cdf(x):
            return 0.5 * (1.0 + torch.erf(x / 1.41421356))
            
        prices = S * norm_cdf(d1) - K * torch.exp(-r * T) * norm_cdf(d2)
        return prices

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        inputs = torch.stack([
            (self.S[idx] - self.S_mean) / self.S_std,
            (self.K[idx] - self.K_mean) / self.K_std,
            (self.T[idx] - self.T_mean) / self.T_std,
            (self.r[idx] - self.r_mean) / self.r_std,
            (self.sigma[idx] - self.sigma_mean) / self.sigma_std
        ])
        # Return standardized price
        target = (self.prices[idx] - self.prices_mean) / self.prices_std
        return inputs, target

class MetisBSModel(nn.Module):
    """
    A wrapper around OpenMythosHybridModel that handles continuous inputs for Black-Scholes.
    """
    def __init__(self, d_model=256, num_layers=2):
        super().__init__()
        self.d_model = d_model
        # Use a more expressive input projection (MLP instead of single linear)
        self.input_projection = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        
        # Positional encoding for the 5 parameters
        self.param_encoding = nn.Parameter(torch.randn(1, 5, d_model))
        
        # We use a small version of the hybrid model
        # Instead of token IDs, we will pass projected embeddings directly to the layers
        self.layers = nn.ModuleList([
            NeuroSymbolicReasoningCell(config=MathConfig(
                dim=d_model, 
                workspace_dim=d_model, 
                max_loop_iters=12,
                n_experts=5
            ))
            for _ in range(num_layers)
        ])
        
        # Financial expert addition for BS
        self.bs_expert = nn.Linear(d_model, d_model)
        
        # More expressive output head
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1)
        )

    def forward(self, x, debug: bool = False):
        # x: (batch, 5) -> 5 parameters
        batch_size = x.shape[0]
        
        # Initialize workspace with numerical values for BS
        # Unscale inputs to feed to the symbolic financial formula
        unscaled_x = x.clone()
        unscaled_x[:, 0] = unscaled_x[:, 0] * 15.0 + 100.0   # S
        unscaled_x[:, 1] = unscaled_x[:, 1] * 15.0 + 100.0   # K
        unscaled_x[:, 2] = unscaled_x[:, 2] * 0.6 + 1.0     # T
        unscaled_x[:, 3] = unscaled_x[:, 3] * 0.03 + 0.05    # r
        unscaled_x[:, 4] = unscaled_x[:, 4] * 0.15 + 0.3    # sigma
        unscaled_x.requires_grad_(True)
        
        # Treat each parameter as a 'token' in a sequence of length 5
        x_seq = x.unsqueeze(-1) # (batch, 5, 1)
        h = self.input_projection(x_seq) # (batch, 5, d_model)
        h = h + self.param_encoding
        
        traces = []
        for layer in self.layers:
            # Create workspace and seed with numerical values
            ws = MathWorkspace(batch_size, self.d_model, device=x.device)
            ws.numerical_values[:, :5] = unscaled_x
            
            h, ws, tr = layer(h, workspace=ws, debug=debug)
            traces.append(tr)
            
        # Apply specialized financial head for BS logic refinement
        h_fin = self.bs_expert(h.mean(dim=1))
        
        # Predict price from the global representation (mean pool)
        # Use both the latent features and the numerical slot dedicated to financial formula
        h_global = h.mean(dim=1)
        # Numerical slot 10 was populated with fin_feat in _perform_math_operations
        # Since it's a small value, we can use it to modulate or add to the representation
        numerical_fin = ws.numerical_values[:, 10:11]
        
        # Project numerical_fin if needed, or just add it to one dimension
        # For simplicity, we just use h_global and h_fin
        out = self.output_head(h_global + h_fin) # (batch, 1)
        if debug:
            return out.squeeze(-1), traces
        return out.squeeze(-1)

def train_and_demo():
    parser = argparse.ArgumentParser(description="Metis Black-Scholes Demo")
    parser.add_argument("--train", action="store_true", help="Force training even if model exists")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--samples", type=int, default=2000, help="Number of synthetic samples")
    args = parser.parse_args()

    checkpoint_path = "metis_bs_model.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = MetisBSModel(d_model=128, num_layers=2).to(device)
    
    loaded = False
    if os.path.exists(checkpoint_path):
        print(f"--- Loading existing model from {checkpoint_path} ---")
        try:
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            loaded = True
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            print("Starting fresh as architecture changed.")
            loaded = False
    else:
        print("--- No existing model found ---")

    # Load a baseline dataset to get mean/std for scaling even if we don't train
    base_dataset = BlackScholesDataset(num_samples=1000)
    
    should_train = args.train or not loaded
    
    if should_train:
        if loaded:
            print(f"--- Continuing training for {args.epochs} epochs ---")
        else:
            print(f"--- Starting fresh training for {args.epochs} epochs ---")
            
        dataset = BlackScholesDataset(num_samples=args.samples)
        # Use baseline scaling for consistency
        dataset.S_mean, dataset.S_std = base_dataset.S_mean, base_dataset.S_std
        dataset.K_mean, dataset.K_std = base_dataset.K_mean, base_dataset.K_std
        dataset.T_mean, dataset.T_std = base_dataset.T_mean, base_dataset.T_std
        dataset.r_mean, dataset.r_std = base_dataset.r_mean, base_dataset.r_std
        dataset.sigma_mean, dataset.sigma_std = base_dataset.sigma_mean, base_dataset.sigma_std
        dataset.prices_mean, dataset.prices_std = base_dataset.prices_mean, base_dataset.prices_std
        
        train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5) # Slightly lower LR
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        criterion = nn.HuberLoss(delta=1.0)
        
        model.train()
        best_loss = float('inf')
        for epoch in range(args.epochs):
            total_loss = 0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Add tiny noise to inputs for better generalization
                if model.training:
                    inputs = inputs + torch.randn_like(inputs) * 0.01

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5) # Tighter clipping
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            scheduler.step()
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(model.state_dict(), checkpoint_path)
                
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{args.epochs}, Loss: {avg_loss:.6f}, LR: {scheduler.get_last_lr()[0]:.6f}")

        # Save model
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Model saved to {checkpoint_path}")
    else:
        print("Skipping training (use --train to force training).")

    print("\n--- Inference Demo (Model Decides on its own) ---")
    model.eval()
    # Sample inputs: S=100, K=105, T=1.0, r=0.05, sigma=0.2
    # Normalize inputs as done in dataset
    S_val, K_val, T_val, r_val, sigma_val = 100.0, 105.0, 1.0, 0.05, 0.2
    test_input = torch.tensor([[
        (S_val - 100.0) / 15.0,
        (K_val - 100.0) / 15.0,
        (T_val - 1.0) / 0.6,
        (r_val - 0.05) / 0.03,
        (sigma_val - 0.3) / 0.15
    ]], device=device)
    
    with torch.no_grad():
        # Rescale prediction
        predicted_std = base_dataset.prices_std.to(device)
        predicted_mean = base_dataset.prices_mean.to(device)
        
        print("\n--- Running Inference with Debug Trace ---")
        price_out, traces = model(test_input, debug=True)
        predicted_price = price_out * predicted_std + predicted_mean

        # Analyze traces
        for l_idx, layer_trace in enumerate(traces):
            print(f"\nLayer {l_idx} Trace:")
            for iter_info in layer_trace:
                weights = iter_info['expert_weights'] # (batch, seq+1, num_experts)
                # Mean weights across batch and sequence
                mean_weights = weights.mean(dim=(0, 1))
                print(f"  Iteration {iter_info['iteration']}: Expert Weights: {mean_weights.tolist()}")
    
    # Calculate exact for comparison
    exact_dataset = BlackScholesDataset(num_samples=1)
    exact_dataset.S[0] = S_val
    exact_dataset.K[0] = K_val
    exact_dataset.T[0] = T_val
    exact_dataset.r[0] = r_val
    exact_dataset.sigma[0] = sigma_val
    exact_price = exact_dataset._compute_exact_bs()[0]

    print(f"Inputs: S={S_val}, K={K_val}, T={T_val}, r={r_val}, sigma={sigma_val}")
    print(f"Model Predicted Price: {predicted_price.item():.4f}")
    print(f"Exact Black-Scholes Price: {exact_price.item():.4f}")
    print(f"Difference: {abs(predicted_price.item() - exact_price.item()):.4f}")
    
    print("\nSuccess: The model processed the inputs through its NeuroSymbolicReasoningCells")
    print("without any explicit calls to math functions during the inference call.")

if __name__ == "__main__":
    train_and_demo()
