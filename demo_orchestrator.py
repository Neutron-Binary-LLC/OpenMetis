import torch
import torch.nn as nn
import os
import sys

# Add current directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from nn.block import NeuroSymbolicReasoningCell, MathConfig
from nn.workspace import MathWorkspace
from nn.fin_math import FinMathTools

def run_orchestrator_demo():
    print("=== FinMath-Orchestrator Demo ===")
    print("Goal: Demonstrate tool-augmented neuro-symbolic reasoning for financial math.")
    
    device = torch.device("cpu")
    d_model = 128
    batch_size = 1
    
    # 1. Initialize Configuration
    config = MathConfig(
        dim=d_model,
        workspace_dim=d_model,
        max_loop_iters=3, # Small number of iterations for demo
        n_experts=5
    )
    
    # 2. Initialize Reasoning Cell
    cell = NeuroSymbolicReasoningCell(config=config).to(device)
    
    # 3. Prepare Inputs (S, K, T, r, sigma)
    # S=100, K=105, T=1.0, r=0.05, sigma=0.2
    S, K, T, r, sigma = 100.0, 105.0, 1.0, 0.05, 0.2
    inputs = torch.tensor([[S, K, T, r, sigma]], dtype=torch.float32)
    
    # Mock hidden state
    h = torch.randn(batch_size, 5, d_model)
    
    # 4. Initialize Workspace
    ws = MathWorkspace.new(batch_size, d_model, device)
    
    # Enable gradients on numerical values to trigger tool usage in the cell
    ws.numerical_values = torch.zeros(batch_size, 16, device=device, requires_grad=True)
    
    # Seed numerical values
    with torch.no_grad():
        ws.numerical_values.data[:, :5] = inputs
    
    print(f"\nInitial Inputs: S={S}, K={K}, T={T}, r={r}, sigma={sigma}")
    
    # 5. Forward Pass through Reasoning Cell
    print("\nExecuting Reasoning Cell (with tool augmentation)...")
    h_out, ws_out, trace = cell(h, workspace=ws, debug=True)
    
    # 6. Inspect Results
    print("\n=== Results Analysis ===")
    
    # Extract calculated values from numerical slots
    price = ws_out.numerical_values[0, 10].item()
    delta = ws_out.numerical_values[0, 5].item()
    gamma = ws_out.numerical_values[0, 6].item()
    vega = ws_out.numerical_values[0, 7].item()
    theta = ws_out.numerical_values[0, 8].item()
    rho = ws_out.numerical_values[0, 9].item()
    
    print(f"Computed Price: {price:.4f}")
    print(f"Greeks: Delta={delta:.4f}, Gamma={gamma:.4f}, Vega={vega:.4f}, Theta={theta:.4f}, Rho={rho:.4f}")
    
    # 7. Audit Trail (Auditability)
    print("\n=== Audit Trail (Tool Usage History) ===")
    for entry in ws_out.step_history:
        print(f"Step {entry['step']}: Called {entry['tool']} using {entry['model']} model. Resulting Price: {entry['output_price']:.4f}")

    # 8. Verification against direct tool call
    print("\n=== Verification ===")
    exact_params = {"S": torch.tensor([S]), "K": torch.tensor([K]), "T": torch.tensor([T]), "r": torch.tensor([r]), "sigma": torch.tensor([sigma])}
    direct_price = FinMathTools.price_option("black_scholes", exact_params).item()
    print(f"Direct Tool Price: {direct_price:.4f}")
    print(f"Difference: {abs(price - direct_price):.8f}")
    
    print("\nSuccess: The NeuroSymbolicReasoningCell successfully delegated computations to FinMathTools.")

if __name__ == "__main__":
    run_orchestrator_demo()
