import torch
import torch.nn as nn
import os
import sys
import argparse
from typing import Dict, Any, List

# Add current directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from nn.block import NeuroSymbolicReasoningCell, MathConfig
from nn.workspace import MathWorkspace
from nn.fin_math import FinMathTools
from visualize_orchestrator import visualize_orchestrator_run

class MetisInterface:
    def __init__(self, checkpoint_path: str = "math_block_checkpoint.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.config = None
        self.last_history = None
        
        # Default config matching train_advanced.py
        self.d_model = 256
        self.workspace_dim = 128
        self.num_iterations = 4
        self.num_experts = 5
        
        self.setup_model()

    def setup_model(self):
        self.config = MathConfig(
            dim=self.d_model,
            workspace_dim=self.workspace_dim,
            max_loop_iters=self.num_iterations,
            n_experts=self.num_experts,
            device=str(self.device)
        )
        self.model = NeuroSymbolicReasoningCell(config=self.config).to(self.device)
        
        if os.path.exists(self.checkpoint_path):
            print(f"[*] Attempting to load checkpoint from {self.checkpoint_path}...")
            try:
                # Use weights_only=False because we trust the local checkpoint
                checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print(f"[+] Successfully loaded checkpoint (Epoch: {checkpoint.get('epoch', 'unknown')})")
            except Exception as e:
                print(f"[!] Warning: Failed to load checkpoint: {e}")
                print("[!] The model will use randomly initialized weights.")
                print("[!] You can use the 'train' command to create a compatible checkpoint.")
        else:
            print(f"[!] Checkpoint {self.checkpoint_path} not found. Using fresh weights.")

    def run_inference(self, params: Dict[str, float]):
        print("\n[~] Running Neuro-Symbolic Inference...")
        
        # Prepare inputs for the workspace
        # Parameters: S, K, T, r, sigma
        S = params.get("S", 100.0)
        K = params.get("K", 100.0)
        T = params.get("T", 1.0)
        r = params.get("r", 0.05)
        sigma = params.get("sigma", 0.2)
        
        inputs = torch.tensor([[S, K, T, r, sigma]], dtype=torch.float32, device=self.device)
        
        # Mock hidden state for the neural part
        h = torch.randn(1, 16, self.d_model, device=self.device)
        
        # Initialize Workspace
        ws = MathWorkspace.new(1, self.workspace_dim, self.device)
        # Numerical values tensor needs to have requires_grad=True to trigger tool usage in the cell
        num_vals = torch.zeros(1, 16, device=self.device)
        num_vals[:, :5] = inputs
        num_vals.requires_grad_(True)
        ws.numerical_values = num_vals
            
        print(f"    Input Parameters: S={S}, K={K}, T={T}, r={r}, sigma={sigma}")
        
        # Forward Pass
        self.model.eval()
        # We need to ensure gradients are enabled for the numerical parts even in eval mode
        with torch.set_grad_enabled(True):
            h_out, ws_out, trace = self.model(h, workspace=ws, debug=False)
        
        # Results
        price = ws_out.numerical_values[0, 10].item()
        greeks = {
            "Delta": ws_out.numerical_values[0, 5].item(),
            "Gamma": ws_out.numerical_values[0, 6].item(),
            "Vega": ws_out.numerical_values[0, 7].item(),
            "Theta": ws_out.numerical_values[0, 8].item(),
            "Rho": ws_out.numerical_values[0, 9].item(),
        }
        
        print("\n[+] Inference Results:")
        print(f"    Computed Price: {price:.4f}")
        for k, v in greeks.items():
            print(f"    {k}: {v:.4f}")
            
        print("\n[+] Audit Trail:")
        if ws_out.step_history:
            for entry in ws_out.step_history:
                print(f"    Step {entry['step']}: {entry['tool']} ({entry['model']}) -> Result: {entry['output_price']:.4f}")
        else:
            print("    (No tool calls recorded in history)")
            
        return ws_out.step_history

    def quick_train(self, epochs: int = 5):
        print(f"\n[*] Starting quick re-training ({epochs} epochs)...")
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        data = torch.randn(10, 16, self.d_model).to(self.device)
        targets = torch.randn(10, 16, self.d_model).to(self.device)
        
        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            output, ws, _ = self.model(data)
            loss = nn.MSELoss()(output, targets)
            loss.backward()
            optimizer.step()
            print(f"    Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
            
        # Save new checkpoint
        checkpoint = {
            'epoch': epochs,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item(),
            'config': {
                'd_model': self.d_model,
                'num_iterations': self.num_iterations,
                'workspace_dim': self.workspace_dim,
                'num_experts': self.num_experts
            }
        }
        torch.save(checkpoint, self.checkpoint_path)
        print(f"[+] Saved compatible checkpoint to {self.checkpoint_path}")

def main():
    print("==================================================")
    print("    OpenMetis Interactive Agent Environment")
    print("==================================================")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="math_block_checkpoint.pth")
    args = parser.parse_args()
    
    interface = MetisInterface(checkpoint_path=args.checkpoint)
    
    while True:
        try:
            print("\nAvailable commands: [run, train, viz, exit]")
            cmd_input = input("metis> ").strip().lower()
            if not cmd_input:
                continue
            
            cmds = cmd_input.split()
            cmd = cmds[0]
            
            if cmd == "exit":
                print("Goodbye!")
                break
            elif cmd == "run":
                print("Enter parameters (press enter for defaults):")
                s = input("  S (Spot) [100.0]: ") or "100.0"
                k = input("  K (Strike) [100.0]: ") or "100.0"
                t = input("  T (Time) [1.0]: ") or "1.0"
                r = input("  r (Rate) [0.05]: ") or "0.05"
                sigma = input("  sigma (Vol) [0.2]: ") or "0.2"
                
                params = {
                    "S": float(s),
                    "K": float(k),
                    "T": float(t),
                    "r": float(r),
                    "sigma": float(sigma)
                }
                
                history = interface.run_inference(params)
                interface.last_history = history
            elif cmd == "train":
                epochs = 5
                if len(cmds) > 1:
                    epochs = int(cmds[1])
                interface.quick_train(epochs=epochs)
            elif cmd == "viz":
                if interface.last_history:
                    visualize_orchestrator_run(interface.last_history)
                else:
                    print("[!] No inference history to visualize. Run 'run' first.")
            else:
                print(f"[!] Unknown command: {cmd}")
        except KeyboardInterrupt:
            print("\nUse 'exit' to quit.")
        except Exception as e:
            print(f"[!] Error: {e}")

if __name__ == "__main__":
    main()
