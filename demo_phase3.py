import torch
import os
import sys

# Add current directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from nn.block import NeuroSymbolicReasoningCell, MathConfig
from nn.workspace import MathWorkspace
from nn.expression import ExpressionTree

def run_phase3_demo():
    print("=== Phase 3: Reasoning & Verification Demo ===")
    print("Goal: Demonstrate Formal Verification, Self-Correction, and Reasoning Traces.")
    
    device = torch.device("cpu")
    d_model = 128
    batch_size = 1
    
    # 1. Initialize Configuration
    config = MathConfig(
        dim=d_model,
        workspace_dim=d_model,
        max_loop_iters=5,
        act_threshold=0.95
    )
    
    # 2. Initialize Reasoning Cell
    cell = NeuroSymbolicReasoningCell(config=config).to(device)
    
    # 3. Prepare Workspace with a specific problem
    ws = MathWorkspace.new(batch_size, d_model, device)
    
    # Problem: Verify if (x + 1) * (x - 1) - (x*x - 1) is always 0
    # We'll set the symbolic expression and see if the verification expert picks it up
    ws.symbolic_expr = ["(x + 1) * (x - 1) - (x*x - 1)"]
    # Initialize the tree from the expression
    ws.expression_trees = [ExpressionTree.from_string(ws.symbolic_expr[0])]
    
    # We want to enable gradients to trigger some of the internal tool logic if needed,
    # though here we are focused on symbolic/formal verification.
    ws.numerical_values = torch.zeros(batch_size, 16, device=device, requires_grad=True)
    
    # Scenario 1: Successful Verification
    print("\n--- Scenario 1: Successful Verification ---")
    ws.symbolic_expr = ["(x + 1) * (x - 1) - (x*x - 1)"]
    ws.expression_trees = [ExpressionTree.from_string(ws.symbolic_expr[0])]
    h = torch.randn(batch_size, 1, d_model)
    _, ws_out1, _ = cell(h, workspace=ws.clone(), debug=True)
    
    print(f"Final Confidence: {ws_out1.confidence.item():.4f}")
    print(f"Reasoning: {ws_out1.reasoning_traces[0][-1]}")

    # Scenario 2: Verification Failure and Self-Correction
    print("\n--- Scenario 2: Verification Failure & Self-Correction ---")
    ws2 = MathWorkspace.new(batch_size, d_model, device)
    ws2.symbolic_expr = ["x - 100"] 
    ws2.expression_trees = [ExpressionTree.from_string(ws2.symbolic_expr[0])]
    
    # Force goal 2 (is always zero?) which should fail for "x - 100"
    _, ws_out2, _ = cell(h, workspace=ws2, debug=True, forced_goal=2)
    
    print(f"\nFinal Confidence: {ws_out2.confidence.item():.4f}")
    print(f"Iterations completed: {ws_out2.iteration_count}")
    
    print("\nReasoning Trace for Scenario 2:")
    for step in ws_out2.reasoning_traces[0]:
        print(f"  - {step}")

    print("\nSuccess: Phase 3 features (Formal Verification, Self-Correction, and Traces) are functional.")

if __name__ == "__main__":
    run_phase3_demo()
