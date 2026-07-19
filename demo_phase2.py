import torch
import os
import sys

# Add current directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from nn.block import NeuroSymbolicReasoningCell, MathConfig
from nn.workspace import MathWorkspace
from nn.expression import ExpressionTree, ExpressionNode

def run_phase2_demo():
    print("=== Phase 2: Enhanced Symbolic Integration Demo ===")
    print("Goal: Demonstrate Hybrid Workspace (Trees + Latents) and Symbolic Expert Rules.\n")
    
    device = torch.device("cpu")
    d_model = 256
    batch_size = 1
    
    # 1. Initialize Configuration
    config = MathConfig(
        dim=d_model,
        workspace_dim=d_model,
        max_loop_iters=2 # We only need a couple of iterations to see simplification
    )
    
    # 2. Initialize Reasoning Cell
    cell = NeuroSymbolicReasoningCell(config=config).to(device)
    
    # 3. Initialize Workspace with a symbolic expression that needs simplification
    # Target: (x + 0) * 1  -> x
    ws = MathWorkspace.new(batch_size, d_model, device)
    ws.symbolic_expr = ["(x + 0) * 1"]
    
    print(f"Step 0: Initial Workspace State")
    print(f"  - Symbolic Expression: {ws.symbolic_expr[0]}")
    print(f"  - Expression Tree: {ws.expression_trees[0]}")
    
    # 4. Forward Pass
    x = torch.randn(batch_size, 4, d_model)
    
    print("\nExecuting Neuro-Symbolic Reasoning Loop...")
    h_out, ws_out, _ = cell(x, workspace=ws, debug=False)
    
    # 5. Inspect Results
    print("\n=== Final Results Analysis ===")
    print(f"  - Final Symbolic Expression: {ws_out.symbolic_expr[0]}")
    
    if ws_out.expression_trees[0]:
        print(f"  - Final Tree Representation (Infix): {ws_out.expression_trees[0].to_infix()}")
        print(f"  - Final Tree Representation (Prefix): {ws_out.expression_trees[0].root}")
    
    # Verification
    if ws_out.symbolic_expr[0] == "x":
        print("\n[SUCCESS] The 'Symbolic Expert' correctly identified and applied algebraic rewrite rules.")
        print("          Rule 1: x + 0 -> x")
        print("          Rule 2: x * 1 -> x")
    else:
        print(f"\n[INFO] The expression evolved to: {ws_out.symbolic_expr[0]}")

    print("\n=== External Expert Integration (Qwen2.5-Math) ===")
    print(f"  - Specialized LLM Expert: {cell.external_llm_expert.model_id}")
    print(f"  - Workspace Audit Log contains {len(ws_out.step_history)} entries.")
    for entry in ws_out.step_history:
        if "tool" in entry:
            print(f"    * Step {entry['step']}: {entry['tool']} ({entry['model']})")

    print("\nConclusion: The workspace has successfully transitioned to a hybrid representation.")
    print("            It now integrates explicit expression trees with rule-based proposals")
    print("            and specialized external LLM experts.")

if __name__ == "__main__":
    run_phase2_demo()
