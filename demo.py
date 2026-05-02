import torch
import torch.nn as nn
from hybrid_math.block import HybridRecurrentMathBlock
from hybrid_math.expression import MathExpression, SymbolicOp

def demo_usage():
    print("--- Initializing HybridRecurrentMathBlock ---")
    d_model = 256
    num_iterations = 4
    block = HybridRecurrentMathBlock(
        d_model=d_model, 
        num_heads=4, 
        num_iterations=num_iterations, 
        workspace_dim=128,
        num_experts=4
    )
    
    # Mock input: (batch, seq_len, d_model)
    # Let's imagine this is an embedding of "Solve for x: 2x + 5 = 15"
    batch_size = 2
    seq_len = 10
    x = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
    
    print(f"Input shape: {x.shape}")
    print(f"Target iterations: {num_iterations}")
    
    # Forward pass
    hidden_out, final_workspace_obj = block(x)
    final_workspace = final_workspace_obj.to_dict()
    
    print("\n--- Forward Pass Results ---")
    print(f"Output hidden state shape: {hidden_out.shape}")
    print(f"Final workspace iteration count: {final_workspace['iteration_count']}")
    print(f"Final workspace latent state shape: {final_workspace['latent_state'].shape}")
    
    # Gradient Flow Verification
    print("\n--- Verifying Gradient Flow ---")
    loss = hidden_out.pow(2).mean() + final_workspace['latent_state'].pow(2).mean()
    loss.backward()
    
    if x.grad is not None:
        grad_norm = x.grad.norm().item()
        print(f"Gradient norm w.r.t input: {grad_norm:.6f}")
        if grad_norm > 0:
            print("✅ Gradient flow confirmed.")
        else:
            print("❌ Zero gradient detected.")
    else:
        print("❌ No gradient found.")

    # Symbolic Operation Example (Differentiable)
    print("\n--- Symbolic Operation Demo ---")
    z = torch.randn(1, requires_grad=True)
    # Equation: y = z^2 + 3z + 1
    y = z**2 + 3*z + 1
    dy_dz = SymbolicOp.differentiate(y, z)
    print(f"Expression: z^2 + 3z + 1 at z={z.item():.2f}")
    print(f"Differentiable derivative dy/dz: {dy_dz.item():.2f} (Expected: 2z + 3 = {2*z.item()+3:.2f})")

    # Foundational Math Functions Demo
    print("\n--- Foundational Math Functions Demo ---")
    test_val = torch.tensor([1.0], requires_grad=True)
    sin_val = block.apply_trigonometric(test_val, "sin")
    exp_val = block.apply_exponential_log(test_val, "exp")
    pow_val = block.apply_power(test_val, 2.0)
    
    print(f"Value: {test_val.item():.2f}")
    print(f"sin(x): {sin_val.item():.4f} (Expected: {torch.sin(test_val).item():.4f})")
    print(f"exp(x): {exp_val.item():.4f} (Expected: {torch.exp(test_val).item():.4f})")
    print(f"pow(x, 2): {pow_val.item():.4f} (Expected: 1.0000)")

    # Inference with different depth
    print("\n--- Inference with Dynamic Depth ---")
    dynamic_iters = 8
    hidden_out_deep, _ = block(x, num_iterations=dynamic_iters)
    print(f"Successfully ran with {dynamic_iters} iterations.")

if __name__ == "__main__":
    demo_usage()
