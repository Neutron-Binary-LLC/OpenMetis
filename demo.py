import torch
import torch.nn as nn
from nn.block import NeuroSymbolicReasoningCell
from nn.expression import MathExpression, SymbolicOp

def demo_usage():
    print("--- Initializing NeuroSymbolicReasoningCell ---")
    d_model = 256
    num_iterations = 4
    block = NeuroSymbolicReasoningCell(
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
    hidden_out, final_workspace_obj, trace = block(x)
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
    hidden_out_deep, _, trace_deep = block(x, num_iterations=dynamic_iters, debug=True)
    print(f"Successfully ran with {dynamic_iters} iterations.")
    if trace_deep:
        print(f"Captured trace for {len(trace_deep)} iterations.")
        print(f"Expert weights for first iteration: {trace_deep[0]['expert_weights'].mean(dim=(0,1)).tolist()}")

def black_scholes_demo(block: NeuroSymbolicReasoningCell):
    print("\n--- Black-Scholes Calculation using NeuroSymbolicReasoningCell ---")
    
    # Parameters: S (Spot), K (Strike), T (Time to maturity), r (Risk-free rate), sigma (Volatility)
    S = torch.tensor([100.0], requires_grad=True)
    K = torch.tensor([105.0], requires_grad=True)
    T = torch.tensor([1.0], requires_grad=True)
    r = torch.tensor([0.05], requires_grad=True)
    sigma = torch.tensor([0.2], requires_grad=True)
    
    # d1 = (ln(S/K) + (r + sigma^2 / 2) * T) / (sigma * sqrt(T))
    # d2 = d1 - sigma * sqrt(T)
    
    # Calculate using block's math operations
    ln_S_K = block.apply_exponential_log(S / K, "log")
    sigma_sq = block.apply_power(sigma, 2.0)
    sqrt_T = block.apply_power(T, 0.5)
    
    d1 = (ln_S_K + (r + sigma_sq / 2.0) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    
    # Normal CDF approximation or torch.special.erf
    def norm_cdf(x):
        return 0.5 * (1.0 + torch.erf(x / torch.sqrt(torch.tensor([2.0]))))
    
    phi_d1 = norm_cdf(d1)
    phi_d2 = norm_cdf(d2)
    
    # Call Price = S * N(d1) - K * exp(-r * T) * N(d2)
    exp_minus_rt = block.apply_exponential_log(-r * T, "exp")
    call_price = S * phi_d1 - K * exp_minus_rt * phi_d2
    
    print(f"Inputs: S={S.item()}, K={K.item()}, T={T.item()}, r={r.item()}, sigma={sigma.item()}")
    print(f"Calculated Call Price: {call_price.item():.4f}")
    
    # Verify gradient flow
    call_price.backward()
    print(f"Vega (dPrice/dSigma): {sigma.grad.item():.4f}")
    print(f"Delta (dPrice/dS): {S.grad.item():.4f}")

if __name__ == "__main__":
    demo_usage()
    
    # Initialize block for Black-Scholes demo
    d_model = 256
    block = NeuroSymbolicReasoningCell(d_model=d_model)
    black_scholes_demo(block)
