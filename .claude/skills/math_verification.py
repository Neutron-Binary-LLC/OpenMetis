import torch
from nn.block import NeuroSymbolicReasoningCell, MathConfig

def verify_gradients(dim=128, max_loop_iters=4):
    """Verifies that gradients flow through the NeuroSymbolicReasoningCell."""
    config = MathConfig(dim=dim, max_loop_iters=max_loop_iters)
    model = NeuroSymbolicReasoningCell(config)
    
    x = torch.randn(1, 10, dim, requires_grad=True)
    output, workspace, _ = model(x)
    
    loss = output.sum() + workspace.latent_state.sum()
    loss.backward()
    
    if x.grad is not None:
        print("Gradient verification successful: Gradients flowed to input.")
        return True
    else:
        print("Gradient verification failed: No gradients found at input.")
        return False

if __name__ == "__main__":
    verify_gradients()
