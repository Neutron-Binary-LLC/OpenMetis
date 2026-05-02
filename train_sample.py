import torch
import torch.nn as nn
import torch.optim as optim
from hybrid_math.block import HybridRecurrentMathBlock

def train_sample():
    # Hyperparameters
    d_model = 128
    num_iterations = 4
    batch_size = 16
    seq_len = 20
    learning_rate = 1e-4
    epochs = 5
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Model
    model = HybridRecurrentMathBlock(
        d_model=d_model, 
        num_heads=4, 
        num_iterations=num_iterations,
        workspace_dim=64
    ).to(device)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    print("Starting sample training loop...")
    
    for epoch in range(epochs):
        model.train()
        
        # Mock data: (batch, seq, d_model)
        # In a real task, these would be embeddings of math problems
        inputs = torch.randn(batch_size, seq_len, d_model).to(device)
        # Targets: (batch, seq, d_model)
        # In a real task, these would be the embeddings of the steps or final answer
        targets = torch.randn(batch_size, seq_len, d_model).to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        # The block returns (last_hidden, last_workspace_obj)
        outputs, workspace_obj = model(inputs)
        workspace = workspace_obj.to_dict()
        
        # Loss calculation
        # 1. Main prediction loss
        pred_loss = criterion(outputs, targets)
        
        # 2. Workspace consistency loss (example: latent state should not explode)
        workspace_reg = workspace['latent_state'].pow(2).mean() * 0.01
        
        total_loss = pred_loss + workspace_reg
        
        # Backward pass and optimization
        total_loss.backward()
        
        # Gradient clipping for recurrent stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        if (epoch + 1) % 1 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss.item():.4f}, Pred Loss: {pred_loss.item():.4f}")

    print("Sample training completed.")

if __name__ == "__main__":
    train_sample()
