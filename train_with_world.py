import torch
import torch.nn as nn
import torch.optim as optim
from nn.block import NeuroSymbolicReasoningCell, MathConfig
from world.environment import FinancialWorld

def train_with_world():
    # Hyperparameters
    d_model = 128
    num_iterations = 4
    batch_size = 32
    learning_rate = 5e-4
    epochs = 100
    task_name = "price"
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. World Initialization
    world = FinancialWorld(device=device)
    
    # 2. Model Initialization
    # We'll use a config for the cell
    config = MathConfig(
        dim=d_model,
        n_heads=4,
        max_loop_iters=num_iterations,
        workspace_dim=64,
        device=str(device)
    )
    model = NeuroSymbolicReasoningCell(config=config).to(device)
    
    # 3. Input Projection: (batch, 7) -> (batch, 1, d_model)
    # We treat the financial parameters as a single multi-feature token
    input_projection = nn.Linear(7, d_model).to(device)
    
    # 4. Output Head: (batch, 1, d_model) -> (batch, 1)
    output_head = nn.Linear(d_model, 1).to(device)
    
    # Optimizer and Loss
    optimizer = optim.Adam(
        list(model.parameters()) + 
        list(input_projection.parameters()) + 
        list(output_head.parameters()), 
        lr=learning_rate
    )
    criterion = nn.MSELoss()
    
    print(f"Starting training on task: {task_name}...")
    
    for epoch in range(epochs):
        model.train()
        
        # Get real data from the world
        inputs, labels, raw_data = world.get_data_and_labels(batch_size, task_name)
        
        # Project inputs to model dimension
        # (batch, 6) -> (batch, d_model) -> (batch, 1, d_model)
        x = input_projection(inputs).unsqueeze(1)
        
        optimizer.zero_grad()
        
        # Forward pass
        # The block returns (last_hidden, last_workspace_obj, trace)
        outputs, workspace_obj, _ = model(x)
        
        # Predict the value (using the representation of the first/only token)
        # outputs shape: (batch, 1, d_model)
        prediction = output_head(outputs.squeeze(1))
        
        # Loss calculation
        loss = criterion(prediction, labels)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")
            # print sample prediction vs label
            print(f"  Sample Pred: {prediction[0].item():.4f}, Label: {labels[0].item():.4f}")

    print("Training with FinancialWorld completed.")

if __name__ == "__main__":
    train_with_world()
