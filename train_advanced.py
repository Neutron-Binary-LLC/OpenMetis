import torch
import torch.nn as nn
import torch.optim as optim
import os
import argparse
from typing import Dict, Any
from nn.block import NeuroSymbolicReasoningCell, MathConfig

def save_checkpoint(model: nn.Module, optimizer: optim.Optimizer, epoch: int, loss: float, path: str):
    """
    Saves a training checkpoint including model weights, optimizer state, and training metadata.
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': {
            'd_model': model.config.dim,
            'num_iterations': model.config.max_loop_iters,
            'workspace_dim': model.config.workspace_dim,
            'num_experts': len(model.experts)
        }
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path} at epoch {epoch}")

def load_checkpoint(path: str, device: torch.device):
    """
    Loads a checkpoint and returns the metadata and state dicts.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"No checkpoint found at {path}")
    
    checkpoint = torch.load(path, map_location=device)
    print(f"Loaded checkpoint from {path} (Epoch: {checkpoint['epoch']}, Loss: {checkpoint['loss']:.4f})")
    return checkpoint

def main():
    parser = argparse.ArgumentParser(description="Advanced training script for NeuroSymbolicReasoningCell")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--seq_len", type=int, default=16, help="Sequence length")
    parser.add_argument("--checkpoint", type=str, default="math_block_checkpoint.pth", help="Path to save/load checkpoint")
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint if it exists")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model parameters
    d_model = 256
    workspace_dim = 128
    num_iterations = 4
    num_experts = 5 # Needs to be 5 because MathExpert list in NeuroSymbolicReasoningCell is hardcoded with 5 types

    config = MathConfig(
        dim=d_model,
        workspace_dim=workspace_dim,
        max_loop_iters=num_iterations,
        n_experts=num_experts,
        device=str(device)
    )

    model = NeuroSymbolicReasoningCell(config=config).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    start_epoch = 0

    if args.resume and os.path.exists(args.checkpoint):
        checkpoint = load_checkpoint(args.checkpoint, device)
        # Verify config matches
        conf = checkpoint['config']
        if conf['d_model'] != d_model or conf['workspace_dim'] != workspace_dim:
            print("Warning: Checkpoint configuration mismatch. Attempting to load weights anyway...")
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1

    # Dummy dataset for demonstration
    # In a real scenario, this would be mathematical sequences/expressions
    data = torch.randn(100, args.seq_len, d_model).to(device)
    targets = torch.randn(100, args.seq_len, d_model).to(device)

    print("Starting training...")
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0
        
        # Simple batching loop
        for i in range(0, len(data), args.batch_size):
            optimizer.zero_grad()
            
            x = data[i:i+args.batch_size]
            y = targets[i:i+args.batch_size]
            
            output, workspace_obj, trace = model(x)
            workspace_dict = workspace_obj.to_dict()
            
            # Use MSE loss as a placeholder for learning to transform/reason
            loss = nn.MSELoss()(output, y)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()

        avg_loss = epoch_loss / (len(data) // args.batch_size)
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {avg_loss:.4f}")

        # Save checkpoint every 5 epochs and at the end
        if (epoch + 1) % 5 == 0 or (epoch + 1) == args.epochs:
            save_checkpoint(model, optimizer, epoch, avg_loss, args.checkpoint)

    print("Training complete.")

    # Demonstration of loading for inference
    print("\nVerifying loading for inference...")
    inference_checkpoint = load_checkpoint(args.checkpoint, device)
    model.load_state_dict(inference_checkpoint['model_state_dict'])
    model.eval()
    
    with torch.no_grad():
        test_input = torch.randn(1, args.seq_len, d_model).to(device)
        test_output, _, _ = model(test_input)
        print(f"Inference output shape: {test_output.shape}")

if __name__ == "__main__":
    main()
