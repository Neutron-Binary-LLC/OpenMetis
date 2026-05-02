import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
import argparse
from typing import List, Tuple
import sys

# Add parent directory to path to import hybrid_math
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from metis_model.model import OpenMetisHybridModel
from hybrid_math.workspace import MathWorkspace

class MathDataset(Dataset):
    """
    Mock dataset for mathematical sequences.
    In production, this would load data from GSM8K, MATH, or LaTeX sources.
    """
    def __init__(self, num_samples: int = 1000, seq_len: int = 32, vocab_size: int = 1000):
        self.data = torch.randint(0, vocab_size, (num_samples, seq_len))
        self.targets = torch.randint(0, vocab_size, (num_samples, seq_len))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

def train():
    parser = argparse.ArgumentParser(description="OpenMetis Training Script")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--vocab_size", type=int, default=1000)
    parser.add_argument("--checkpoint", type=str, default="mythos_checkpoint.pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # Initialize model
    model = OpenMetisHybridModel(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        num_layers=args.num_layers,
        device=device
    )

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # Data
    dataset = MathDataset(num_samples=200, seq_len=16, vocab_size=args.vocab_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    print("Starting Training Loop...")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            logits, workspaces = model(inputs)
            
            # Loss: Language modeling loss + Workspace regularization
            loss = criterion(logits.view(-1, args.vocab_size), targets.view(-1))
            
            # Add workspace latent regularization (encourage stability)
            ws_reg = sum([ws.latent_state.pow(2).mean() for ws in workspaces]) * 0.001
            total_loss_step = loss + ws_reg
            
            total_loss_step.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += total_loss_step.item()
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {avg_loss:.4f}")

    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'args': args
    }, args.checkpoint)
    print(f"Model saved to {args.checkpoint}")

if __name__ == "__main__":
    train()
