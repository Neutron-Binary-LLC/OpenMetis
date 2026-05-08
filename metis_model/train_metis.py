import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
import argparse
import yaml
from typing import List, Tuple, Dict, Any
import sys
import time

# Add parent directory to path to import nn
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from metis_model.model import OpenMetisHybridModel
from nn.workspace import MathWorkspace

class SyntheticMathDataset(Dataset):
    """
    Enhanced synthetic dataset for mathematical reasoning.
    Simulates sequences of tokenized math problems and solutions.
    """
    def __init__(self, num_samples: int = 1000, seq_len: int = 64, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        
        # Simulate some structure: [PROBLEM_TOKENS] [EQUAL] [ANSWER_TOKENS]
        # For simplicity, we just generate random tokens but can be extended
        self.data = torch.randint(1, vocab_size, (num_samples, seq_len))
        
        # Targets are shifted inputs for language modeling
        self.targets = torch.roll(self.data, shifts=-1, dims=1)
        self.targets[:, -1] = 0 # End token or padding

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

def load_config(config_path: str) -> Dict[str, Any]:
    if not os.path.isabs(config_path):
        # Try to find it relative to the project root (parent of metis_model)
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        potential_path = os.path.join(root_dir, config_path)
        if os.path.exists(potential_path):
            config_path = potential_path
            
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def train():
    parser = argparse.ArgumentParser(description="OpenMetis Large-Scale Training Script")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config")
    parser.add_argument("--device", type=str, default=None, help="Force device (cuda/cpu/mps)")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    train_cfg = config['training']
    model_cfg = config['model']
    data_cfg = config['dataset']

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    
    print(f"Training on: {device}")
    print(f"Model Variant: {model_cfg['variant']}")

    # Initialize model with variant and overrides
    model = OpenMetisHybridModel(
        variant=model_cfg['variant'],
        num_layers=model_cfg['num_layers'],
        vocab_size=model_cfg['vocab_size'],
        max_seq_len=model_cfg['max_seq_len'],
        dropout=model_cfg['dropout'],
        device=str(device)
    )

    optimizer = optim.AdamW(
        model.parameters(), 
        lr=float(train_cfg['lr']), 
        weight_decay=float(train_cfg['weight_decay'])
    )
    
    # Use Cosine Annealing scheduler for better convergence in large scale
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_cfg['epochs'])
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # Data
    dataset = SyntheticMathDataset(
        num_samples=data_cfg['num_samples'], 
        seq_len=data_cfg['seq_len'], 
        vocab_size=model_cfg['vocab_size']
    )
    dataloader = DataLoader(dataset, batch_size=train_cfg['batch_size'], shuffle=True, num_workers=0)

    print(f"Starting Training Loop ({train_cfg['epochs']} epochs, {len(dataset)} samples)...")
    
    best_loss = float('inf')
    
    for epoch in range(train_cfg['epochs']):
        model.train()
        total_loss = 0
        start_time = time.time()
        
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            logits, workspaces, _ = model(inputs)
            
            # Loss: Language modeling loss
            # Reshape: (batch * seq_len, vocab_size) vs (batch * seq_len)
            loss = criterion(logits.view(-1, model_cfg['vocab_size']), targets.view(-1))
            
            # Add workspace latent regularization (encourage stability)
            # This is key for neuro-symbolic recurrent models
            ws_reg = sum([ws.latent_state.pow(2).mean() for ws in workspaces]) * 0.001
            total_loss_step = loss + ws_reg
            
            total_loss_step.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg['grad_clip'])
            
            optimizer.step()
            total_loss += total_loss_step.item()
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}, Batch {batch_idx+1}/{len(dataloader)}, Loss: {total_loss_step.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        scheduler.step()
        
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{train_cfg['epochs']} finished in {epoch_time:.2f}s, Avg Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

        # Save checkpoint
        if (epoch + 1) % train_cfg['save_every'] == 0 or avg_loss < best_loss:
            checkpoint_path = train_cfg['checkpoint_path']
            if avg_loss < best_loss:
                best_loss = avg_loss
                checkpoint_path = checkpoint_path.replace(".pth", "_best.pth")
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'config': config
            }, checkpoint_path)
            print(f"  Checkpoint saved to {checkpoint_path}")

    print("Training Complete.")

if __name__ == "__main__":
    train()
