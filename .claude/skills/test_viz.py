import torch
from workspace_viz import plot_workspace_confidence
from nn.workspace import MathWorkspace

def test_viz():
    batch_size = 1
    workspace_dim = 256
    device = torch.device('cpu')
    
    history = []
    for i in range(5):
        ws = MathWorkspace.new(batch_size, workspace_dim, device)
        ws.confidence = torch.full((batch_size, 1), i / 5.0)
        history.append(ws)
    
    plot_workspace_confidence(history)

if __name__ == "__main__":
    test_viz()
