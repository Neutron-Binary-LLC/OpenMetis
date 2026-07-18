import torch
import matplotlib.pyplot as plt
from nn.workspace import MathWorkspace

def plot_workspace_confidence(history: list):
    """Plots the confidence score evolution from a list of MathWorkspace objects."""
    confidences = [ws.confidence.mean().item() for ws in history]
    plt.figure(figsize=(10, 5))
    plt.plot(confidences, marker='o')
    plt.title("Workspace Confidence Evolution")
    plt.xlabel("Iteration")
    plt.ylabel("Mean Confidence")
    plt.grid(True)
    plt.savefig("docs/tutorials/confidence_plot.png")
    print("Plot saved to docs/tutorials/confidence_plot.png")

# Note: This is a placeholder for actual integration into the training/eval loop.
