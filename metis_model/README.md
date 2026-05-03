# OpenMetis Hybrid Architecture

OpenMetis is a deep neuro-symbolic model built upon the `NeuroSymbolicReasoningCell`. It stacks multiple recurrent layers, each maintaining its own mathematical workspace, to solve complex, multi-step symbolic reasoning tasks.

## Architecture Overview

The model follows a hierarchical structure where each layer processes the sequence and updates a layer-specific `MathWorkspace`. This allows the model to maintain different levels of abstraction (e.g., Layer 1 focuses on algebra, Layer 2 on calculus).

### High-Level Diagram

```mermaid
graph TD
    Input[Token IDs] --> Emb[Embedding + Pos Encoding]
    
    subgraph MythosStack [OpenMetis Stacked Layers]
        direction TB
        Layer1[HybridRecurrentBlock 1]
        Layer2[HybridRecurrentBlock 2]
        LayerN[HybridRecurrentBlock N]
        
        Layer1 --> Layer2
        Layer2 -->|...| LayerN
    end
    
    subgraph Workspaces [Hierarchical Workspaces]
        direction TB
        WS1[Workspace 1]
        WS2[Workspace 2]
        WSN[Workspace N]
    end
    
    Emb --> Layer1
    Layer1 <--> WS1
    Layer2 <--> WS2
    LayerN <--> WSN
    
    LayerN --> Norm[LayerNorm]
    Norm --> Head[Language Model Head]
    Head --> Output[Logits]
```

### Sequence Flow

```mermaid
sequenceDiagram
    participant U as User/Data
    participant M as OpenMetisModel
    participant L as HybridLayers
    participant W as LayerWorkspaces
    
    U->>M: input_ids (Tokens)
    M->>M: Embedding + PosEncoding
    
    loop For each Layer i
        M->>L: Hidden State (x_i)
        L->>W: Fetch WS_i
        L->>L: Recurrent Loop (N iterations)
        L->>W: Update WS_i (Latent + Numerical)
        L->>M: Output Hidden State (x_i+1)
    end
    
    M->>U: Final Logits + All Workspaces
```

## Key Components

- **Hierarchical Reasoning**: By stacking layers, the model can decompose a problem into sub-tasks, with each layer's workspace acting as a specialized "scratchpad".
- **Dynamic Iterations**: The number of recurrent iterations can be adjusted globally or per-layer at inference time to handle problems of varying complexity.
- **Symbolic Regularization**: The training script includes penalties for unstable workspace latent states, ensuring the neuro-symbolic loop converges.

## Setup and Training

### Installation
Ensure you have the core `hybrid_math` package available in your python path.

```bash
# From the project root
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

### Training
The training environment supports standard language modeling objectives augmented with workspace-aware losses.

```bash
python3 metis_model/train_metis.py --epochs 10 --num_layers 4 --d_model 512
```

## Recommended Experts
For high-performance mathematical reasoning, we recommend integrating the following open-source experts into the `MathExpert` slots:
- **Qwen2.5-Math-7B**: State-of-the-art for reasoning steps.
- **DeepSeek-Math-Base**: Excellent for symbolic manipulation.
- **Llama-3-Math-Intermediate**: Good for general mathematical context.

## Data Sources
For effective training of the OpenMythos architecture, utilize:
- **GSM8K**: Grade school math word problems.
- **MATH Dataset**: Advanced high school math.
- **AMPS (Algebraic Manipulation, Proofs, and Symbols)**: Large-scale dataset for symbolic reasoning.
