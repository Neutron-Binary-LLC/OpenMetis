# Hybrid Neuro-Symbolic Recurrent Block

This project implements a `NeuroSymbolicReasoningCell` in PyTorch, designed for advanced mathematical reasoning by combining neural transformer-style processing with a persistent, differentiable mathematical workspace.

## Core Architecture

The architecture features a recurrent loop that carries both a hidden state (neural) and a mathematical workspace (symbolic/latent) across multiple iterations.

### Architecture Diagram

The architecture is divided into the high-level macro-flow and the detailed internal recurrent processing.

#### 1. Macro-Flow
Overview of the data flow from input embeddings through the recurrent stack to the final output.

```mermaid
graph LR
    subgraph Input_Space [Input Layer]
        direction TB
        X[Input x<br/>Batch, Seq, D_model]
        InitWS[Initial Workspace]
    end

    subgraph Recurrent_Stack [Recurrent Processing]
        direction TB
        Prelude[Prelude Layers]
        Loop["Recurrent Loop<br/>(N Iterations)"]
        Coda[Coda Layers]
        
        Prelude --> Loop --> Coda
    end

    subgraph Output_Space [Output Layer]
        direction TB
        Hidden[Final Hidden State]
        FinalWS[Final Workspace]
    end

    X --> Prelude
    InitWS --> Loop
    Coda --> Hidden
    Loop --> FinalWS

%%    style Recurrent_Stack fill:#1a1a1a,stroke:#aaaaaa,stroke-width:2px,color:#ffffff
%%    style Input_Space fill:#141414,stroke:#999999,stroke-width:2px,color:#ffffff
```

#### 2. Recurrent Loop Detail
Internal view of the processing within the `NeuroSymbolicReasoningCell`, divided into functional stages.

##### Master Diagram: Functional Stages
High-level interaction between the neural core, symbolic bridge, and mathematical workspace within a single iteration.

```mermaid
graph TD
    subgraph Iteration ["Single Recurrent Iteration"]
        direction TB
        
        NPU[Neural Processing Unit<br/>Attention & MoE FFN]
        Bridge[Neuro-Symbolic Bridge<br/>Projector & Expert Router]
        Heads[Symbolic Heads<br/>Math-Specific Updates]
        WS[(Math Workspace<br/>Latent & Numerical)]

        WS -.->|Project| Bridge
        Bridge -->|Inject| NPU
        NPU -->|Refine| Bridge
        Bridge -->|Route| Heads
        Heads -->|Update| WS
    end

%%    style Iteration fill:#1a1a1a,stroke:#aaaaaa,stroke-width:2px,color:#ffffff
%%    style WS fill:#2a2a2a,stroke:#e0e0e0,stroke-width:2px,color:#ffffff
```

##### A. Neural Processing Unit (NPU)
Handles standard transformer-style hidden state transformation.

```mermaid
graph LR
    subgraph Neural_Core [Neural Processing Unit]
        direction LR
        subgraph AttnBlock [Self-Attention]
            direction LR
            Norm_A[LN] --> MHA[MHA] --> Drop_A[DO] --> Add_A[Add]
        end
        subgraph FFNBlock [FFN / MoE]
            direction LR
            Norm_F[LN] --> Lin1[L+G] --> Drop_F1[DO] --> Lin2[L] --> Drop_F2[DO] --> Add_F[Add]
        end
        Add_A --> Norm_F
    end
```

##### B. Neuro-Symbolic Bridge & Experts
Integrates the symbolic workspace into the neural flow and routes to specialized math experts.

```mermaid
graph LR
    subgraph Bridge_Workspace [Neuro-Symbolic Core]
        direction LR
        subgraph NeuroSymbolic_Bridge [NS Bridge]
            direction TB
            Proj[Projector]
            Aug[Augmentor]
            Router[Router]
        end
        subgraph Experts [Math Experts MoE]
            direction LR
            Alg[Alg] --- Calc[Calc] --- Num[Num] --- Ver[Ver]
        end
        
        Latent[Latent State] --> Proj --> Aug
        Aug --> Router --> Experts
    end
```

##### C. Symbolic Heads & Workspace Updates
Processes the refined hidden state to propose and apply updates to the persistent mathematical state.

```mermaid
graph LR
    subgraph Workspace_Update_Flow [Symbolic Update Cycle]
        direction TB
        subgraph Math_Heads [Symbolic Heads]
            direction LR
            Deriv[Deriv] --- Integ[Integ] --- Simp[Simp] --- Trig[Trig] --- ExpLog[ExpLog] --- Pow[Pow] --- Update[Update]
        end
        subgraph Workspace_State [Math Workspace]
            direction LR
            Latent[Latent] --- NumSlots[NumSlots] --- Conf[Confidence] --- Iter[Iteration]
        end

        Math_Heads -->|State Delta| Latent
        Math_Heads -->|Numerical Proj| NumSlots
        Math_Heads -->|Confidence Update| Conf
    end

%%    style Workspace_State fill:#2a2a2a,stroke:#e0e0e0,stroke-width:2px,color:#ffffff
%%    style Math_Heads fill:#303030,stroke:#f0f0f0,stroke-width:2px,color:#ffffff
```

### Sequence Diagram

```mermaid
sequenceDiagram
    autonumber
    participant I as Input (x)
    participant B as HybridRecurrentBlock
    participant P as Workspace Projector
    participant N as Neural Core (Attn/FFN)
    participant E as MoE Experts
    participant H as Math Heads
    participant W as MathWorkspace

    I->>B: forward(x, math_state_init)
    B->>W: Initialize State (Latent, NumSlots, Conf)
    
    loop num_iterations
        W->>P: Fetch Latent State
        P->>N: Inject Projected Latent into x
        N->>N: Self-Attention & FFN Processing
        N->>E: Route to Domain Experts (Algebra/Calc/etc)
        E->>N: Return Refined Hidden State
        N->>H: Propose Symbolic Operations
        H->>W: Update Numerical Slots (Gradients/Values)
        H->>W: Update Latent State via Workspace Updater
        H->>W: Update Confidence Score
        W->>W: Increment Iteration Count
    end
    
    B->>I: Return (Output, Final Workspace)
```

## Key Features

- **Recurrent Depth**: Shared weights across configurable iterations (default 4-8).
- **Mathematical Workspace**: Persistent state carrying latent mathematical context, numerical values, and confidence scores.
    - **Latent State**: High-dimensional vector representing the abstract mathematical context.
    - **Numerical Slots**: Dedicated slots for storing constants, variables, and their gradients (e.g., for differentiation).
    - **Confidence Score**: Scalar indicating the model's certainty in its current mathematical state.
    - **Iteration Counter**: Tracks the number of recurrent steps taken.
- **MoE Experts**: Specialized layers for different mathematical domains (Algebra, Calculus, etc.).
- **Symbolic Heads**: Dedicated linear layers (`deriv_head`, `integral_head`, `simplify_head`, `trig_head`, `exp_log_head`, `pow_head`) that propose modifications to the mathematical workspace.
- **Differentiable Symbolic Ops**: Native support for operations like differentiation, integration approximation, trigonometric functions, exponentials, and power laws within the recurrent loop.
- **Stability**: Residual connections, layer normalization, and LTI-style gating to ensure gradient flow across depth.

## Installation

```bash
pip install torch
```

## Usage

```python
import torch
from nn.block import NeuroSymbolicReasoningCell

# Initialize the block
block = NeuroSymbolicReasoningCell(d_model=512, num_iterations=6)

# Input tensor (Batch, Seq, Dim)
x = torch.randn(2, 10, 512)

# Forward pass
output_hidden, workspace_obj = block(x)
final_workspace = workspace_obj.to_dict()

print(f"Final Iteration Count: {final_workspace['iteration_count']}")
```

## Advantages and Disadvantages

### Advantages
- **Structured Reasoning**: The persistent mathematical workspace allows the model to maintain state and "reason" over multiple steps, similar to how humans use scratchpads.
- **Hybrid Flexibility**: Combines the pattern recognition strengths of Transformers with the precision of symbolic-like operations and specialized MoE experts.
- **Recurrent Efficiency**: Uses shared weights across iterations, reducing parameter count compared to deep feed-forward models while allowing for variable computation time (dynamic depth).
- **Differentiability**: The entire loop is end-to-end differentiable, enabling the use of standard gradient-based optimization even for neuro-symbolic tasks.
- **Stability**: Built-in mechanisms like residual connections, layer normalization, and LTI-style gating help mitigate vanishing/exploding gradients in the recurrent loop.

### Disadvantages
- **Computational Overhead**: Multiple iterations per block increase the training and inference time compared to a single-pass layer.
- **Memory Consumption**: Backpropagating through many recurrent iterations (BPTT) can be memory-intensive.
- **Complexity**: The architecture is more complex to implement and tune than standard Transformer blocks, requiring careful balancing between neural and symbolic components.
- **Convergence Sensitivity**: Recurrent models can be more sensitive to hyperparameter choices (like learning rate and weight initialization) to maintain stability.
- **Expert Routing**: The MoE router adds another layer of complexity, potentially leading to expert collapse if not properly regularized.

## Open-Source Math Experts

To enhance the `NeuroSymbolicReasoningCell`, you can integrate pre-trained mathematical models as specialized experts. Some recommended open-source models and resources include:

- **Qwen2.5-Math**: A state-of-the-art mathematical LLM series (1.5B to 72B) optimized for reasoning and problem-solving. [Hugging Face](https://huggingface.co/Qwen/Qwen2.5-Math-7B).
- **MathBERT**: A BERT-based model pre-trained on a large corpus of mathematical texts, ideal for extracting features from mathematical expressions. [Hugging Face](https://huggingface.co/tbs17/MathBERT).
- **Llama-3-Math-70B**: Various community-tuned versions of Llama-3 specifically for competitive mathematics and reasoning.
- **DeepSeek-Math**: A specialized model for mathematical reasoning that achieves high performance on benchmarks like GSM8K and MATH.

These can be integrated by replacing the default `MathExpert` with a wrapper around these pre-trained models.

## Training and Checkpoints

We provide two scripts for training and experimentation:

1.  `train_sample.py`: A basic template for a training loop.
2.  `train_advanced.py`: An advanced training script with support for:
    - **Checkpointing**: Saves and loads weights, optimizer states, and epoch metadata.
    - **Resuming**: Continue training from a previous checkpoint using the `--resume` flag.
    - **Inference**: Demonstrates how to load a model for inference after training.

### Usage: Advanced Training

```bash
# Start a new training session
python3 train_advanced.py --epochs 10 --lr 1e-4

# Resume training from a checkpoint
python3 train_advanced.py --epochs 20 --resume --checkpoint math_block_checkpoint.pth
```

## Metis Black-Scholes Demo

The `demo_metis_bs.py` script demonstrates a higher-level application: training a Metis-based model to perform Black-Scholes option pricing autonomously.

### Key Features:
- **Neuro-Symbolic Dataset**: Generates synthetic Black-Scholes data (S, K, T, r, sigma) and exact prices.
- **Autonomous Inference**: Once trained, the model predicts prices by processing inputs through its `NeuroSymbolicReasoningCells` without calling explicit math functions.
- **Checkpoint Persistence**: Automatically saves and loads `metis_bs_model.pth`.
- **Continuous Training**: Supports loading an existing model and performing additional training epochs.

### Usage:

```bash
# Run inference using existing model (skips training if checkpoint exists)
python3 demo_metis_bs.py

# Force training even if model exists
python3 demo_metis_bs.py --train

# Customize training (e.g., 50 epochs with 5000 samples)
python3 demo_metis_bs.py --train --epochs 50 --samples 5000
```

## Large-Scale Training and Configuration

The `metis_model/train_metis.py` script has been enhanced to support large-scale training using a YAML-based configuration system.

### Key Enhancements:
- **YAML Configuration**: Externalized all hyperparameters (training, model, dataset) into `config.yaml`.
- **Model Variants**: Support for predefined variants (`tiny`, `small`, `medium`, `large`) with easy overrides.
- **Enhanced Dataset**: `SyntheticMathDataset` for simulating structured mathematical sequences.
- **Improved Training Loop**:
    - **Cosine Annealing LR**: For smoother convergence.
    - **Gradient Clipping**: To maintain stability in recurrent loops.
    - **Workspace Regularization**: Encourages stability in latent state transitions.
    - **Checkpointing**: Saves both periodic and "best" models based on loss.

### Usage:

```bash
# Train using the default config.yaml
python3 metis_model/train_metis.py

# Specify a custom config and device
python3 metis_model/train_metis.py --config my_large_config.yaml --device cuda
```

### Configuration Example (`config.yaml`):

```yaml
training:
  epochs: 10
  batch_size: 16
  lr: 2.0e-4
  grad_clip: 1.0
  checkpoint_path: "openmetis_large.pth"

model:
  variant: "small"
  num_layers: 4
  vocab_size: 5000

dataset:
  num_samples: 5000
  seq_len: 64
```

## Roadmap

The development of the `NeuroSymbolicReasoningCell` is planned across several phases to evolve from a latent-state recycler to a full-fledged neuro-symbolic engine.

### Phase 1: Foundation (Current)
- [x] Recurrent block architecture with latent workspace.
- [x] Mixture-of-Experts (MoE) routing for domain specialization.
- [x] Advanced math-specific heads (Differentiation, Integration, Simplification).
- [x] Differentiable symbolic operations and numerical slots.
- [x] Advanced configuration and model variants (OpenMythos-style).
- [x] Prelude/Coda layers and LoRA depth adaptation.
- [x] Advanced training script with checkpointing and persistence.

### Phase 2: Enhanced Symbolic Integration
- [ ] **Tree-Based Workspace**: Transition from purely latent vectors to a hybrid representation involving explicit expression trees.
- [ ] **Rule-Based Proposals**: Integrate a library of fixed algebraic rewrite rules (simplification, expansion) as a "Symbolic Expert".
- [ ] **Dynamic Expert Scaling**: Support for plugging in external LLMs (e.g., Qwen2.5-Math) as specialized experts via API or local weights.

### Phase 3: Reasoning & Verification
- [ ] **Self-Correction Loop**: Implement a verification-driven update where the model retries iterations if the "Verification Expert" reports low confidence.
- [ ] **Formal Verification**: Integrate with formal solvers like Z3 or Lean for hard constraint satisfaction within the loop.
- [ ] **Curriculum Learning**: Training pipeline for gradually increasing mathematical complexity (from arithmetic to calculus).

### Phase 4: Scaling & Deployment
- [ ] **Flash-Recurrence**: Optimize the recurrent loop for faster inference and reduced memory footprint during training.
- [ ] **Multi-Block Stacking**: Research on stacking multiple `NeuroSymbolicReasoningCells` with hierarchical workspaces.
- [ ] **Interpretable Reasoning Traces**: Tools to visualize and export the symbolic "scratchpad" evolution in human-readable LaTeX.

## Project Structure

- `hybrid_math/`
    - `block.py`: Main `NeuroSymbolicReasoningCell` implementation.
    - `workspace.py`: `MathWorkspace` class managing state.
    - `expression.py`: `MathExpression` and `SymbolicOp` utilities.
- `metis_model/`
    - `model.py`: `OpenMythosHybridModel` implementation.
    - `train_metis.py`: Main training script for Metis models.
- `demo.py`: Basic demonstration script showing math heads and gradient verification.
- `demo_metis_bs.py`: Black-Scholes pricing demo with autonomous model inference.
- `train_sample.py`: A basic training script template.
- `train_advanced.py`: Advanced training script with checkpointing and resuming capabilities.
