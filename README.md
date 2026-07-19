# OpenMetis: Agentic Neuro-Symbolic Workspace

OpenMetis is a state-of-the-art framework for neural-symbolic integration, designed for advanced mathematical reasoning and financial engineering. It combines neural transformer-style processing with a persistent, differentiable mathematical workspace and deterministic tool augmentation.

## Core Architecture

OpenMetis features a hierarchical architecture where multiple `NeuroSymbolicReasoningCell` layers are stacked to enable deep, multi-stage reasoning. Each layer maintains its own mathematical workspace and can delegate complex computations to specialized tools.

### Architecture Diagram

The architecture is divided into the high-level macro-flow (stacked layers) and the detailed internal recurrent processing within each cell.

#### 1. Macro-Flow (Stacked Hybrid Model)
Overview of the data flow through multiple stacked reasoning layers, interacting with the mathematical tools and the world environment.

```mermaid
graph TD
    subgraph World_Env [Financial World]
        Data[Market Data / Tasks]
    end

    subgraph Stack [OpenMetis Hybrid Stack]
        direction TB
        L1[Layer 1: Reasoning Cell]
        L2[Layer 2: Reasoning Cell]
        LN[Layer N: Reasoning Cell]
        
        L1 --> L2 --> LN
    end

    subgraph Tools [Deterministic Tools]
        FMT[FinMathTools]
        Sym[Symbolic Engine]
    end

    Data --> L1
    Stack <--> Tools
    LN --> Output[Refined Result / Trace]

%%    style Stack fill:#1a1a1a,stroke:#aaaaaa,stroke-width:2px,color:#ffffff
%%    style Tools fill:#141414,stroke:#999999,stroke-width:2px,color:#ffffff
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

- **Hierarchical Stacking**: Stack multiple hybrid layers (`OpenMetisHybridModel`) for deep reasoning.
- **Tool-Augmented Reasoning (Orchestrator)**: The `NeuroSymbolicReasoningCell` acts as an orchestrator, delegating complex calculations to `FinMathTools`.
- **Mathematical Workspace**: Persistent state carrying latent mathematical context, numerical values, and expression trees.
    - **Hybrid Representation**: Combines high-dimensional latent vectors with explicit **Expression Trees** for symbolic transparency.
    - **Numerical Slots**: 16 dedicated slots for storing constants, variables, and their gradients.
    - **Symbolic Expert**: Differentiable routing to algebraic rewrite rules for expression simplification.
    - **Audit Trail**: Full history of tool calls, symbolic updates, and external LLM interactions (`step_history`).
- **Recurrent Depth**: Shared weights across configurable iterations (default 4-8) per layer.
- **MoE Experts**: Specialized layers for different mathematical domains (Algebra, Calculus, etc.).
- **Differentiable Symbolic Ops**: Native support for differentiation, integration approximation, and elementary functions.

## Financial Mathematics World

OpenMetis is integrated with a specialized `world` package for financial mathematics:
- **FinancialWorld**: High-level environment for generating market data and evaluating tasks.
- **DataSources**: Synthetic generation of option pricing data (S, K, T, r, sigma).
- **Task Library**: Built-in tasks for Option Pricing, Greeks calculation, and Implied Volatility (IV) estimation.

## Agentic Workspace

This repository follows an **Agentic Workspace** workflow (see [AGENTS.md](AGENTS.md)). Specialized AI agents collaborate on the development:
- **ContextIngestor**: Maps neural-symbolic patterns.
- **CodeGenerator**: Implements neural blocks and math logic.
- **MathAgent**: Ensures numerical stability and correctness.
- **VisualizationAgent**: Provides insights into latent states and tool usage.
- **ProjectScaffolder**: Maintaining repository structure.
- **Reviewer**: Code quality and performance review.
- **TestGenerator**: Automated unit and integration testing.
- **HealthcheckAgent**: Workspace integrity.

## Installation

We recommend using [uv](https://github.com/astral-sh/uv) for fast dependency management:

```bash
uv pip install torch matplotlib pyyaml
# Or using the lockfile
uv sync
```

Alternatively, use standard pip:

```bash
pip install torch matplotlib pyyaml
```

## Usage

```python
import torch
from nn.block import NeuroSymbolicReasoningCell, MathConfig

# Initialize Configuration
config = MathConfig(
    dim=512,
    workspace_dim=128,
    max_loop_iters=6,
    n_experts=5
)

# Initialize the block
block = NeuroSymbolicReasoningCell(config=config)

# Input tensor (Batch, Seq, Dim)
x = torch.randn(2, 10, 512)

# Forward pass
output_hidden, workspace_obj, trace = block(x)
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

## Demos

We provide several demos to showcase the framework's capabilities:

1.  **Orchestrator Demo**: `demo_orchestrator.py` - Shows tool-augmented reasoning for Black-Scholes pricing and Greeks calculation with a full audit trail.
2.  **Visualization Demo**: `visualize_orchestrator.py` - Generates `orchestrator_trace.png` showing the consistency of tool-based reasoning.
3.  **Metis Black-Scholes Demo**: `demo_metis_bs.py` - Demonstrates autonomous model training for option pricing.
4.  **Symbolic Phase 2 Demo**: `demo_phase2.py` - Demonstrates the hybrid (tree + latent) workspace and rule-based symbolic experts.
5.  **Interactive Shell**: `interact.py` - Allows interactive exploration of the model's reasoning process and tool calls.

### Running the Orchestrator Demo

```bash
python3 demo_orchestrator.py
```

## Training and Checkpoints

We provide scripts for large-scale training using a YAML-based configuration system:

1.  `train_advanced.py`: Advanced training script with checkpointing and persistence.
2.  `metis_model/train_metis.py`: Main training script for stacked Metis models using `config.yaml`.
3.  `train_with_world.py`: Training script integrated with `FinancialWorld` for financial reasoning tasks.

### Usage: Large-Scale Training

```bash
# Train using the default config.yaml
python3 metis_model/train_metis.py

# Specify a custom config and device
python3 metis_model/train_metis.py --config config.yaml --device cpu
```

## Project Structure

- `nn/`: Core neural-symbolic components.
    - `block.py`: `NeuroSymbolicReasoningCell` implementation.
    - `workspace.py`: `MathWorkspace` managing state and history.
    - `expression.py`: `ExpressionTree` and symbolic utilities.
    - `symbolic_expert.py`: Algebraic rewrite rules and simplification.
    - `external_llm.py`: Integration with external math LLMs.
    - `fin_math.py`: `FinMathTools` for deterministic financial calculations.
    - `variants.py`: Predefined model configurations (tiny, small, medium, large).
- `metis_model/`: High-level models and training logic.
    - `model.py`: `OpenMetisHybridModel` (stacked layers).
    - `train_metis.py`: Config-driven training loop.
- `world/`: Financial mathematics environment.
    - `environment.py`: `FinancialWorld` for task management.
    - `data_source.py`: Synthetic financial data generation.
    - `tasks.py`: Financial task definitions (Pricing, Greeks, IV).
- `docs/`: Documentation and architectural insights.
- `config.yaml`: Centralized hyperparameter configuration.

## Roadmap

### Phase 1: Foundation (Completed)
- [x] Recurrent block architecture with latent workspace.
- [x] Mixture-of-Experts (MoE) routing for domain specialization.
- [x] Stacked hierarchical layers (`OpenMetisHybridModel`).
- [x] Tool-augmented reasoning (FinMath-Orchestrator).
- [x] Audit trail and reasoning traces.

### Phase 2: Enhanced Symbolic Integration (Completed)
- [x] **Tree-Based Workspace**: Transition to hybrid representation involving explicit expression trees.
- [x] **Rule-Based Proposals**: Integrate algebraic rewrite rules as a "Symbolic Expert".
- [x] **External LLM Integration**: Support for Qwen2.5-Math as specialized experts.

### Phase 3: Reasoning & Verification (In Progress)
- [ ] **Self-Correction Loop**: Retrying iterations if verification confidence is low.
- [ ] **Formal Verification**: Integration with Z3 or Lean for constraint satisfaction.
- [ ] **Reasoning Traces for LLMs**: Generating chain-of-thought data for fine-tuning.
