# OpenMetis Architectural Overview

This document provides a technical deep-dive into the OpenMetis framework, designed for agents to understand the core neuro-symbolic integration.

## Core Components

### 1. NeuroSymbolicReasoningCell (`nn/block.py`)
A recurrent transformer-style block that iterates over a hidden state and a `MathWorkspace`.
- **Recurrence**: Shared weights across $N$ iterations (default 4-8).
- **NS Bridge**: Projects the `MathWorkspace` into the hidden state and vice-versa.
- **Math Heads**: Specific linear layers for:
  - `deriv_head`: Differentiation proposals.
  - `integral_head`: Integration proposals.
  - `simplify_head`: Expression simplification.
  - `trig_head`, `exp_log_head`, `pow_head`: Elementary functions.

### 2. MathWorkspace (`nn/workspace.py`)
A persistent container for the mathematical state.
- **Latent State**: High-dimensional vector representing abstract math context.
- **Numerical Values**: Tensor of slots (default 16) for constants and evaluation points.
- **Confidence**: Scalar [0, 1] indicating the model's certainty.
- **Symbolic History**: Tracks the evolution of expressions across iterations.

## Data Flow
1. **Input**: Initial hidden state $h_0$ and empty `MathWorkspace`.
2. **Iteration**:
   - $h_{i}' = \text{Attention}(h_{i-1} + \text{Project}(\text{Workspace}_{i-1}))$
   - $\text{Update} = \text{MathHeads}(h_{i}')$
   - $\text{Workspace}_{i} = \text{Update}(\text{Workspace}_{i-1})$
3. **Output**: Final hidden state and final `MathWorkspace`.

## Specialized Math Experts
The `MoERouter` directs the hidden state to specialized experts (`MathExpert`):
- **Algebra**: Basic operations and simplification.
- **Calculus**: Derivatives and integrals.
- **Numerical**: Floating point precision and stabilization.

## Visual Testing & Conceptualization
To rapidly test changes, use the `VisualizationAgent` hooks to plot:
- Latent state trajectories.
- Symbolic head activation patterns.
- Numerical value evolution in the workspace slots.
