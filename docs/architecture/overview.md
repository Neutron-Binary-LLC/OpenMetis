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
A persistent container for the mathematical state, now using a hybrid representation.
- **Latent State**: High-dimensional vector representing abstract math context.
- **Numerical Slots**: Tensor of slots (default 16) for constants, variables, and gradients.
- **Expression Trees**: AST-based symbolic representation for formal reasoning and interpretability.
- **Confidence**: Vector [0, 1] indicating the model's certainty for each batch item.
- **Reasoning Traces**: Step-by-step history of symbolic updates, verification statuses, and tool calls.

## Reasoning & Verification Logic

### Formal Verification (Phase 3)
The framework integrates **Z3 SMT Solver** to rigorously verify mathematical properties proposed by the neural core.
- **Goals**: The model can select verification goals such as *Consistency*, *Positivity*, or *Equality*.
- **SMT Translation**: Expression trees are recursively translated into Z3 symbolic expressions.

### Self-Correction Loop
If the verification confidence falls below a critical threshold (e.g., 0.3), the `NeuroSymbolicReasoningCell` performs:
1. **Backtracking**: Reverts the workspace to the state before the failed proposal.
2. **Perturbation**: Adds a small noise to the hidden state to encourage exploration of alternative reasoning branches.
3. **Retry**: Re-runs the iteration with the perturbed state.

## Data Flow
1. **Input**: Initial hidden state $h_0$ and empty `MathWorkspace`.
2. **Iteration**:
   - $h_{i}' = \text{Attention}(h_{i-1} + \text{Project}(\text{Workspace}_{i-1}))$
   - $\text{Update} = \text{MathHeads}(h_{i}')$
   - $\text{Workspace}_{i} = \text{Update}(\text{Workspace}_{i-1})$
3. **Output**: Final hidden state and final `MathWorkspace`.

## Specialized Math Experts
The `MoERouter` directs the hidden state to specialized experts (`MathExpert`):
- **Algebra**: Basic operations and simplification (`nn/symbolic_expert.py`).
- **Calculus**: Derivatives and integrals.
- **Financial**: Black-Scholes, Greeks, and market dynamics (`nn/fin_math.py`).
- **Numerical**: Floating point precision and stabilization.

## World Environment & Tasks (`world/`)
OpenMetis interacts with a simulated financial environment to ground its reasoning:
- **DataSource**: Generates synthetic or real market data (prices, volatility, rates).
- **Environment**: Manages the state of financial instruments and task execution.
- **Tasks**: Defines reasoning goals such as "Calculate Delta" or "Verify Put-Call Parity".

## Visual Testing & Conceptualization
To rapidly test changes, use the `VisualizationAgent` hooks to plot:
- Latent state trajectories.
- Symbolic head activation patterns.
- Numerical value evolution in the workspace slots.
