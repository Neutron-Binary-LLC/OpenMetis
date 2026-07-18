# CLAUDE.md — OpenMetis Technical Overview

## Project Mission
Developing neural-symbolic integration by combining transformer-style neural processing with a persistent, differentiable mathematical workspace.

## Tech Stack
- **Language**: Python 3.14+
- **Framework**: PyTorch 2.11.0+
- **Configuration**: YAML (via `config.yaml`)
- **Package Manager**: uv (implied by `uv.lock`)

## Core Architecture
- **NeuroSymbolicReasoningCell**: Recurrent block with shared weights across iterations.
- **MathWorkspace**: Persistent state for latent context, numerical values, and confidence scores.
- **MoE Experts**: Specialized layers for Algebra, Calculus, etc.
- **Symbolic Heads**: Linear layers proposing updates for differentiation, integration, etc.

## Development Commands
- **Basic Demo**: `python demo.py`
- **Black-Scholes Demo**: `python demo_metis_bs.py`
- **Advanced Training**: `python train_advanced.py --epochs 10 --lr 1e-4`
- **Resume Training**: `python train_advanced.py --resume --checkpoint math_block_checkpoint.pth`
- **Large-Scale Training**: `python metis_model/train_metis.py`
- **Interactive Shell**: `python interact.py`

## Project Structure
- `nn/`: Core neural-symbolic components.
    - `block.py`: `NeuroSymbolicReasoningCell`.
    - `workspace.py`: `MathWorkspace`.
    - `expression.py`: Symbolic utilities.
- `metis_model/`: High-level model implementations.
- `docs/`: Comprehensive documentation and agent knowledge.
- `.claude/`: Agentic workspace definitions.

## Coding Standards
- Use type hints for all function signatures.
- Maintain differentiability in `apply_*` methods.
- Follow existing patterns in `NeuroSymbolicReasoningCell` for new math heads.
