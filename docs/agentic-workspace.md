# Agentic Workspace for OpenMetis

The OpenMetis Agentic Workspace is a multi-agent environment designed to accelerate the development of neural-symbolic systems and financial mathematical models. It leverages specialized agents to handle different aspects of the reasoning pipeline, from raw data ingestion to complex visualization.

## Multi-Agent Pipeline

The workflow is orchestrated through a series of handoffs between agents, ensuring that every architectural decision and implementation detail is verified and documented:

1.  **Ingestion & Mapping (`ContextIngestor`)**:
    - Scans `nn/`, `world/`, and `metis_model/` to understand current capabilities.
    - Extracts financial math rules and latent state update logic from `nn/fin_math.py`.
2.  **Mathematical Design (`MathAgent` & `ProjectScaffolder`)**:
    - Validates proposed changes for numerical stability and financial model accuracy.
    - Ensures symbolic rewrite rules in `nn/symbolic_expert.py` are mathematically sound.
    - Scaffolds the necessary file structures for new experiments.
3.  **Implementation (`CodeGenerator`)**:
    - Writes PyTorch code for new reasoning cells, symbolic experts, or financial tools.
    - Integrates complex math logic into the recurrent loop of the `NeuroSymbolicReasoningCell`.
4.  **Verification (`Reviewer` & `TestGenerator`)**:
    - Checks for gradient flow, differentiability, and PyTorch best practices.
    - Generates and runs unit tests covering extreme market scenarios.
5.  **Insights (`VisualizationAgent`)**:
    - Visualizes latent state transitions and financial reasoning traces.
    - Generates orchestrator maps to conceptualize multi-stage reasoning.
6.  **Integrity Monitor (`HealthcheckAgent`)**:
    - Ensures all documentation is consistent with the latest code changes.
    - Validates that all agents have the correct context for the next development cycle.

## Benefits of this Setup
- **Domain Expertise**: Specialized agents for Math and Visualization prevent common deep learning pitfalls in financial engineering.
- **Differentiable Logic**: Focused testing on differentiability ensures the hybrid model can be trained end-to-end.
- **Rapid Iteration**: Automated scaffolding and testing allow for quick exploration of new neural-symbolic architectures.
