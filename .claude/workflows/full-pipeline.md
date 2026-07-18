# OpenMetis Full Development Pipeline

This workflow defines the sequential collaboration of agents for any enhancement or bug fix in the OpenMetis project.

## Workflow Stages

### 1. Discovery (The "Map")
**Lead Agent**: `ContextIngestor`
- **Action**: Analyze current `nn/` implementation and identify the target for enhancement.
- **Output**: Context report in `docs/architecture/`.

### 2. Design (The "Formula")
**Lead Agent**: `MathAgent`
- **Co-Agent**: `ProjectScaffolder`
- **Action**: Define the mathematical requirements and necessary structural changes.
- **Output**: Updated `docs/architecture/overview.md` with new math logic specs.

### 3. Implementation (The "Code")
**Lead Agent**: `CodeGenerator`
- **Action**: Write PyTorch code for the new component. Update `nn/block.py` or `nn/workspace.py`.
- **Output**: Pull request or file updates in `nn/`.

### 4. Verification (The "Gate")
**Lead Agent**: `Reviewer`
- **Co-Agent**: `TestGenerator`
- **Action**: Conduct code review and run automated tests (including `gradcheck`).
- **Output**: Test results and review approval.

### 5. Visualization (The "Insight")
**Lead Agent**: `VisualizationAgent`
- **Action**: Generate plots or demos showing the new component in action.
- **Output**: Visual artifacts in `docs/tutorials/` or `plots/`.

### 6. Delivery (The "Healthcheck")
**Lead Agent**: `HealthcheckAgent`
- **Action**: Verify workspace integrity and finalize documentation.
- **Output**: Final project status report.
