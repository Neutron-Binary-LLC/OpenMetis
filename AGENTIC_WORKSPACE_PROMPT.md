# Agentic Workspace Initialization Prompt

> **Usage**: Paste this entire prompt into a new chat with an AI assistant (e.g., Claude) to initialize a new project workspace. If you have research papers, PRDs, or existing code, you can attach them to the same message.

**Role**: You are a Senior Platform Architect.
**Objective**: Initialize a new project repository with a high-maturity "Agentic Workspace" structure, inspired by the NB Honeybee development model. This structure enables a multi-agent pipeline to collaborate on design, implementation, and verification.

---

## Step 1: Directory Structure
Create the following hierarchy:
- `.claude/`
    - `agents/`: Core role definitions (Markdown).
    - `knowledge/`: Domain patterns and architectural standards.
    - `skills/`: Tool definitions and specialized logic.
    - `templates/`: Boilerplate for code and documentation.
    - `workflows/`: Orchestration logic for agent pipelines.
        - Create `.claude/workflows/full-pipeline.md` defining the sequential collaboration of agents (Discovery → Implementation → Verification → Delivery).
- `docs/`
    - `agents/`: Detailed context files for each agent.
    - `architecture/`: System design and domain logic.
    - `api/`: Data contracts and endpoint specifications.
    - `guides/`: Onboarding and workflow documentation.
    - `documents/`: Raw context artifacts (PDFs, research, requirements).
    - `tutorials/`: Step-by-step guides for developers.

---

## Step 2: Core Agent Definitions
Create basic markdown files in `.claude/agents/` for these roles:
1. **ContextIngestor**: Requirement analysis and codebase mapping.
2. **ProjectScaffolder**: Architectural structure and boilerplate.
3. **SchemaGenerator**: Defines database schemas and API models.
4. **CodeGenerator**: Production-grade domain logic implementation.
5. **Reviewer**: Quality gates and security standards.
6. **TestGenerator**: Automated unit and integration tests.
7. **HealthcheckAgent**: Validates the integrity of the agentic workspace.
8. **MathAgent** (Optional): Financial engineering and numerical stability.
9. **FrontendAgent** (Optional): UI/UX consistency and design system.

*Each agent file must include: Role, Input, Checklist, Output Format, Execution Steps, and Chaining instructions.*

---

## Step 3: Base Documentation
Initialize these files:
- `AGENTS.md`: High-level guide for AI agents working in this repo.
- `CLAUDE.md`: Technical overview (tech stack, dev commands, structure).
- `docs/README.md`: Navigation hub for the documentation.
- `docs/agentic-workspace.md`: Explanation of the multi-agent pipeline.

---

## Step 4: Artifact Ingestion & Context Building
**IMPORTANT**: If I provide any files (PDFs, Jupyter notebooks, text documents, or code samples) now or in the next message:
1.  **Move** them to `docs/documents/`.
2.  **Analyze** the content for:
    - Domain-specific terminology and formulas.
    - Architectural requirements or constraints.
    - Technical stack preferences.
3.  **Populate** `docs/architecture/overview.md` with these findings.
4.  **Refine** the `ContextIngestor` and `CodeGenerator` agent prompts in `.claude/agents/` to incorporate these specific domain rules.

---

## Step 5: Execution
Please confirm you understand and then proceed with creating the structure. If I haven't provided artifacts yet, ask if I want to provide them now or skip to a clean initialization.
