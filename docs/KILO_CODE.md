# Kilo Code integration and delegation guide

**Status:** intended agent workflow, not an assertion that any agent has run locally. The project's root `AGENTS.md` and `TASKS.md` remain canonical.

## Supported configuration

Current Kilo VS Code extension and CLI support project-specific Markdown agent files under `.kilo/agents/` (also `.kilo/agent/`) with YAML frontmatter. `mode: primary` makes an agent selectable, while `mode: subagent` delegates via the `task` tool. See official documentation: https://kilo.ai/docs/customize/custom-modes and https://kilo.ai/docs/customize/custom-rules .

This repository deliberately uses `.kilo/agents/*.md`, **not** legacy `.kilocodemodes` or `.agents/`. Avoid a second project orchestrator config or redundant rules. Kilo loads root `AGENTS.md` automatically in supported versions.

## Current roles

| Agent | Mode | Scope |
|---|---|---|
| orchestrator | primary | Inspect TASKS.md, coordinate, delegate, gather evidence; no edits |
| simulation-engineer | subagent | world transitions and Gymnasium adapter |
| rl-engineer | subagent | agents/training/experiments pipeline |
| research-evaluator | subagent | read-only statistical and reward review |
| qa-reviewer | subagent | read-only implementation/CI and regression review |
| observatory-designer | subagent | read-only visualization design, later milestone |

Specialist `permission.edit` rules are intentionally scoped. An `ask` permission requires a user decision; `deny` must not be bypassed by switching tools or writing indirectly. Role prompts are guardrails, not substitutes for branch protection and review. The specific runtime permission behavior should be validated against the installed Kilo version.

## Recommended project workflow

1. Checkout the approved implementation branch; ensure documentation PR #1 has been integrated or deliberately base new work on its docs branch.
2. Select the `orchestrator` agent in Kilo. It reads `AGENTS.md`, `TASKS.md` and relevant canonical docs.
3. Give a narrow instruction such as: **"Analyze DL-010. Delegate simulation implementation only within its acceptance criteria, request independent QA, and report evidence. Do not merge."**
4. Orchestrator delegates only in-scope subtasks. Role owner implements, research-evaluator assesses experiment soundness, QA reviews actual diff.
5. Run focused tests and appropriate local quality gates, documenting executed commands/results, not predictions.
6. Report changed files, potential contract migrations, reproducibility artifacts, remaining blockers. Update TASKS.md boxes only with complete evidence.
7. Commits, pushes, PRs, merges and other irreversible actions require the owner's explicit authorization.

## Ownership / handoff matrix

- DL-010, DL-011, DL-012: simulation-engineer implementation; qa-reviewer acceptance review.
- DL-013 and DL-015: research-evaluator defines baseline/evaluation conditions; implementation scoped to appropriate code owner; qa-reviewer checks train/test leakage.
- DL-014: rl-engineer implementation; research-evaluator and qa-reviewer review outputs.
- DL-016: replay implementation coordinated with simulation owner, then QA.
- DL-017: orchestrator collects measured evidence; independent evaluation review.
- DL-050 onward: observatory-designer becomes relevant; no premature frontend implementation.

The current specialist allow-list intentionally excludes broad project scaffolding, so DL-001–DL-003 need an explicitly user-authorized coding session or a dedicated scoped bootstrap mode added when work starts. Never relax roles globally merely to get around this limitation.

## Important scientific boundaries

- Kilo's LLM coding agents **do not** constitute the simulated agents.
- The RL organism sees only the specified 199-element observation and chooses 1 of 6 actions.
- The authoritative simulation remains deterministic given the documented inputs and supported CPU runtime.
- Training results, generalization and emergent behavior require actual measurements, not agent-written claims.
- A subagent cannot change reward/termination/observation contracts to make tests easier without versioning and review.

## Local verification checklist

When Kilo is installed on the user's machine: open the repo in Kilo, confirm `orchestrator` appears in the agent selector, confirm subagent definitions can be delegated to, verify one read-only reviewer cannot edit, and verify edit permission prompts for specialist-owned paths. This validation has **not** been executed remotely by merely adding Markdown files.

## Compatibility

If using an older Kilo release that only recognizes legacy custom modes, upgrade the extension/CLI rather than duplicating agent definitions in `.kilocodemodes`. Keep YAML fields limited to documented `description`, `mode` and `permission` unless validated locally.
