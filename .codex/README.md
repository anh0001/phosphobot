# Codex Setup

This directory is the Codex-native adapter layer for `phosphobot`.

## Files

- `config.toml`: project-local Codex defaults, MCP recommendations, and multi-agent role wiring
- `AGENTS.md`: Codex-specific supplement to the root repo guidance
- `agents/`: focused multi-agent role definitions
- `prompts/`: reusable workflow prompts aligned with the existing `.claude/commands/` surface
- `research/ecc-codex-adaptation.md`: what was adopted from ECC and what was intentionally skipped

## Usage

Run Codex from the repo root so it automatically loads:

- `AGENTS.md`
- `.codex/AGENTS.md`
- `.codex/config.toml`

For workflow-heavy tasks, point Codex at the relevant prompt file, for example:

- `Follow .codex/prompts/feature-development.md for this change.`
- `Use .codex/prompts/python-review.md on my current diff.`
- `Plan this work using .codex/prompts/plan.md and wait for confirmation.`

## Design Choice

The repo already has a mature `.claude/` layout. This Codex layer stays intentionally thin and reuses those assets by reference rather than cloning them.
