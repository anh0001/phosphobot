# ECC Codex Adaptation for phosphobot

This note records what was applied from `everything-claude-code` and why.

## What Was Reused

The following ECC ideas fit this repo well:

1. A thin Codex supplement layered on top of the root `AGENTS.md`
2. Project-local `config.toml` with multi-agent role wiring
3. Small, role-specific agent configs instead of one generic subagent setup
4. Instruction-based workflow prompts in place of slash commands
5. An explicit statement of what is intentionally not ported

## What Was Applied Here

### 1. Thin Codex Adapter

Added:

- `.codex/AGENTS.md`
- `.codex/README.md`

This mirrors ECC's "root AGENTS plus Codex supplement" pattern, but keeps the repo-specific guidance centered on the existing `AGENTS.md`.

### 2. Project-Local Codex Config

Added:

- `.codex/config.toml`

Adopted from ECC:

- `approval_policy`, `sandbox_mode`, and `web_search` defaults
- optional MCP recommendations for GitHub, Context7, Playwright, and Sequential Thinking
- `features.multi_agent = true`
- project-local agent role registration

Adapted for phosphobot:

- simulation-first persistent instructions
- roles tailored to robotics, Python backend review, and browser E2E coverage

### 3. Multi-Agent Role Files

Added:

- `.codex/agents/explorer.toml`
- `.codex/agents/planner.toml`
- `.codex/agents/reviewer.toml`
- `.codex/agents/python-reviewer.toml`
- `.codex/agents/docs-researcher.toml`
- `.codex/agents/e2e-runner.toml`

These are direct descendants of ECC's role-based approach, but reshaped around this repo's actual work:

- `planner` mirrors the existing planning workflow
- `python-reviewer` matches the Python-heavy backend
- `e2e-runner` maps to the web surfaces already present
- every role calls out simulation-first safety when relevant

### 4. Instruction-Based Prompt Files

Added:

- `.codex/prompts/feature-development.md`
- `.codex/prompts/plan.md`
- `.codex/prompts/tdd.md`
- `.codex/prompts/e2e.md`
- `.codex/prompts/python-review.md`
- `.codex/prompts/verify.md`

This follows ECC's "Codex is instruction-driven" practice. Instead of cloning the whole ECC command catalog, these prompts map directly to the workflows already maintained under `.claude/commands/`.

## What Was Intentionally Not Applied

These ECC assets were deliberately skipped:

1. Global sync and installer scripts
   - not appropriate for a project repo that already has local guidance
2. Global git hooks
   - useful in ECC, but out of scope for a thin repo-local Codex layer
3. Full skill and rules duplication
   - the repo already has `.claude/skills/` and `.claude/rules/`
   - Codex can read those files directly when needed
4. Huge generic command catalogs
   - the repo benefits more from a small set of prompts tied to actual current workflows
5. Notification or desktop-specific config
   - portable repo defaults matter more than host-specific extras

## Net Result

The Codex setup now follows ECC's strongest reusable practices:

- additive guidance instead of replacement
- role-based multi-agent support
- explicit workflow prompts
- instruction-first adaptation instead of harness-specific assumptions

But it stays repo-native by reusing the existing phosphobot documentation and safety rules rather than importing ECC wholesale.
