# Codex Prompt: Plan

Use this when the task needs explicit planning before implementation.

## Source Material

- Existing Claude workflow: `.claude/commands/plan.md`
- Planner role: `.codex/agents/planner.toml`

## Instructions

1. Restate the request in concrete repo terms.
2. Identify affected packages, files, and interfaces.
3. Surface risks and blockers early, especially:
   - robot-safety or calibration risk
   - backend/frontend integration risk
   - dependency, packaging, or CI impact
4. Produce a step-by-step plan with:
   - exact file paths where possible
   - dependencies between steps
   - validation per step
5. Stop after the plan and wait for confirmation before touching code.
