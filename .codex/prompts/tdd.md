# Codex Prompt: TDD

Use this when you want a strict test-first workflow.

## Source Material

- Existing Claude command shim: `.claude/commands/tdd.md`
- Maintained skill body: `.claude/skills/tdd-workflow/SKILL.md`

## Instructions

Follow RED -> GREEN -> REFACTOR.

- Write or update the smallest failing test first.
- Make the smallest code change to pass.
- Refactor only after the test passes.
- Re-run the narrowest relevant verification before broadening scope.

## Repo-Native Test Targets

- Backend: targeted `pytest` under `phosphobot/tests/`
- API: `make test_server`, then `cd phosphobot && uv run pytest tests/api/ -s -v`
- Dashboard: `cd dashboard && npm run test`

## Guardrails

- Keep hardware changes simulation-first even when following TDD.
- Do not add a new test harness if the touched app already has one.
