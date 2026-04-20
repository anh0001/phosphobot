# Codex Prompt: Python Review

Use this for a Python-focused review of the current diff or a target change set.

## Source Material

- Existing Claude command: `.claude/commands/python-review.md`
- Existing reviewer doc: `.claude/agents/python-reviewer.md`
- Codex role: `.codex/agents/python-reviewer.toml`

## Review Priorities

1. Security:
   - unsafe shelling out
   - path traversal
   - unsafe deserialization
   - hardcoded secrets
   - broad exception swallowing that hides failures
2. Correctness:
   - blocking work in async paths
   - weak request or response validation
   - mutable defaults
   - race conditions and resource handling mistakes
3. Type safety:
   - missing or weak annotations on public interfaces
   - mypy regressions
   - misuse of `Any`
4. Maintainability:
   - deep nesting
   - duplicated logic
   - missing tests around changed behavior

## Repo-Native Checks

- `make types`
- `make sort`
- `make tests`
- API tests when endpoints changed

## Output

Lead with findings ordered by severity and include concrete file references.
