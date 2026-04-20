---
description: Comprehensive PR review using specialized agents
---

Run a comprehensive multi-perspective review of a pull request.

## Usage

`/review-pr [PR-number-or-URL] [--focus=comments|tests|errors|types|code|simplify]`

If no PR is specified, review the current branch's PR. If no focus is specified, run the full review stack.

## Steps

1. Identify the PR:
   - use `gh pr view` to get PR details, changed files, and diff
2. Find project guidance:
   - read `AGENTS.md`, `CLAUDE.md`, and relevant repo rules first
3. Run specialized review agents:
   - `code-reviewer`
   - `python-reviewer` for Python-heavy diffs
   - `typescript-reviewer` for frontend or TS-heavy diffs
   - `pr-test-analyzer`
   - `silent-failure-hunter`
   - `security-reviewer`
   - `vla-reviewer` when inference, robotics, or VLA integration code changed
4. Aggregate results:
   - dedupe overlapping findings
   - rank by severity
5. Report findings grouped by severity

## Confidence Rule

Only report issues with confidence >= 80:

- Critical: bugs, security, data loss
- Important: missing tests, safety problems, packaging/integration fallout
- Advisory: suggestions only when explicitly requested
