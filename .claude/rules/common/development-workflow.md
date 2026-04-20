# Development Workflow

> This file extends [common/git-workflow.md](./git-workflow.md) with the repo-specific workflow that happens before git operations.

The feature implementation workflow for `phosphobot` is: inspect local context, choose the smallest safe change, verify with the narrowest relevant checks, then prepare git operations.

## Feature Implementation Workflow

0. **Inspect Before Editing** _(mandatory)_
   - Read `AGENTS.md`, `CLAUDE.md`, and the relevant local package/app first.
   - Check whether a similar endpoint, hardware driver, model integration, or frontend component already exists.
   - Prefer reusing repo patterns over importing new abstractions.

1. **Plan to the Right Depth**
   - For larger or riskier changes, create a short implementation plan before editing.
   - Call out integration points across backend, frontend, simulation, and packaged assets when relevant.
   - Surface safety risks early for hardware-facing changes.

2. **Implementation**
   - Prefer the smallest coherent change that satisfies the request.
   - Keep backend, dashboard, and hardware concerns separated unless the feature requires coordinated edits.
   - Do not manually edit generated dashboard output in `phosphobot/resources/dist`.

3. **Verification**
   - Backend logic: run the narrowest relevant `pytest` target first.
   - Type-sensitive Python changes: run `make types`.
   - Dashboard changes: run `npm run build`, and `npm run test` when utilities/state logic changed.
   - Next.js app changes: run `npm run build`.
   - Hardware/control changes: validate in simulation first.

4. **Review**
   - Review the actual diff, not just changed snippets in isolation.
   - Focus on behavior regressions, safety issues, missing tests, and packaging/integration fallout.

5. **Pre-PR Checks**
   - Match local checks to the touched surfaces and CI expectations.
   - Rebuild dashboard assets when backend-shipped UI changed.
   - See [git-workflow.md](./git-workflow.md) for commit message and PR process details.
