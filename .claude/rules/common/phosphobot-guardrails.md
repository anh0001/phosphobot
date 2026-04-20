# Phosphobot Guardrails

These repo-specific guardrails complement the shared coding, testing, and security rules.

## Monorepo Boundaries

- Keep changes scoped to the smallest relevant area: `phosphobot/`, `dashboard/`, `dataset-viewer/`, `cloud/frontend/`, or `modal/`.
- Do not introduce cross-package abstractions unless the task clearly needs them.
- Avoid incidental lockfile churn when dependencies did not change.

## Robotics Safety

- Treat motor, calibration, CAN, URDF, and controller changes as safety-sensitive.
- Default to simulation-only, headless simulation, or pybullet GUI validation.
- Do not assume physical hardware is connected.

## Frontend Integration

- Do not hand-edit `phosphobot/resources/dist`.
- If dashboard source changes need to ship with the backend, rebuild via `make` or `make build_frontend`.
- Preserve existing design/system patterns unless the task explicitly asks for a redesign.

## Verification

- Run the narrowest relevant checks first.
- Backend logic changes should usually verify with targeted `pytest`.
- If Python types are affected, run `make types`.
- Dashboard changes should run `npm run build`, plus `npm run test` for tested logic.
- Next.js app changes should run `npm run build`.

## Worktree Safety

- Do not overwrite unrelated local changes.
- Prefer additive or surgical edits over broad refactors.
- Keep generated artifacts intentional and easy to explain in the final summary.
