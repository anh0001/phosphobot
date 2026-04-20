---
description: Run repo-appropriate verification for the touched surface in phosphobot.
---

# Verification Command

Use this when you want a focused verification pass before handing work back or preparing a PR.

## Arguments

`$ARGUMENTS`

## Verification Matrix

- Backend Python logic:
  - run the narrowest relevant `pytest` target first
  - run `make types` if types changed
- API endpoints:
  - start `make test_server`
  - run `cd phosphobot && uv run pytest tests/api/ -s -v`
- Dashboard:
  - run `cd dashboard && npm run build`
  - run `cd dashboard && npm run test` when tested utilities/state changed
  - if backend-shipped assets changed, rebuild with `make build_frontend` or `make`
- Next.js apps:
  - run `npm run build` in `dataset-viewer/` or `cloud/frontend/`
- Hardware/control/URDF/calibration:
  - verify in simulation first
  - do not assume physical hardware access

## Output

- Report exactly what ran, what passed, what failed, and what remains unverified.
- Do not claim full verification if only a subset ran.
