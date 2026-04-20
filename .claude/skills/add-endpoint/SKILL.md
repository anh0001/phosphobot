---
name: add-endpoint
description: Use when adding or modifying a FastAPI route, changing request/response schemas, or touching files under phosphobot/phosphobot/endpoints/. Fires on "add endpoint", "new API route", "change response schema".
---

# Adding a FastAPI endpoint

Routes live in [phosphobot/phosphobot/endpoints/](../../../phosphobot/phosphobot/endpoints/), grouped by concern: `auth.py`, `camera.py`, `chat.py`, `control.py`, `networking.py`, `pages.py`, `recording.py`, `training.py`, `update.py`. App wiring: [phosphobot/phosphobot/app.py](../../../phosphobot/phosphobot/app.py).

## Constraints

- Pick the existing module that matches the concern; don't create a new module for a single route.
- Request/response models use Pydantic. Define them in the same module unless shared.
- Full type annotations are required (strict mypy). No `Any` returns without justification.
- If the endpoint is consumed by the dashboard, update the TS client in [dashboard/src/](../../../dashboard/src/) in the same PR — grep for a sibling endpoint's path to find where fetch calls live.

## Gotchas

- Long-running handlers (training, recording) should stream or background-task, not block the event loop.
- Auth: check whether the route needs the existing auth dependency (look at other routes in the same module).
- Integration tests in `phosphobot/tests/api/` need `make test_server` running — add a test there for any new route.

## Verification

1. `make types`
2. `make test_server` then `cd phosphobot && uv run pytest tests/api/ -s -v -k <your_route>`
3. If frontend-consumed: `cd dashboard && npm run test`
