---
name: dashboard-dev
description: Use when modifying the React/Vite dashboard (dashboard/ directory), debugging UI behavior, or when changes need hot-reload instead of a full backend rebuild. Fires on "dashboard", "frontend", "UI bug", "React component".
---

# Dashboard (frontend) work

Stack: React 19 + TypeScript + Vite 6 + Tailwind 4 + Radix UI + Zustand. Source: [dashboard/src/](../../../dashboard/src/). Build output: [phosphobot/resources/dist/](../../../phosphobot/resources/dist/) (served by the backend).

## Workflow

- **Hot-reload dev:** `cd dashboard && npm run dev` — fastest loop for UI-only changes.
- **Full stack dev:** `make local` — needed when your change depends on backend behavior.
- **Backend-only iteration after frontend is built:** `make prod_back`.

## Constraints

- Tailwind 4 — no `tailwind.config.js` in the v3 style; utility tokens live in CSS. Don't reintroduce v3 config.
- Radix UI for primitives; Zustand for state. Don't pull in Redux or a second UI kit.
- Prettier + ESLint 9 are authoritative. Run before committing.

## Gotchas

- Production UI is served from the backend at built paths; a change not visible after `make prod` likely means the frontend didn't rebuild. Check `phosphobot/resources/dist/` mtimes.
- Camera stream URLs and auth session flows have been touched recently (see recent commits) — when editing either, retest in-browser, don't trust types alone.
- Visual/UX bugs: take a screenshot and include it in the conversation. Don't guess from code alone.

## Verification

1. `cd dashboard && npm run test`
2. Load the feature in a browser (`npm run dev` or `make local`), walk the golden path + one edge case.
3. Report explicitly if you couldn't test in-browser.
