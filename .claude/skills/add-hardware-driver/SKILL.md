---
name: add-hardware-driver
description: Use when adding support for a new robot/arm, modifying robot motion primitives (move_relative, move_absolute, teleop), or touching files under phosphobot/phosphobot/hardware/. Covers driver scaffolding, URDF registration, and controller-orientation mapping conventions.
---

# Adding or modifying a hardware driver

Drivers live in [phosphobot/phosphobot/hardware/](../../../phosphobot/phosphobot/hardware/). Each robot has its own module inheriting from `base.py`. Existing: `so100.py`, `koch11.py`, `wx250s.py`, `piper.py`, `go2.py`, `lekiwi.py`, plus `sim.py` (PyBullet), `phosphobot.py` (phospho arm).

## Constraints

- New driver = new module in `hardware/`, subclass of the base in `base.py`. Register it wherever the existing drivers are registered (grep for an existing driver name like `so100` to find all registration sites).
- URDFs go under [phosphobot/resources/urdf/](../../../phosphobot/resources/urdf/). Load via `urdfloader.py`.
- Controller-orientation mapping (see `move_relative` on `piper.py`) is per-robot — do not assume SO-100 axes apply to a new arm.
- Motor communication code belongs under `hardware/motors/`, not inline in the driver.

## Gotchas

- Simulation vs. real: `sim.py` shadows real drivers in `make local` mode. Test both paths before claiming a motion fix works.
- `make types` is strict — every new method needs full annotations.
- Hardware tests are gated by physical presence. Unit-test pure logic (coordinate transforms, limits) without the hardware.
- A driver change often needs a matching frontend change in [dashboard/](../../../dashboard/) (robot picker, teleop UI). Grep for the robot name in `dashboard/src/`.

## Verification

1. `make types`
2. `make tests`
3. `make local` → sanity-check in simulation before touching real hardware.
