"""Mid-rollout disturbance: nudge the movable object nearest the eef laterally.

CONFIRMED mechanics (see preempt/_introspect.py):
  - robosuite env  = lerobot_env._env.env ;  sim = rs.sim
  - movable objects = FREE joints (jnt_type == 0); qpos slice [adr:adr+3] is xyz
  - eef pos        = rs._get_observations()["robot0_eef_pos"]
  - nudge: sim.data.qpos[adr] += DELTA ; sim.forward()
"""
from __future__ import annotations

import numpy as np

_MJ_JNT_FREE = 0


def _movable_free_joints(sim) -> list[tuple[str, int]]:
    out = []
    for jid in range(sim.model.njnt):
        if int(sim.model.jnt_type[jid]) == _MJ_JNT_FREE:
            name = sim.model.joint_id2name(jid)
            adr = int(sim.model.jnt_qposadr[jid])
            out.append((name, adr))
    return out


def perturb_object_by_name(lerobot_env, object_name: str, delta: float = 0.05) -> dict:
    """Nudge a SPECIFIC named free-joint object +delta in x (control for perturbed_start).

    Used so the episode-start control displaces the exact object the mid-rollout
    disturbance targeted (read from the recovery-run records).
    """
    rs = lerobot_env._env.env
    sim = rs.sim
    match = [(n, a) for (n, a) in _movable_free_joints(sim) if n == object_name]
    if not match:
        return {"perturbed": False, "reason": f"object_not_found:{object_name}"}
    name, adr = match[0]
    before = np.asarray(sim.data.qpos[adr:adr + 3]).copy()
    sim.data.qpos[adr] += delta
    sim.forward()
    after = np.asarray(sim.data.qpos[adr:adr + 3]).copy()
    return {"perturbed": True, "object": name, "delta": delta,
            "before_xyz": before.tolist(), "after_xyz": after.tolist()}


def perturb_object_by_name_vec(lerobot_env, object_name: str, dxy: tuple[float, float]) -> dict:
    """Nudge a SPECIFIC named free-joint object by (dx, dy) in the world XY plane.

    Generalizes perturb_object_by_name to arbitrary planar directions (reach-field
    grid). Returns before/after xyz so the analysis can use the REALIZED
    displacement (collision resolution may shift the commanded one).
    """
    rs = lerobot_env._env.env
    sim = rs.sim
    match = [(n, a) for (n, a) in _movable_free_joints(sim) if n == object_name]
    if not match:
        return {"perturbed": False, "reason": f"object_not_found:{object_name}"}
    name, adr = match[0]
    before = np.asarray(sim.data.qpos[adr:adr + 3]).copy()
    sim.data.qpos[adr] += dxy[0]
    sim.data.qpos[adr + 1] += dxy[1]
    sim.forward()
    after = np.asarray(sim.data.qpos[adr:adr + 3]).copy()
    return {"perturbed": True, "object": name, "dxy": [float(dxy[0]), float(dxy[1])],
            "before_xyz": before.tolist(), "after_xyz": after.tolist(), "qpos_adr": adr}


def object_xyz(lerobot_env, qpos_adr: int) -> np.ndarray:
    """Current xyz of a free-joint object given its qpos address."""
    sim = lerobot_env._env.env.sim
    return np.asarray(sim.data.qpos[qpos_adr:qpos_adr + 3]).copy()


def perturb_nearest_object(lerobot_env, delta: float = 0.05) -> dict:
    """Find the movable object nearest the eef in XY and nudge it +delta in x.

    Returns a small dict describing what was moved (for logging / debugging).
    Mutates the live MuJoCo sim in place (this IS the disturbance).
    """
    rs = lerobot_env._env.env
    sim = rs.sim
    eef = np.asarray(rs._get_observations()["robot0_eef_pos"])

    free = _movable_free_joints(sim)
    if not free:
        return {"perturbed": False, "reason": "no_free_joints"}

    def _xy_dist(adr: int) -> float:
        pos = np.asarray(sim.data.qpos[adr:adr + 3])
        return float(np.linalg.norm(pos[:2] - eef[:2]))

    name, adr = min(free, key=lambda t: _xy_dist(t[1]))
    before = np.asarray(sim.data.qpos[adr:adr + 3]).copy()
    sim.data.qpos[adr] += delta
    sim.forward()
    after = np.asarray(sim.data.qpos[adr:adr + 3]).copy()
    return {
        "perturbed": True,
        "object": name,
        "delta": delta,
        "before_xyz": before.tolist(),
        "after_xyz": after.tolist(),
        "eef_xyz": eef.tolist(),
        "d_eef_xy": _xy_dist(adr),
    }
