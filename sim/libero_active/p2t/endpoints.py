"""P2T M0 — contact/lift-time endpoints, extracted OFFLINE from reach-field npz.

The pregrasp endpoint (eef at the first gripper-close command) captures the
PLANNED reach and can fire before the arm physically reaches the object — an
anchored policy that still succeeds via basin tolerance shows an anchored
pregrasp but must, by definition, have contacted+lifted the object at its true
location on successes. Measuring contact/lift therefore turns "proj is weakly
tied to success" (Codex audit) into positive evidence: pregrasp anchored (u≈1)
+ lift necessarily at the object (u→0) on successes = the anchoring is a
planning bias, not a measurement artifact, and success comes from late
correction/basin, not grounded planning.

Definitions (physical, threshold-calibrated on clean rollouts):
  contact = first step where the object first MOVES from rest
            (||obj_xyz - obj_xyz[0]|| > CONTACT_M); endpoint = eef at that step.
  lift    = first step where the object rises (obj_z - obj_z[0] > LIFT_M);
            endpoint = eef at that step.
Both fall back to None if the object never moved/lifted (failure to engage).

Usage:
  .venv/bin/python p2t/endpoints.py --logs p2t/eval_A_gain_logs --calibrate
  from p2t.endpoints import endpoints_from_npz, enrich_eval
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

CONTACT_M = 0.01   # object displaced 1 cm from rest = something touched it
LIFT_M = 0.03      # object risen 3 cm = picked up (rest z ~0.90, lift ~0.99)


def endpoints_from_arrays(eef: np.ndarray, obj: np.ndarray, act: np.ndarray) -> dict:
    """(contact, lift) eef xyz + step indices from one rollout's arrays."""
    n = len(eef)
    obj0 = obj[0]
    # 3D displacement (not xy-only) so a vertical pick registers contact at/before
    # lift; guarantees t_contact <= t_lift since any z-rise>LIFT_M implies
    # 3D-disp>LIFT_M>CONTACT_M. NaN object tracking -> no crossing -> None.
    disp = np.linalg.norm(obj - obj0, axis=1)
    rise = obj[:, 2] - obj0[2]
    moved = np.where(disp > CONTACT_M)[0]
    lifted = np.where(rise > LIFT_M)[0]
    t_contact = int(moved[0]) if len(moved) else None
    t_lift = int(lifted[0]) if len(lifted) else None
    return {
        "contact": eef[t_contact].tolist() if t_contact is not None else None,
        "lift": eef[t_lift].tolist() if t_lift is not None else None,
        "t_contact": t_contact, "t_lift": t_lift,
    }


def endpoints_from_npz(path: str | Path) -> dict:
    d = np.load(path)
    return endpoints_from_arrays(d["eef_xyz"], d["obj_xyz"], d["exec_action"])


def _key_to_npz(logs_dir: Path, rec: dict) -> Path | None:
    """Map an eval jsonl record to its npz (reach_field key = t{task}_s{seed}_{cond})."""
    key = rec.get("key")
    if key is None:
        # round-2/eval records may store task_id/seed/cond separately
        t, s, c = rec.get("task_id"), rec.get("seed"), rec.get("cond")
        if t is None or s is None or c is None:
            return None
        key = f"t{t}_s{s}_{c}"
    p = logs_dir / f"{key}.npz"
    return p if p.exists() else None


def enrich_eval(eval_jsonl: str, logs_dir: str) -> list[dict]:
    """Attach contact/lift endpoints to each displaced record from its npz."""
    logs = Path(logs_dir)
    out = []
    for line in Path(eval_jsonl).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if "error" in r:
            continue
        npz = _key_to_npz(logs, r)
        if npz is not None:
            ep = endpoints_from_npz(npz)
            r.setdefault("endpoints", {}).update(
                {"contact": ep["contact"], "lift": ep["lift"],
                 "t_contact": ep["t_contact"], "t_lift": ep["t_lift"]})
        out.append(r)
    return out


def calibrate(logs_dir: str, n: int = 8) -> None:
    """Sanity: on clean rollouts, contact precedes lift and both are defined."""
    fs = sorted(glob.glob(f"{logs_dir}/*clean*.npz"))[:n]
    if not fs:
        fs = sorted(glob.glob(f"{logs_dir}/*.npz"))[:n]
    print(f"[calibrate] {len(fs)} rollouts from {logs_dir}")
    ok = 0
    for f in fs:
        ep = endpoints_from_npz(f)
        d = np.load(f)
        grip = d["exec_action"][:, 6]
        close = np.where((grip[1:] > 0) & (grip[:-1] <= 0))[0]
        t_pg = int(close[0]) + 1 if len(close) else None
        order = (t_pg, ep["t_contact"], ep["t_lift"])
        sane = (ep["t_contact"] is not None and ep["t_lift"] is not None
                and ep["t_contact"] <= ep["t_lift"])
        ok += sane
        print(f"  {Path(f).name}: pregrasp={t_pg} contact={ep['t_contact']} "
              f"lift={ep['t_lift']} {'OK' if sane else 'CHECK'}")
    print(f"[calibrate] {ok}/{len(fs)} sane (contact defined, contact<=lift)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", required=True)
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--enrich", help="eval jsonl to enrich; writes <name>.enriched.jsonl")
    args = ap.parse_args()
    if args.calibrate:
        calibrate(args.logs)
    if args.enrich:
        recs = enrich_eval(args.enrich, args.logs)
        out = args.enrich.replace(".jsonl", ".enriched.jsonl")
        Path(out).write_text("\n".join(json.dumps(r) for r in recs))
        nc = sum(r["endpoints"].get("contact") is not None for r in recs
                 if "endpoints" in r)
        print(f"enriched {len(recs)} recs ({nc} with contact) -> {out}")


if __name__ == "__main__":
    main()
