"""Generate reach-field pair metadata for a LIBERO suite (env-only, no policy).

For each (task, seed) we reset the env, read the task instruction and the movable
free-joint objects, and match the instruction's target-object phrase to a free-joint
name (normalized: lowercased, digits/underscores stripped). We record that object's
name and canonical (reset) xyz. Output mirrors the records format `load_pair_meta`
reads (records[i] with variant -> perturb.object), so reach_field/object_local_paste
can consume it via --src.

This is the libero_object analog of the libero_spatial e1_recovery_d005.json obj_map.
The target object is an ENVIRONMENT property (the task's object), so no checkpoint is
needed. Run with E_SUITE set to the target suite.

Usage:
  E_SUITE=libero_object env -u PYTHONPATH MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
    python preempt/gen_pair_meta.py --tasks 0,1,2,3,4,5,6,7,8,9 \
      --seeds 1000,1001,1002,1003,1004 --out preempt/pair_meta_libero_object.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preempt.harness_lib import SUITE_NAME, make_single_env  # noqa: E402
from preempt.perturb import _movable_free_joints  # noqa: E402

STOP = {"the", "a", "an", "up", "pick", "place", "put", "and", "it", "on", "in",
        "into", "to", "of", "from", "between", "next", "left", "right", "front",
        "back", "top", "bottom", "side", "your", "that", "this", "object"}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z]", "", s.lower())


def target_phrase(instruction: str) -> str:
    """Extract the PICK target span, excluding the place destination.

    LIBERO-object instructions are 'pick up the X and place it in the basket';
    we must match X, not the basket. Take the span after 'pick up [the]' up to
    the first 'and'/'place'/'put'/'into'/'in '.
    """
    s = instruction.lower()
    m = re.search(r"pick up\s+(?:the\s+)?(.*?)(?:\s+and\b|\s+place\b|\s+put\b|\s+into\b|\s+in\s+|$)", s)
    if m and m.group(1).strip():
        return m.group(1).strip()
    m2 = re.search(r"pick up\s+(?:the\s+)?(.*)", s)
    return m2.group(1).strip() if m2 else s


def match_object(instruction: str, names: list[str]) -> str | None:
    """Pick the free-joint name whose normalized form best matches the PICK target.

    Strategy: score each object name by how many of its (normalized) word-tokens
    appear as substrings of the normalized target phrase; tie-break on longest match.
    """
    instr_n = _norm(target_phrase(instruction))
    best, best_score = None, (0, 0)
    for name in names:
        toks = [t for t in re.split(r"[_\d]+", name.lower()) if t and t not in STOP]
        if not toks:
            continue
        hits = sum(1 for t in toks if _norm(t) and _norm(t) in instr_n)
        cover = sum(len(_norm(t)) for t in toks if _norm(t) in instr_n)
        if (hits, cover) > best_score and hits > 0:
            best, best_score = name, (hits, cover)
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--seeds", default="1000,1001,1002,1003,1004")
    ap.add_argument("--out", default="preempt/pair_meta.json")
    args = ap.parse_args()
    tasks = [int(x) for x in args.tasks.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]

    records = []
    n_ok = n_miss = 0
    for t in tasks:
        for si, s in enumerate(seeds):
            env = make_single_env(task_id=t, episode_index=si)
            raw_obs, _ = env.reset(seed=s)
            instr = env.task_description
            sim = env._env.env.sim
            free = _movable_free_joints(sim)
            names = [n for n, _ in free]
            obj = match_object(instr, names)
            if obj is None:
                n_miss += 1
                print(f"[MISS] t{t}_s{s}: '{instr}' | objs={names}", flush=True)
                env.close()
                continue
            adr = dict(free)[obj]
            xyz = np.asarray(sim.data.qpos[adr:adr + 3]).tolist()
            # records format consumed by load_pair_meta (variant -> perturb.object)
            records.append({"task_id": t, "seed": s, "variant": "open_loop",
                            "task": instr, "perturb": {"object": obj, "before_xyz": xyz}})
            records.append({"task_id": t, "seed": s, "variant": "clean", "success": True})
            n_ok += 1
            print(f"[ok] t{t}_s{s}: obj={obj}  xyz={[round(v,3) for v in xyz]}  | '{instr[:50]}'", flush=True)
            env.close()

    Path(args.out).write_text(json.dumps(
        {"config": {"suite": SUITE_NAME, "source": "gen_pair_meta (env-only instruction match)"},
         "records": records}, indent=1))
    print(f"\n[gen_pair_meta] suite={SUITE_NAME} ok={n_ok} miss={n_miss} -> {args.out}")


if __name__ == "__main__":
    main()
