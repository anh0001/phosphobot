"""E2r — object-local paste probe (mislocalization SUFFICIENCY instrument).

Successor to the pilot's `oracle_warp.py`, whose GLOBAL frame-translate was OOD
(warp_clean broke 87% of healthy episodes) and so could not adjudicate sufficiency.
Here we move ONLY the target object's *appearance*, leaving the rest of the scene
intact, via a renderer-local virtual object render (Codex xhigh spec):

  per macro step:
    1. save target object free-joint qpos
    2. set object xy -> VISUAL location V; sim.forward(); _refresh_obs -> virtual imgs
    3. restore object qpos -> PHYSICAL location; sim.forward()
    4. feed the policy the virtual image(s) + REAL proprio; step REAL physics

Conditions = 2x2 factorial (physical loc x visual loc) + specificity, d=50mm:
  CC  physical C, visual C            sham / machinery no-op  (||E_CC - E_clean|| should be tiny)
  CT  physical C, visual T            PRIMARY sufficiency arm (does seeing object at T steer reach->T?)
  TC  physical T, visual C            counter-paste sanity (non-identifying)
  TT  physical T, visual T            E1r-compatible baseline (no-op render at real pos)
  CR  physical C, visual C+R90(d)     specificity (paste artifact control)

Primary metric (analysis): per (task,seed,dir), S = dot(E_CT - E_CC, d)/||d||^2 on
the pregrasp endpoint. Sufficiency passes if median S>=0.40, task-cluster boot lower
95% > 0.15, >=55% of CT endpoints closer to visual-T than canonical-C, CR < CT, and
CC is a no-op vs the clean rollout.

Usage:
  env -u PYTHONPATH MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
    python preempt/object_local_paste.py --ckpt <strong_ckpt> \
      [--smoke] [--pairs-from preempt/e0_eval_fullexp_100k.jsonl] \
      [--mags 0.05] [--dirs px,nx,py,ny] [--conditions CC,CT,TC,TT,CR] \
      [--edit-cams image,image2] [--out ...] [--log-dir ...]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preempt.harness_lib import (  # noqa: E402
    GRIPPER_DIM,
    Pipeline,
    _refresh_obs,
    build_pipeline,
    build_policy_batch,
    make_single_env,
)
from preempt.perturb import object_xyz, perturb_object_by_name_vec  # noqa: E402
from preempt.reach_field import load_pair_meta, run_rollout  # noqa: E402
from lerobot.utils.constants import ACTION  # noqa: E402
from lerobot.utils.random_utils import set_seed  # noqa: E402

DELTA = 0.05
DIRS = {"px": (1.0, 0.0), "nx": (-1.0, 0.0), "py": (0.0, 1.0), "ny": (0.0, -1.0)}
SEEDS = [1000, 1001, 1002, 1003, 1004]
SRC = "preempt/e1_recovery_d005.json"
# condition -> (physical_loc, visual_loc); locs in {C, T, R(=C+R90(d))}
COND = {"CC": ("C", "C"), "CT": ("C", "T"), "TC": ("T", "C"),
        "TT": ("T", "T"), "CR": ("C", "R")}


def clean_success_pairs(pairs_from: str) -> list[tuple[int, int]]:
    """(task,seed) whose CLEAN rollout succeeded on the strong-ckpt gate eval."""
    out = []
    for line in Path(pairs_from).read_text().splitlines():
        r = json.loads(line)
        if r.get("success"):
            out.append((r["task_id"], r["seed"]))
    return sorted(set(out))


def _rot90(dx: float, dy: float) -> tuple[float, float]:
    return (-dy, dx)


def _set_obj_xy(sim, adr: int, xy: np.ndarray) -> np.ndarray:
    saved = np.asarray(sim.data.qpos[adr:adr + 3]).copy()
    sim.data.qpos[adr] = xy[0]
    sim.data.qpos[adr + 1] = xy[1]
    sim.forward()
    return saved


def virtual_object_imgs(env, adr: int, visual_xy: np.ndarray,
                        cams: tuple[str, ...]) -> dict:
    """Render the scene with the target object kinematically moved to visual_xy.

    Robot/other qpos untouched, so the only visual change is the object's
    appearance. Object qpos is restored before returning (no physics step).
    """
    sim = env._env.env.sim
    saved = _set_obj_xy(sim, adr, visual_xy)
    vobs = _refresh_obs(env)
    imgs = {k: np.asarray(vobs["pixels"][k]).copy() for k in cams}
    sim.data.qpos[adr:adr + 3] = saved
    sim.forward()
    _refresh_obs(env)
    return imgs


def _paste_obs(raw_obs: dict, virt_imgs: dict) -> dict:
    """Policy obs = real proprio + virtual object image(s)."""
    out = {"pixels": dict(raw_obs["pixels"]),
           "robot_state": {g: dict(sub) for g, sub in raw_obs["robot_state"].items()}}
    for k, im in virt_imgs.items():
        out["pixels"][k] = im
    return out


def _eef_xyz(raw_obs) -> np.ndarray:
    return np.asarray(raw_obs["robot_state"]["eef"]["pos"], dtype=np.float64).copy()


def run_condition(pipe: Pipeline, env, seed: int, obj_name: str, cond: str,
                  dirname: str, cams: tuple[str, ...],
                  smoke_dump: Path | None = None) -> dict:
    phys, vis = COND[cond]
    dx, dy = (DIRS[dirname][0] * DELTA, DIRS[dirname][1] * DELTA)

    set_seed(seed)
    pipe.policy.reset()
    raw_obs, _ = env.reset(seed=seed)
    task_desc = env.task_description

    # Establish C + qpos_adr, and place the object at its PHYSICAL location.
    if phys == "T":
        pinfo = perturb_object_by_name_vec(env, obj_name, (dx, dy))
        C = np.asarray(pinfo["before_xyz"], dtype=np.float64)
        phys_xyz = np.asarray(pinfo["after_xyz"], dtype=np.float64)
    else:  # physical C: zero-probe to read C + adr without moving the object
        pinfo = perturb_object_by_name_vec(env, obj_name, (0.0, 0.0))
        C = np.asarray(pinfo["before_xyz"], dtype=np.float64)
        phys_xyz = C.copy()
    adr = pinfo["qpos_adr"]
    raw_obs = _refresh_obs(env)

    T = C + np.array([dx, dy, 0.0])
    if vis == "C":
        V = C.copy()
    elif vis == "T":
        V = T.copy()
    else:  # R: canonical + 90-deg-rotated displacement
        rx, ry = _rot90(dx, dy)
        V = C + np.array([rx, ry, 0.0])
    visual_xy = V[:2]
    # no-op render (V == physical) still goes through the machinery, for a fair sham
    do_virtual = True

    max_steps = env._max_episode_steps
    n_action_steps = pipe.policy.config.n_action_steps
    eef_traj, exec_actions = [], []
    chunk = None
    chunk_pos = 0
    macro_calls = 0
    success = False
    t0 = time.time()

    for step in range(max_steps):
        eef_traj.append(_eef_xyz(raw_obs))
        if chunk is None or chunk_pos >= n_action_steps:
            if do_virtual:
                virt = virtual_object_imgs(env, adr, visual_xy, cams)
                obs_for_policy = _paste_obs(raw_obs, virt)
            else:
                obs_for_policy = raw_obs
            if smoke_dump is not None and macro_calls == 0:
                np.savez_compressed(
                    smoke_dump,
                    true_image=np.asarray(raw_obs["pixels"]["image"]),
                    policy_image=np.asarray(obs_for_policy["pixels"]["image"]),
                    true_wrist=np.asarray(raw_obs["pixels"].get("image2", np.zeros(1))),
                    policy_wrist=np.asarray(obs_for_policy["pixels"].get("image2", np.zeros(1))),
                    C=C, T=T, V=V, phys_xyz=phys_xyz,
                )
            batch = build_policy_batch(pipe, obs_for_policy, task_desc)
            with torch.inference_mode():
                chunk = pipe.policy.predict_action_chunk(batch)
            chunk_pos = 0
            macro_calls += 1
        a = chunk[:, chunk_pos, :]
        chunk_pos += 1
        action = pipe.postprocessor(a)
        action = pipe.env_postprocessor({ACTION: action})[ACTION]
        action_np = action.to("cpu").numpy()
        raw_obs, reward, terminated, truncated, info = env.step(action_np[0])
        exec_actions.append(action_np[0].astype(np.float64))
        if bool(info.get("is_success", False)):
            success = True
        if terminated:
            break

    n_steps = len(exec_actions)
    eef = np.asarray(eef_traj)
    grip = np.asarray(exec_actions)[:, GRIPPER_DIM]
    # pregrasp = eef at first gripper-close; closest = nearest approach to PHYSICAL object
    close = np.where((grip[1:] > 0.0) & (grip[:-1] <= 0.0))[0]
    t_pg = (int(close[0]) + 1) if len(close) else (0 if (len(grip) and grip[0] > 0) else None)
    t_cl = int(np.argmin(np.linalg.norm(eef[:, :2] - phys_xyz[:2], axis=1)))

    return {
        "cond": cond, "dir": dirname, "success": success, "n_steps": n_steps,
        "macro_calls": macro_calls, "wall_s": round(time.time() - t0, 1),
        "object": obj_name, "phys_loc": phys, "vis_loc": vis,
        "canonical_xyz": C.tolist(), "true_xyz": T.tolist(),
        "phys_xyz": phys_xyz.tolist(), "visual_xyz": V.tolist(),
        "d_xy": [dx, dy], "edit_cams": list(cams),
        "endpoints": {
            "pregrasp": eef[t_pg].tolist() if t_pg is not None else None,
            "closest": eef[t_cl].tolist(), "final": eef[n_steps - 1].tolist(),
            "t_pregrasp": t_pg, "t_closest": t_cl},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--pairs-from", default="preempt/e0_eval_fullexp_100k.jsonl")
    ap.add_argument("--mags", default="0.05")  # E2r primary is 50mm only
    ap.add_argument("--dirs", default=",".join(DIRS.keys()))
    ap.add_argument("--conditions", default="CC,CT,TC,TT,CR")
    ap.add_argument("--edit-cams", default="image,image2")
    ap.add_argument("--out", default="preempt/e2r_object_local_paste_50mm.jsonl")
    ap.add_argument("--log-dir", default="preempt/e2r_object_local_paste_50mm_logs")
    ap.add_argument("--smoke", action="store_true",
                    help="8 pairs x 2 dirs x {CC,CT,TC,CR} = 64 rollouts")
    ap.add_argument("--src", default=SRC, help="pair-metadata (obj_map source)")
    args = ap.parse_args()

    global DELTA
    mags = [float(x) for x in args.mags.split(",")]
    assert len(mags) == 1, "E2r primary uses a single magnitude"
    DELTA = mags[0]
    dirs = [d for d in args.dirs.split(",")]
    conds = [c for c in args.conditions.split(",")]
    cams = tuple(args.edit_cams.split(","))

    pairs = clean_success_pairs(args.pairs_from)
    obj_map, _ = load_pair_meta(args.src)
    pairs = [p for p in pairs if p in obj_map]

    if args.smoke:
        pairs = pairs[:8]
        dirs = dirs[:2]
        conds = [c for c in ("CC", "CT", "TC", "CR") if c in conds] or ["CC", "CT", "TC", "CR"]

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out)
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            try:
                r = json.loads(line)
                done.add((r["task_id"], r["seed"], r["cond"], r["dir"]))
            except Exception:
                pass

    jobs = [(t, s, c, dn) for (t, s) in pairs for c in conds for dn in dirs
            if (t, s, c, dn) not in done]
    print(f"[e2r] ckpt={args.ckpt or 'default'} pairs={len(pairs)} conds={conds} "
          f"dirs={dirs} cams={cams} jobs={len(jobs)} (done {len(done)})", flush=True)

    pipe = build_pipeline(ckpt=args.ckpt) if args.ckpt else build_pipeline()
    n = 0
    t_start = time.time()
    cur_key, env = None, None
    for (task_id, seed, cond, dn) in jobs:
        if cur_key != (task_id, seed):
            if env is not None:
                env.close()
            env = make_single_env(task_id=task_id, episode_index=SEEDS.index(seed))
            cur_key = (task_id, seed)
            # clean warm-up (position discipline + E_clean baseline for the CC no-op check)
            wu = run_rollout(pipe, env, seed, obj_map[(task_id, seed)], None, flow_k=0)
            with out_path.open("a") as f:
                f.write(json.dumps({"task_id": task_id, "seed": seed, "cond": "clean_warmup",
                                    "dir": "-", "success": wu["success"],
                                    "n_steps": wu["n_steps"],
                                    "endpoints": wu["endpoints"]}) + "\n")
        dump = (log_dir / f"smoke_t{task_id}_s{seed}_{cond}_{dn}.npz") if args.smoke else None
        r = run_condition(pipe, env, seed, obj_map[(task_id, seed)], cond, dn, cams,
                          smoke_dump=dump)
        r.update({"task_id": task_id, "seed": seed})
        with out_path.open("a") as f:
            f.write(json.dumps(r) + "\n")
        n += 1
        el = time.time() - t_start
        print(f"[{n}/{len(jobs)}] t{task_id}_s{seed} {cond}:{dn} succ={r['success']} "
              f"steps={r['n_steps']} (eta {(el / n) * (len(jobs) - n) / 3600:.2f}h)", flush=True)
    if env is not None:
        env.close()
    print("E2R_PASTE_DONE", flush=True)


if __name__ == "__main__":
    main()
