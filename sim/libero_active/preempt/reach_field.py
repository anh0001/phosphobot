"""Pilot (a) — reach-field grid: WHERE does the policy reach under controlled
episode-start object displacement?  (vla-multiscopic-v2 flagship, K1 gate.)

For each (task, seed) pair we run one CLEAN rollout plus a grid of displaced
rollouts (magnitude x planar direction applied to the task-relevant object at
episode start), with normal chunked execution, logging full kinematics so the
endpoint-error vector field and the displacement-vector-recovery (DVR)
statistics can be computed offline (reach_field_analysis.py).

Pre-registered endpoint definitions (analysis computes all three; PRIMARY for
K1 = pregrasp where defined, else closest-approach-to-true):
  - pregrasp : eef xyz at the first step whose EXECUTED gripper command crosses
               from open (<=0) to close (>0)
  - closest  : eef xyz at argmin_t ||eef_xy(t) - T_xy||  (T = displaced object)
  - final    : eef xyz at the last step

Per-rollout logs (npz under --log-dir):
  eef_xyz (S,3) | obj_xyz (S,3) | exec_action (S,7) | norm_action (S,7)
  chunk_steps (M,) | chunks (M,50,7) | flow_samples (M,K,50,7)  [RNG-isolated]
JSONL summary record per rollout (append; the run is resumable by key).

Object identity + clean-success conditioning conventions follow
mislocalization_probe.py (source: e1_recovery_d005.json).

Usage (full pilot grid, ~850 rollouts):
  env -u PYTHONPATH MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
    python preempt/reach_field.py --out preempt/reach_field_results.jsonl
Smoke:
  ... python preempt/reach_field.py --smoke
"""
from __future__ import annotations

import argparse
import json
import os
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
from lerobot.utils.constants import ACTION  # noqa: E402
from lerobot.utils.random_utils import set_seed  # noqa: E402

SRC = "preempt/e1_recovery_d005.json"
SEEDS = [1000, 1001, 1002, 1003, 1004]
TASKS = list(range(10))
MAGS = [0.025, 0.05, 0.075, 0.10]
DIRS = {"px": (1.0, 0.0), "nx": (-1.0, 0.0), "py": (0.0, 1.0), "ny": (0.0, -1.0)}
FLOW_K = 5  # extra RNG-isolated chunk re-samples per macro call (dispersion log)
# P2T degradation-null hook: additive Gaussian noise (std, normalized action units)
# on the executed xyz action deltas. 0.0 = no-op (backward compatible). Set via
# --action-noise / env P2T_ACTION_NOISE to build a competence-matched null that
# shares the base policy's grounding but is degraded only by execution noise.
ACTION_NOISE_STD = float(os.environ.get("P2T_ACTION_NOISE", "0.0"))
_NOISE_RNG = np.random.default_rng(int(os.environ.get("P2T_NOISE_SEED", "12345")))


def load_pair_meta(src_path: str) -> tuple[dict, dict]:
    """(task,seed) -> task-relevant object name; (task,seed) -> prior clean success."""
    src = json.load(open(src_path))
    pairs: dict = {}
    for r in src["records"]:
        pairs.setdefault((r["task_id"], r["seed"]), {})[r["variant"]] = r
    obj_map, clean_map = {}, {}
    for k, v in pairs.items():
        clean_map[k] = bool(v.get("clean", {}).get("success", False))
        for vary in ("open_loop", "replan_only", "preempt_hold", "recovery_probe"):
            o = v.get(vary, {}).get("perturb", {}).get("object")
            if o:
                obj_map[k] = o
                break
    return obj_map, clean_map


def _eef_xyz(raw_obs) -> np.ndarray:
    return np.asarray(raw_obs["robot_state"]["eef"]["pos"], dtype=np.float64).copy()


def _sample_flow_chunks(pipe: Pipeline, batch: dict, k: int) -> np.ndarray:
    """K extra chunk samples with the policy RNG saved/restored (no contamination)."""
    rng_cpu = torch.get_rng_state()
    rng_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    out = []
    with torch.inference_mode():
        for _ in range(k):
            out.append(pipe.policy.predict_action_chunk(batch)[0].float().cpu().numpy())
    torch.set_rng_state(rng_cpu)
    if rng_cuda is not None:
        torch.cuda.set_rng_state_all(rng_cuda)
    return np.stack(out)  # (K, 50, 7)


def run_rollout(pipe: Pipeline, env, seed: int, obj_name: str,
                dxy: tuple[float, float] | None, flow_k: int) -> dict:
    """One chunked rollout (clean if dxy is None, else displaced at episode start)."""
    set_seed(seed)
    pipe.policy.reset()
    raw_obs, _ = env.reset(seed=seed)
    task_desc = env.task_description

    pinfo: dict = {"perturbed": False}
    qpos_adr = None
    if dxy is not None:
        pinfo = perturb_object_by_name_vec(env, obj_name, dxy)
        if not pinfo.get("perturbed"):
            return {"error": pinfo.get("reason", "perturb_failed")}
        qpos_adr = pinfo["qpos_adr"]
        raw_obs = _refresh_obs(env)
    else:
        # locate the object for trajectory logging in the clean arm too
        probe = perturb_object_by_name_vec(env, obj_name, (0.0, 0.0))
        if probe.get("perturbed"):
            qpos_adr = probe["qpos_adr"]
            pinfo = {"perturbed": False, "object": obj_name,
                     "before_xyz": probe["before_xyz"], "after_xyz": probe["before_xyz"]}
            raw_obs = _refresh_obs(env)

    max_steps = env._max_episode_steps
    n_action_steps = pipe.policy.config.n_action_steps

    eef_traj, obj_traj, exec_actions, norm_actions = [], [], [], []
    chunk_steps, chunks, flow_samples = [], [], []
    chunk = None
    chunk_pos = 0
    macro_calls = 0
    success = False
    t0 = time.time()

    for step in range(max_steps):
        eef_traj.append(_eef_xyz(raw_obs))
        obj_traj.append(object_xyz(env, qpos_adr) if qpos_adr is not None
                        else np.full(3, np.nan))
        if chunk is None or chunk_pos >= n_action_steps:
            batch = build_policy_batch(pipe, raw_obs, task_desc)
            if flow_k > 0:
                flow_samples.append(_sample_flow_chunks(pipe, batch, flow_k))
            with torch.inference_mode():
                chunk = pipe.policy.predict_action_chunk(batch)
            chunk_steps.append(step)
            chunks.append(chunk[0].float().cpu().numpy())
            chunk_pos = 0
            macro_calls += 1

        a = chunk[:, chunk_pos, :]
        chunk_pos += 1
        norm_actions.append(a[0].float().cpu().numpy())
        # inline execute_action so the EXECUTED (post-processed) action is logged
        action = pipe.postprocessor(a)
        action = pipe.env_postprocessor({ACTION: action})[ACTION]
        action_np = action.to("cpu").numpy()
        if ACTION_NOISE_STD > 0.0:  # degradation-null: noise on xyz deltas only
            action_np[0, :3] += _NOISE_RNG.normal(0.0, ACTION_NOISE_STD, size=3)
        raw_obs, reward, terminated, truncated, info = env.step(action_np[0])
        exec_actions.append(action_np[0].astype(np.float64))
        if bool(info.get("is_success", False)):
            success = True
        if terminated:
            break

    n_steps = len(exec_actions)
    eef = np.asarray(eef_traj)
    obj = np.asarray(obj_traj)
    ex = np.asarray(exec_actions)

    # ---- endpoints (3 pre-registered definitions) ----
    T = np.asarray(pinfo["after_xyz"], dtype=np.float64)   # displaced/true (== C if clean)
    C = np.asarray(pinfo["before_xyz"], dtype=np.float64)  # canonical
    grip = ex[:, GRIPPER_DIM]
    close_idx = np.where((grip[1:] > 0.0) & (grip[:-1] <= 0.0))[0]
    t_pregrasp = int(close_idx[0] + 1) if len(close_idx) else (0 if grip[0] > 0 else None)
    t_closest = int(np.argmin(np.linalg.norm(eef[:, :2] - T[:2], axis=1)))
    endpoints = {
        "pregrasp": eef[t_pregrasp].tolist() if t_pregrasp is not None else None,
        "closest": eef[t_closest].tolist(),
        "final": eef[n_steps - 1].tolist(),
    }

    return {
        "success": success, "n_steps": n_steps, "macro_calls": macro_calls,
        "wall_s": round(time.time() - t0, 1), "task": task_desc,
        "object": obj_name, "perturb": pinfo,
        "canonical_xyz": C.tolist(), "true_xyz": T.tolist(),
        "d_real_xy": (T[:2] - C[:2]).tolist(),
        "t_pregrasp": t_pregrasp, "t_closest": t_closest,
        "endpoints": endpoints,
        "obj_drift_cm": float(np.linalg.norm(obj[min(t_closest, n_steps - 1), :2] - T[:2]) * 100)
        if qpos_adr is not None else None,
        "_arrays": {
            "eef_xyz": eef, "obj_xyz": obj, "exec_action": ex,
            "norm_action": np.asarray(norm_actions),
            "chunk_steps": np.asarray(chunk_steps, dtype=np.int32),
            "chunks": np.asarray(chunks, dtype=np.float32),
            "flow_samples": (np.asarray(flow_samples, dtype=np.float32)
                             if flow_samples else np.zeros((0,), dtype=np.float32)),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=",".join(map(str, TASKS)))
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--mags", default=",".join(map(str, MAGS)))
    ap.add_argument("--dirs", default=",".join(DIRS.keys()))
    ap.add_argument("--out", default="preempt/reach_field_results.jsonl")
    ap.add_argument("--log-dir", default="preempt/reach_field_logs")
    ap.add_argument("--flow-k", type=int, default=FLOW_K)
    ap.add_argument("--no-flow-samples", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="task0/seed1000, mags {0.05,0.10}, dirs {px,py} + clean")
    ap.add_argument("--ckpt", default=None,
                    help="checkpoint path or hub id (default: harness CKPT)")
    ap.add_argument("--src", default=SRC,
                    help="pair-metadata file (obj_map source); default libero_spatial")
    args = ap.parse_args()

    tasks = [int(x) for x in args.tasks.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    mags = [float(x) for x in args.mags.split(",")]
    dirs = {k: DIRS[k] for k in args.dirs.split(",")}
    if args.smoke:
        tasks, seeds, mags = [0], [1000], [0.05, 0.10]
        dirs = {k: DIRS[k] for k in ("px", "py")}
    flow_k = 0 if args.no_flow_samples else args.flow_k

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out)

    done: set[str] = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            try:
                done.add(json.loads(line)["key"])
            except Exception:
                pass

    obj_map, prior_clean = load_pair_meta(args.src)
    conditions = [("clean", None)] + [
        (f"d{int(m * 1000)}mm_{dn}", (m * dv[0], m * dv[1]))
        for m in mags for dn, dv in dirs.items()
    ]
    total = sum(1 for t in tasks for s in seeds if (t, s) in obj_map) * len(conditions)
    print(f"[reach_field] grid: {len(tasks)}T x {len(seeds)}S x {len(conditions)}cond "
          f"= {total} rollouts ({len(done)} already done) | flow_k={flow_k}", flush=True)

    pipe = build_pipeline(ckpt=args.ckpt) if args.ckpt else build_pipeline()
    n_run = 0
    t_start = time.time()
    for task_id in tasks:
        for si, seed in enumerate(seeds):
            key_ts = (task_id, seed)
            if key_ts not in obj_map:
                continue
            todo = [(cn, d) for cn, d in conditions
                    if f"t{task_id}_s{seed}_{cn}" not in done]
            if not todo:
                continue
            env = make_single_env(task_id=task_id, episode_index=si)
            for cond_name, dxy in todo:
                key = f"t{task_id}_s{seed}_{cond_name}"
                r = run_rollout(pipe, env, seed, obj_map[key_ts], dxy, flow_k)
                if "error" in r:
                    rec = {"key": key, "task_id": task_id, "seed": seed,
                           "cond": cond_name, "error": r["error"]}
                else:
                    arrays = r.pop("_arrays")
                    np.savez_compressed(log_dir / f"{key}.npz", **arrays)
                    rec = {"key": key, "task_id": task_id, "seed": seed,
                           "cond": cond_name, "prior_clean_success": prior_clean[key_ts],
                           **r}
                with out_path.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
                n_run += 1
                el = time.time() - t_start
                print(f"[{n_run}/{total - len(done)}] {key} succ={rec.get('success')} "
                      f"steps={rec.get('n_steps')} wall={rec.get('wall_s')}s "
                      f"({el / 60:.0f}m elapsed, eta {(el / n_run) * (total - len(done) - n_run) / 3600:.1f}h)",
                      flush=True)
            env.close()
    print("REACH_FIELD_DONE", flush=True)


if __name__ == "__main__":
    main()
