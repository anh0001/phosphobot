"""Pilot (b) — oracle virtual-frame re-anchoring (causal-sufficiency probe, K2 gate).

Frame lie (identity counter-warp by the relative-action equivariance fact,
Wang et al. 2505.13431 Props 1-2): present the policy with frame V = world - w:
  - agentview image translated by the pixel projection of -w at the object plane
  - perceived eef position = true eef position - w
  - wrist camera (image2) untouched (exactly invariant under the frame shift)
  - actions executed verbatim (relative deltas are translation-invariant)

Model prediction (canonical-anchoring, tau~=0 from pilot a): the pre-grasp
endpoint of EVERY arm lands at C + w (canonical location plus applied warp):
  nowarp        w=0          -> endpoint C            (misses T by -d)   [from grid]
  warp_full     w=d          -> endpoint C+d = T      (RESTORATION)
  warp_image    w=d img-only -> partial (channel attribution)
  warp_proprio  w=d pro-only -> partial (channel attribution)
  warp_random   w=R90(d)     -> endpoint C+R90(d)     (NO restoration; specificity)
  warp_clean    no displ, w  -> endpoint C+w          (breaks a healthy episode!)
  sham_zero     displ, w=0   -> identical to grid nowarp record (machinery no-op)
  midroll       displ@t=25   -> post-replan endpoint still C (mid-rollout tau~=0)

Arms run on the clean-success pairs only (conditioning set; ~14 pairs).
nowarp@5cm baselines reuse the pilot-(a) grid records (no re-run).

Usage:
  env -u PYTHONPATH MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
    python preempt/oracle_warp.py [--smoke] [--out preempt/oracle_warp_results.jsonl]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Import BEFORE any set_seed: lazy-importing inside the rollout advances RNG
# state between seeding and the first inference (caught by the sham_zero no-op
# check in the first smoke).
from robosuite.utils.camera_utils import (  # noqa: E402
    get_camera_transform_matrix,
    project_points_from_world_to_camera,
)

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
from preempt.reach_field import _sample_flow_chunks, load_pair_meta, run_rollout  # noqa: E402
from lerobot.utils.constants import ACTION  # noqa: E402
from lerobot.utils.random_utils import set_seed  # noqa: E402

GRID_RESULTS = "preempt/reach_field_results_v2.jsonl"  # post-obs-fix grid only
DELTA = 0.05
DIRS = {"px": (1.0, 0.0), "nx": (-1.0, 0.0), "py": (0.0, 1.0), "ny": (0.0, -1.0)}
ROT90 = {"px": "py", "py": "nx", "nx": "ny", "ny": "px"}
MIDROLL_STEP = 25  # before typical pre-grasp (~35); replan sees it at step 50
FLOW_K = 5


def clean_success_pairs(grid_path: str) -> list[tuple[int, int]]:
    out = []
    for line in Path(grid_path).read_text().splitlines():
        r = json.loads(line)
        if r.get("cond") == "clean" and r.get("success"):
            out.append((r["task_id"], r["seed"]))
    return sorted(out)


def pixel_shift_for_warp(env, obj_xyz_now: np.ndarray, w_xyz: np.ndarray) -> tuple[int, int]:
    """(drow, dcol) IN THE RAW IMAGE FRAME so content at p(O) moves to p(O - w).

    The raw LiberoEnv render is 180-degree rotated relative to the projection
    convention (LiberoProcessorStep flips it back downstream), so the raw-frame
    translation is the NEGATION of the projection-frame pixel shift (verified
    visually in the smoke: clean/displaced/warped PNG triplet).
    """
    sim = env._env.env.sim
    M = get_camera_transform_matrix(sim, "agentview", 360, 360)
    pts = np.stack([obj_xyz_now - w_xyz, obj_xyz_now])
    pix = project_points_from_world_to_camera(pts, M, 360, 360)
    d = pix[1] - pix[0]  # negated for the 180-rotated raw frame
    return int(round(d[0])), int(round(d[1]))


def empirical_pixel_shift(env, qpos_adr: int, w_xyz: np.ndarray,
                          base: tuple[int, int]) -> tuple[tuple[int, int], float]:
    """Calibrate the image shift by actually re-rendering with the object moved.

    The camera-matrix projection gets the DIRECTION right but its scale is ~5x
    off for the LiberoEnv render pipeline (smoke: 5cm -> ~4px rendered vs 20px
    predicted). So: render with object at its current pose O and at O - w, then
    fit the scalar s minimizing the masked diff between translate(img_O, s*base)
    and img_{O-w}. Restores qpos exactly (kinematic-only, no physics step).
    Returns ((drow, dcol), s).
    """
    sim = env._env.env.sim
    img_now = np.asarray(_refresh_obs(env)["pixels"]["image"], dtype=np.int16)
    saved = np.asarray(sim.data.qpos[qpos_adr:qpos_adr + 3]).copy()
    sim.data.qpos[qpos_adr] -= w_xyz[0]
    sim.data.qpos[qpos_adr + 1] -= w_xyz[1]
    sim.forward()
    img_back = np.asarray(_refresh_obs(env)["pixels"]["image"], dtype=np.int16)
    sim.data.qpos[qpos_adr:qpos_adr + 3] = saved
    sim.forward()
    _refresh_obs(env)

    mask = np.abs(img_now - img_back).sum(2) > 40
    if mask.sum() < 50:  # object not visibly moved; fall back to matrix shift
        return base, 1.0
    ys, xs = np.where(mask)
    r0, r1 = max(ys.min() - 5, 0), min(ys.max() + 6, 360)
    c0, c1 = max(xs.min() - 5, 0), min(xs.max() + 6, 360)
    best = (1e18, base, 1.0)
    for s in np.arange(0.05, 2.01, 0.05):
        dr, dc = int(round(s * base[0])), int(round(s * base[1]))
        shifted = translate_image(img_now, dr, dc)
        err = float(np.abs(shifted[r0:r1, c0:c1] - img_back[r0:r1, c0:c1]).mean())
        if err < best[0]:
            best = (err, (dr, dc), float(s))
    return best[1], best[2]


def translate_image(img: np.ndarray, drow: int, dcol: int) -> np.ndarray:
    """Shift content by (drow, dcol) with edge replication (clamped indexing)."""
    h, w = img.shape[:2]
    src_r = np.clip(np.arange(h) - drow, 0, h - 1)
    src_c = np.clip(np.arange(w) - dcol, 0, w - 1)
    return img[src_r][:, src_c]


def warp_obs(raw_obs: dict, w_xyz: np.ndarray, pix: tuple[int, int],
             do_image: bool, do_proprio: bool) -> dict:
    """Warped copy for INFERENCE ONLY (true obs is kept for logging/stepping)."""
    out = {"pixels": dict(raw_obs["pixels"]),
           "robot_state": {g: dict(sub) for g, sub in raw_obs["robot_state"].items()}}
    if do_image and (pix[0] or pix[1]):
        out["pixels"]["image"] = translate_image(
            np.asarray(raw_obs["pixels"]["image"]), pix[0], pix[1])
    if do_proprio and np.linalg.norm(w_xyz) > 0:
        out["robot_state"]["eef"]["pos"] = (
            np.asarray(raw_obs["robot_state"]["eef"]["pos"], dtype=np.float64) - w_xyz)
    return out


def _eef_xyz(raw_obs) -> np.ndarray:
    return np.asarray(raw_obs["robot_state"]["eef"]["pos"], dtype=np.float64).copy()


def run_arm(pipe: Pipeline, env, seed: int, obj_name: str, arm: str, dirname: str,
            flow_k: int, smoke_dump: Path | None = None) -> dict:
    dvec = np.array([*DIRS[dirname], 0.0]) * DELTA
    displace = arm not in ("warp_clean",)
    perturb_step = MIDROLL_STEP if arm == "midroll" else 0
    if arm in ("warp_full", "warp_image", "warp_proprio"):
        w = dvec.copy()
    elif arm == "warp_random":
        w = np.array([*DIRS[ROT90[dirname]], 0.0]) * DELTA
    elif arm == "warp_clean":
        w = dvec.copy()
    else:  # sham_zero, midroll
        w = np.zeros(3)
    do_image = arm not in ("warp_proprio",)
    do_proprio = arm not in ("warp_image",)

    set_seed(seed)
    pipe.policy.reset()
    raw_obs, _ = env.reset(seed=seed)
    task_desc = env.task_description

    # Mirror reach_field's exact call sequence (validated by determinism):
    # displaced arms = ONE perturb call + refresh; clean-scene arms = zero-probe + refresh.
    pinfo: dict = {"perturbed": False}
    pix = (0, 0)
    pix_matrix = (0, 0)
    calib_scale = 0.0
    warp_active = False
    if displace and perturb_step == 0:
        pinfo = perturb_object_by_name_vec(env, obj_name, (dvec[0], dvec[1]))
        qpos_adr = pinfo["qpos_adr"]
        C = np.asarray(pinfo["before_xyz"], dtype=np.float64)
        T = np.asarray(pinfo["after_xyz"], dtype=np.float64)
        raw_obs = _refresh_obs(env)
        if np.linalg.norm(w) > 0:
            pix_matrix = pixel_shift_for_warp(env, object_xyz(env, qpos_adr), w)
            pix, calib_scale = empirical_pixel_shift(env, qpos_adr, w, pix_matrix)
        warp_active = True
    else:
        probe = perturb_object_by_name_vec(env, obj_name, (0.0, 0.0))
        qpos_adr = probe["qpos_adr"]
        C = np.asarray(probe["before_xyz"], dtype=np.float64)
        T = C.copy()
        raw_obs = _refresh_obs(env)
        if arm == "warp_clean":
            pix_matrix = pixel_shift_for_warp(env, object_xyz(env, qpos_adr), w)
            pix, calib_scale = empirical_pixel_shift(env, qpos_adr, w, pix_matrix)
            warp_active = True

    max_steps = env._max_episode_steps
    n_action_steps = pipe.policy.config.n_action_steps
    eef_traj, exec_actions = [], []
    chunk_steps, chunks, flow_samples = [], [], []
    chunk = None
    chunk_pos = 0
    macro_calls = 0
    success = False
    t0 = time.time()

    for step in range(max_steps):
        if displace and perturb_step > 0 and step == perturb_step:
            pinfo = perturb_object_by_name_vec(env, obj_name, (dvec[0], dvec[1]))
            T = np.asarray(pinfo["after_xyz"], dtype=np.float64)
            raw_obs = _refresh_obs(env)
            # midroll: NO warp; stale chunk continues until the step-50 replan
        eef_traj.append(_eef_xyz(raw_obs))
        if chunk is None or chunk_pos >= n_action_steps:
            obs_for_policy = (warp_obs(raw_obs, w, pix, do_image, do_proprio)
                              if warp_active else raw_obs)
            if smoke_dump is not None and macro_calls == 0:
                np.savez_compressed(
                    smoke_dump,
                    true_image=np.asarray(raw_obs["pixels"]["image"]),
                    policy_image=np.asarray(obs_for_policy["pixels"]["image"]),
                    pix=np.array(pix),
                    true_eef=_eef_xyz(raw_obs),
                    policy_eef=np.asarray(obs_for_policy["robot_state"]["eef"]["pos"]),
                )
            batch = build_policy_batch(pipe, obs_for_policy, task_desc)
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
    ex = np.asarray(exec_actions)
    grip = ex[:, GRIPPER_DIM]

    def endpoints_from(t0_idx: int) -> dict:
        g = grip[t0_idx:]
        close = np.where((g[1:] > 0.0) & (g[:-1] <= 0.0))[0]
        t_pg = (t0_idx + int(close[0]) + 1) if len(close) else (
            t0_idx if (len(g) and g[0] > 0) else None)
        t_cl = t0_idx + int(np.argmin(
            np.linalg.norm(eef[t0_idx:, :2] - T[:2], axis=1)))
        return {"pregrasp": eef[t_pg].tolist() if t_pg is not None else None,
                "closest": eef[t_cl].tolist(), "final": eef[n_steps - 1].tolist(),
                "t_pregrasp": t_pg, "t_closest": t_cl}

    rec = {
        "arm": arm, "dir": dirname, "success": success, "n_steps": n_steps,
        "macro_calls": macro_calls, "wall_s": round(time.time() - t0, 1),
        "object": obj_name, "canonical_xyz": C.tolist(), "true_xyz": T.tolist(),
        "d_real_xy": (T[:2] - C[:2]).tolist(), "w_xyz": w.tolist(),
        "pix_shift": list(pix), "pix_shift_matrix": list(pix_matrix),
        "calib_scale": calib_scale, "perturb_step": perturb_step,
        "endpoints": endpoints_from(0),
    }
    if arm == "midroll":
        rec["endpoints_post_replan"] = endpoints_from(min(50, n_steps - 1))
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="preempt/oracle_warp_results.jsonl")
    ap.add_argument("--flow-k", type=int, default=FLOW_K)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    pairs = clean_success_pairs(GRID_RESULTS)
    obj_map, _ = load_pair_meta("preempt/e1_recovery_d005.json")
    seeds = [1000, 1001, 1002, 1003, 1004]

    jobs: list[tuple[int, int, str, str]] = []
    if args.smoke:
        t, s = (3, 1000) if (3, 1000) in pairs else pairs[0]
        jobs = [(t, s, "sham_zero", "px"), (t, s, "warp_full", "px"),
                (t, s, "warp_clean", "px"), (t, s, "midroll", "px")]
    else:
        for (t, s) in pairs:
            jobs.append((t, s, "sham_zero", "px"))
            for arm in ("warp_full", "warp_image", "warp_proprio", "warp_random",
                        "warp_clean", "midroll"):
                for dn in DIRS:
                    jobs.append((t, s, arm, dn))


    # Env-warm-up effect (found in smoke): the FIRST rollout on a freshly
    # constructed LiberoEnv diverges from all later ones; the pilot-(a) grid ran
    # 'clean' first on every env, so all its displaced records sit at position
    # >= 2. We reproduce that structure: one clean warm-up rollout per env,
    # doubling as a per-pair determinism check against the grid clean record.
    grid_recs = {}
    for line in Path(GRID_RESULTS).read_text().splitlines():
        r = json.loads(line)
        grid_recs[(r["task_id"], r["seed"], r["cond"])] = r

    out_path = Path(args.out)
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            try:
                r = json.loads(line)
                done.add((r["task_id"], r["seed"], r["arm"], r["dir"]))
            except Exception:
                pass
    jobs = [j for j in jobs if j not in done]
    print(f"[oracle_warp] pairs={len(pairs)} jobs={len(jobs)} (done {len(done)})", flush=True)

    pipe = build_pipeline()
    smoke_dir = Path("preempt/oracle_warp_smoke_logs")
    if args.smoke:
        smoke_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    t_start = time.time()
    cur_env_key, env = None, None
    for (task_id, seed, arm, dn) in jobs:
        if cur_env_key != (task_id, seed):
            if env is not None:
                env.close()
            env = make_single_env(task_id=task_id, episode_index=seeds.index(seed))
            cur_env_key = (task_id, seed)
            wu = run_rollout(pipe, env, seed, obj_map[(task_id, seed)], None,
                             flow_k=0)
            ref = grid_recs.get((task_id, seed, "clean"))
            match = (ref is not None and wu["success"] == ref["success"]
                     and wu["n_steps"] == ref["n_steps"])
            print(f"[warmup] t{task_id}_s{seed} clean succ={wu['success']} "
                  f"steps={wu['n_steps']} grid_match={match}", flush=True)
            with out_path.open("a") as f:
                f.write(json.dumps({"task_id": task_id, "seed": seed,
                                    "arm": "warmup_clean", "dir": "-",
                                    "success": wu["success"], "n_steps": wu["n_steps"],
                                    "endpoints": wu["endpoints"],
                                    "grid_match": match}) + "\n")
        dump = (smoke_dir / f"smoke_{arm}_{dn}.npz") if args.smoke else None
        r = run_arm(pipe, env, seed, obj_map[(task_id, seed)], arm, dn,
                    args.flow_k, smoke_dump=dump)
        r.update({"task_id": task_id, "seed": seed})
        with out_path.open("a") as f:
            f.write(json.dumps(r) + "\n")
        n += 1
        el = time.time() - t_start
        print(f"[{n}/{len(jobs)}] t{task_id}_s{seed} {arm}:{dn} succ={r['success']} "
              f"steps={r['n_steps']} pix={r['pix_shift']} "
              f"(eta {(el / n) * (len(jobs) - n) / 3600:.1f}h)", flush=True)
    if env is not None:
        env.close()
    print("ORACLE_WARP_DONE", flush=True)


if __name__ == "__main__":
    main()
