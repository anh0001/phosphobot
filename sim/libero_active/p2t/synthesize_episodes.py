"""P2T step 3 — counterfactual episode synthesis (DemoGen-in-sim, success-filtered).

For each allocated (task, dir, mag) cell: take source demos from the ORIGINAL
LIBERO HDF5 (full sim states -> exact init, no init-state mapping needed),
displace the target object by d at episode start, replay the demo actions with
geometric retargeting, and record real renders + retargeted labels as a new
episode. Only successful episodes are kept.

Retargeting (relative OSC deltas, output_max = 0.05 m/step, from the demo
env_args): the xy displacement is spread over the pre-grasp segment with
per-step carry (clipping-safe); the transport segment gets -d (restores the
place target); post-release actions are unchanged.

Usage:
  env -u PYTHONPATH MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
    .venv/bin/python p2t/synthesize_episodes.py --alloc p2t/acquisition_maps.json \
      --condition A_gain --out-dir p2t/staging [--clean-check N] [--smoke]
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from libero.libero import benchmark  # noqa: E402

from p2t.p2t_lib import env_state8  # noqa: E402
from preempt.harness_lib import SUITE_NAME, _refresh_obs  # noqa: E402
from preempt.perturb import object_xyz, perturb_object_by_name_vec  # noqa: E402

DEMO_DIR = Path(__file__).parent / "libero_demos/libero_spatial"
STEP_MAX_M = 0.05           # OSC_POSE output_max (demo env_args)
DIRS = {"px": (1.0, 0.0), "nx": (-1.0, 0.0), "py": (0.0, 1.0), "ny": (0.0, -1.0)}
OBS_HW = 256                # native dataset resolution (no resize blur)
JITTER_ANG_DEG = 15.0
JITTER_MAG_FRac = 0.2


def make_gen_env(task_id: int):
    """256x256 generator env (no policy; init overwritten from demo states)."""
    from lerobot.envs.libero import LiberoEnv
    suite = benchmark.get_benchmark_dict()[SUITE_NAME]()
    return LiberoEnv(
        task_suite=suite, task_id=task_id, task_suite_name=SUITE_NAME,
        obs_type="pixels_agent_pos", observation_width=OBS_HW,
        observation_height=OBS_HW, init_states=True, episode_index=0,
        control_mode="relative")


def demo_file(task_id: int) -> Path:
    suite = benchmark.get_benchmark_dict()[SUITE_NAME]()
    name = suite.get_task(task_id).name
    p = DEMO_DIR / f"{name}_demo.hdf5"
    assert p.exists(), p
    return p


def task_object(task_id: int) -> str:
    import json as _json
    src = _json.load(open("preempt/e1_recovery_d005.json"))
    for r in src["records"]:
        if r["task_id"] == task_id:
            o = r.get("perturb", {}).get("object")
            if o:
                return o
    raise KeyError(task_id)


def dataset_view(raw_obs: dict, cam: str) -> np.ndarray:
    """Env render -> dataset convention (180-deg un-rotation), native size."""
    return np.asarray(raw_obs["pixels"][cam])[::-1, ::-1].copy()


def spread_with_carry(actions: np.ndarray, seg: slice, d_xy: np.ndarray) -> np.ndarray:
    """Add d_xy (meters) across actions[seg,(0,1)] in normalized units with
    per-step carry so clipping never loses displacement."""
    out = actions.copy()
    idx = range(*seg.indices(len(actions)))
    n = len(idx)
    if n == 0:
        return out
    per_step = d_xy / STEP_MAX_M / n  # normalized units per step
    carry = np.zeros(2)
    for i in idx:
        want = out[i, :2] + per_step + carry
        clipped = np.clip(want, -1.0, 1.0)
        carry = want - clipped
        out[i, :2] = clipped
    return out


LEAD = 6  # finish injecting d this many steps BEFORE the segment end: the OSC
# controller tracks with a first-order lag, so displacement commanded in the
# last few pre-grasp steps is not yet realized when the gripper closes. The
# naive spread (inject through t_grasp) collapsed yield at 75-100 mm
# (2026-07-07: 90/128 exhausted at 100 mm) — the pre-registered yield<30%
# trigger; the lead gives the controller room to converge before the grasp.


def retarget(actions: np.ndarray, d_xy: np.ndarray) -> tuple[np.ndarray, int | None, int | None]:
    g = actions[:, 6]
    close = np.where((g[1:] > 0.0) & (g[:-1] <= 0.0))[0]
    t_grasp = int(close[0]) + 1 if len(close) else None
    if t_grasp is None:
        return actions.copy(), None, None
    opens = np.where((g[t_grasp:-1] > 0.0) & (g[t_grasp + 1:] <= 0.0))[0]
    t_release = (t_grasp + 1 + int(opens[0])) if len(opens) else len(actions)
    reach_end = max(t_grasp - LEAD, max(t_grasp // 2, 1))
    trans_end = max(t_release - LEAD, t_grasp + max((t_release - t_grasp) // 2, 1))
    out = spread_with_carry(actions, slice(0, reach_end), d_xy)
    out = spread_with_carry(out, slice(t_grasp, trans_end), -d_xy)
    return out, t_grasp, t_release


RAMP_P = 0.5  # shift ramp exponent for the reach segment: p<1 front-loads the
# lateral shift so the arm arrives ABOVE the displaced object before the final
# descent (late lateral motion during descent knocks the object).


def shift_at(t: int, t_grasp: int, t_release: int, n: int, d_xy: np.ndarray) -> np.ndarray:
    """Waypoint shift schedule: ramp 0->d over the reach, d->0 over transport."""
    if t <= t_grasp:
        return d_xy * (t / max(t_grasp, 1)) ** RAMP_P
    if t < t_release:
        return d_xy * (1.0 - (t - t_grasp) / max(t_release - t_grasp, 1))
    return np.zeros(2)


def synth_episode(env, h5_demo, obj_name: str, d_xy: np.ndarray) -> dict:
    """Init from demo state[0], displace object by d_xy, and SERVO the demo's
    recorded eef trajectory (obs/ee_pos), shifted by the ramp schedule.

    The naive open-loop delta-spread accumulated OSC tracking lag with no
    correction and collapsed yield at 75-100 mm (0/8 even with LEAD). Here the
    xyz action at each step is feedforward+feedback toward the shifted demo
    waypoint (error cannot accumulate); rotation + gripper stay verbatim.
    Labels = the executed servo commands, so (frame, action) pairs stay
    physically consistent.
    """
    actions = np.asarray(h5_demo["actions"], dtype=np.float64)
    ee_ref = np.asarray(h5_demo["obs"]["ee_pos"], dtype=np.float64)  # (T,3)
    state0 = np.asarray(h5_demo["states"][0], dtype=np.float64)
    env.reset(seed=0)
    env._env.set_init_state(state0)
    raw_obs = _refresh_obs(env)

    pinfo = perturb_object_by_name_vec(env, obj_name, (d_xy[0], d_xy[1]))
    if not pinfo.get("perturbed"):
        return {"ok": False, "reason": pinfo.get("reason", "perturb_failed")}
    qpos_adr = pinfo["qpos_adr"]
    raw_obs = _refresh_obs(env)

    _, t_grasp, t_release = retarget(actions, np.zeros(2))
    if t_grasp is None:
        return {"ok": False, "reason": "no_grasp_in_demo"}
    t_release = t_release if t_release is not None else len(actions)

    frames, frames2, states8, exec_actions = [], [], [], []
    success = False
    pregrasp_offset_cm = None
    T = len(actions)
    for t in range(T):
        frames.append(dataset_view(raw_obs, "image"))
        frames2.append(dataset_view(raw_obs, "image2"))
        states8.append(env_state8(raw_obs))
        eef = np.asarray(raw_obs["robot_state"]["eef"]["pos"], dtype=np.float64)
        if t == t_grasp:
            obj = object_xyz(env, qpos_adr)
            pregrasp_offset_cm = float(np.linalg.norm(eef[:2] - obj[:2]) * 100)
        a = actions[t].copy()
        t_next = min(t + 1, T - 1)
        wp = ee_ref[t_next].copy()
        wp[:2] += shift_at(t_next, t_grasp, t_release, T, d_xy)
        a[:3] = np.clip((wp - eef) / STEP_MAX_M, -1.0, 1.0)
        raw_obs, reward, terminated, truncated, info = env.step(a)
        exec_actions.append(a.copy())
        if bool(info.get("is_success", False)):
            success = True
        if terminated:
            break
    new_actions = np.asarray(exec_actions)
    n = len(frames)
    return {
        "ok": True, "success": success, "n_steps": n,
        "t_grasp": t_grasp, "t_release": t_release,
        "pregrasp_offset_cm": pregrasp_offset_cm,
        "canonical_xyz": [float(x) for x in pinfo["before_xyz"]],
        "displaced_xyz": [float(x) for x in pinfo["after_xyz"]],
        "frames": np.asarray(frames[:n], dtype=np.uint8),
        "frames2": np.asarray(frames2[:n], dtype=np.uint8),
        "states8": np.asarray(states8[:n], dtype=np.float32),
        "actions": new_actions[:n].astype(np.float32),
    }


def jittered_d(dirname: str, mag_mm: int, rng: np.random.Generator) -> np.ndarray:
    base = np.asarray(DIRS[dirname], dtype=np.float64)
    ang = np.deg2rad(rng.uniform(-JITTER_ANG_DEG, JITTER_ANG_DEG))
    rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    mag = (mag_mm / 1000.0) * (1.0 + rng.uniform(-JITTER_MAG_FRac, JITTER_MAG_FRac))
    return rot @ base * mag


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alloc", required=True, help="acquisition_maps.json")
    ap.add_argument("--condition", required=True,
                    choices=["A_gain", "C_uniform", "D_failure", "E_sham"])
    ap.add_argument("--out-dir", default="p2t/staging")
    ap.add_argument("--max-attempts", type=int, default=3,
                    help="source demos tried per requested episode")
    ap.add_argument("--clean-check", type=int, default=0,
                    help="ALSO run N d=0 replays per task (fidelity gate)")
    ap.add_argument("--smoke", action="store_true", help="first 4 cells only")
    args = ap.parse_args()

    alloc = json.load(open(args.alloc))["alloc"][args.condition]
    cells = sorted(alloc.items())
    if args.smoke:
        cells = cells[:4]
    out_dir = Path(args.out_dir) / args.condition
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "meta.jsonl"
    done = set()
    if meta_path.exists():
        for line in meta_path.read_text().splitlines():
            r = json.loads(line)
            # exhausted keys are retried on rerun (e.g. after a retarget fix);
            # only successes and clean-check records are final.
            if r.get("success") or r.get("cell") == "clean":
                done.add(r["key"])

    # group cells per task so each env is built once
    by_task: dict[int, list] = {}
    for cell, count in cells:
        t = int(cell.split("_")[0][1:])
        by_task.setdefault(t, []).append((cell, count))

    rng = np.random.default_rng(hash(args.condition) % (2 ** 32))
    t_start = time.time()
    n_ok = n_try = 0
    for task_id, task_cells in sorted(by_task.items()):
        env = make_gen_env(task_id)
        obj_name = task_object(task_id)
        with h5py.File(demo_file(task_id)) as h5:
            demo_keys = sorted(h5["data"].keys(), key=lambda s: int(s.split("_")[1]))
            if args.clean_check:
                for k in range(args.clean_check):
                    key = f"t{task_id}_clean{k}"
                    if key in done:
                        continue
                    r = synth_episode(env, h5["data"][demo_keys[k]], obj_name, np.zeros(2))
                    rec = {"key": key, "cell": "clean", "task_id": task_id,
                           "demo": demo_keys[k], "d_xy": [0.0, 0.0],
                           "success": r.get("success"), "n_steps": r.get("n_steps"),
                           "pregrasp_offset_cm": r.get("pregrasp_offset_cm")}
                    with meta_path.open("a") as f:
                        f.write(json.dumps(rec) + "\n")
                    print(f"[clean] {key} demo={demo_keys[k]} succ={r.get('success')} "
                          f"off={r.get('pregrasp_offset_cm')}cm", flush=True)
            for cell, count in task_cells:
                _, dirname, mag_s = cell.split("_")
                mag_mm = int(mag_s[:-2])
                demo_order = rng.permutation(len(demo_keys))
                di = 0
                for k in range(count):
                    key = f"{cell}_k{k}"
                    if key in done:
                        continue
                    saved = False
                    for _ in range(args.max_attempts):
                        demo = demo_keys[demo_order[di % len(demo_keys)]]
                        di += 1
                        d_xy = jittered_d(dirname, mag_mm, rng)
                        r = synth_episode(env, h5["data"][demo], obj_name, d_xy)
                        n_try += 1
                        if r.get("ok") and r["success"]:
                            np.savez_compressed(
                                out_dir / f"{key}.npz", frames=r["frames"],
                                frames2=r["frames2"], states8=r["states8"],
                                actions=r["actions"])
                            rec = {"key": key, "cell": cell, "task_id": task_id,
                                   "demo": demo, "d_xy": d_xy.round(4).tolist(),
                                   "success": True, "n_steps": r["n_steps"],
                                   "t_grasp": r["t_grasp"], "t_release": r["t_release"],
                                   "pregrasp_offset_cm": r["pregrasp_offset_cm"],
                                   "canonical_xyz": r["canonical_xyz"],
                                   "displaced_xyz": r["displaced_xyz"]}
                            with meta_path.open("a") as f:
                                f.write(json.dumps(rec) + "\n")
                            n_ok += 1
                            saved = True
                            el = time.time() - t_start
                            print(f"[{n_ok}] {key} demo={demo} |d|={np.linalg.norm(d_xy)*100:.1f}cm "
                                  f"off={r['pregrasp_offset_cm']}cm "
                                  f"yield={n_ok}/{n_try} ({el/60:.0f}m)", flush=True)
                            break
                    if not saved:
                        with meta_path.open("a") as f:
                            f.write(json.dumps({"key": key, "cell": cell,
                                                "task_id": task_id, "success": False,
                                                "exhausted": True}) + "\n")
                        print(f"[FAIL] {key} exhausted {args.max_attempts} attempts", flush=True)
        env.close()
    print(f"SYNTH_DONE cond={args.condition} ok={n_ok} attempts={n_try} "
          f"yield={n_ok/max(n_try,1):.0%}", flush=True)


if __name__ == "__main__":
    main()
