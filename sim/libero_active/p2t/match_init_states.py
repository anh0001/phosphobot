"""P2T step 1 — map each libero_spatial dataset episode to its benchmark init state.

The dataset carries no init metadata, so we match each episode's FIRST FRAME
(agentview image) against the rendered frame of each of the 50 benchmark init
states per task. Robot proprio is useless here (the arm always resets to home;
init states differ in object placement — verified in the first smoke: 8-D state
matching was degenerate, min second/best gap 1.0x). A valid image match requires
the best MAE to sit far below the second-best (bimodal gap).

Fidelity gate: replay the dataset actions of --replay-per-task episodes from the
matched init state; record `is_success` + max frame-vs-render MAE.

Usage:
  env -u PYTHONPATH MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
    .venv/bin/python p2t/match_init_states.py [--tasks 0,1,...] [--replay-per-task 2] \
      [--out p2t/init_state_map.json]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from p2t.p2t_lib import (  # noqa: E402
    decode_image,
    episode_frames,
    spatial_episode_map,
)
from preempt.harness_lib import _refresh_obs, make_single_env  # noqa: E402
from preempt.reach_field import load_pair_meta  # noqa: E402

N_INIT = 50


def task_object_map() -> dict[int, str]:
    """task_id -> task-relevant object name (same source as the instruments)."""
    obj_map, _ = load_pair_meta("preempt/e1_recovery_d005.json")
    out: dict[int, str] = {}
    for (task_id, _seed), obj in obj_map.items():
        out.setdefault(task_id, obj)
    return out


def render_dataset_view(raw_obs: dict, hw: int = 256) -> np.ndarray:
    """Env agentview -> dataset convention (un-rotate 180deg, resize to hw)."""
    from PIL import Image
    ren = np.asarray(raw_obs["pixels"]["image"])[::-1, ::-1]
    return np.asarray(Image.fromarray(ren).resize((hw, hw)))


def init_frames_table(task_id: int) -> tuple[np.ndarray, "object"]:
    """(N_INIT, 256, 256, 3) rendered first frame per init state; returns (table, env)."""
    env = make_single_env(task_id=task_id, episode_index=0)
    rows = []
    for j in range(N_INIT):
        env.init_state_id = j
        raw_obs, _ = env.reset(seed=0)
        rows.append(render_dataset_view(raw_obs))
    return np.asarray(rows, dtype=np.int16), env


def replay_episode(env, init_j: int, ep: int, obj_name: str | None = None,
                   dump_dir: Path | None = None) -> dict:
    """Replay dataset actions from init state init_j.

    Reports task success, frame MAE samples, and — the sharp label-quality
    gate — the xy distance between the eef and the target object at the
    pregrasp step (a wrong init match shows up as a pregrasp offset even when
    basin tolerance still lets the task succeed).
    """
    from preempt.perturb import object_xyz, perturb_object_by_name_vec

    frames = episode_frames(ep)
    actions = np.stack(frames["action"].to_numpy())
    t_grasp = None
    g = actions[:, 6]
    close = np.where((g[1:] > 0.0) & (g[:-1] <= 0.0))[0]
    if len(close):
        t_grasp = int(close[0]) + 1
    env.init_state_id = init_j
    raw_obs, _ = env.reset(seed=0)
    qpos_adr = None
    if obj_name:
        probe = perturb_object_by_name_vec(env, obj_name, (0.0, 0.0))
        if probe.get("perturbed"):
            qpos_adr = probe["qpos_adr"]
        raw_obs = _refresh_obs(env)
    success = False
    mae_samples = []
    pregrasp_offset_cm = None
    check_ts = {0, len(actions) // 2, len(actions) - 1}
    for t, a in enumerate(actions):
        if t in check_ts:
            ref = decode_image(frames.iloc[t]["observation.images.image"])
            ren = render_dataset_view(raw_obs, hw=ref.shape[0])
            mae = float(np.abs(ren.astype(np.int16) - ref.astype(np.int16)).mean())
            mae_samples.append({"t": t, "mae": round(mae, 2)})
            if dump_dir is not None:
                np.savez_compressed(dump_dir / f"ep{ep}_t{t}.npz", render=ren, ref=ref)
        if t == t_grasp and qpos_adr is not None:
            eef = np.asarray(raw_obs["robot_state"]["eef"]["pos"], dtype=np.float64)
            obj = object_xyz(env, qpos_adr)
            pregrasp_offset_cm = float(np.linalg.norm(eef[:2] - obj[:2]) * 100)
        raw_obs, reward, terminated, truncated, info = env.step(a.astype(np.float64))
        if bool(info.get("is_success", False)):
            success = True
        if terminated:
            break
    return {"episode": ep, "init_j": init_j, "success": success,
            "n_actions": len(actions), "n_stepped": t + 1,
            "pregrasp_offset_cm": None if pregrasp_offset_cm is None
            else round(pregrasp_offset_cm, 2),
            "frame_mae": mae_samples}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=",".join(map(str, range(10))))
    ap.add_argument("--replay-per-task", type=int, default=2)
    ap.add_argument("--out", default="p2t/init_state_map.json")
    ap.add_argument("--dump-frames", action="store_true")
    args = ap.parse_args()

    ep_map = spatial_episode_map()
    result: dict = {"tasks": {}, "replays": []}
    t0 = time.time()
    for task_id in [int(x) for x in args.tasks.split(",")]:
        eps = ep_map[task_id]
        table, env = init_frames_table(task_id)
        first_frames = np.asarray([
            decode_image(episode_frames(ep).iloc[0]["observation.images.image"])
            for ep in eps], dtype=np.int16)
        # High-diff pixel count against every init render, restricted to the
        # variance mask (pixels that actually differ ACROSS init states = the
        # object-placement regions). Global MAE and unmasked counts are both
        # degenerate: render/codec mismatch between the conversion pipeline and
        # our renderer puts a ~10%-of-pixels systematic floor everywhere
        # (verified in smokes 2-3); the mask keeps only discriminative pixels.
        var_mask = table.std(axis=0).sum(axis=2) > 20  # (256,256)
        diff = np.abs(first_frames[:, None] - table[None, :]).sum(axis=4)
        D = (diff > 40)[:, :, var_mask].mean(axis=2)
        best = D.argmin(1)
        d_sorted = np.sort(D, axis=1)
        gap = d_sorted[:, 1] / np.maximum(d_sorted[:, 0], 1e-12)
        dup = len(set(best.tolist())) < len(best)
        result["tasks"][task_id] = {
            "episodes": eps, "init_j": best.tolist(),
            "best_mae": d_sorted[:, 0].round(3).tolist(),
            "second_over_best": gap.round(2).tolist(),
            "duplicate_matches": bool(dup),
        }
        print(f"[task {task_id}] n_eps={len(eps)} best_mae max={d_sorted[:,0].max():.2f} "
              f"min_gap={gap.min():.2f}x dup={dup}", flush=True)
        dump = Path("p2t/replay_frames") if args.dump_frames else None
        if dump is not None:
            dump.mkdir(parents=True, exist_ok=True)
        obj_name = task_object_map().get(task_id)
        for k in range(min(args.replay_per_task, len(eps))):
            r = replay_episode(env, int(best[k]), eps[k], obj_name=obj_name, dump_dir=dump)
            r["task_id"] = task_id
            result["replays"].append(r)
            print(f"  replay ep{eps[k]} init{best[k]} succ={r['success']} "
                  f"steps={r['n_stepped']}/{r['n_actions']} "
                  f"pregrasp_off={r['pregrasp_offset_cm']}cm mae={r['frame_mae']}", flush=True)
        env.close()
    n_ok = sum(r["success"] for r in result["replays"])
    result["replay_success_rate"] = n_ok / max(len(result["replays"]), 1)
    Path(args.out).write_text(json.dumps(result, indent=1))
    print(f"DONE replays {n_ok}/{len(result['replays'])} ok "
          f"({time.time()-t0:.0f}s) -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
