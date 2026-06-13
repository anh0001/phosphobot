"""E0 candidate screening: clean success of a SmolVLA checkpoint on libero_spatial.

Position-disciplined (fresh env per (task,seed); clean rollout first — matches the
reach-field grid convention). flow_k=0 (no extra sampling).

Usage:
  env -u PYTHONPATH MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
    python preempt/eval_clean.py --ckpt <hub_id_or_path> --out preempt/e0_<name>.jsonl
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preempt.harness_lib import build_pipeline, make_single_env  # noqa: E402
from preempt.reach_field import load_pair_meta, run_rollout  # noqa: E402

SEEDS = [1000, 1001, 1002, 1003, 1004]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tasks", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--out", required=True)
    ap.add_argument("--unflip", action="store_true",
                    help="pre-flip images so the LiberoProcessorStep 180-flip cancels "
                         "(for checkpoints trained on unflipped frames)")
    args = ap.parse_args()

    tasks = [int(x) for x in args.tasks.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    obj_map, _ = load_pair_meta("preempt/e1_recovery_d005.json")
    out_path = Path(args.out)
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            r = json.loads(line)
            done.add((r["task_id"], r["seed"]))

    # auto-rename image2 -> whatever wrist key the checkpoint expects, and find
    # the training dataset repo for normalization-stats fallback
    rename, stats_repo = None, None
    try:
        from lerobot.configs.policies import PreTrainedConfig
        feats = PreTrainedConfig.from_pretrained(args.ckpt).input_features
        wrist = [k for k in feats if "wrist" in k]
        if wrist and "observation.images.image2" not in feats:
            rename = {"observation.images.image2": wrist[0]}
            print(f"[eval_clean] rename_map: {rename}")
        import json as _json
        from huggingface_hub import hf_hub_download
        tc = _json.load(open(hf_hub_download(args.ckpt, "train_config.json")))
        stats_repo = (tc.get("dataset") or {}).get("repo_id")
        print(f"[eval_clean] stats_repo: {stats_repo}")
    except Exception as e:
        print(f"[eval_clean] hub peek failed ({e}); local checkpoint assumed")
    pipe = build_pipeline(ckpt=args.ckpt, rename_map=rename, stats_repo=stats_repo)
    if args.unflip:
        import torch
        orig_pre = pipe.env_preprocessor

        def _unflip_then(obs):
            out = dict(obs)
            for k, v in out.items():
                if k.startswith("observation.images.") and isinstance(v, torch.Tensor) and v.dim() == 4:
                    out[k] = torch.flip(v, dims=[2, 3])
            return orig_pre(out)

        pipe.env_preprocessor = _unflip_then
        print("[eval_clean] UNFLIP active")
    n_succ = n_tot = 0
    t0 = time.time()
    for t in tasks:
        for si, s in enumerate(seeds):
            if (t, s) in done or (t, s) not in obj_map:
                continue
            env = make_single_env(task_id=t, episode_index=si)
            r = run_rollout(pipe, env, s, obj_map[(t, s)], None, flow_k=0)
            env.close()
            r.pop("_arrays", None)
            rec = {"task_id": t, "seed": s, "ckpt": args.ckpt, **{k: r[k] for k in
                   ("success", "n_steps", "macro_calls", "endpoints", "task")}}
            with out_path.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            n_tot += 1
            n_succ += bool(r["success"])
            print(f"[{n_tot}] t{t}_s{s} succ={r['success']} steps={r['n_steps']} "
                  f"(running {n_succ}/{n_tot} = {n_succ / n_tot:.0%}, "
                  f"{(time.time() - t0) / 60:.0f}m)", flush=True)
    print(f"CLEAN_EVAL_DONE ckpt={args.ckpt} success={n_succ}/{n_tot}")


if __name__ == "__main__":
    main()
