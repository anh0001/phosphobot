"""P2T step 4 — pack ONE mixed LeRobot v3 dataset: 432 originals + all synthetic.

Single dataset => every condition trains with identical normalization stats and
pipeline; conditions select episode subsets via --dataset.episodes (manifest
written next to the dataset). Images stored as dtype 'image' (PNG-in-parquet),
matching HuggingFaceVLA/libero.

Usage:
  env -u PYTHONPATH .venv/bin/python p2t/make_mixed_dataset.py \
      --staging p2t/staging --out-root p2t/datasets --name p2t_mixed \
      [--originals-only] [--smoke]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from p2t.p2t_lib import decode_image, episode_frames, spatial_episode_map  # noqa: E402

FEATURES = {
    "observation.images.image": {"dtype": "image", "shape": (256, 256, 3),
                                 "names": ["height", "width", "channel"]},
    "observation.images.image2": {"dtype": "image", "shape": (256, 256, 3),
                                  "names": ["height", "width", "channel"]},
    "observation.state": {"dtype": "float32", "shape": (8,), "names": ["state"]},
    "action": {"dtype": "float32", "shape": (7,), "names": ["actions"]},
}
CONDITIONS = ("A_gain", "C_uniform", "D_failure", "E_sham")


def task_language(task_id: int) -> str:
    import os
    from libero.libero import benchmark
    suite = benchmark.get_benchmark_dict()[os.environ.get("E_SUITE", "libero_spatial")]()
    return suite.get_task(task_id).language


def add_original(ds, ep: int, lang: str) -> None:
    fr = episode_frames(ep)
    for _, row in fr.iterrows():
        ds.add_frame({
            "observation.images.image": decode_image(row["observation.images.image"]),
            "observation.images.image2": decode_image(row["observation.images.image2"]),
            "observation.state": np.asarray(row["observation.state"], dtype=np.float32),
            "action": np.asarray(row["action"], dtype=np.float32),
            "task": lang,
        })
    ds.save_episode()


def add_synthetic(ds, npz_path: Path, lang: str) -> None:
    d = np.load(npz_path)
    n = len(d["actions"])
    for t in range(n):
        ds.add_frame({
            "observation.images.image": d["frames"][t],
            "observation.images.image2": d["frames2"][t],
            "observation.state": d["states8"][t].astype(np.float32),
            "action": d["actions"][t].astype(np.float32),
            "task": lang,
        })
    ds.save_episode()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--staging", default="p2t/staging")
    ap.add_argument("--out-root", default="p2t/datasets")
    ap.add_argument("--name", default="p2t_mixed")
    ap.add_argument("--originals-only", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="2 originals + up to 2 synthetic per condition")
    args = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = Path(args.out_root) / args.name
    if root.exists():
        print(f"[warn] removing existing {root}")
        shutil.rmtree(root)

    ds = LeRobotDataset.create(
        repo_id=f"local/{args.name}", fps=10, features=FEATURES, root=root,
        robot_type="panda", use_videos=False, image_writer_threads=8)

    manifest: dict = {"originals": [], "conditions": {c: [] for c in CONDITIONS}}
    ep_map = spatial_episode_map()
    ep_idx = 0

    originals = [(t, ep) for t in sorted(ep_map) for ep in ep_map[t]]
    if args.smoke:
        originals = originals[:2]
    lang_cache = {t: task_language(t) for t in sorted(ep_map)}
    for i, (t, ep) in enumerate(originals):
        add_original(ds, ep, lang_cache[t])
        manifest["originals"].append(ep_idx)
        ep_idx += 1
        if i % 25 == 0:
            print(f"[orig {i+1}/{len(originals)}]", flush=True)

    if not args.originals_only:
        # Matched-budget equalization (design deviation log, 2026-07-07): yields
        # differ across conditions, so every condition is capped to the minimum
        # delivered count via a seeded shuffle (preserves each condition's
        # realized placement distribution).
        cond_recs: dict[str, list] = {}
        for cond in CONDITIONS:
            meta = Path(args.staging) / cond / "meta.jsonl"
            if not meta.exists():
                print(f"[skip] {cond}: no staging meta")
                continue
            recs = [json.loads(l) for l in meta.read_text().splitlines()]
            cond_recs[cond] = [r for r in recs
                               if r.get("success") and r.get("cell") != "clean"]
        if cond_recs:
            cap = 2 if args.smoke else min(len(v) for v in cond_recs.values())
            print(f"[equalize] per-condition cap = {cap} "
                  f"(delivered: {{k: len(v) for k, v in cond_recs.items()}})".replace(
                      "{k: len(v) for k, v in cond_recs.items()}",
                      str({k: len(v) for k, v in cond_recs.items()})), flush=True)
            rng = np.random.default_rng(0)
            for cond, recs in cond_recs.items():
                keep = [recs[i] for i in rng.permutation(len(recs))[:cap]]
                for r in sorted(keep, key=lambda r: r["key"]):
                    add_synthetic(ds, Path(args.staging) / cond / f"{r['key']}.npz",
                                  lang_cache[r["task_id"]])
                    manifest["conditions"][cond].append(ep_idx)
                    ep_idx += 1
                print(f"[{cond}] {cap} episodes (of {len(recs)} delivered)", flush=True)

    ds.finalize()
    (root / "p2t_manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"DATASET_DONE root={root} episodes={ep_idx}", flush=True)


if __name__ == "__main__":
    main()
