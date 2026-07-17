#!/usr/bin/env bash
# Resumable per-file object-demo downloader. hf_hub_download auto-resumes partial
# ('.incomplete') files across network drops. Loops each file until it lands.
cd "$(dirname "$0")/.."
env -u PYTHONPATH .venv/bin/python -u - <<'PY'
import time
from huggingface_hub import HfApi, hf_hub_download
api = HfApi()
files = [s.rfilename for s in api.repo_info('yifengzhu-hf/LIBERO-datasets',
         repo_type='dataset').siblings if s.rfilename.startswith('libero_object/')
         and s.rfilename.endswith('.hdf5')]
print(f"[dl] {len(files)} object demo files", flush=True)
done = 0
for f in files:
    for attempt in range(1, 41):  # up to 40 resume attempts per file
        try:
            p = hf_hub_download('yifengzhu-hf/LIBERO-datasets', repo_type='dataset', filename=f)
            done += 1
            print(f"[dl] {done}/{len(files)} OK {f.split('/')[-1]}", flush=True)
            break
        except Exception as e:
            print(f"[dl] {f.split('/')[-1]} attempt {attempt} failed: {type(e).__name__} — resuming in 10s", flush=True)
            time.sleep(10)
    else:
        print(f"[dl] GAVE UP on {f}", flush=True)
print("DEMOS_DONE" if done == len(files) else f"DEMOS_PARTIAL {done}/{len(files)}", flush=True)
PY
