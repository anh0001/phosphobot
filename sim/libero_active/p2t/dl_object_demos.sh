#!/usr/bin/env bash
cd "$(dirname "$0")/.."
for i in $(seq 1 20); do
  n=$(ls p2t/libero_demos/libero_object/*.hdf5 2>/dev/null | wc -l)
  [ "$n" -ge 10 ] && { echo "DEMOS_DONE ($n/10)"; exit 0; }
  .venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('yifengzhu-hf/LIBERO-datasets', repo_type='dataset', allow_patterns=['libero_object/*'], local_dir='p2t/libero_demos', max_workers=2)" 2>/dev/null
  sleep 5
done
echo "DEMOS_FAILED after 20 attempts ($(ls p2t/libero_demos/libero_object/*.hdf5 2>/dev/null | wc -l)/10)"
