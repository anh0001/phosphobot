#!/usr/bin/env bash
cd "$(dirname "$0")/.."
# wait for null-search to release GPU 1
while pgrep -f "reach_field.py --ckpt.*null_search" >/dev/null 2>&1 || pgrep -af "P2T_ACTION_NOISE" | grep -q null_search; do sleep 30; done
while pgrep -f "null_search.sh" >/dev/null 2>&1; do sleep 30; done
echo "[chain] null-search done, launching m2stage $(date '+%F %T')"
bash p2t/m2stage.sh
