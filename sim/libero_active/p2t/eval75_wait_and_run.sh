#!/usr/bin/env bash
# Wait until GPU 1 has >= 8 GB free, then run the 75mm sweep (resumable).
cd "$(dirname "$0")/.."
while true; do
  USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1)
  FREE=$((49140 - USED))
  if [ "$FREE" -ge 8000 ]; then
    echo "GPU1 free ${FREE}MiB -> starting sweep $(date '+%F %T')"
    bash p2t/eval75_sweep.sh
    exit $?
  fi
  sleep 120
done
