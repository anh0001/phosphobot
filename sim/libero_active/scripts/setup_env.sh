#!/usr/bin/env bash
# Environment setup for the LIBERO active-query pipeline (PS.0).
# Creates a dedicated Python 3.10 venv (LIBERO/robosuite pin to 3.10) and installs
# LeRobot with the LIBERO extra. Kept separate from phosphobot's uv environment.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

DEPS_DIR="$HERE/.deps"
VENV_DIR="$HERE/.venv"
LEROBOT_DIR="$DEPS_DIR/lerobot"

echo "[setup] project root: $HERE"
mkdir -p "$DEPS_DIR"

# 1. Python 3.12 venv via uv (lerobot 0.5.x requires Python >= 3.12).
if [ ! -d "$VENV_DIR" ]; then
  echo "[setup] creating Python 3.12 venv at $VENV_DIR"
  uv venv --python 3.12 "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# 2. LeRobot from source (the [libero] extra requires the source tree).
if [ ! -d "$LEROBOT_DIR" ]; then
  echo "[setup] cloning LeRobot"
  git clone --depth 1 https://github.com/huggingface/lerobot.git "$LEROBOT_DIR"
fi

echo "[setup] installing lerobot[libero] (this is the long step)"
uv pip install -e "${LEROBOT_DIR}[libero]"

# 3. PEFT (LoRA) + extras used by the active-query code.
#    num2words is an undeclared runtime dep of the SmolVLM processor (SmolVLA backbone).
uv pip install peft scikit-learn num2words

# 4. Pre-seed the LIBERO config so first import does not block on an interactive
#    input() prompt (LIBERO asks about a custom dataset path on first run).
LIBERO_PKG="$VENV_DIR/lib/python3.12/site-packages/libero/libero"
if [ -d "$LIBERO_PKG" ] && [ ! -f "$HOME/.libero/config.yaml" ]; then
  echo "[setup] seeding ~/.libero/config.yaml"
  mkdir -p "$HOME/.libero"
  cat > "$HOME/.libero/config.yaml" <<EOF
benchmark_root: $LIBERO_PKG
bddl_files: $LIBERO_PKG/bddl_files
init_states: $LIBERO_PKG/init_files
datasets: $LIBERO_PKG/../datasets
assets: $LIBERO_PKG/assets
EOF
fi

# 5. Sanity check.
echo "[setup] verifying imports"
python - <<'PY'
import torch, lerobot
print("torch:", torch.__version__, "cuda:", torch.cuda.is_available(),
      "devices:", torch.cuda.device_count())
print("lerobot:", getattr(lerobot, "__version__", "unknown"))
try:
    import libero
    print("libero: import OK")
except Exception as e:  # noqa: BLE001
    print("libero import FAILED:", e)
PY

echo "[setup] done. activate with: source $VENV_DIR/bin/activate"
