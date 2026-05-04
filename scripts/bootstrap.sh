#!/usr/bin/env bash
# Saamid one-command bootstrap.
#
# Run on a fresh clone to get from "git clone" to "ready to demo" with
# zero further commands.  Idempotent — safe to re-run.
#
#   ./scripts/bootstrap.sh
#
# After it succeeds:
#   uv run python -m scripts.serve                  # live mode
#   uv run python -m scripts.serve --simulate data/abdulrahman/DroneAbdulrahman.wav  # demo mode
#
# Then open http://localhost:8000 in a browser.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

bold()   { printf "\033[1m%s\033[0m\n" "$*"; }
green()  { printf "\033[32m%s\033[0m\n" "$*"; }
yellow() { printf "\033[33m%s\033[0m\n" "$*"; }
red()    { printf "\033[31m%s\033[0m\n" "$*" >&2; }

# ----------------------------------------------------------------------------
# 1. uv (Python package manager)
# ----------------------------------------------------------------------------
bold "[1/5] Checking uv"
if ! command -v uv >/dev/null 2>&1; then
  yellow "  uv not found — installing"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # uv installs to ~/.local/bin or ~/.cargo/bin; make it visible for this run
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi
green "  uv $(uv --version | awk '{print $2}')"

# ----------------------------------------------------------------------------
# 2. system audio libraries
# ----------------------------------------------------------------------------
bold "[2/5] Checking system audio libraries (portaudio, ffmpeg)"
case "$(uname -s)" in
  Darwin)
    if ! command -v brew >/dev/null 2>&1; then
      yellow "  Homebrew not installed.  Install audio deps manually:"
      yellow "    brew install portaudio ffmpeg"
    else
      for pkg in portaudio ffmpeg; do
        if ! brew list --formula "$pkg" >/dev/null 2>&1; then
          yellow "  installing $pkg"
          brew install "$pkg"
        fi
      done
    fi
    green "  audio libs OK (macOS)"
    ;;
  Linux)
    missing=()
    pkg-config --exists portaudio-2.0 2>/dev/null || missing+=("portaudio19-dev / portaudio-devel")
    command -v ffmpeg >/dev/null 2>&1   || missing+=("ffmpeg")
    if [ ${#missing[@]} -gt 0 ]; then
      yellow "  Missing system packages: ${missing[*]}"
      yellow "  On Debian/Ubuntu:  sudo apt install portaudio19-dev libsndfile1 ffmpeg"
      yellow "  On Fedora/RHEL  :  sudo dnf install portaudio-devel libsndfile  ffmpeg"
    else
      green "  audio libs OK (Linux)"
    fi
    ;;
  *)
    yellow "  Unknown OS '$(uname -s)' — install portaudio + ffmpeg yourself"
    ;;
esac

# ----------------------------------------------------------------------------
# 3. Python dependencies via uv
# ----------------------------------------------------------------------------
bold "[3/5] Resolving Python dependencies"
uv sync
green "  dependencies installed into .venv"

# ----------------------------------------------------------------------------
# 4. Pre-download the AST model (so demo day doesn't need internet)
# ----------------------------------------------------------------------------
bold "[4/5] Caching detection model (~430 MB, one-time)"
HUB_ID="${SAAMID_HUB_ID:-Rashidbm/samid-drone-detector}"
uv run python - <<PY
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
hub = "${HUB_ID}"
print(f"  downloading {hub}")
AutoFeatureExtractor.from_pretrained(hub)
AutoModelForAudioClassification.from_pretrained(hub)
print("  model cached at ~/.cache/huggingface")
PY
green "  model cached"

# ----------------------------------------------------------------------------
# 5. Self-test: run inference against a known recording
# ----------------------------------------------------------------------------
bold "[5/5] Self-test — inferring on data/abdulrahman/DroneAbdulrahman.wav"
SELFTEST_WAV="${SAAMID_SELFTEST_WAV:-data/abdulrahman/DroneAbdulrahman.wav}"
if [ ! -f "$SELFTEST_WAV" ]; then
  yellow "  $SELFTEST_WAV not found — skipping inference self-test"
else
  result=$(uv run python scripts/standalone_inference.py --wav "$SELFTEST_WAV" --threshold 0.25 \
    | grep -E "smoothed max|VERDICT" || true)
  if [ -z "$result" ]; then
    red "  inference produced no output — install is broken"
    exit 1
  fi
  echo "$result" | sed 's/^/    /'
  if echo "$result" | grep -q "DRONE DETECTED"; then
    green "  self-test passed"
  else
    red "  self-test FAILED — model did not detect a known drone recording"
    red "  check that $HUB_ID is accessible and reachable"
    exit 1
  fi
fi

echo
bold "✓ Saamid is ready"
echo
echo "Next steps:"
echo "  • Demo (no hardware):"
echo "      uv run python -m scripts.serve --simulate data/abdulrahman/DroneAbdulrahman.wav"
echo "  • Live mics:"
echo "      uv run python -m scripts.serve"
echo "  • Open the dashboard at http://localhost:8000"
echo
echo "macOS only: pre-grant microphone permission to your terminal in"
echo "System Settings → Privacy & Security → Microphone."
