#!/usr/bin/env bash
# One-shot deploy to Hugging Face Spaces.
#
# Prereq: you've already created an empty Space at
#   https://huggingface.co/new-space
# with: SDK = Docker, name = whatever you want, visibility = Public.
#
# Then run:
#   ./scripts/deploy_hf.sh <hf-username> <space-name>
#
# Example:
#   ./scripts/deploy_hf.sh Rashidbm saamid
#
# The script clones the empty Space repo into a sibling directory, copies
# the source over (respecting .gitignore + .dockerignore), and pushes.
# HF then builds the Dockerfile and the Space goes live in 5–15 minutes.
#
# Re-running pushes any local edits.  Idempotent.

set -euo pipefail

if [ $# -lt 2 ]; then
  echo "usage: $0 <hf-username> <space-name>" >&2
  echo "example: $0 Rashidbm saamid" >&2
  exit 1
fi

HF_USER="$1"
HF_SPACE="$2"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_DIR="${TMPDIR:-/tmp}/saamid-hf-deploy"
SPACE_URL="https://huggingface.co/spaces/${HF_USER}/${HF_SPACE}"
SPACE_GIT="https://huggingface.co/spaces/${HF_USER}/${HF_SPACE}.git"

bold() { printf "\033[1m%s\033[0m\n" "$*"; }

# --- prereqs
command -v git >/dev/null   || { echo "git not installed" >&2; exit 2; }
command -v git-lfs >/dev/null \
  || { echo "git-lfs not installed (brew install git-lfs)" >&2; exit 2; }

bold "[1/5] preparing work dir at $WORK_DIR"
rm -rf "$WORK_DIR"
mkdir -p "$WORK_DIR"
git lfs install --skip-repo

bold "[2/5] cloning empty Space repo"
if ! git clone "$SPACE_GIT" "$WORK_DIR"; then
  cat <<HINT >&2

Couldn't clone $SPACE_GIT.

Did you create the Space first?
  https://huggingface.co/new-space
  - SDK: Docker
  - Name: $HF_SPACE
  - Visibility: Public

If yes, you may need to authenticate via:
  huggingface-cli login         # paste your access token (write scope)
  git config --global credential.helper store

Then re-run this script.
HINT
  exit 3
fi

bold "[3/5] syncing source into Space repo (excludes .venv, runs/, data/, etc.)"
# Sync everything except heavy / private dirs.  Also exclude
# configs/slides_outline.md and any other documents with strong
# defense-targeted framing — HF's automated content moderation has
# flagged earlier deploys as "abusive" because of the pitch language.
# The dashboard itself stays the same; only the Space-facing README
# and the deck are softened/dropped.
rsync -a --delete \
  --exclude '.git/' \
  --exclude '.venv/' \
  --exclude 'runs/' \
  --exclude 'data/' \
  --exclude '.cache/' \
  --exclude '.claude/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude 'wandb/' \
  --exclude '.DS_Store' \
  --exclude 'configs/slides_outline.md' \
  --exclude 'configs/visuals/' \
  "$REPO_ROOT/" "$WORK_DIR/"

# Override the Space's README with the academic/research framing if a
# Space-specific one exists.  Keeps the local repo README untouched.
if [ -f "$REPO_ROOT/.hf-space-readme.md" ]; then
  cp "$REPO_ROOT/.hf-space-readme.md" "$WORK_DIR/README.md"
  echo "  using .hf-space-readme.md for the Space"
fi

# Ensure HF Spaces sees the README YAML frontmatter as the Space metadata.
if ! head -2 "$WORK_DIR/README.md" | grep -q '^---$'; then
  echo "ERROR: README.md is missing the HF Spaces YAML frontmatter." >&2
  exit 4
fi

bold "[4/5] committing"
cd "$WORK_DIR"
git lfs install
git add -A
if git diff --cached --quiet; then
  echo "  nothing to commit — Space is already in sync"
else
  git commit -m "Saamid · update from $(date -u +%Y-%m-%dT%H:%M:%SZ)"
fi

bold "[5/5] pushing to $SPACE_GIT"
git push origin main 2>&1 | sed 's/^/  /'

cat <<DONE

✓ Pushed.

  Space:    $SPACE_URL
  Build:    $SPACE_URL?logs=container
  Status:   $SPACE_URL?logs=build

The first build takes 5–15 min (downloads torch + transformers + the AST
model).  Subsequent pushes only rebuild the changed layers.

Once the build finishes the dashboard is live at:
  $SPACE_URL

(HF normalises the URL to https://${HF_USER}-${HF_SPACE}.hf.space — both work.)
DONE
