#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HF_REPO_ID="${HF_REPO_ID:-huigecheng/pasketti-phonetic-weights}"
REVISION="${REVISION:-main}"
TARGET_DIR="${TARGET_DIR:-$REPO_ROOT/working/online/17}"
TREE_TARGET_DIR="${TREE_TARGET_DIR:-$REPO_ROOT/src/tree_reranker}"

usage() {
  cat <<'EOF'
Usage:
  HF_REPO_ID=owner/repo bash scripts/download_weights.sh

Optional environment variables:
  HF_REPO_ID       Hugging Face model repo id (default: huigecheng/pasketti-phonetic-weights)
  REVISION         Branch / tag / commit to download from (default: main)
  TARGET_DIR       Destination for ASR checkpoints (default: working/online/17)
  TREE_TARGET_DIR  Destination for reranker artifacts (default: src/tree_reranker)

Expected Hugging Face repo layout:
  online/17/<model_name>/model.pt
  online/17/<model_name>/flags.json
  online/17/<model_name>/nemo_model_slim.nemo
  tree_reranker/reranker_meta.json
  tree_reranker/tree_cb_fold0/model.json
  ...
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "ERROR: huggingface-cli not found." >&2
  echo "Install it with: python -m pip install huggingface_hub" >&2
  exit 1
fi

mkdir -p "$TARGET_DIR" "$TREE_TARGET_DIR"

echo "==> Downloading model checkpoints from $HF_REPO_ID@$REVISION"
huggingface-cli download "$HF_REPO_ID" \
  --repo-type model \
  --revision "$REVISION" \
  --local-dir "$TARGET_DIR" \
  --include 'online/17/*'

echo "==> Downloading tree reranker artifacts"
huggingface-cli download "$HF_REPO_ID" \
  --repo-type model \
  --revision "$REVISION" \
  --local-dir "$REPO_ROOT/src" \
  --include 'tree_reranker/*'

echo "==> Done"
echo "  checkpoints: $TARGET_DIR"
echo "  reranker:    $TREE_TARGET_DIR"