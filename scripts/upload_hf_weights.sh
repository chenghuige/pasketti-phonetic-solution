#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODELS_FILE="${MODELS_FILE:-$REPO_ROOT/src/models.txt}"
SOURCE_MODEL_ROOT="${SOURCE_MODEL_ROOT:-/home/gezi/pikachu/projects/drivendata/pasketti-phonetic/working/online/9}"
SOURCE_RERANKER_DIR="${SOURCE_RERANKER_DIR:-/home/gezi/pikachu/projects/drivendata/pasketti-phonetic/working/offline/9/ensemble.feat_nemo_group.feat_tdt_group.feat_wavlm_group.0407/0}"
HF_REPO_ID="${HF_REPO_ID:-huigecheng/pasketti-phonetic-weights}"
REVISION="${REVISION:-main}"
STAGE_DIR="${STAGE_DIR:-$REPO_ROOT/.hf_upload_stage}"
PRIVATE_REPO="${PRIVATE_REPO:-1}"
UPLOAD_NOW="${UPLOAD_NOW:-0}"
NUM_WORKERS="${NUM_WORKERS:-8}"
KEEP_STAGE="${KEEP_STAGE:-1}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/upload_hf_weights.sh

Environment variables:
  HF_REPO_ID           Target Hugging Face repo id.
  SOURCE_MODEL_ROOT    Directory containing final online model dirs.
  SOURCE_RERANKER_DIR  Directory containing reranker_meta.json and tree_cb_fold*/.
  STAGE_DIR            Temporary staging directory.
  PRIVATE_REPO         1=create/use private repo if needed, 0=public.
  UPLOAD_NOW           1=run huggingface-cli upload-large-folder after staging.
  NUM_WORKERS          Upload workers for huggingface-cli.
  KEEP_STAGE           1=keep staged files after upload, 0=delete stage dir.

Examples:
  HF_REPO_ID=huigecheng/pasketti-phonetic-weights bash scripts/upload_hf_weights.sh
  HF_REPO_ID=huigecheng/pasketti-phonetic-weights UPLOAD_NOW=1 bash scripts/upload_hf_weights.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "ERROR: huggingface-cli not found" >&2
  exit 1
fi

if [[ ! -f "$MODELS_FILE" ]]; then
  echo "ERROR: models file not found: $MODELS_FILE" >&2
  exit 1
fi

rm -rf "$STAGE_DIR"
mkdir -p "$STAGE_DIR/online/17" "$STAGE_DIR/tree_reranker"

copy_model_dir() {
  local model_name="$1"
  local src_dir="$SOURCE_MODEL_ROOT/$model_name/0"
  local dst_dir="$STAGE_DIR/online/17/$model_name"
  local flags_src=""

  if [[ ! -d "$src_dir" ]]; then
    echo "ERROR: model source dir missing: $src_dir" >&2
    exit 1
  fi

  mkdir -p "$dst_dir"
  cp "$src_dir/model.pt" "$dst_dir/"

  if [[ -f "$src_dir/flags.json" ]]; then
    flags_src="$src_dir/flags.json"
  elif [[ -f "$(dirname "$src_dir")/flags.json" ]]; then
    flags_src="$(dirname "$src_dir")/flags.json"
  else
    echo "ERROR: flags.json missing for $model_name" >&2
    exit 1
  fi
  cp "$flags_src" "$dst_dir/flags.json"

  for optional_file in nemo_model_slim.nemo model_meta.json preprocessor_config.json config.json; do
    if [[ -f "$src_dir/$optional_file" ]]; then
      cp "$src_dir/$optional_file" "$dst_dir/"
    fi
  done

  echo "staged model: $model_name"
}

while IFS= read -r line; do
  model_name="${line%%#*}"
  model_name="$(echo "$model_name" | xargs)"
  [[ -z "$model_name" ]] && continue
  copy_model_dir "$model_name"
done < "$MODELS_FILE"

if [[ ! -d "$SOURCE_RERANKER_DIR" ]]; then
  echo "ERROR: reranker source dir missing: $SOURCE_RERANKER_DIR" >&2
  exit 1
fi

for required_file in reranker_meta.json reranker_experiment.json reranker_features.txt; do
  if [[ ! -f "$SOURCE_RERANKER_DIR/$required_file" ]]; then
    echo "ERROR: missing reranker file: $SOURCE_RERANKER_DIR/$required_file" >&2
    exit 1
  fi
  cp "$SOURCE_RERANKER_DIR/$required_file" "$STAGE_DIR/tree_reranker/"
done

for tree_dir in "$SOURCE_RERANKER_DIR"/tree_cb_fold*; do
  [[ -d "$tree_dir" ]] || continue
  cp -r "$tree_dir" "$STAGE_DIR/tree_reranker/"
done

python - <<PY
from pathlib import Path
stage = Path('$STAGE_DIR')
print('stage_dir', stage)
print('stage_size_bytes', sum(p.stat().st_size for p in stage.rglob('*') if p.is_file()))
print('n_models', len(list((stage / 'online' / '17').iterdir())))
print('n_tree_dirs', len(list((stage / 'tree_reranker').glob('tree_cb_fold*'))))
PY

echo "Staged upload payload at: $STAGE_DIR"
echo "Suggested upload command:"
echo "  HF_REPO_ID=$HF_REPO_ID UPLOAD_NOW=1 bash scripts/upload_hf_weights.sh"

if [[ "$UPLOAD_NOW" != "1" ]]; then
  exit 0
fi

upload_args=(upload-large-folder "$HF_REPO_ID" "$STAGE_DIR" --repo-type model --revision "$REVISION" --num-workers "$NUM_WORKERS")
if [[ "$PRIVATE_REPO" == "1" ]]; then
  upload_args+=(--private)
fi

echo "==> Uploading to Hugging Face repo: $HF_REPO_ID"
huggingface-cli "${upload_args[@]}"

if [[ "$KEEP_STAGE" == "0" ]]; then
  rm -rf "$STAGE_DIR"
fi

echo "==> Upload complete: https://huggingface.co/$HF_REPO_ID"