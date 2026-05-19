#!/usr/bin/env bash
# Reproduce the second-stage CatBoost tree reranker from the 11 offline fold0
# model eval artifacts listed in models.txt.
#
# Prerequisite:
#   bash reproduce_offline_fold0.sh
#
# Required per model under ../working/offline/9/<model_name>/0/:
#   model.pt, flags.json, eval.csv, ctc_logprobs.pt
# Optional but used when available:
#   dual_head_preds.pt, aux_meta_preds.pt, pred_score columns in eval.csv
#
# Usage:
#   bash reproduce_tree_reranker.sh
#   DRY_RUN=1 bash reproduce_tree_reranker.sh
#
# Env:
#   GPU=0
#   PYTHON=python
#   EXTRA_ARGS=""      # appended to ensemble.py command
#   COPY_TO_RELEASE=1  # copy artifacts to src/tree_reranker for pack_submission.sh
set -euo pipefail

cd "$(dirname "$0")"

PYTHON="${PYTHON:-python}"
GPU="${GPU:-0}"
DRY_RUN="${DRY_RUN:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
COPY_TO_RELEASE="${COPY_TO_RELEASE:-1}"
RUN_ROOT="../working/offline/9"
TREE_MNS="ensemble.feat_nemo_group.feat_tdt_group.feat_wavlm_group.0407"
TREE_DIR="${TREE_DIR:-$RUN_ROOT/$TREE_MNS/0}"
RELEASE_TREE_DIR="${RELEASE_TREE_DIR:-tree_reranker}"

missing=0
while IFS= read -r raw; do
  line="${raw%%#*}"
  mn="$(echo "$line" | xargs || true)"
  [[ -n "$mn" ]] || continue
  model_dir="$RUN_ROOT/$mn/0"
  for f in model.pt flags.json eval.csv ctc_logprobs.pt; do
    if [[ ! -s "$model_dir/$f" ]]; then
      echo "ERROR: missing $model_dir/$f" >&2
      missing=1
    fi
  done
  if [[ ! -s "$model_dir/dual_head_preds.pt" ]]; then
    echo "WARN: missing optional $model_dir/dual_head_preds.pt" >&2
  fi
done < models.txt

if [[ "$missing" == "1" ]]; then
  echo "Run first: bash reproduce_offline_fold0.sh" >&2
  if [[ "$DRY_RUN" != "1" ]]; then
    exit 1
  fi
fi

cmd="PYTHONPATH=_compat:\$PYTHONPATH CUDA_VISIBLE_DEVICES=$GPU $PYTHON ensemble.py --feat_nemo_group --feat_tdt_group --feat_wavlm_group --mns=.0407"
if [[ -n "$EXTRA_ARGS" ]]; then
  cmd="$cmd $EXTRA_ARGS"
fi

echo "+ $cmd"
if [[ "$DRY_RUN" != "1" ]]; then
  eval "$cmd"
fi

if [[ "$COPY_TO_RELEASE" == "1" ]]; then
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "+ copy $TREE_DIR -> $RELEASE_TREE_DIR"
  else
    mkdir -p "$RELEASE_TREE_DIR"
    for f in reranker_meta.json reranker_features.txt reranker_experiment.json reranker_feature_importance.csv metrics.csv eval.csv; do
      [[ -f "$TREE_DIR/$f" ]] && cp "$TREE_DIR/$f" "$RELEASE_TREE_DIR/"
    done
    for d in "$TREE_DIR"/tree_*_fold*; do
      [[ -d "$d" ]] || continue
      rm -rf "$RELEASE_TREE_DIR/$(basename "$d")"
      cp -r "$d" "$RELEASE_TREE_DIR/"
    done
    echo "Copied tree reranker artifacts to $RELEASE_TREE_DIR"
  fi
fi
