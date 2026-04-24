# Pasketti Phonetic ASR — DrivenData prize-winner solution

> Reproduction code for the **Pasketti Phonetic** track of the
> DrivenData [Pasketti Speech Recognition Challenge](https://www.drivendata.org/competitions/),
> public LB **0.2539**, private LB **0.2559**.

This repository is intentionally minimal: it bundles the exact training
and inference code used to produce the leaderboard score, packaged so it
can be run end-to-end without any of the author's internal libraries. A
small compatibility layer under `src/_compat/` provides just enough of
the `gezi` / `melt` / `lele` interface that the project files expect, so
the model code itself is unchanged from the development repository.

The deeper write-up of the modeling choices is in
[`docs/SOLUTION.md`](docs/SOLUTION.md). The word-track solution is **not**
included.

---

## 1. Solution at a glance

| Component                  | Choice                                                           |
| -------------------------- | ---------------------------------------------------------------- |
| Acoustic backbones         | NeMo Parakeet-TDT-0.6B (TDT + CTC), WavLM-Large (CTC)            |
| Output units               | IPA phoneme set (dual-head IPA + word-BPE during training)       |
| Augmentation               | SpecAugment, concat-mix (up to 8 clips), light noise overlay     |
| Decoder                    | Beam-search CTC + TDT, top-10 N-best per model                   |
| Model averaging            | EMA (decay 0.999) saved as the final checkpoint                  |
| Ensemble                   | 11 models → cross-model CTC log-prob rescore → CatBoost LambdaRank |
| Final reranker             | CatBoost, 5-fold, ≈200 features                                  |

The final ensemble model list is in [`src/models.txt`](src/models.txt).

---

## 2. Repository layout

```
pasketti-phonetic-solution/
├── Makefile               # one-command targets (setup / train / ensemble / pack)
├── Dockerfile             # mirrors the DrivenData runtime (for local end-to-end tests)
├── requirements.txt
├── docs/SOLUTION.md       # detailed methodology
├── notebooks/
│   └── 02_run_inference.ipynb   # all-in-one single-model demo
├── scripts/
│   ├── pack_submission.sh       # build submission.zip
│   ├── _resolve_models.py       # resolve names in models.txt to dirs
│   └── sync_core_from_pikachu.sh# (maintainer-only) re-sync core files
├── src/
│   ├── train.py           # standalone training entry (was main.py upstream)
│   ├── train_loop.py      # hand-written AMP / EMA / cosine-LR loop
│   ├── config.py / config_base.py   # absl flag definitions
│   ├── dataset.py         # data + collate + bucket sampler
│   ├── eval.py            # IPA CER metric (matches official scorer)
│   ├── ctc_decode.py      # beam search
│   ├── submit.py          # Docker-runtime entry (renamed to main.py at pack time)
│   ├── ensemble.py        # cross-model rescore + CatBoost reranker
│   ├── tree_reranker/     # saved CatBoost reranker artifacts for final inference
│   ├── models/            # base.py + nemo.py + wav2vec2.py
│   ├── flags/             # versioned flag files (base, v8 … v17)
│   ├── models.txt         # names of the 11 models in the final ensemble
│   └── _compat/           # tiny gezi / melt / lele / husky shims
└── working/               # populated by training (model checkpoints, logs, metrics)
```

`src/_compat/` is the only piece of "infrastructure" code in this
repository — it implements about ~400 lines of helpers (a `Globals`
singleton, EMA-aware checkpoint loader, length-bucketed sampler, etc.)
so that the project files can keep using their original imports
(`from gezi.common import *`, `import lele as le`, `melt.init`).

> **Why `absl.flags` instead of a plain `class FLAGS`?** The training
> code defines roughly 500 flags split across `config_base.py` and
> `config.py` with default-overrides per `flags/v*` file. Switching to a
> hand-rolled config object would have meant rewriting every flag file
> as well. Using absl keeps the surface identical to the development
> setup while remaining a single ~30-line dependency.

---

## 3. Reproducing the result

### 3.1 Hardware

* GPU with ≥ 24 GB VRAM (training was done on a single A100-80GB).
* ~80 GB disk for raw audio + intermediate features.
* CUDA 12.1 / 12.4 with cuDNN.

### 3.2 Environment

```bash
make setup          # pip install -r requirements.txt
make data           # prints the expected layout under ../input/pasketti
make smoke          # import-only sanity check, no GPU required
```

The competition data is expected under `../input/pasketti/` (audio in
`audio/<id>.wav`, plus `train.csv` and `submission_format.csv`). Symlink
or set `DATA_DIR` to point elsewhere.

### 3.3 Train a single model (fold 0)

```bash
make train-fold0 GPU=1            # -> working/offline/17/v17.fold0/0/
```

Each `flags/v*` file is incremental: `v17` chains all the way back to
`base` via `--flagfile`. The full ensemble retrains the same recipe with
different backbones and seeds — see `src/models.txt` for the exact list
and use the corresponding `--flagfile` and `--mn` from each line's
suffix.

For the production submission, every model is also re-trained with
`--online` (no held-out fold) using `make train-online`.

### 3.4 Build the ensemble + reranker

After all 11 models in `src/models.txt` are trained:

```bash
make ensemble                     # cross-model rescore + CatBoost reranker
make pack                         # bundles submission.zip from src/models.txt
```

The tree reranker code is already included in the repository:

* `src/ensemble.py` trains the CatBoost reranker and writes the saved tree artifacts.
* `src/reranker_features.py` builds the online/offline feature frame.
* `src/submit.py` loads the packed tree model(s) at inference time.

For the final leaderboard submission, the saved reranker artifacts must be
available under `src/tree_reranker/` before `make pack` is run.

`make pack` copies `submit.py` to `main.py` (the runtime entry expected
by the DrivenData container), tarballs `src/_compat/` as
`pikachu_utils.tar.gz`, and zips everything together with the model
weight directories. If `src/tree_reranker/` exists, it is copied into the
submission bundle as well. The runtime extracts the tar onto `sys.path`
automatically — no edits to `submit.py` are needed.

---

## 4. Pre-trained weights

> **Not published yet.** Weight checkpoints (model.pt + flags.json per
> model directory) are required to reproduce the final leaderboard score
> without retraining. Until a public download URL is available, the only
> fully reproducible path is to retrain all 11 models from scratch
> (~1 GPU-day per model on an A100).

When the weights are published, place each one under:

```
working/online/17/<model_name>/model.pt
working/online/17/<model_name>/flags.json
working/online/17/<model_name>/nemo_model_slim.nemo   # NeMo backbones only

src/tree_reranker/reranker_meta.json
src/tree_reranker/reranker_features.txt
src/tree_reranker/reranker_experiment.json
src/tree_reranker/tree_cb_fold0/model.pkl
src/tree_reranker/tree_cb_fold1/model.pkl
src/tree_reranker/tree_cb_fold2/model.pkl
src/tree_reranker/tree_cb_fold3/model.pkl
src/tree_reranker/tree_cb_fold4/model.pkl
```

where `<model_name>` matches an entry in `src/models.txt`.

Recommended publication layout:

* Put the large neural-network checkpoints in a separate Hugging Face model
  repository, for example `huigecheng/pasketti-phonetic-weights`, then tell
  users to download/symlink them into `working/online/17/`.
* Keep this GitHub repository focused on code plus the small reranker
  artifacts. The CatBoost reranker files are tiny compared with the ASR
  checkpoints, so committing `src/tree_reranker/` directly to GitHub is fine.
* If you prefer to keep everything out of git, store `src/tree_reranker/` in
  the same cloud bundle as the model weights and extract it into the repo
  before running `make pack`.

Recommended Hugging Face repo layout:

```
online/17/<model_name>/model.pt
online/17/<model_name>/flags.json
online/17/<model_name>/nemo_model_slim.nemo
tree_reranker/reranker_meta.json
tree_reranker/reranker_features.txt
tree_reranker/reranker_experiment.json
tree_reranker/tree_cb_fold0/model.pkl
tree_reranker/tree_cb_fold1/model.pkl
tree_reranker/tree_cb_fold2/model.pkl
tree_reranker/tree_cb_fold3/model.pkl
tree_reranker/tree_cb_fold4/model.pkl
```

Once the Hugging Face repo is live, the intended download flow is:

```bash
python -m pip install -r requirements.txt
HF_REPO_ID=huigecheng/pasketti-phonetic-weights bash scripts/download_weights.sh
make pack
```

To stage and upload the exact final 11-model bundle from the original training
workspace to Hugging Face, use:

```bash
HF_REPO_ID=huigecheng/pasketti-phonetic-weights UPLOAD_NOW=1 bash scripts/upload_hf_weights.sh
```

If you want to keep the reranker artifacts in GitHub instead of Hugging Face,
commit the whole `src/tree_reranker/` directory and the download script will
still work for the neural-network checkpoints only.

---

## 5. License

* Project source code: MIT (see `LICENSE`).
* Pre-trained NeMo Parakeet-TDT-0.6B and WavLM-Large weights: see their
  upstream license terms (CC-BY-NC for Parakeet, MIT for WavLM).
