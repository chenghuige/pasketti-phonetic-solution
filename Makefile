# =============================================================================
# Pasketti Phonetic ASR — DrivenData prize-winner reproduction
# =============================================================================
# All paths are repo-relative. Targets assume CUDA is available.
# Edit DATA_DIR below if your competition data lives elsewhere.
# =============================================================================
PYTHON   ?= python
GPU      ?= 0
DATA_DIR ?= ../input/pasketti
WORKING  ?= ./working
RUN      ?= 17

.PHONY: help setup data smoke train-fold0 train-online ensemble pack notebook clean

help:
	@echo "Targets:"
	@echo "  setup        pip install -r requirements.txt"
	@echo "  data         show instructions for staging competition data"
	@echo "  smoke        import-only sanity check (no GPU)"
	@echo "  train-fold0  CUDA_VISIBLE_DEVICES=$(GPU) train v17 on fold 0"
	@echo "  train-online CUDA_VISIBLE_DEVICES=$(GPU) train v17 on full data"
	@echo "  ensemble     run cross-model rescore + CatBoost reranker"
	@echo "  pack         build submission.zip from src/models.txt"
	@echo "  notebook     launch JupyterLab"

setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

data:
	@echo "Place the official competition data under: $(DATA_DIR)"
	@echo "Expected layout:"
	@echo "  $(DATA_DIR)/audio/<id>.wav"
	@echo "  $(DATA_DIR)/train.csv"
	@echo "  $(DATA_DIR)/submission_format.csv"

smoke:
	@PYTHONPATH=src/_compat:$$PYTHONPATH $(PYTHON) -c \
	  "from gezi.common import *; from src import config; print('imports OK')"

train-fold0:
	cd src && PYTHONPATH=_compat:$$PYTHONPATH \
	  CUDA_VISIBLE_DEVICES=$(GPU) $(PYTHON) train.py \
	    --flagfile=flags/v17 --mn=v17.fold0 --fold=0

train-online:
	cd src && PYTHONPATH=_compat:$$PYTHONPATH \
	  CUDA_VISIBLE_DEVICES=$(GPU) $(PYTHON) train.py \
	    --flagfile=flags/v17 --mn=v17.online --online

ensemble:
	cd src && PYTHONPATH=_compat:$$PYTHONPATH \
	  CUDA_VISIBLE_DEVICES=$(GPU) $(PYTHON) ensemble.py \
	    --models_file=models.txt

pack:
	bash scripts/pack_submission.sh ensemble src/models.txt submission.zip

notebook:
	$(PYTHON) -m jupyterlab notebooks/

clean:
	rm -rf $(WORKING)/tmp __pycache__ src/__pycache__ src/_compat/*/__pycache__
