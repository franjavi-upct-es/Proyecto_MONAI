# lungseg Makefile (uv)
# =====================

SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.ONESHELL:
.NOTPARALLEL:
.DEFAULT_GOAL := help

UV ?= uv
UV_SYNC_FLAGS ?= --locked
EXTRAS ?= --all-extras

PYTHON := $(UV) run python
LUNGSEG := $(UV) run python -m lungseg.cli

CONFIG ?= config
SANITY_CONFIG ?= sanity
SEED ?= 42
N_FOLDS ?= 5
FOLDS ?= 0 1 2 3 4
PHASE4_FOLDS ?= $(FOLDS)
PHASE6_FOLDS ?= $(FOLDS)

TRAINING ?= local_5060
MODEL ?= vit_25d_lung
FOLD ?= 0

# Estrategia de freeze/unfreeze del encoder MAE pre-entrenado.
# `-1` (default) = encoder congelado durante toda la ejecución.
UNFREEZE_EPOCH ?= -1
UNFREEZE_LR_FACTOR ?= 0.1

TASK06_ROOT ?= data/raw/Task06_Lung
TASK06_JSON ?= $(TASK06_ROOT)/dataset.json
SPLITS_DIR ?= data/splits
LUNG_MASK_DIR ?= data/cache/lung_masks
LIDC_MANIFEST ?= data/processed/lidc/nodule_manifest.csv

RUN_ID := $(shell date +%Y%m%d-%H%M%S)
OUTPUTS_ROOT ?= outputs/full-pipeline/$(RUN_ID)

# Variables de rendimiento Phase 4 / Phase 6.
PHASE4_MAX_ITER ?=
PHASE4_VAL_EVERY ?=
PHASE4_PATIENCE ?=
PHASE4_RESUME_FROM ?=
PHASE4_SW_BATCH_SIZE ?=
PHASE4_CACHE_MODE ?= ram
PHASE4_CACHE_RATE ?= 1.0
PHASE4_CACHE_WORKERS ?= 4
PHASE4_NUM_WORKERS ?= 4
PHASE4_PIN_MEMORY ?= true
PHASE6_MAX_ITER ?=
PHASE6_VAL_EVERY ?=
PHASE6_PATIENCE ?=
PHASE6_SW_BATCH_SIZE ?=
PHASE6_CACHE_MODE ?= ram
SANITY_MAX_ITER ?= 20
SANITY_VAL_EVERY ?= 10

FRACTIONS ?= 0.25 0.5 1.0
AUGS ?= none standard
SEEDS ?= 0 1 2

CHECKPOINT ?=
IMAGE ?=
PRED_OUT ?= $(OUTPUTS_ROOT)/prediction.nii.gz
PHASE5_E2E ?= 0

.PHONY: help bootstrap install check-uv check-data check-splits check-lung-masks doctor qa lint test
.PHONY: smoke smoke-unfreeze splits precompute-masks phase4-fold phase4-all phase6 phase5
.PHONY: predict-one pipeline full summary

help:
	@printf "\nlungseg Makefile (uv) — pipeline 2.5D ViT con prior pulmonar\n"
	printf "===========================================================\n\n"
	printf "Targets principales:\n"
	printf "  make bootstrap          Sincroniza el entorno con uv (%s %s).\n" "$(UV_SYNC_FLAGS)" "$(EXTRAS)"
	printf "  make doctor             Comprueba uv, Task06, splits y máscaras pulmonares.\n"
	printf "  make qa                 Ejecuta ruff + pytest con uv.\n"
	printf "  make precompute-masks   Pre-computa máscaras pulmonares (lungmask R231) en %s.\n" "$(LUNG_MASK_DIR)"
	printf "  make smoke              Sanity run corto: overfit de 1 batch.\n"
	printf "  make smoke-unfreeze     Smoke con encoder unfreeze tras 2 epochs.\n"
	printf "  make splits             Regenera data/splits/fold_{0..4}.json desde Task06.\n"
	printf "  make phase4-fold FOLD=N Entrena Phase 4 sobre el fold N.\n"
	printf "  make phase4-all         Entrena Phase 4 en folds: %s.\n" "$(PHASE4_FOLDS)"
	printf "  make phase6             Sweep de ablación en folds: %s.\n" "$(PHASE6_FOLDS)"
	printf "  make pipeline           Todo Task06 de principio a fin (incluye máscaras).\n"
	printf "  make summary            Muestra summaries bajo OUTPUTS_ROOT.\n\n"
	printf "Pipeline mínimo de Phase 4:\n"
	printf "  make bootstrap && make splits && make precompute-masks\n"
	printf "  make phase4-fold FOLD=0\n\n"
	printf "Variables útiles:\n"
	printf "  MODEL=%s\n" "$(MODEL)"
	printf "  TRAINING=%s\n" "$(TRAINING)"
	printf "  UNFREEZE_EPOCH=%s UNFREEZE_LR_FACTOR=%s\n" "$(UNFREEZE_EPOCH)" "$(UNFREEZE_LR_FACTOR)"
	printf "  PHASE4_CACHE_MODE=%s (ram|disk|none)\n" "$(PHASE4_CACHE_MODE)"
	printf "  OUTPUTS_ROOT=%s\n" "$(OUTPUTS_ROOT)"

check-uv:
	@command -v "$(UV)" >/dev/null || { echo "ERROR: uv no esta instalado o no esta en PATH."; exit 1; }
	$(UV) --version

install: check-uv
	@echo "==> Sincronizando entorno uv"
	$(UV) sync $(UV_SYNC_FLAGS) $(EXTRAS)

bootstrap: install

check-data:
	@echo "==> Comprobando dataset Task06 en $(TASK06_ROOT)"
	test -f "$(TASK06_JSON)" || { echo "ERROR: falta $(TASK06_JSON)"; exit 1; }
	test -d "$(TASK06_ROOT)/imagesTr" || { echo "ERROR: falta $(TASK06_ROOT)/imagesTr"; exit 1; }
	test -d "$(TASK06_ROOT)/labelsTr" || { echo "ERROR: falta $(TASK06_ROOT)/labelsTr"; exit 1; }
	images=$$(find "$(TASK06_ROOT)/imagesTr" -maxdepth 1 -name '*.nii.gz' | wc -l | tr -d ' ')
	labels=$$(find "$(TASK06_ROOT)/labelsTr" -maxdepth 1 -name '*.nii.gz' | wc -l | tr -d ' ')
	echo "Task06 OK: $$images imagenes de train, $$labels etiquetas."

check-splits:
	@echo "==> Comprobando splits en $(SPLITS_DIR)"
	missing=0
	for fold in $(FOLDS); do
		if [[ ! -f "$(SPLITS_DIR)/fold_$${fold}.json" ]]; then
			echo "Falta $(SPLITS_DIR)/fold_$${fold}.json"
			missing=1
		fi
	done
	if [[ "$$missing" -ne 0 ]]; then exit 1; fi

check-lung-masks:
	@echo "==> Comprobando máscaras pulmonares en $(LUNG_MASK_DIR)"
	if [[ ! -d "$(LUNG_MASK_DIR)" ]]; then
		echo "ERROR: $(LUNG_MASK_DIR) no existe. Ejecuta 'make precompute-masks' primero."
		exit 1
	fi
	count=$$(find "$(LUNG_MASK_DIR)" -maxdepth 1 -name '*.nii.gz' | wc -l | tr -d ' ')
	if [[ "$$count" -eq 0 ]]; then
		echo "ERROR: no hay máscaras en $(LUNG_MASK_DIR). Ejecuta 'make precompute-masks'."
		exit 1
	fi
	echo "Máscaras pulmonares OK: $$count volúmenes cacheados."

doctor: check-uv check-data check-splits check-lung-masks

lint: check-uv
	@echo "==> Ruff"
	$(UV) run ruff check src tests

test: check-uv
	@echo "==> Pytest"
	$(UV) run pytest tests/ -q

qa: lint test

splits: check-uv check-data
	@echo "==> Regenerando splits patient-level"
	$(PYTHON) -c "from pathlib import Path; from lungseg.data.splits import make_splits; paths = make_splits(Path('$(TASK06_JSON)'), Path('$(SPLITS_DIR)'), seed=$(SEED), k=$(N_FOLDS)); print('Splits generados:'); [print(' -', p) for p in paths]"

precompute-masks: check-uv check-data check-splits
	@echo "==> Pre-computando máscaras pulmonares (lungmask R231) en $(LUNG_MASK_DIR)"
	$(LUNGSEG) precompute-lung-masks

smoke: check-data check-splits check-lung-masks
	@echo "==> Sanity run corto (overfit 1 batch)"
	$(LUNGSEG) train --config-name=$(SANITY_CONFIG) \
		paths.outputs=$(OUTPUTS_ROOT)/sanity \
		data.cache.rate=0.0 \
		data.cache.num_workers=0 \
		training.sanity.max_iterations=$(SANITY_MAX_ITER) \
		training.sanity.val_every=$(SANITY_VAL_EVERY)

smoke-unfreeze: check-data check-splits check-lung-masks
	@echo "==> Smoke con unfreeze del encoder en epoch 2 ($(MODEL))"
	$(LUNGSEG) train --config-name=$(SANITY_CONFIG) \
		model=$(MODEL) \
		training.unfreeze_epoch=2 \
		training.unfreeze_lr_factor=0.5 \
		paths.outputs=$(OUTPUTS_ROOT)/smoke-unfreeze \
		data.cache.rate=0.0 \
		training.sanity.max_iterations=10 \
		training.sanity.val_every=5

phase4-fold: check-data check-splits check-lung-masks
	@echo "==> Phase 4 fold $(FOLD) con $(MODEL)"
	extra=()
	if [[ -n "$(PHASE4_MAX_ITER)" ]]; then extra+=("experiment.max_iterations=$(PHASE4_MAX_ITER)"); fi
	if [[ -n "$(PHASE4_VAL_EVERY)" ]]; then extra+=("experiment.val_every=$(PHASE4_VAL_EVERY)"); fi
	if [[ -n "$(PHASE4_RESUME_FROM)" ]]; then extra+=("training.resume_from=$(PHASE4_RESUME_FROM)"); fi
	$(LUNGSEG) train --config-name=$(CONFIG) \
		experiment=phase4_full \
		training=$(TRAINING) \
		model=$(MODEL) \
		fold=$(FOLD) \
		training.unfreeze_epoch=$(UNFREEZE_EPOCH) \
		training.unfreeze_lr_factor=$(UNFREEZE_LR_FACTOR) \
		paths.outputs=$(OUTPUTS_ROOT)/phase4/fold_$(FOLD) \
		data.cache.mode=$(PHASE4_CACHE_MODE) \
		data.cache.rate=$(PHASE4_CACHE_RATE) \
		"$${extra[@]}"

phase4-all: check-data check-splits check-lung-masks
	@for fold in $(PHASE4_FOLDS); do
		$(MAKE) phase4-fold FOLD=$$fold
	done

phase6: check-data check-splits check-lung-masks
	@for fold in $(PHASE6_FOLDS); do
		for frac in $(FRACTIONS); do
			for aug in $(AUGS); do
				for seed in $(SEEDS); do
					echo "---- Phase 6 fold=$$fold fraction=$$frac aug=$$aug seed=$$seed ----"
					$(LUNGSEG) ablate --config-name=$(CONFIG) \
						training=$(TRAINING) \
						model=$(MODEL) \
						fold=$$fold \
						data_fraction=$$frac \
						aug_regime=$$aug \
						seed=$$seed \
						training.unfreeze_epoch=$(UNFREEZE_EPOCH) \
						paths.outputs=$(OUTPUTS_ROOT)/phase6/fold_$$fold \
						data.cache.mode=$(PHASE6_CACHE_MODE)
				done
			done
		done
	done

phase5:
	@echo "==> Phase 5 LIDC radiomics/classification"
	if [[ ! -f "$(LIDC_MANIFEST)" ]]; then exit 0; fi
	$(LUNGSEG) classify --config-name=$(CONFIG) paths.outputs=$(OUTPUTS_ROOT)/phase5

predict-one: check-uv
	@if [[ -z "$(CHECKPOINT)" || -z "$(IMAGE)" ]]; then exit 2; fi
	mkdir -p "$$(dirname "$(PRED_OUT)")"
	$(LUNGSEG) predict --checkpoint "$(CHECKPOINT)" --image "$(IMAGE)" --output "$(PRED_OUT)"

pipeline: bootstrap splits precompute-masks qa phase4-all phase6 phase5 summary

full: pipeline

summary:
	@find "$(OUTPUTS_ROOT)" -name summary.json -print | sort | xargs cat
