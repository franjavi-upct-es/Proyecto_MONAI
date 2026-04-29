# lungseg Makefile (uv)
# =====================

SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.ONESHELL:
.NOTPARALLEL:
.DEFAULT_GOAL := help

UV ?= uv
UV_SYNC_FLAGS ?= --locked
EXTRAS ?= --extra dev --extra radiomics

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
MODEL ?= segresnet_lung
FOLD ?= 0

TASK06_ROOT ?= data/raw/Task06_Lung
TASK06_JSON ?= $(TASK06_ROOT)/dataset.json
SPLITS_DIR ?= data/splits
LIDC_MANIFEST ?= data/processed/lidc/nodule_manifest.csv

RUN_ID := $(shell date +%Y%m%d-%H%M%S)
OUTPUTS_ROOT ?= outputs/full-pipeline/$(RUN_ID)

# Variables de rendimiento (Punto Medio)
PHASE4_MAX_ITER ?=
PHASE4_VAL_EVERY ?=
PHASE4_PATIENCE ?=
PHASE4_CACHE_RATE ?= 1.0
PHASE4_CACHE_WORKERS ?= 2
PHASE4_NUM_WORKERS ?= 2
PHASE4_PIN_MEMORY ?= true
PHASE6_MAX_ITER ?=
PHASE6_VAL_EVERY ?=
PHASE6_PATIENCE ?=
SANITY_MAX_ITER ?= 20
SANITY_VAL_EVERY ?= 10

FRACTIONS ?= 0.25 0.5 1.0
AUGS ?= none standard
SEEDS ?= 0 1 2

CHECKPOINT ?=
IMAGE ?=
PRED_OUT ?= $(OUTPUTS_ROOT)/prediction.nii.gz
PHASE5_E2E ?= 0

.PHONY: help bootstrap install check-uv check-data check-splits doctor qa lint test smoke
.PHONY: splits phase4-fold phase4-all phase6 phase5 predict-one pipeline full summary

help:
	@printf "\nlungseg Makefile (uv)\n"
	printf "=====================\n\n"
	printf "Targets principales:\n"
	printf "  make bootstrap      Instala el entorno con uv (%s %s).\n" "$(UV_SYNC_FLAGS)" "$(EXTRAS)"
	printf "  make doctor         Comprueba uv, Task06 y splits.\n"
	printf "  make qa             Ejecuta ruff + pytest con uv.\n"
	printf "  make smoke          Run corto de sanity sobre un batch real.\n"
	printf "  make splits         Regenera data/splits/fold_{0..4}.json desde Task06.\n"
	printf "  make phase4-all     Entrena Phase 4 en folds: %s.\n" "$(PHASE4_FOLDS)"
	printf "  make phase6         Ejecuta sweep de ablacion en folds: %s.\n" "$(PHASE6_FOLDS)"
	printf "  make phase5         Clasificacion LIDC si existe %s; si no, salta.\n" "$(LIDC_MANIFEST)"
	printf "  make pipeline       Todo Task06 de principio a fin: bootstrap, splits, QA, Phase 4, Phase 6, Phase 5 opcional.\n"
	printf "  make summary        Muestra summaries bajo OUTPUTS_ROOT.\n\n"
	printf "Uso habitual:\n"
	printf "  make pipeline\n"
	printf "  make smoke\n"
	printf "  make phase4-fold FOLD=0\n"
	printf "  make phase6 PHASE6_FOLDS=\"0\" PHASE6_MAX_ITER=1000\n\n"
	printf "Variables utiles:\n"
	printf "  OUTPUTS_ROOT=%s\n" "$(OUTPUTS_ROOT)"
	printf "  TRAINING=%s MODEL=%s CONFIG=%s\n" "$(TRAINING)" "$(MODEL)" "$(CONFIG)"
	printf "  PHASE4_MAX_ITER=<vacio usa config> PHASE6_MAX_ITER=<vacio usa config>\n"
	printf "  PHASE4_NUM_WORKERS=%s PHASE4_PIN_MEMORY=%s PHASE4_CACHE_RATE=%s\n" "$(PHASE4_NUM_WORKERS)" "$(PHASE4_PIN_MEMORY)" "$(PHASE4_CACHE_RATE)"
	printf "  FRACTIONS=\"%s\" AUGS=\"%s\" SEEDS=\"%s\"\n\n" "$(FRACTIONS)" "$(AUGS)" "$(SEEDS)"
	printf "Aviso: make pipeline es largo por diseno: 5 folds x Phase 4 + sweep Phase 6.\n\n"

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
	if [[ "$$labels" -eq 0 ]]; then
		echo "ERROR: no hay etiquetas en $(TASK06_ROOT)/labelsTr"
		exit 1
	fi

check-splits:
	@echo "==> Comprobando splits en $(SPLITS_DIR)"
	missing=0
	for fold in $(FOLDS); do
		if [[ ! -f "$(SPLITS_DIR)/fold_$${fold}.json" ]]; then
			echo "Falta $(SPLITS_DIR)/fold_$${fold}.json"
			missing=1
		fi
	done
	if [[ "$$missing" -ne 0 ]]; then
		echo "Ejecuta: make splits"
		exit 1
	fi
	echo "Splits OK: $(FOLDS)"

doctor: check-uv check-data check-splits

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
	missing=0
	for fold in $(FOLDS); do
		if [[ ! -f "$(SPLITS_DIR)/fold_$${fold}.json" ]]; then
			echo "Falta $(SPLITS_DIR)/fold_$${fold}.json tras regenerar"
			missing=1
		fi
	done
	if [[ "$$missing" -ne 0 ]]; then exit 1; fi

smoke: check-data check-splits
	@echo "==> Sanity run corto"
	$(LUNGSEG) train --config-name=$(SANITY_CONFIG) \
		paths.outputs=$(OUTPUTS_ROOT)/sanity \
		data.cache.rate=0.0 \
		data.cache.num_workers=0 \
		training.sanity.max_iterations=$(SANITY_MAX_ITER) \
		training.sanity.val_every=$(SANITY_VAL_EVERY)

phase4-fold: check-data check-splits
	@echo "==> Phase 4 fold $(FOLD)"
	extra=()
	if [[ -n "$(PHASE4_MAX_ITER)" ]]; then extra+=("experiment.max_iterations=$(PHASE4_MAX_ITER)"); fi
	if [[ -n "$(PHASE4_VAL_EVERY)" ]]; then extra+=("experiment.val_every=$(PHASE4_VAL_EVERY)"); fi
	if [[ -n "$(PHASE4_PATIENCE)" ]]; then extra+=("experiment.patience=$(PHASE4_PATIENCE)"); fi
	$(LUNGSEG) train --config-name=$(CONFIG) \
		experiment=phase4_full \
		training=$(TRAINING) \
		model=$(MODEL) \
		fold=$(FOLD) \
		paths.outputs=$(OUTPUTS_ROOT)/phase4/fold_$(FOLD) \
		data.cache.rate=$(PHASE4_CACHE_RATE) \
		data.cache.num_workers=$(PHASE4_CACHE_WORKERS) \
		training.num_workers=$(PHASE4_NUM_WORKERS) \
		training.pin_memory=$(PHASE4_PIN_MEMORY) \
		"$${extra[@]}"

phase4-all: check-data check-splits
	@echo "==> Phase 4 en folds: $(PHASE4_FOLDS)"
	extra=()
	if [[ -n "$(PHASE4_MAX_ITER)" ]]; then extra+=("experiment.max_iterations=$(PHASE4_MAX_ITER)"); fi
	if [[ -n "$(PHASE4_VAL_EVERY)" ]]; then extra+=("experiment.val_every=$(PHASE4_VAL_EVERY)"); fi
	if [[ -n "$(PHASE4_PATIENCE)" ]]; then extra+=("experiment.patience=$(PHASE4_PATIENCE)"); fi
	for fold in $(PHASE4_FOLDS); do
		echo "---- Phase 4 fold $$fold ----"
		$(LUNGSEG) train --config-name=$(CONFIG) \
			experiment=phase4_full \
			training=$(TRAINING) \
			model=$(MODEL) \
			fold=$$fold \
			paths.outputs=$(OUTPUTS_ROOT)/phase4/fold_$$fold \
			data.cache.rate=$(PHASE4_CACHE_RATE) \
			data.cache.num_workers=$(PHASE4_CACHE_WORKERS) \
			training.num_workers=$(PHASE4_NUM_WORKERS) \
			training.pin_memory=$(PHASE4_PIN_MEMORY) \
			"$${extra[@]}"
	done

phase6: check-data check-splits
	@echo "==> Phase 6 ablation"
	extra=()
	if [[ -n "$(PHASE6_MAX_ITER)" ]]; then extra+=("experiment.max_iterations=$(PHASE6_MAX_ITER)"); fi
	if [[ -n "$(PHASE6_VAL_EVERY)" ]]; then extra+=("experiment.val_every=$(PHASE6_VAL_EVERY)"); fi
	if [[ -n "$(PHASE6_PATIENCE)" ]]; then extra+=("experiment.patience=$(PHASE6_PATIENCE)"); fi
	for fold in $(PHASE6_FOLDS); do
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
						paths.outputs=$(OUTPUTS_ROOT)/phase6/fold_$$fold \
						"$${extra[@]}"
				done
			done
		done
	done

phase5:
	@echo "==> Phase 5 LIDC radiomics/classification"
	if [[ ! -f "$(LIDC_MANIFEST)" ]]; then
		echo "Saltando Phase 5: no existe $(LIDC_MANIFEST)."
		echo "Task06 no tiene etiquetas benigno/maligno; Phase 5 requiere LIDC-IDRI."
		exit 0
	fi
	extra=()
	if [[ "$(PHASE5_E2E)" == "1" ]]; then extra+=("--e2e"); fi
	$(LUNGSEG) classify --config-name=$(CONFIG) \
		"$${extra[@]}" \
		paths.outputs=$(OUTPUTS_ROOT)/phase5

predict-one: check-uv
	@if [[ -z "$(CHECKPOINT)" || -z "$(IMAGE)" ]]; then
		echo "Uso: make predict-one CHECKPOINT=outputs/.../best.pt IMAGE=data/raw/Task06_Lung/imagesTr/lung_001.nii.gz"
		exit 2
	fi
	mkdir -p "$$(dirname "$(PRED_OUT)")"
	$(LUNGSEG) predict --config-name=$(CONFIG) \
		--checkpoint "$(CHECKPOINT)" \
		--image "$(IMAGE)" \
		--output "$(PRED_OUT)"

pipeline: bootstrap splits qa phase4-all phase6 phase5 summary

full: pipeline

summary:
	@echo "==> Summaries en $(OUTPUTS_ROOT)"
	if [[ ! -d "$(OUTPUTS_ROOT)" ]]; then
		echo "No existe $(OUTPUTS_ROOT)."
		exit 0
	fi
	found=0
	while IFS= read -r path; do
		found=1
		echo
		echo "---- $$path ----"
		sed -n '1,120p' "$$path"
	done < <(find "$(OUTPUTS_ROOT)" -name summary.json -print | sort)
	if [[ "$$found" -eq 0 ]]; then
		echo "No hay summary.json todavia."
	fi
