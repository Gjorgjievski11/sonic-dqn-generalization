CONFIG ?= configs/dqn_sonic.yaml

.DEFAULT_GOAL := help

.PHONY: help
help:
	@echo ""
	@echo "  Sonic DQN Generalization"
	@echo "  ──────────────────────────────────────────"
	@echo "  make build-cpu      build CPU Docker image"
	@echo "  make build-gpu      build GPU Docker image"
	@echo "  make import-rom     import Sonic ROM into retro"
	@echo "  make train-cpu      train on CPU"
	@echo "  make train-gpu      train on GPU"
	@echo "  make evaluate       evaluate latest checkpoint"
	@echo ""

.PHONY: build-cpu
build-cpu:
	docker compose build train-cpu

.PHONY: build-gpu
build-gpu:
	docker compose build train-gpu

.PHONY: import-rom
import-rom:
	docker compose run --rm train-cpu python -m retro.import roms/

.PHONY: train-cpu
train-cpu:
	docker compose run --rm train-cpu

.PHONY: train-gpu
train-gpu:
	docker compose run --rm train-gpu

.PHONY: evaluate
evaluate:
	docker compose run --rm evaluate


