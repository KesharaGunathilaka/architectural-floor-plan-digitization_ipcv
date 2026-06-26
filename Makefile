# ── Architectural Floor Plan Digitization ───────────────────────────────────
# Usage: make <target>
# Run `make help` to see all available commands.

.PHONY: help install install-dev setup audit-svg dataset-stats select-data \
        convert verify preprocess train tune evaluate postprocess \
        test lint clean sync-ubuntu

# Default target
help:
	@echo ""
	@echo "Architectural Floor Plan Digitization — Project Commands"
	@echo "─────────────────────────────────────────────────────────"
	@echo "  make install        Install production dependencies"
	@echo "  make install-dev    Install all dependencies (+ dev tools)"
	@echo "  make setup          Full first-time setup"
	@echo ""
	@echo "  make audit-svg      Phase 1: Audit SVG structure"
	@echo "  make dataset-stats  Phase 1: Compute dataset statistics"
	@echo "  make select-data    Phase 1: Select dataset subset"
	@echo "  make convert        Phase 2: Convert SVG → YOLO labels"
	@echo "  make verify         Phase 2: Verify annotations visually"
	@echo "  make preprocess     Phase 3: Preprocess images"
	@echo "  make train          Phase 5: Train the model"
	@echo "  make tune           Phase 5: Hyperparameter tuning sweep"
	@echo "  make evaluate       Phase 6: Evaluate on test set"
	@echo "  make postprocess    Phase 7: Export detections to JSON/DXF"
	@echo ""
	@echo "  make test           Run unit tests"
	@echo "  make lint           Run code linter (ruff)"
	@echo "  make clean          Remove generated artifacts"
	@echo "  make sync-ubuntu    Sync project to Ubuntu training machine"
	@echo ""

# ── Environment Setup ────────────────────────────────────────────────────────

install:
	pip install -r requirements.txt
	pip install -e .

install-dev:
	pip install -r requirements-dev.txt
	pip install -e .

setup: install-dev
	@if not exist .env (copy .env.example .env)
	@echo "✓ Setup complete. Edit .env with your dataset paths."

# ── Pipeline Stages ──────────────────────────────────────────────────────────

audit-svg:
	python src/data/audit_svg.py

dataset-stats:
	python src/data/dataset_stats.py

select-data:
	python src/data/select_subset.py

convert:
	python src/data/convert_annotations.py

verify:
	python src/data/verify_annotations.py

preprocess:
	python src/preprocessing/preprocess_pipeline.py

train:
	python src/training/train_yolo.py

tune:
	python src/training/hyperparameter_tuning.py

evaluate:
	python src/evaluation/evaluate_test.py

postprocess:
	python src/export/export_cad.py

# ── Quality Assurance ────────────────────────────────────────────────────────

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

lint:
	ruff check src/ tests/

# ── Utilities ────────────────────────────────────────────────────────────────

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -delete
	rm -rf htmlcov/ .coverage

sync-ubuntu:
	@echo "Syncing project to Ubuntu machine (edit UBUNTU_HOST in .env first)"
	rsync -avz --exclude='data/raw/' --exclude='data/processed/' \
	      --exclude='runs/' --exclude='.venv/' \
	      ./ $(UBUNTU_USER)@$(UBUNTU_HOST):$(UBUNTU_PATH)