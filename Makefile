.PHONY: lint test dry-run smoke train-small evaluate reproduce-table-1 check-sbatch validate-sources env-info check

export PYTHONPATH := src
PYTHON ?= python3

lint:
	$(PYTHON) -m ruff check src scripts tests .cursor/hooks
	$(PYTHON) -m compileall -q src scripts tests

test:
	$(PYTHON) -m unittest discover -s tests -v

check-sbatch:
	$(PYTHON) scripts/check_sbatch.py

validate-sources:
	$(PYTHON) scripts/validate_sources.py --config configs/sources.yaml

dry-run:
	$(PYTHON) scripts/train.py --config configs/baseline.yaml --dry-run --output-root outputs

evaluate:
	$(PYTHON) scripts/evaluate.py --config configs/baseline.yaml

env-info:
	$(PYTHON) -m openchange.provenance

# Everything that runs without Isambard, GPUs, or network access.
check: lint test check-sbatch validate-sources dry-run evaluate

train-small: dry-run

reproduce-table-1:
	@echo "Table 1 does not exist yet. Run make check."

smoke:
	@echo "Submit this yourself on the login node after the SIF exists:"
	@echo "  sbatch slurm/00_smoke.sbatch"
	@echo "  squeue --me"
