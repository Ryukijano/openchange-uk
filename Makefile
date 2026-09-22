.PHONY: lint test dry-run smoke train-small evaluate reproduce-table-1 check-sbatch

export PYTHONPATH := src

lint:
	python -m compileall -q src scripts tests

test:
	python -m unittest discover -s tests -v

check-sbatch:
	python scripts/check_sbatch.py

dry-run:
	python scripts/train.py --config configs/baseline.yaml --dry-run

evaluate:
	python scripts/evaluate.py --config configs/baseline.yaml

train-small: dry-run

reproduce-table-1:
	@echo "Table 1 does not exist yet. Run make test and make dry-run."

smoke:
	@echo "Submit this yourself on the login node after the SIF exists:"
	@echo "  sbatch slurm/00_smoke.sbatch"
	@echo "  squeue --me"
