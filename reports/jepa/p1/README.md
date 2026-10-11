# P1 synthetic evidence

These are CPU plumbing checks, not EO research results. No imagery was fetched,
no source was approved and no cluster job ran. The manifests retain the pre-commit
git SHA and dirty state; they do not claim the implementation existed at that SHA.

- `test-final.log`: 49 JEPA tests passed, including Gloo gradients and updates.
- `check-final.log`: torch-free base checks passed; torch tests skipped.
- `fit-seed0/`: original 50-step ViT-Ti synthetic run (32 items, 4 CPU threads).
- `resume-seed0/`: steps 26-50 resumed from the original step-25 checkpoint.

Runtime was 79.81 seconds; 284,210 post-drop encoder tokens were counted. No
non-finite events occurred. Resume numerical metrics agreed exactly, excluding
wall times and RSS. Checkpoints are not committed.

The original fit's checkpoint index lists step 50 twice. This evidence is retained
unchanged. The trainer's duplicate final-save bug was subsequently fixed and tested.

Run locally from the repo root (x86 CPU only; Isambard uses its NGC container):

```bash
uv venv --python 3.11 .venv-jepa
uv pip install --python .venv-jepa/bin/python --torch-backend=cpu -e '.[dev,jepa]'
make test-jepa PYTHON=.venv-jepa/bin/python
make jepa-synthetic-fit PYTHON=.venv-jepa/bin/python
```

Exact restart requires the same resolved config, torch version and CPU threads.
Training arrays are synthetic only. Real-data adapters are not implemented.
