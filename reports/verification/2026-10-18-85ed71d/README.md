# Verification of 85ed71d

Tested commit: `85ed71d2a6e0439769944d396547a0d0d36f896d`.

This is local code verification, **not** execution of the proposed Gate 1 or
Gate 2 training experiments. No new Slurm job was submitted.

## Executed

The complete tracked tree was checked out in a clean, detached Git worktree.
The existing CPU Torch environment was reused:

```bash
make check PYTHON=/home/ubuntu/repos/openchange-uk/.venv-jepa/bin/python
```

- Exit status: 0.
- Unit tests: **185 discovered, 183 passed, 2 skipped** (CUDA unavailable).
- Ruff, compileall, all nine Slurm-template checks, source validation, and the
  baseline/JEPA dry-runs completed successfully.
- Baseline evaluation correctly refused to report scores for the provisional
  split; the dry-runs are plumbing checks, not training or model-quality results.
- [Full command output](make-check.log), SHA-256:
  `6d9507bcb6d3e17ead9d681436c36f80954320c5c0bbc26f8b62dcc12968464f`.

Environment: Linux x86_64, Python 3.11.17, torch 2.14.1+cpu, NumPy 2.4.6,
PyYAML 6.0.3, Ruff 0.17.0. No GPU was available locally.

The earlier count of 193 includes eight tests from an untracked coastal-inventory
test module. Those unrelated files were not included in this verification or
publication. The original working tree was left untouched.

## Proposed GPU gates: not executed

### Gate 1: four-arm EO objective comparison

The approved 480-pair `pilot_v1` and the 26.09 SIF are present on Isambard, and
certificate-based SSH access worked during inspection. However:

- `scripts/train_jepa_distributed.py` still calls only `losses.jepa_loss`.
  It has no objective selector or training path for the split-subspace/flow,
  TESSERA mix-up, or VICReg arms.
- No four-arm experiment configuration or post-training evaluation runner is
  present. The proposed criteria list diagnostics but do not define numerical
  acceptance thresholds. Prefix classification retention also needs labels and
  a fixed evaluation protocol, not merely the training-pair manifest.
- `lejepa_identifiability_bound` emits a diagnostic calculation. Computing it on
  S1/S2 embeddings alone does not establish the latent-transition assumptions of
  the cited theorem or validate a recovery guarantee on real EO data.

Repeating the existing baseline job would not test Gate 1.

### Gate 2: WAM training and closed-loop control

- The approved source registry has SSL4EO pilot sources only, not Push-T or
  RoboCasa365. The inspected cluster data directory contains `pilot_v1` only.
- No end-to-end WAM training, simulator evaluation, or matched-history experiment
  runner is present in `scripts/`.
- `slurm/06_wam_and_science_gpu.sbatch` reserves four GPUs but invokes ordinary
  unit tests; it is not a four-rank WAM trainer or closed-loop benchmark.

Gate 2 therefore needs an explicitly approved, pinned robotics pilot and an
evaluation protocol/runner before it can be executed. No datasets were fetched,
held-out groups changed, or allocation hours spent on substitute experiments.
