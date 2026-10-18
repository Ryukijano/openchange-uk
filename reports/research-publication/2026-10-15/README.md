# Research publication bundle — 15 October 2026

## Scope

Publish the EO research history and coastal evidence in one navigable place,
including negative outcomes. Start with [the progress summary](../../../docs/PROGRESS.md)
and [reading guide](../../../docs/research/eo-task-research/README.md).

Included: authored synthesis notes, available independent reviewer reports,
self-contained HTML/PDF visuals, historical experiment scripts, protocols,
aggregate metrics, available original logs, environments, source manifests and
licence/acquisition evidence. Older programme/JEPA work already on GitHub is
linked rather than duplicated. The separate
[implementation PR #2](https://github.com/Ryukijano/openchange-uk/pull/2) is not
merged or republished as part of this PR.

Excluded deliberately: survey NetCDF/ZIP payloads; crop-label GeoJSON and
per-parcel data; embedding arrays; derived residual/prediction CSVs; weights;
third-party source clones and downloaded paper/web-page dumps; caches/virtual
environments; intermediate HTML body fragments; internal working-state notes;
duplicate screenshots/patch archives; and the pre-existing unrelated untracked
`reports/cuda/` directory. Source links, checksums and environments are retained
instead. No secret is required to read or verify the bundle.

## Provenance

The original coastal commits were made locally before publication. This branch
starts at `origin/main` and cherry-picks only the three coastal archive commits,
so the unrelated open JEPA implementation does not enter its diff:

| Original local SHA | Publication-branch SHA | Content |
|---|---|---|
| `926a40c` | `bcae01c` | Stopped three-site audit |
| `8702b6f` | `f41f892` | Narrabeen + Duck registration and realised gate |
| `5f82715` | `87da5f5` | Two-fold runner and negative outcome |

Subsequent commits add this guide and the selected material that previously
existed only outside the repository. Original protocols, scripts, logs and
scores are not rewritten. Publication does not retroactively make a local
protocol an independently preregistered study.

[source_inventory.json](source_inventory.json) records the local source path,
source hash, destination and size of the newly collected artifacts. Absolute
paths are historical provenance, not required clone locations. Most artifacts
are copied byte-for-byte. The **latest results HTML only** has publication edits:
correct title, readable non-overlapping charts, current publication status,
accurate selection/calibration wording and explicit separation of hypotheses
from measured findings. Its metric values are unchanged. Older HTML/PDF guides
remain dated historical snapshots.

## Verification and what it does not prove

From the repository root, verify all published bundle bytes with:

```bash
sha256sum -c reports/research-publication/2026-10-15/SHA256SUMS
```

`SHA256SUMS` covers the new research tree, coastal and TESSERA evidence, progress
summary and this publication directory (except the checksum list itself).
[verification.log](verification.log) records publication checks. Repository
`make check` verifies software scaffolding, not coastal predictive skill.

This is **not a clean-room, end-to-end reproduced experiment release**. No raw
data are bundled and no training is rerun for this documentation/publication
task. Existing scripts have workspace-specific assumptions. Environment dumps
are historical snapshots, not new portable dependency pins. The run evidence is
uneven: Stage 0 has an original stdout log; the two-site run has saved fold JSON
but no full original stdout/per-scene prediction archive. The original three-site
audit includes failed acquisition and tool attempts as well as successful checks.

For the two-site result, [input_snapshot.json](input_snapshot.json) hashes the
still-preserved local matched inputs and original runner **at publication time**.
This is additional provenance, not a fabricated original-run manifest or a
certificate that the data never changed before this snapshot.

### Offline replay, with separately held inputs

- Stage 0: follow its [workspace instructions](../../coastal/2026-10-13-stage0/README.md).
- Three-site audit: follow the [audit replay notes](../../coastal/2026-10-13/README.md).
- Two-site model: restore approved matched CSVs named
  `matched_observations.csv` and `matched_observations_DUCK.csv` into a writable
  workspace with the byte-identical archived `run_folds.py`, then use the original
  CPU environment and `OMP_NUM_THREADS=1 "$PYTHON" "$WORKSPACE/run_folds.py"`.
  The runner writes `fold_results.json` into that workspace; do not overwrite the
  immutable evidence copy. Compare metrics/counts, excluding start/elapsed time.
- TESSERA: inputs and rights are unresolved; no clean replay is claimed.

These are replay instructions, not new download/training approvals. No cluster,
GPU inference, additional dataset acquisition, pretraining or Space deployment
occurs in this publication.
