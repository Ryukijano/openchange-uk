# Coastal survey audit: insufficient data for the registered comparison

This is a data-feasibility audit, not a three-site model result. **All three
registered folds stop at the minimum-data gate. No model was fitted and no
primary model-comparison outcome was computed.** The original windows, profiles,
regions and temporal cutoffs were not changed.

## Decision and counts

The protocol requires at least 30 matched rows and 10 scene groups in each
affected partition. Scene groups are dated satellite observations, not individual
transects. With the registered ±48-hour matching window:

| Site | Fit rows / scenes | Calibration rows / scenes | Test rows / scenes |
|---|---:|---:|---:|
| Narrabeen | 138 / 31 | 194 / 45 | 329 / 68 |
| Truc Vert | **1 / 1** | **2 / 1** | **7 / 3** |
| Torrey Pines | Not available | Not available | Not available |

Truc Vert provides only 10 matched rows across 5 scenes in total; the ±36-hour
sensitivity window gives 1 row / 1 scene. Every fixed fold uses Truc Vert for
fitting, calibration or testing, so none is eligible even if Torrey arrives.
Hold off on that upload for this registered comparison.

The independent checker uses Python's `csv` and `datetime`, not pandas or
`merge_asof`, and reproduces the ±48-hour counts and temporal partitions. Even
before contour-quality exclusions there are only 14 potentially matched rows
across 5 scenes. Removing those exclusions would not rescue the scene-count gate.
Survey dates without a time are treated as midnight UTC, as in the recorded
audit; these counts do not establish exact acquisition-time ground truth.

This establishes **insufficient data under this contract**, not that learned
shoreline quality prediction cannot work. It does not justify coastal backbone
pretraining. A replacement experiment needs a separately approved protocol and
adequate independently measured labels, checked before fitting anything.

## Acquisition and physical checks

- Approved payload budget: 300,000,000 bytes. Acquired Truc Vert payload:
  162,611,806 bytes (294 monitoring NetCDFs plus `Grids.nc`), checked against
  publisher SHA-256 values. No survey payload is in this archive.
- OSF Truc Vert and Dryad Torrey metadata identify CC0. Source URLs, attribution,
  expected bytes and approval are preserved in `evidence/sources.yaml` and licence
  snapshots. Torrey's 115,224,176-byte archive was not downloaded: the API returned
  401 and the publisher's public download route returned 403. Those failures are
  preserved; no bypass or alternate mirror was used.
- Truc Vert's grid elevation already uses MSL (0.4 m was subtracted from NGF by
  the provider). Do not subtract it again. The grid-x origin differs by about
  20–20.15 m from the supplied benchmark transect chainage. Chainage comes from
  projected geometry, not a translation fitted to shoreline residuals.
- Contours require exactly one downward MSL crossing and no upward crossing;
  missing cells are not bridged and there is no extrapolation. There are 1,032
  accepted and 144 rejected Truc Vert profile/date combinations.
- Torrey's proposed datum conversion and loader remain unvalidated against its
  unavailable payload. Their presence in the script is not evidence of support.

## Evidence and provenance

`code/` contains byte-identical snapshots of the audit, tests, independent count
checker and acquisition script. `evidence/` contains the original protocol,
source manifests, licence snapshots, transfer ledger, environment, failed and
successful run logs, count outputs and repository checks. `archive_manifest.json`
records the original paths, sizes and hashes of every copied file. Derived contour
CSVs, matched residual CSVs, diagnostic plots, model weights and raw NC/ZIP files
are excluded.

The original run manifest names OpenChange's parent commit and the SDS_Benchmark
commit used at execution time. Its `audit_scripts_uncommitted` flag and the
environment's equivalent flag truthfully describe that historical run; archiving
does not retroactively change them. This directory's Git commit supplies the
subsequent local evidence archive. The protocol's statement that no commit was
authorized is likewise historical: the user subsequently approved a local-only
audit commit, with no push, PR or payloads. No scientific split was amended.

The archived `requirements.lock` is a complete environment snapshot, not a
portable minimal install recipe; it contains an unrelated editable GeoTessera
path. The acquisition snapshot also retains its original OpenChange import path.
These are recorded limitations of the historical scripts, not release-ready
packaging. No new dependencies are needed in the existing research environment.

## Replaying the audit offline

Use the preserved CPU Python 3.12 research environment and original workspace
layout: a `coastal-three-site` directory holding the four scripts, protocol,
sources and `raw/TRUCVERT`, next to the pinned `sds-benchmark` clone. Both are
already present on the session VM. For another machine, restore the separately
held approved inputs and byte-identical archived scripts into that layout first.
No download is part of replay. Executing the audit regenerates derived outputs
in the workspace, not in this immutable archive.

```bash
# Set WORKSPACE and PYTHON to your existing survey workspace and CPU interpreter.
"$PYTHON" "$WORKSPACE/audit_surveys.py" && \
  "$PYTHON" "$WORKSPACE/verify_match_counts.py"
```

Expected count output: the table above and `agrees_with_primary_audit: true`.
The audit's environment timestamp and absolute paths will differ if relocated.
The independent checker only validates count matching from reconstructed
contours; it is not an independent reimplementation of contour geometry.

Code verification is recorded separately: Ruff, 10 audit unit tests, and the
repository `make check` (43 tests, 8 optional skips). Those checks do not validate
a model, Torrey support, operational safety or new-site measurement accuracy.

The eventual Hugging Face demo remains the goal. This audit stops an inadequately
labelled test rather than manufacturing a model result to fill that demo.
