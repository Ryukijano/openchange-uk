# Inspect the training setup before changing it

This is a **human-run, host-only inspection**, not a training or GPU readiness test.
Run it after you log in normally; do not start an agent on a login node.
Nothing here connects to a cluster, submits work, installs software or downloads data.

The first report tells us which checkout, container path and data directories actually
exist. We can then choose the missing setup steps from evidence instead of installing
an unneeded stack. It does not establish that any model is ready to train.

## Shortest path

1. In your own cluster terminal, go to the training checkout containing these files
   on `docs/eo-progress-and-coastal-results` ([PR #3](https://github.com/Ryukijano/openchange-uk/pull/3)).
   They are not yet on `main`. Review the commit you use before running anything.
2. Run this one command from that checkout:

   ```bash
   python3 scripts/inspect_training_setup.py \
     --config configs/training_inspection.json \
     --output "${PROJECTDIR:?PROJECTDIR must be set}/training-inspection-$(date -u +%Y%m%dT%H%M%SZ).json"
   ```

   Use the site's documented Python environment if your current Python is too old;
   the repository targets Python 3.11+. The inspector itself uses only the standard
   library, so there is no package-install step. It runs in the foreground and exits.
3. Review the resulting JSON, then upload that JSON to the research session. It contains paths and
   filenames; remove anything you do not want shared. Do not upload credentials.

If using the separately supplied inspection bundle instead, extract it **outside
the training checkout** and run the same command from the training checkout, using
absolute paths to the bundle's script and config. This avoids introducing untracked
helper files into the clean checkout required by the later GPU job. No credentials
are needed by the inspector.

## What the default run inspects

| Area | Recorded facts | Deliberately not done |
|---|---|---|
| Host | Architecture, OS, interpreter version and executable | No claim that this is a compute node |
| Checkout | Git SHA, dirty flag including ordinary untracked files, optional expected-SHA comparison, presence/sizes of known setup files | No remotes, diffs, file contents, fetch, pull or commit |
| Software | Executable locations, loaded module names, selected **host** package versions from distribution metadata | No package imports, version subprocesses, installation or module changes |
| Storage | Project/scratch/local paths and up to 50 immediate entries in each, filesystem free space, access hints | No recursion through these roots, no `du`, no quota/accounting query or write probe |
| SIF | Presence, size and a read-access hint for the configured file | No container execution, full-file hash, Arm64 verification or CUDA check |
| Data | None by default; only explicitly selected roots on a second run | No automatic dataset discovery beyond the shallow storage samples |

The script's only subprocesses are short, read-only `git` queries with optional Git
locks disabled. It does not execute Clifton, SSH, Slurm, `nvidia-smi`, a shell or an
arbitrary command from configuration.

Missing host Torch, CUDA compiler or geospatial packages do **not** automatically
mean the training environment is broken: training uses the staged Arm64 NGC container,
not the login node's Python. Container package imports and GPU execution belong in a
compute allocation. No global or host-side Torch installation is proposed here.

Filesystem free space is **not** your project quota or remaining allocation.
`writable_hint` is an access check, not proof that a write or large checkpoint will
succeed. Container existence is **not** an image identity or compatibility check.

## Inspect selected data directories

After the first report identifies likely locations, copy the JSON config **outside
the training checkout** and edit that copy (or edit the external bundle's config).
Pass the copy's absolute path to `--config`; keep `repo_path` pointing at the training
checkout, or leave it as `.` and run from that checkout. This preserves a clean
reviewed tree for later GPU checks. For example, set:

```json
"data_roots": ["${PROJECTDIR}/ACTUAL_DATASET_SUBDIRECTORY"]
```

Use real, explicitly selected dataset subdirectories, not `/`, your home directory,
or all of `/projects`. At most four roots are accepted. Only `HOME`, `PROJECTDIR`,
`SCRATCHDIR` and `LOCALDIR` variables may be expanded in configured paths. An unset
storage variable is recorded as missing instead of guessed.

By default **each selected root** has a two-level descent limit, 500 visited entries
and 100 returned files. The run has a 45-second best-effort deadline, capped at 60
seconds by validation. A stalled network filesystem can exceed a process deadline;
this is not a scheduler-enforced limit. This is a bounded enumeration-order sample,
not a total count, random sample, completeness audit or estimate of dataset size.

For regular files it records relative path, suffix and size. For `.npy` files it also
reads at most 8 KiB of header to return shape and dtype; it never reads array payloads,
loads pickles or opens checkpoints. Structured dtype field names are not exported.
TIFF, NetCDF, Zarr, HDF5, CSV and other formats receive file metadata only in this
first pass. Their bands, CRS, dimensions, missingness and values remain uninspected.
Format-aware checks can be selected after we know the actual files and container.

Hidden paths, credential-like names, symlinks, sockets and named pipes are excluded
from data inventories. The exclusions are a precaution, not a guarantee that all
filenames or metadata are non-sensitive. Review the report before sharing it.

No whole-home scan or recursive storage traversal occurs. The report is created with
owner-only permissions, cannot replace an existing file, and records the config and
inspector SHA-256. If the deadline is hit, a partial report remains explicitly
`inspection_incomplete`. Even a completed report says
`inspection_complete_not_training_ready`.

## What happens after you return the report

1. Reconcile the actual checkout with the reviewed code and required configuration.
   If known, set `expected_git_sha` to the reviewed full 40-character SHA in a later
   inspection. Leaving it null means the checkout is **not pinned** by this report.
2. Locate the staged `nvcr.io/nvidia/pytorch:25.05-py3` Arm64 SIF, or prepare a separate
   reviewed container-staging step if absent. A filename is not proof of image identity.
3. Use the [GPU readiness runbook](GPU_READINESS.md): verify the staged SIF identity
   and hash, then run the explicit one-GH200, ten-minute check from a clean reviewed
   checkout. Review its output before the two-GPU and optional four-GPU stages.
4. Choose the **actual workload** and its explicit config. Generic CUDA success does
   not validate JEPA, a frozen EO encoder, data decoding or a scientific hypothesis.
   The historical SyncBatchNorm failure remains unresolved.
5. Specify only the missing dependencies, staged data and weights required by that
   workload. Any future downloads must pass the existing source/licence gates.
   No installation or download happens inside training jobs.

This inspection prepares setup decisions. No remote environment is changed by this
kit, and no training, data readiness or GPU result is claimed without its own evidence.

## Local verification and publication

Local test coverage exercises traversal bounds, symlink and credential exclusions,
NPY header parsing without payloads, output permissions, timeout reporting and refusal
to overwrite prior reports. `reports/gpu-readiness/preparation/inspection-local-checks.log`
is the local development check log, **not a BriCS run log**. The helper additions and
that log remain uncommitted until publication is explicitly approved.

Official sources checked for this workflow:

- [Using AI agents](https://docs.isambard.ac.uk/user-documentation/guides/using_ai_agents/)
- [Modules](https://docs.isambard.ac.uk/user-documentation/guides/modules/)
- [Storage and quotas](https://docs.isambard.ac.uk/user-documentation/information/system-storage/)
- [Containers and Arm compatibility](https://docs.isambard.ac.uk/user-documentation/guides/containers/)
