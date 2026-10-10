# Clean JEPA on Sentinel-1/2: selected pilot design

Status: **design selected for implementation, not preregistered or trained**.
This is the current EO research track. It replaces TORAX and the earlier generic
loss-versus-planning proposal as the next study in this repository. It does not
replace the OpenChange-UK benchmark or accept its proposed regions and cutoff.
Updating these documents is not approval to fetch a dataset or use an allocation.

## Read in this order

1. This page: decisions and boundaries.
2. [Formulation](FORMULATION.md): notation, exact losses, gradients and distributed convention.
3. [Implementation runbook](IMPLEMENTATION.md): work packages, interfaces, tests and exit gates.
4. [Sources and corrections](SOURCES.md): primary links, provenance and errors in earlier reviews.
5. [Training diagram](training-diagram.html): self-contained HTML with selectable nodes.
   GitHub shows HTML source; download the file and open it locally to use the diagram.
6. [Review archive](reviews/README.md): supporting reports, not the implementation specification.

## What we are testing

**Q1 — Label-free selection:** within a fixed training setting, does lower clean-JEPA
validation loss rank useful EO representations? Separately, does the ranking survive
changes in model/data/compute scale? Compare matched Barlow Twins and MAE, not numbers
copied from unrelated papers. Selection labels are unavailable to the selector; the
downstream probes use labels to audit it.

**Q2 — Version compatibility:** can a calibration map carry features and a frozen
probe between independently trained versions? Orthogonal Procrustes in projector
space tests a theory-motivated hypothesis; backbone-space probe transfer tests the
applied question. Neither implies that every old archive can avoid re-embedding.

**Q3 — Sensor substitution:** what performance is lost when optical S2 is missing
or occluded and only SAR S1 remains? Report clean, degraded, single-sensor and joint
inputs with real and synthetic conditions separated.

**Method candidate — Availability-stratified SIGReg:** test whether a Gaussianity
penalty computed separately for availability strata reduces harmful sensor dependence
without destroying task-relevant information. This is an experiment, not a guarantee.

Temporal positives are an ablation: invariance can erase real change. Action-conditioned
rollout and intervention are absent, so this is not a world-action model. WAM remains
a later research objective, not a claim made by this EO experiment.

## Decisions made

| Item | Selected pilot design | Why / limit |
|---|---|---|
| Repository | Keep the EO track here, isolated in research docs now and `openchange.jepa` when implemented | Reuse source gates, split contracts and provenance; do not silently turn benchmark scripts into pretraining |
| Architecture | Separate S1/S2 patch stems, shared ViT and projector; no teacher, EMA, predictor or stop-gradient | A specified clean arm, not a claim about every upstream launch configuration |
| Loss convention | Mean over samples, views **and coordinates** for invariance; per-view EP | Removes ambiguity in sum-versus-mean over projector width |
| Projector | K=128; K=64 sensitivity; hidden widths 2048, 2048 | Fixed within each Q1 population |
| Main-scale batch | N=1024, four quotas of 256, stratum weights 1/4 | Fix counts across comparisons; the CPU debug batch is smaller and not comparable |
| Views | Two 224-pixel globals; six 96-pixel locals | Globals use all currently available sensors; local sensor subsets drawn only from available sensors |
| Token drop | Global 0.9, local 0.5, after position embedding; keep CLS | Avoid reducing a single-sensor local to four tokens |
| Gaussian target | Keep N(0,I) in all strata as the **pilot candidate** | Compare pooled and stratified losses; do not introduce an unvalidated covariance/shared-subspace method yet |
| Missing-sensor failure gate | Stop the method scale-up if its robustness gain costs unacceptable clean-task utility or only creates nuisance features | A low availability-probe score alone is insufficient; see runbook |
| Mean penalty | Main arm coefficient 0; weighted, sample-count-scaled T_mu as an ablation | A slice-free first-moment check, not a separate guarantee |
| Optimizer | AdamW, betas (0.9,0.95), lr 5e-4 at N=1024, wd 0.05, clip norm 3 | Study choices; pilot tunes stability |
| Drop-path | 0 initially; sweep 0, 0.05, 0.1, 0.2 later | Our choice. Current upstream MINIMAL uses 0.1, contrary to the earlier summary |
| Precision | bf16 forwards where supported; explicitly disable autocast for fp32 EP; no scaler for bf16 | Our choice. Current upstream MINIMAL does instantiate a GradScaler |
| Schedule | Cosine baseline; WSD/branched cooldowns only after a matched pilot | Neither WSD nor the proposed EO recipe is an official universal default |
| Corpus | Source-gated SSL4EO-S12 v1.1 pilot first | Copernicus-Pretrain expands the location pool later; it is not assumed time-paired |
| Major TOM | Excluded from main arm | Different licence and preprocessing; optional separately approved study |
| Compute | No priced main grid yet | Old GPU-hour totals are unvalidated FLOP scenarios, not measured budgets |

For the missing-sensor decision: retain the simple full-Gaussian candidate to **test**
the information conflict, with pooled SIGReg as the control. If it fails, write a new
protocol for a shared-subspace or reduced-covariance target; do not change the target
mid-grid and present the runs as one method.

## Current implementation status

- Existing: source approval validation, provisional geographic/temporal split contract,
  run manifests, local checks and capped Slurm scaffolds.
- Missing: JEPA encoder/loss, view loader, distributed equivalence tests, checkpointing,
  benchmark adapters, selection/Procrustes experiments and measured throughput.
- No dataset entry has been approved by this update. `configs/sources.yaml` remains empty.
- `make train-small` is still a dry-run, and `make evaluate` is still a stub. Neither
  trains/evaluates JEPA. No new runnable training command is advertised here.
- These documents report no new research result. Archived synthetic checks are
  reviewer observations, not an EO training run or an Isambard smoke pass.

## Permission and resource gates

Repository planning records 5,000 NHR through 15 March 2027. That is not a verified
remaining balance or proof that the award covers this pretraining study. The existing
benchmark budget already allocates the whole planning envelope; do not add the JEPA
estimate to it as though there were a second allocation. Obtain award scope/balance,
then replace the combined ledger using measured throughput.

Agents work locally and never use SSH/Clifton or submit cluster jobs. A human-run
one-GPU smoke stays at or below 30 minutes. Data approval, accepted split contract,
and a frozen evaluation protocol are separate gates that this design decision does
not waive. See [AGENTS.md](../../../AGENTS.md) and [WORKFLOW.md](../../WORKFLOW.md).
