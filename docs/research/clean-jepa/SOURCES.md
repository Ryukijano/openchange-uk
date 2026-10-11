# Primary sources, unresolved facts and corrections

This is a source register, not a dataset approval file. Nothing listed here is
authorized for access by this page. Pin immutable revisions, licence text and
checksums in the implementation manifest before using code/data. Provider default
branches can change; paper equations and examples are not identical recipes.

The LeJEPA `main` reference resolved to `c293d291ca87cd4fddee9d3fffe4e914c7272052`
during this update (`git ls-remote`). [Pinned MINIMAL](https://github.com/rbalestr-lab/lejepa/blob/c293d291ca87cd4fddee9d3fffe4e914c7272052/MINIMAL.md)
and [pinned EP](https://github.com/rbalestr-lab/lejepa/blob/c293d291ca87cd4fddee9d3fffe4e914c7272052/lejepa/univariate/epps_pulley.py)
anchor the code observations; other sources still need explicit pins before use.

## Primary links

| Purpose | Source | What to verify before implementation |
|---|---|---|
| Clean loss and minimal example | [LeJEPA paper](https://arxiv.org/abs/2511.08544), [MINIMAL.md](https://github.com/rbalestr-lab/lejepa/blob/main/MINIMAL.md) | Per-coordinate MSE, per-view EP, projector/normalization, licence and code commit |
| EP and distributed statistic | [epps_pulley.py](https://github.com/rbalestr-lab/lejepa/blob/main/lejepa/univariate/epps_pulley.py), [slicing.py](https://github.com/rbalestr-lab/lejepa/blob/main/lejepa/multivariate/slicing.py) | Autograd-aware reduction, directions, global N and quadrature convention |
| Aggressive token dropping | [LeVJEPA paper](https://arxiv.org/abs/2608.27395), [project page](https://levjepa.github.io/) | Exact source revision and token-drop implementation; our EO global/local choices are not universal defaults |
| Identifiability hypothesis | [arXiv:2605.26379](https://arxiv.org/abs/2605.26379) | Latent dimension, transition and observation assumptions; no direct EO theorem claim |
| Paired pilot data | [SSL4EO-S12 v1.1 card](https://huggingface.co/datasets/embed2scale/SSL4EO-S12-v1.1), [repository](https://github.com/DLR-MF-DAS/SSL4EO-S12-v1.1), [report](https://arxiv.org/abs/2503.00168) | Full revision, CC-BY-4.0 terms, acquisition gaps/date order, L1C/mask compatibility, units/dtypes |
| Location-pool expansion | [Copernicus-Pretrain card](https://huggingface.co/datasets/wangyi111/Copernicus-Pretrain), [Copernicus-FM code](https://github.com/zhu-xlab/Copernicus-FM), [paper](https://arxiv.org/abs/2503.11849) | Deduplication, S1/S2 timing, CC-BY-4.0 terms, band/acquisition metadata |
| Optional corpus, not main arm | [Major TOM paper](https://arxiv.org/abs/2402.12095), [Core-S2L2A](https://huggingface.co/datasets/Major-TOM/Core-S2L2A), [Core-S1RTC](https://huggingface.co/datasets/Major-TOM/Core-S1RTC) | CC-BY-SA obligations, sensor preprocessing and unrelated acquisition times |
| Candidate downstream tasks | [Copernicus-Bench](https://github.com/zhu-xlab/Copernicus-FM/tree/main/Copernicus-Bench), [PANGAEA](https://github.com/VMarsocci/pangaea-bench) | Exact tasks/licences/official splits, footprint decontamination, rerun baselines |
| Cross-regime selection context | [TESSERA v2](https://arxiv.org/abs/2607.03949) | Compare its population and metric definition, not just a reported correlation |
| Distributed implementation | [PyTorch DDP docs](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) | Mean reduction, autograd collectives, synchronized normalization and uneven batch handling |
| Compute/runtime | [Isambard ML packages](https://docs.isambard.ac.uk/user-documentation/applications/ML-packages/), [storage](https://docs.isambard.ac.uk/user-documentation/information/system-storage/), [agent policy](https://docs.isambard.ac.uk/user-documentation/guides/using_ai_agents/) | Current GH200/aarch64 container route, limits, storage and human-only cluster operations |

Recent-review numbers (counts, byte sizes, cloud proportions, pairing-gap statistics,
throughput scenarios) are archived with their sources in [reviews](reviews/README.md).
They need reproducible local manifests/logs before being called study measurements.
Copernicus' ~18.7M total is multisensor, not 18.7M time-matched S1/S2 examples.
The two corpora overlap; adding advertised counts does not create new unique locations.

## Corrections to earlier messages and archived reviews

1. **Drop-path and GradScaler:** the current LeJEPA MINIMAL read during this repository
   update sets `drop_path_rate=0.1` and instantiates/enables `GradScaler` even with bf16
   autocast. The previous claim that all official recipes use drop-path 0 / no scaler
   is incorrect. Drop-path 0 and bf16 without a scaler are our selected pilot choices.
2. **Invariance translation:** zero sum of gradients across all views means common
   translation is unconstrained. It does not imply the mean of global views cannot move.
3. **DDP:** a differentiable SUM of local sums is not independent of DDP averaging.
   Replicated loss backward and DDP mean reduction must be considered together; test
   parameter gradients and one update, not only the value of the statistic.
4. **Gaussian null:** ~1.05 is the expected EP statistic on independent Gaussian draws,
   not an irreducible floor or a target plateau for optimized/repeated finite batches.
5. **Weights:** prevalence weights are not invalid mathematics. Uniform weights are
   our equal-stratum risk choice; with four equal quotas the two choices coincide.
6. **Mean term:** the raw sum of squared stratum means and the N_m/K-scaled T_mu are
   distinct. The raw toy-gap penalty can be K-independent; scaled T_mu is not a universal
   K-independent loss. Its useful property is a direct first-moment gradient.
7. **Token counts:** ~208 tokens/item applies to a joint item under the proposed views.
   Single-sensor items use ~148; equal four-stratum sampling averages ~178. Derive steps
   from realized counts, not the earlier universal 208 or 100 estimate.
8. **S1 uint16:** quantizing floating-point dB to uint16 is not lossless. Specify codec
   scale/offset and error before calling it packing; do not claim storage savings for free.
9. **SAR predictability:** S1 can correlate with NDVI and optical properties through shared
   vegetation/geography. A blanket "no better than physics"/no-NDVI pass criterion is not
   operationally defensible. Use shared/private synthetic controls and held-out residual
   targets against S1-only baselines, without calling correlation optical recovery.
10. **Procrustes:** the junk-coordinate residual formulas are toy-model results. Evaluation
    must use locations separate from calibration; fitting and scoring on the same matrix
    can make compatibility look better. A linear map, not just a rotation, may be needed.
11. **Q1 statistics:** independent configurations and correlated checkpoints are not the
    same sample size. Negate loss for positive higher-is-better rank correlations; use
    a common validation view bank. No guaranteed power from "64 runs" without simulation.
12. **Budget:** 35% MFU/H100-like FLOP totals do not price data decoding, memory, projector,
    EP, full baseline compute, evaluation or cluster efficiency. The 750/1100 NHR figures
    are superseded as budget commitments. Award scope/remaining balance remains unverified.
13. **Upstream launch flags:** a `teacher_student` or `patch_mask_ratio` flag alone does
    not establish its semantics in an unavailable script. Our teacher-free definition
    follows the inspected minimal loss/code; do not generalize to every published run.

## What remains unresolved (not design defaults)

- Full immutable source revisions, exact approval/attribution, numerical normalization,
  dtype/nodata conventions, mask/date alignment and permitted pair gap.
- Human acceptance of OpenChange-UK's geographic/time split; the conservative pretrain
  UK exclusion is additional protection, not acceptance of a test specification.
- Geographic footprints and date decontamination for each global benchmark task.
- Empirical viability of full conditional Gaussianity when information is missing.
- Measured CPU/GPU throughput, loss-memory footprint, distributed gradients and grid price.
- Award permission, balance and combined benchmark+pretraining resource envelope.
- Novelty/prior-art audit before a paper claim; these are research questions, not
  promised conference acceptance or guaranteed new methods.
