# Scaling a clean JEPA: what to train on, what to claim

Date: 2026-10-09. Status: proposal for discussion, no experiment run. "Clean JEPA" below means the LeJEPA family: a single encoder, a predictive/invariance loss plus SIGReg, no EMA teacher, no stop-gradient, no schedulers ([LeJEPA, arXiv:2511.08544](https://arxiv.org/abs/2511.08544); [LeVJEPA for video, 27 Aug 2026](https://arxiv.org/abs/2608.27395); [identifiability theory, arXiv:2605.26379](https://arxiv.org/abs/2605.26379)).

## 1. What LeJEPA claims that nobody has tested at scale

| LeJEPA claim | Evidence in the papers | Open at scale? |
|---|---|---|
| Training loss tracks downstream quality, so models can be selected without labels | One plot, ImageNet-1k, ViT-B | Yes. The only large controlled EO scaling study ([TESSERA v2, 395 runs on 1,024 GH200, Barlow Twins](https://arxiv.org/abs/2607.03949)) found the opposite for its loss: Pearson r below 0.2 |
| In-domain clean SSL beats transfer from frontier models (DINOv2/v3) | Galaxy10, a 17k-image dataset | Yes. Never tested on a domain with tens of millions of samples and a strong in-domain frontier baseline |
| Stable without heuristics up to 1.8B | ImageNet | Partly. No data-scaling or compute-optimal fit exists for any LeJEPA model |
| Gaussian latents are linearly identifiable, which enables planning | Theory plus toy and small control tasks | Yes for any real domain |
| LeVJEPA: cost is governed by tokens seen, 95% token dropping works | Kinetics/SSv2-style video | Yes outside natural video |

The robot-video version of a JEPA scaling law is taken: [RoboJEPA (7 Oct 2026)](https://arxiv.org/abs/2610.10515) fits compute laws from 22M to 8B on 23 public datasets with V-JEPA 2.1 features. Molecules and weather are covered by JEPA-Anything ([arXiv:2609.20800](https://arxiv.org/abs/2609.20800)) and by existing weather scaling laws. Astronomy already has [AstroPT](https://arxiv.org/abs/2405.14930) scaling 1M to 2.1B on 8.6M DESI stamps.

## 2. Where the massive open data is

| Domain | Dataset | Size | Licence | Who already scaled on it |
|---|---|---|---|---|
| Earth observation, all Sentinel sensors | [Copernicus-Pretrain](https://huggingface.co/datasets/wangyi111/Copernicus-Pretrain) | 18.7M aligned images, 8 modalities (S1 SAR, S2 optical, S3, four S5P gases, DEM), ~310k 0.25 degree grids aligned to ERA5, 20.7 TB | Copernicus legal notice: free, full and open; reproduction, distribution and adaptation allowed with attribution ([text](https://cds.climate.copernicus.eu/licences/ec-sentinel)) | Copernicus-FM (ICCV 2025), MAE-style; no JEPA, no scaling law |
| Earth observation, S2 + S1 + DEM on one grid | [Major TOM](https://github.com/ESA-PhiLab/Major-TOM) Core-S2L2A 23 TB, Core-S1RTC 16 TB, Core-DEM 1 TB | 2.2M S2 patches of 1068x1068 at 10 m | Copernicus data, redistributed by ESA | PhilEO scaling study (44M to 300M, 0.5 to 23 TB, [arXiv:2506.14765](https://arxiv.org/abs/2506.14765)); supervised-style, no JEPA |
| Earth observation, analysis-ready multi-season | [SSL4EO-S12 v1.1](https://huggingface.co/datasets/embed2scale/SSL4EO-S12-v1.1) | 246k locations x 4 seasons, S1+S2 aligned, 2.26 TB, Zarr | CC-BY-4.0 | Many MAE/DINO baselines; small by today's standards |
| Robot video | OXE, DROID (CC-BY 4.0); [AgiBot World 2026](https://huggingface.co/datasets/agibot-world/AgiBotWorld2026) 13.6 TB | 10k+ hours | AgiBot is CC-BY-NC-SA | RoboJEPA, OpenWAM, InternW0 |
| Molecules | [OMol25](https://fair-chem.github.io/omol25/) | 100M+ DFT calculations | CC-BY-4.0, gated download | Meta UMA; JEPA-Anything for MD |
| Astronomy | DESI Legacy DR8 stamps | 8.6M | Open | AstroPT scaling laws |

Earth observation is the only domain on this list with (a) tens of terabytes of licence-clean, redistributable data, (b) several aligned sensors so "translation" has a literal meaning, (c) no JEPA scaling study, (d) a direct point of comparison on the loss-vs-downstream question (TESSERA v2), and (e) a fit to the u6xn award if it names OpenChange-UK. It is also where the field admits it cannot rank models: the 152-paper audit [No One Knows the State of the Art in GFMs](https://arxiv.org/abs/2605.12678) found 46 cross-paper disagreements of 10+ points and 39% of papers releasing no weights. A controlled, fully released study is a contribution in itself there.

## 3. Three "translation" angles, with closest prior work

**T1. Cross-sensor latent translation.** Train one clean JEPA where the context is one sensor and the target embedding is another (S1 SAR to S2 optical, S2 to S3, S2 to S5P gases), on Copernicus-Pretrain. Measure translation error and downstream value as a function of model size, data and compute. Closest prior: [X-JEPA (WACV 2026)](https://openaccess.thecvf.com/content/WACV2026/html/Choudhury_X-JEPA_A_Novel_Joint_Learning_Cross-Modal_Predictive_Alignment_Framework_for_WACV_2026_paper.html) and [CR-JEPA](https://arxiv.org/abs/2606.00706), both small and retrieval-only; [TerraMind (ICCV 2025)](https://openaccess.thecvf.com/content/ICCV2025/html/Jakubik_TerraMind_Large-Scale_Generative_Multimodality_for_Earth_Observation_ICCV_2025_paper.html) and [GeoSET (Sep 2026)](https://arxiv.org/html/2609.37496), which translate in pixel/token space and score with FID. Nobody has asked whether latent translation gets predictably better with scale, or whether a SAR-conditioned latent reaches the downstream accuracy of a native optical latent (the cloud-cover question that matters for UK coasts).

**T2. Temporal translation, the EO world model.** LeVJEPA on satellite image time series: predict the embedding of the next acquisition from past acquisitions, no generation. Closest prior: [Sat-JEPA-Diff (Mar 2026)](https://arxiv.org/html/2603.13943v1), which predicts AlphaEarth embeddings to steer a diffusion decoder, and [RS-WorldModel](https://arxiv.org/abs/2603.14941v1), a text-guided generative model. Neither is a scaling study and neither is heuristics-free.

**T3. Geographic and temporal transfer.** Pretrain globally, score on held-out UK coastal regions and future dates. That is exactly the OpenChange-UK benchmark, so the harness, splits and leakage rules already exist.

If "translation" meant something else (sim to real, or carrying scaling laws from one domain to another), say so and this section changes.

## 4. The paper I would write

Working title: *Does a heuristics-free JEPA scale cleanly on Earth observation? Loss, translation and transfer from 20M to 1B parameters.*

Pre-registered questions, each with a yes/no answer that is publishable either way:

1. **Loss as a selector.** Across a controlled grid (4 to 5 model sizes, 3 data scales, matched token budgets), does the LeJEPA training loss predict downstream performance on a standard EO suite (PANGAEA or Copernicus-Bench plus the OpenChange-UK tasks)? Report Pearson and rank correlation per task against TESSERA v2's r below 0.2 for Barlow Twins. If the answer is yes, label-free model selection in EO becomes real; if no, LeJEPA's central practical claim fails outside ImageNet.
2. **In-domain versus frontier transfer.** Does a clean JEPA trained on Copernicus data beat frozen [DINOv3 SAT-493M](https://huggingface.co/facebook/dinov3-vit7b16-pretrain-sat493m) (7B, 493M Maxar images at 0.6 m, RGB) on 10 m multispectral tasks, and at what compute? The honest caveat is resolution and band mismatch; it has to be reported as such, not hidden.
3. **Translation scaling.** Do cross-sensor (T1) and temporal (T2) prediction errors follow the same compute exponents as same-sensor prediction, and does the gap between SAR-conditioned and optical-native downstream accuracy close with scale?
4. **Held-out prediction.** Fit the laws on the small grid, write down the predicted loss and downstream score of the largest model before training it, then train it once.

Controls that reviewers will demand: same tokenizer and patching across sizes; same evaluation protocol for every run (the GFM audit is the reason); released weights, manifests and logs for every grid point; geographic and temporal splits fixed before any pretraining; a supervised baseline and a MAE baseline at one size so the comparison is not JEPA-only.

Natural venues: ICLR or NeurIPS for the scaling and label-free selection claims; CVPR EarthVision or an ISPRS/IEEE TGRS journal for the EO-specific results. Both audiences read the GFM audit.

## 5. Budget shape, to be priced by a pilot

TESSERA v2 used 1,024 GH200 for 395 runs. We cannot match that, and should not try. A grid of 4 sizes (20M, 90M, 300M, 1B) x 3 data scales x 1 seed, plus 3 seeds at the two smallest sizes, with LeVJEPA-style 90 to 95% token dropping, is the shape. The real numbers come from a one-GPU pilot: tokens per second per size, tokens needed to saturate, and the data pipeline rate from Zarr or parquet on Isambard storage. Until that pilot runs, no GPU-hour figure in this document should be quoted. Known constraints: 32-GPU project cap, 24 h job cap, 30 min template cap in the repo until a human raises it, scratch quota unknown.

## 6. What this does to the earlier plans

This replaces the TORAX decision-versus-compute study as the lead candidate for project B. The TORAX work stays as a documented fallback. It reuses OpenChange-UK's splits, source gating and Slurm guardrails, so projects A and B merge into one repository and one award question rather than two.

## 7. Open questions before any compute

1. Confirm the meaning of "translation" (section 3).
2. u6xn award wording: does it name Earth observation? If so this plan is in scope; the TORAX plan was not obviously so.
3. Which downstream suite is the primary endpoint: PANGAEA, Copernicus-Bench, or the OpenChange-UK tasks? One has to be primary before pretraining starts.
4. Storage: Copernicus-Pretrain is 20.7 TB and Major TOM S2+S1 is 39 TB. Which subset fits the scratch quota, and is downloading it to Isambard allowed under the source-approval rule?

## 8. First week, local, no cluster

- Pull a 500-sample Copernicus-Pretrain slice through the existing source-approval path; check alignment of S1/S2/S3/S5P per grid cell.
- Implement LeJEPA (SIGReg plus invariance loss) and the cross-sensor variant on a 20M ViT; verify loss goes down and does not collapse on CPU.
- Wire PANGAEA or Copernicus-Bench evaluation to the OpenChange-UK harness; run it on one public checkpoint to confirm the protocol reproduces published numbers within a point.
- Write the pre-registration draft for questions 1 to 4.

