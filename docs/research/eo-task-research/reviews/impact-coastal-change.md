# Coastal change: when is a retreat claim physically supportable?

**13 October 2026 | topic_id: coastal_change | CONDITIONAL-GO**

## 1. One problem, one falsifiable question

**Problem:** distinguish displacement of a sandy beach’s surveyed, local-mean-sea-level contour from displacement caused by tide/waves, changing shoreline proxies, sensor bias and georegistration. This is contour change—not automatically sediment loss, permanent erosion, or attribution to climate or an intervention. Estuaries and cliffs are outside the initial claim.

**Hypothesis:** explicitly representing ambiguous optical proxies and scene-correlated registration/water-level errors will yield more correctly signed, non-abstained contour-change decisions than strong physical-error baselines at the same false-claim rate on geographically and temporally held-out coasts. A calibrated interval spanning zero means “direction unresolved,” not “unchanged.”

The contribution would be an **identifiability benchmark and selective-inference procedure**: specify the physical target, propagate correlated uncertainties across dates, test competing proxy interpretations, and refuse a physical claim if plausible interpretations disagree. This is not another confidence raster. It must outperform a conservative physical uncertainty envelope; otherwise the learned component dies. Neither JEPA nor foundation-model fine-tuning is required.

Success would address a coastal scientist’s substantive conclusion, rather than encoder-upgrade continuity. That is potentially greater impact. Independent calibration, retained coverage and practitioner use remain unproved.

## 2. User, job and documented burden

A coastal analyst screens satellite detections, corrects water levels, intersects transects, and decides whether retreat warrants further survey or investigation. [USGS’s 2025 practitioner report](https://pubs.usgs.gov/publication/ofr20251054/full), read through methods, comparisons and operational discussion, describes weeks of initial setup. At difficult Barter Island, manual screening alone could consume **8–10 hours**. This is a reported case, not a universal labor benchmark; learning the software and assisting development contributed to setup time. Completed processing took minutes to hours.

[Mikkelsen et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC13439185/) retained 511 of 1,437 available images: 392 automatic exclusions and **534 manual removals**. Their mid-resolution detections coincided with the waterline only about 50% of the time. The proposed instrument targets this claim-validation/QC step, not imagery acquisition or an entire consulting workflow. No cost savings or adoption are established.

## 3. Five closest primary works

The first, third and fourth were read through methods/data/evaluation.

| Work | Evidence and limitation | Proposed delta |
|---|---|---|
| [Vos et al., Communications Earth & Environment, 2023](https://www.nature.com/articles/s43247-023-01001-2) | Five SDS algorithms; four surveyed beaches; approximately 10 m accuracy at microtidal sites, much worse at Truc Vert. Common MSL target, nearest survey within ten days. | Evaluate calibrated **paired physical-change claims**, with explicit timing uncertainty and abstention, not only positional RMSE. |
| [Fitzpatrick et al., CoastSeg, JOSS, 2024](https://joss.theoj.org/papers/10.21105/joss.06683) | Accessible, reproducible extraction/inspection/tidal-correction workflow. An incumbent interface, not evidence that every physical claim is identifiable. | Export a falsifiable claim and its unresolved nuisance alternatives; hosting CoastSeg is insufficient. |
| [USGS Open-File Report 2025–1054](https://pubs.usgs.gov/publication/ofr20251054/full) | Comparisons across diverse U.S. settings; individual SDS uncertainty can exceed traditional observations even where trends agree. Refining scalar slopes gave limited gains. | Test selective safety under new-site transfer, rather than assume another slope estimate solves uncertainty. |
| [Mikkelsen et al., Scientific Reports, 26 May 2026](https://doi.org/10.1038/s41598-026-54336-z) | 128 near-weekly Waikīkī surveys; corrected RMSE 2.83 m PlanetScope–5.54 m Sentinel-2. Proxy inconsistency persisted; wave correction could worsen mid-resolution errors. | Explicitly model/refuse proxy ambiguity and validate physical-change decisions. |
| [CoastXplain, IEEE Access, January 2026](https://doi.org/10.1109/ACCESS.2026.3656095) | [Accepted author text](https://elib.dlr.de/222681/1/CoastXplain_Explainable_Unsupervised_Time-Series_Modeling_of_Coastline_Changes.pdf): Sentinel-2 LDA/GMM segmentation, certainty, temporal summaries and human inspection. | Pixel certainty is not demonstrated datum-contour change coverage; that independent calibration is the proposed gate. |

[NatShore, IJAEOG 2026](https://doi.org/10.1016/j.jag.2026.105277) already claims cross-mission, tidal-aware extraction and pixel confidence; author documentation/abstract were inspected, not full methods. This blocks generic novelty claims. Targeted 2024–2026 CVPR/ICLR/NeurIPS searches found adjacent segmentation/distance work, including [LIGHTHOUSE](https://iclr.cc/virtual/2025/36768), an **ICLR 2025 workshop poster**, not a main-track physical-change validation paper. No claim of exhaustive novelty follows from six search rounds.

## 4. Existing evidence, access and rights

All inspection was documentation/metadata only; no imagery, datasets or weights were downloaded. Use measured elevation contours, not segmentation masks, as truth.

| Source / subset path | Exact support and rights |
|---|---|
| [SDS_Benchmark](https://github.com/kvos/SDS_Benchmark) | Duck, Narrabeen, Torrey Pines and Truc Vert; 4–5 selected transects/site; Landsat/Sentinel-2, slope, tide/wave inputs and MSL positions. Existing benchmark reuse is not an independent untouched discovery dataset; rights derive from upstream sources, not a repository licence. |
| [Narrabeen Dryad](https://doi.org/10.5061/dryad.28g01) | Five monthly profile lines, April 1976–February 2016; 0 m AHD is approximately MSL, requiring datum handling. Archived tide/wave coverage ends in 2014. Dryad API verifies **CC0**; the paper is separately CC BY. |
| [San Diego Dryad](https://doi.org/10.5061/dryad.n5qb383) | 2001–2016 surveys at Torrey Pines, Imperial and Cardiff–Solana; 165 transects across sites. Select `torrey_*`, survey metadata, water levels and wave products. Upstream **CC0** verified via Dryad API. |
| [Truc Vert OSF](https://osf.io/jftw8/) / [descriptor](https://www.nature.com/articles/s41597-020-00750-5) | 326 DEMs plus raw measurements, September 2003–December 2019, including intensive 2008 surveys; hourly waves and ten-minute tides from 2000. NGF-IGN69 converted to MSL by subtracting 0.4 m. OSF node’s linked licence verifies **CC0**; Figshare 13070153 is only a metadata record. |
| [Waikīkī collection](https://doi.org/10.6084/m9.figshare.c.6911899.v1) | 500 m reef-lined beach, April 2018–July 2021; 128 surveys. Upstream items **25366456** (MSL/MHHW/toe vectors), **24715056** (0.5 m DTMs) and **24715062** (surveyed toe points) individually verify **CC0**. Local MSL uses NOAA station 1612340; toe is not MSL. Article’s CC BY-NC-ND is separate. |

[Landsat](https://www.usgs.gov/faqs/are-there-any-restrictions-use-or-redistribution-landsat-data) has unrestricted USGS reuse; [Sentinel](https://dataspace.copernicus.eu/terms-and-conditions) uses its full/open data legal notice, not the portal-content licence. Landsat multispectral 30 m and Sentinel-2 10 m pixels impose limits: subpixel extraction does not imply metre-accurate physical truth. PlanetScope is excluded from public inputs.

[CoastSat](https://github.com/kvos/CoastSat), CoastSeg and SDS_Benchmark code are GPLv3. Optional [CoastSeg SAR_3_band_model](https://huggingface.co/2320sharon/SAR_3_band_model) declares CC0: U-Net/ResNet50 initialized from SSL4EO-S12 Sentinel-1 MOCO; 792 U.S. scenes, 679/113 site-disjoint split. Exact training-site overlap and upstream weight provenance need checking. Its reported 60.1 m shoreline RMSE against optical references is not physical-survey validation. SAR remains out-of-scope initially. [FES2022](https://www.aviso.altimetry.fr/en/data/products/auxiliary-products/global-tide-fes.html) access/licensing is separate; do not redistribute its grids automatically.

## 5. A ≤500-unit falsifier

Proposed cap: **four sites × four fixed transects × twenty sensor-specific acquisition times = 320 endpoint units**; ten non-overlapping endpoint pairs per transect/site add **160 change units: 480 total**. Repeated sensors/dates count separately; these are only 80 scene clusters, not 480 independent samples. Rejected detections remain scored abstentions. No scored extras outside this budget.

Freeze source versions, geographic IDs and dates before fitting: Narrabeen 2013–February 2016 and Torrey Pines 2015–2016 for development/calibration; Truc Vert 2017–2019 and Waikīkī 2018–2021 exclusively for testing. Use Landsat-8/Sentinel-2 where available, pairing within mission without reused endpoints. Training and calibration dates form separate contiguous blocks. Matching requires surveys within 48 hours, recorded timing-error bounds, and no unbounded intervening storm change. Availability is **unverified**: inability to fill the manifest without relaxing these rules kills this pilot design. No ten-day pairing silently treated as simultaneous truth.

Baselines: CoastSat/CoastSeg standard QC plus tide/average-slope correction; optional wave/runup correction; conservative Monte Carlo physical-error propagation including shared registration, survey and timing errors; robust trend/no-change and always-abstain references. Test the learned selector only after these baselines.

Score contour MAE/bias, paired-change interval coverage/width, false signed claims, risk–coverage curves, class calibration and retained coverage by site/sensor. Bootstrap date clusters, report sites separately, and do not assume distribution-free coverage under site shift. Truth also carries survey/datum uncertainty. Small contrasts do not establish multidecadal trend skill. A later blinded timed review is necessary for labor claims.

Predeclared decision rules—not results: kill a claim-capable model if test-site 90% intervals cover under 85%, false directional claims exceed 10% of issued claims, or retained coverage stays below 25%. Kill the learned novelty if it offers no better retained coverage than physical bounds at matched error. These thresholds are provisional research gates, not a regulatory standard. Publish a negative benchmark only if it exposes reproducible failure/resolution limits beyond existing SDS_Benchmark results.

## 6. Space as a bounded scientific instrument

**Input → inference → evidence:** no-login curated paired observations, or a validated numeric CSV containing transect positions, acquisition times, datum, sensor and nuisance bounds → deterministic CPU uncertainty propagation plus a frozen selector → signed change interval or abstention, dominant ambiguity, downloadable claim/provenance JSON and CSV, and benchmark truth for replay verification.

Curated observations are visibly **archived replay**. User inputs receive no accuracy guarantee outside validated support. Cap at 100 endpoints and 1 MB; reject URLs, archives, executable formats, extra fields and malformed values. No imagery/GEE credentials. Process transiently without persistent upload logging; public Spaces are inappropriate for sensitive coordinates. Bound CPU work; ZeroGPU is unnecessary initially. No pretrained robustness assumed.

This enables researchers to reproduce or contest a specific physical claim, not merely inspect a map. Fail closed on missing datum, uncertain slope/proxy, cloud/haze, unresolved registration drift or storm-time mismatch. Reefs/structures need explicit supported subsets; inlets, estuary mouths, cliffs and SAR wet-sand/speckle boundaries abstain. Providing conservative limits may be useful even when most apparent changes are unresolvable, but usefulness must be tested.

## 7. Judgment and gates

**Subjective scores:** impact **4/5**—physical-claim reliability; openness **3/5**—narrow gap, crowded incumbents; feasibility **3/5**—public surveys but uncertain close-time pairs; demo usefulness **4/5**—checkable claim/rejection export; WAM fit **2/5**—could eventually choose a more informative observation, but neither dynamics nor physical intervention is modeled.

**CONDITIONAL-GO:** fund metadata/split feasibility and the physical-baseline pilot, not a production service or large encoder programme. Open gate: unseen-site identifiability remains plausible, not proven. Contribution gate: selective claims must beat cheap physical bounds. Feasibility gate: usable matched data, datum semantics and derivative rights. Proceed smoke-test → pilot → full evaluation → robustness only after each gate passes. Critical unknowns are sample timing, correlated uncertainty, proxy identifiability, optional-weight overlap, tide/wave packaging and actual practitioner acceptance. If physical bounds already suffice, **PIVOT** to a narrowly justified reproducible resolution-limit protocol; if no useful supported coverage or new scientific finding remains, **NO-GO**.
