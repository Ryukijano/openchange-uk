# event_search: training plan audit

Burn-scar analogue retrieval from one example, including "no match". Research-only audit, 13 Oct 2026. Nothing was downloaded beyond web pages and HF metadata/READMEs. Nothing was trained.

**Verdict: NO_TRAINING_NEEDED** for the problem as assigned (retrieving analogues by burn-scar morphology and severity pattern). A trained model becomes worth considering only if a cheap human-judged pilot shows the non-learned pipeline fails, or if the target is changed from morphology to post-fire outcome (Section 9).

## 1. What the dossier got right and what I would change

The dossier's caution is right: retrieval is crowded, demand is undocumented, and two absent queries cannot validate "no match". Three things I would change:

1. **Relevance labels for morphology are not independent of the inputs.** "Similar morphology" is a function of the burn mask and the severity raster. MTBS already ships both for every gallery fire. A metric learned on MTBS perimeters/severity would learn to reproduce shape and severity descriptors computed from the same products. That is circular. The only independent relevance signals are (a) blind human judgements and (b) field plots (CBI), which cover severity at points, not shape.
2. **The gallery side needs no learning.** Every MTBS fire has a perimeter, a mask, dNBR/RdNBR and a thematic severity map (VERIFIED, [product descriptions](https://www.mtbs.gov/product-descriptions)). Shape and severity-pattern descriptors are deterministic computations on those files.
3. **The query side is also mostly solved.** For a fire not in MTBS, dNBR is a fixed spectral index, and a public, Apache-2.0 fine-tuned segmentation model exists (Prithvi-EO-2.0-300M-BurnScars). The dossier's 460-unit design spends most of its effort on Landsat triplets and footprint IoU, which tests mask extraction rather than retrieval.

The dossier's suggested baselines (dNBR distribution kNN, shape descriptors) are, in my reading, the likely *answer*, not just baselines.

## 2. The model most worth training

**For the assigned problem: none.** The pipeline is:

- Input (query): either an MTBS fire ID, or a user-supplied pre/post 6-band HLS/Landsat GeoTIFF pair, or a perimeter polygon plus optional dNBR raster.
- Mask: MTBS mask, or dNBR thresholding, or Prithvi-EO-2.0-300M-BurnScars inference for a user image.
- Descriptor (~30 numbers): log area; perimeter compactness (isoperimetric ratio); elongation (minimum rotated rectangle aspect); fractal dimension of the boundary; number, total fraction and size distribution of unburned islands; severity class proportions; high-severity patch count/size distribution and largest-patch fraction; distance of high-severity patches from the edge; dNBR and RdNBR quantiles.
- Retrieval: standardised-descriptor nearest neighbours over the MTBS catalogue, with optional metadata filters (year, region, assessment type).
- "No match": a distance threshold calibrated on a held-out region with split-conformal quantiles, reported with counts and an interval, not as a probability.

Why not train: there is no learnable target independent of the inputs. Frozen foundation-model embeddings are a poor fit (Section 3): they work on fixed tiles (Prithvi uses 224 x 224 pixels, about 6.7 km at 30 m), many MTBS fires are larger than one tile, and these embeddings mostly encode land cover. Land cover is a confound here: two fires in the same forest type would look similar even if their shapes differ.

**Conditional model (only if pilot kill criterion K3 opens):** a light ranking head (about 1-5 M parameters) on frozen Prithvi-EO-2.0-300M-TL features concatenated with the descriptor vector. It would be trained on human relevance judgements, which are the only independent label. That gives a few hundred pairs at most, so a light head is the ceiling. Fine-tuning a 300M backbone on that is not justified.

## 3. Existing methods and the measured gap

| Work | What I checked | Relevance |
|---|---|---|
| [Prithvi-EO-2.0-300M-BurnScars](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-BurnScars) | Card: Apache-2.0; fine-tuned on HLS Burn Scars, UNet decoder, single timestamp; **card's numbers: test IoU burned 87.52, test mIoU 93.00**; links a demo Space (not opened) | Solves query-side mask extraction for CONUS-like HLS chips. Its splits are not geographically held out as far as the card says, so transfer outside the US is unknown. |
| [GeoQuery, arXiv 2605.05405](https://arxiv.org/html/2605.05405v3) | Frozen CLAY RGB embeddings with text proxy. **Source's numbers: 20 US wildfire queries, 0% within 50 km, 40% within 100 km**; authors attribute failure to burn scars being "spectrally subtle without near-infrared bands" | Direct evidence that generic frozen RGB embeddings do not capture burn scars. Multispectral embeddings are untested in that paper. |
| [Earth Index docs](https://docs.earthindex.ai/concepts/running-predictions) | Nearest-neighbour search over foundation-model tile embeddings from user positive/negative labels, confidence slider, iterative updates | Generic "find more like this" exists. It works on tiles, not on whole events with shape. |
| [ESIA, arXiv 2609.20662](https://arxiv.org/abs/2609.20662) | Abstract: category-agnostic change localisation plus open-vocabulary attribute recognition (category, area, damage severity); test-time tuning from one reference pair; applied to 2025 Palisades Fire burn severity | Adjacent: one-example adaptation for anomalies, not archive retrieval of analogues. Abstract only; numbers not checked in the body. |
| [arXiv 2509.07852](https://arxiv.org/abs/2509.07852) | Siamese U-Net on AlphaEarth data, trained on MTBS in CONUS, evaluated on 17 European regions; **source's numbers: IoU 0.6, F1 74%** | Shows MTBS-trained masks transfer imperfectly abroad; relevant to non-US queries. |
| RSICRC, Text-ITSR, composed EO retrieval (2605.24442), OpenDPR | From the dossier; I only confirmed 2605.24442 appears in an arXiv listing | Caption/text or same-scene retrieval, not cross-site event morphology. |

Six arXiv searches ("burn scar retrieval", "wildfire similarity search", "burned area content-based retrieval", "fire perimeter shape similarity", "analogue wildfire satellite", "wildfire embedding retrieval") returned no paper on burn-scar morphology analogue retrieval. That was a bounded search, so it does not show the gap is absent from the literature.

**Measured gap:** no published number exists for analogue retrieval of burn events, by any method. The gap is an evaluation gap (no relevance judgements, no "no match" protocol), not a method gap that training would close.

## 4. Data contract

| Item | Facts | Status |
|---|---|---|
| MTBS scope | All large US fires (wildfire, wildland fire use, prescribed) 1984-present, all ownerships; Landsat TM/ETM+/OLI, Sentinel-2 MSI when Landsat quality is poor | VERIFIED [FAQ](https://www.mtbs.gov/faqs) |
| MTBS per-fire products | Pre/post reflectance GeoTIFF clipped to 3 km-buffered bbox; NBR/dNBR/RdNBR; 6-class severity (incl. mask class); burn boundary and mask shapefiles; FGDC metadata; PDF/KMZ maps | VERIFIED [product descriptions](https://www.mtbs.gov/product-descriptions) |
| MTBS national layers | Burned-area boundaries (CONUS, AK, HI, PR), fire occurrence points, annual severity mosaics | VERIFIED same page |
| MTBS on Earth Engine | `USFS/GTAC/MTBS/burned_area_boundaries/v1`, availability 1984-01-01 to 2024-12-31; attributes include assessment type, acreage, centroid, dNBR offset, thresholds | VERIFIED [GEE catalog](https://developers.google.com/earth-engine/datasets/catalog/USFS_GTAC_MTBS_burned_area_boundaries_v1) |
| MTBS label nature | Perimeter digitised on screen at 1:24k-1:50k; severity thresholds "subjective and dependent on analyst interpretation"; limited field data | VERIFIED [mapping methods](https://www.mtbs.gov/mapping-methods) |
| MTBS size thresholds (>=1,000 acres west / >=500 east) | From dossier; the FAQ text I retrieved was truncated before this | UNVERIFIED by me |
| MTBS number of fires | Not found on pages read | UNVERIFIED |
| MTBS licence | GEE "Terms of Use" is a USFS no-warranty disclaimer; no licence statement seen. US federal works are usually public domain, but Sentinel-2-based fires contain Copernicus data | UNVERIFIED (check FGDC metadata use constraints) |
| MTBS on HF | HF dataset search "mtbs" returned nothing | VERIFIED (search, 13 Oct) |
| HLS Burn Scars | 804 scenes, 512 x 512, 6 HLS bands, 2018-2021, CONUS; masks from MTBS shapefiles, manually filtered; random 2/3-1/3 split; CC-BY-4.0; on HF | VERIFIED [card](https://huggingface.co/datasets/ibm-nasa-geospatial/hls_burn_scars) |
| ibm-nasa-geospatial/burn_intensity | HLS pre/during/post 224 x 224 chips plus 0-4 severity classes; card licence MIT | VERIFIED [card](https://huggingface.co/datasets/ibm-nasa-geospatial/burn_intensity); upstream imagery rights unchecked |
| CaBuAr | Sentinel-2 pre/post, CAL FIRE masks, 5 random folds, CDLA-permissive-2.0 | VERIFIED [card](https://huggingface.co/datasets/DarthReca/california_burned_areas) |
| CBI field plots | 30 m plots rated 0-3 across five strata; data 1994-2017; ScienceBase download | VERIFIED [portal](https://burnseverity.cr.usgs.gov/products/cbi); plot count UNVERIFIED |
| AlphaEarth Satellite Embedding | Annual embeddings, CC-BY 4.0 with attribution | VERIFIED [GEE](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL) |
| Prithvi-EO-2.0-300M / -TL | Apache-2.0; MAE ViT; pretrained on 4.2M HLS V2 samples; TL variants add lat/lon and date embeddings | VERIFIED [card](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M); overlap with MTBS sites UNVERIFIED |
| Clay | Apache-2.0 (HF card metadata) | VERIFIED [HF](https://huggingface.co/made-with-clay/Clay) |

Native timestamps: MTBS pre/post scene dates are in the file names and metadata (VERIFIED, file-name pattern). Initial vs extended assessments use different post-fire timing (VERIFIED), so the descriptor must be computed per assessment type or the comparison will mix one-season and next-season effects.

## 5. Recommended model (as specified)

- **Primary system (no training):** descriptors plus kNN plus a conformal threshold. Zero parameters learned, apart from about 30 standardisation constants and one threshold.
- **Query-side mask, optional:** Prithvi-EO-2.0-300M-BurnScars as released (Apache-2.0, about 300M parameters), used for inference only.
- **Conditional head (only if K3 opens):** frozen Prithvi-EO-2.0-300M-TL encoder (Apache-2.0); input is pre and post HLS tiles covering the fire (multiple tiles mean-pooled within the perimeter); features concatenated with descriptors; 2-layer MLP projection (about 1-5M parameters); pairwise or listwise ranking loss on human relevance grades, with a separate "none" calibration on held-out queries.
- **Compute (estimates, not measured):** descriptors for the whole MTBS catalogue take CPU-hours, dominated by I/O. Frozen-embedding baselines for about 700 pilot fires take under 1 GPU-hour on an H100/GH200. Conditional head: pilot under 5 GPU-hours; full run with embedding extraction for every MTBS fire 20-60 GPU-hours, mostly data loading. Training from scratch is not justified at any point.

## 6. Baselines the system (or any trained model) must beat

- B0 metadata filter: same region/ecoregion, size class and year range, ranked by acreage difference.
- B1 dNBR/RdNBR quantiles plus severity class proportions, kNN.
- B2 shape-only descriptors, kNN.
- B3 B1+B2 combined (the proposed system).
- B4 frozen Prithvi-EO-2.0-300M embedding of pre/post difference, pooled within perimeter, kNN.
- B5 AlphaEarth annual embedding difference (year after minus year before) pooled within perimeter, kNN. Free CC-BY features, so the cheapest learned-feature baseline.
- B6 random fire from the same ecoregion (floor).

A trained head must beat B3 and the better of B4/B5, not just B0/B6.

## 7. Pilot experiment

**Units:** 40 query fires, at most 480 pair judgements.

- **Split.** Gallery: MTBS fires 1984-2014 from all states except the test and calibration states. Calibration queries: 10 fires, Oregon/Washington, 2015-2017. Test queries: 30 fires, Colorado/Utah/New Mexico, 2018-2020. Exclude any gallery fire within 100 km of a query and any re-burn overlapping a query perimeter. This holds out both geography and time, which matches the use case (a recent fire against the historical record).
- **Query choice.** 30 test queries stratified by descriptor rarity: 10 common, 10 intermediate, 10 from the tail (for example the highest elongation, or the highest unburned-island fraction). The list is frozen before any system is run.
- **Pooling.** Union of the top-5 from B1, B2, B3, B4 and B5, deduplicated (about 12 per query on average, capped at 12). This is TREC-style pooled judging. Unpooled gallery items count as non-relevant; this is stated as a limitation.
- **Judging.** Two raters, blind to system and metadata, see perimeter plus severity map for query and candidate side by side, scaled to the same width with area printed. Grades: 0 not similar, 1 partly similar, 2 clearly similar morphology and severity arrangement. Disagreements are adjudicated.
- **"No match".** A query is "absent" if no pooled candidate gets grade 2 from both raters. The threshold is set on the 10 calibration queries and applied unchanged to test.
- **Metrics.** Inter-rater weighted kappa; P@5 (grade >=1) and nDCG@5 per system, with query-level bootstrap (30 queries, not 480 pairs); for "no match", false-accept count on absent test queries and coverage on non-absent ones.

**Kill / decision criteria (pre-declared; thresholds are proposals, not results):**

- **K1** Weighted kappa < 0.4 on the first 60 pairs: morphology similarity is not consistently judgeable. Stop this framing (NO_GO for morphology search).
- **K2** B3 nDCG@5 within 0.05 of, or above, the best of B4/B5, and P@5 >= 0.6: ship the non-learned system; no training.
- **K3** Open the conditional head only if B3 P@5 < 0.4, *and* B4 or B5 beats B3 by >= 0.15 nDCG@5 on calibration queries, *and* raters' free-text notes name a failure that descriptors cannot represent. Otherwise improve descriptors, not models.
- **K4** "No match": if at the calibrated threshold more than one-third of absent test queries are falsely accepted, or fewer than half of non-absent queries keep a result, the Space shows ranked distances only and no "no match" verdict. With roughly 5-15 absent queries this can only rule out gross failure; it does not establish a reliable abstention rate.

Rater cost (estimate): about 480 judgements at about 20 s each, roughly 3 hours per rater.

## 8. Public Space

What a non-specialist can do: pick a US fire from the MTBS list or a map (or upload a perimeter GeoJSON, optionally with a dNBR GeoTIFF); see its shape and severity summary in plain words ("long and narrow; 18% unburned islands; high-severity core in one block"); see the five closest historical fires with perimeter/severity thumbnails, dates, distances and links to the MTBS fire page; get "no close match in the catalogue" when the distance exceeds the calibrated threshold (only if K4 passes). It must be labelled as similarity of mapped shape and severity, not of cause, behaviour or recovery.

- **CPU feasibility:** a descriptor index for the whole catalogue is tens of thousands x about 30 floats, and thumbnails can be precomputed. This runs on the free CPU tier (estimate).
- **ZeroGPU:** only needed for optional Prithvi BurnScars inference on uploaded imagery. Per [HF docs](https://huggingface.co/docs/hub/spaces-zerogpu), hosting a ZeroGPU Space needs a PRO account or a Team/Enterprise org (VERIFIED). Fetching HLS on demand needs Earthdata/Planetary Computer access from the Space, which is an unverified operational risk. Prefer upload-only.
- **Redistribution risk:** MTBS-derived thumbnails are likely federal public-domain data, but no licence statement was seen (UNVERIFIED). Fires mapped with Sentinel-2 carry Copernicus terms, so start with Landsat-mapped fires only. HLS Burn Scars needs CC-BY attribution. Do not use the Global Flood Database (CC-BY-NC 4.0, VERIFIED) in a public product.

## 9. Different event type or different target?

- **Floods:** the [Global Flood Database](https://developers.google.com/earth-engine/datasets/catalog/GLOBAL_FLOOD_DB_MODIS_EVENTS_V1) covers 2000-02-17 to 2018-12-10, is MODIS-derived and is CC-BY-NC 4.0 (VERIFIED). Labels are not better and the licence is worse for a public product.
- **Landslides and other disturbances:** not checked in this session.
- **Fire with an outcome target (stronger long-term option):** keep MTBS events but define an analogue as a past fire whose *later* trajectory matched. The target is a 1-5 year post-fire NBR/NDVI recovery curve from later Landsat Collection 2 observations, inside the perimeter by severity class. It is learnable, and it is independent of the query's pre/post input because it comes from future observations. It answers a question metadata cannot ("what happened after fires like this one"), and the mtbs.gov menu lists an NDVI Profile Tool and RECOVER, which suggests recovery tracking is an existing workflow (menu labels only; not opened). This would be LIGHT_HEAD_ON_FROZEN: a ranking head on frozen Prithvi-EO-2.0-TL plus descriptors plus climate covariates, trained so that embedding distance predicts trajectory distance. It must beat (i) descriptor kNN, (ii) per-ecoregion-by-severity mean recovery and (iii) a direct regression on the same features. If direct regression wins, retrieval adds only explanation. This is a different project and needs its own audit; it is not the verdict here.

## 10. Critical unknowns and cheapest checks

1. **Can people judge morphology similarity consistently?** Two raters, 60 pairs, weighted kappa (about 30 minutes each).
2. **MTBS licence and Sentinel-2 share.** Read the FGDC metadata of three fires (a few KB each); count Landsat vs Sentinel-2 mapped fires server-side on GEE (no bulk download).
3. **How rare are the "rare" phenotypes?** Compute the descriptor distributions server-side on GEE from the boundaries table. The national perimeter zip probably exceeds this audit's download limit (size UNVERIFIED).
4. **Do multispectral frozen embeddings add anything over descriptors?** The pilot's B4/B5 vs B3 comparison. GeoQuery's failure was RGB-only.
5. **Query masks outside the US.** Run Prithvi BurnScars on about 20 non-US fires with independent perimeters (for example EFFIS; licence unchecked) and measure IoU.
6. **Is there demand for morphology analogues as opposed to outcome analogues?** No documentation found. Only the user can establish this (for example by asking fire-effects colleagues); I did not contact anyone.
7. **Is a perimeter-mapping artefact being retrieved instead of a fire phenomenon?** Severity thresholds are analyst-set (VERIFIED). Recompute severity classes from dNBR with fixed thresholds and check that the neighbours do not change much.
