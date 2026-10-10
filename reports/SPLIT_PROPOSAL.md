# Split design

**Status: PROPOSED.** Nothing here is accepted. `src/openchange/splits.py` keeps placeholder regions and `TEMPORAL_CUTOFF = "unset"` until a human signs this off. Facts marked *verify* need checking against the primary source before acceptance.

## Unit of grouping

Group by coastal sediment cell or sub-cell, not by image, tile, or patch. Cells are largely self-contained for sediment transport, so a cell in train and a neighbouring cell in test share less process and less spatial autocorrelation than two patches cut from one scene.

- England and Wales: Shoreline Management Plan (SMP2) cells and sub-cells (Environment Agency, Natural Resources Wales). *Verify* the current boundary dataset and its licence.
- Scotland: Dynamic Coast coastal cells. *Verify* licence.
- Northern Ireland: no shared cell scheme found yet; use NI Coastal Observatory sites or hand-drawn units. *Verify*.

Each group carries one coast-type label, so train and test both cover every type.

## Candidate groups

| Group | Location | Coast type | Why it is useful |
|---|---|---|---|
| G01 | Holderness, East Yorkshire | soft cliff | fast, well-documented cliff retreat |
| G02 | North and north-east Norfolk (Happisburgh to Blakeney) | soft cliff, barrier, saltmarsh | erosion plus marsh change |
| G03 | Suffolk (Covehithe, Dunwich, Orford Ness) | soft cliff, shingle spit | erosion and spit change |
| G04 | Dungeness and Romney Marsh, Kent/East Sussex | shingle foreland | slow, distinct morphology |
| G05 | Solent and Isle of Wight | saltmarsh, landslide | marsh loss, Undercliff |
| G06 | Chesil to Lyme Regis, Dorset/East Devon | shingle barrier, landslide | episodic slips |
| G07 | North Devon and North Cornwall (Braunton Burrows) | dune, rocky | dunes against hard coast |
| G08 | Severn Estuary and Bristol Channel | mudflat, saltmarsh | very large tidal range |
| G09 | Cardigan Bay and Pembrokeshire, Wales | beach, dune, rocky | Welsh coast, different light |
| G10 | Sefton coast and Dee/Ribble estuaries | dune, saltmarsh | dune and marsh change |
| G11 | Morecambe Bay | intertidal sand/mud | channel migration (tide-sensitive) |
| G12 | Solway Firth | saltmarsh, intertidal | cross-border estuary |
| G13 | Montrose Bay to the Tay, Scotland | beach, dune | Dynamic Coast hotspot |
| G14 | Moray Firth (Culbin) | dune, shingle | northern dunes |
| G15 | Uists, Outer Hebrides | machair, beach | far north-west, cloudy |
| G16 | Dundrum Bay (Murlough), Northern Ireland | dune | NI coverage |

The pilot needs only about four groups: two train, one validation, one test, each mixing at least two coast types.

## How to assign groups (options)

- **S1, pre-registered random draw (recommended).** Stratify by coast type. Draw test and validation groups with a fixed seed written in this file *before* any model runs. Commit the result and its SHA. This removes choice from the person who will later see results.
- **S2, nation holdout.** Train on England and Wales, test on Scotland and Northern Ireland. A harder, easy-to-explain transfer, but it mixes geography with sensor angle, sun angle, and cloud climate.
- **S3, hand-picked.** Reviewers choose test groups for coverage. Most flexible, and most open to bias. Write the reasons down before any model runs.

Validation groups come only from non-test groups and are the only data used for model selection, early stopping, and threshold or temperature tuning.

## Temporal cutoff options

Sentinel-2A data start mid-2015, 2B from 2017, and 2C from late 2024 (*verify*). ESA processing baseline 04.00 (January 2022, *verify*) added a reflectance offset to L2A products, so mixing pre- and post-2022 data needs consistent harmonisation.

| Option | Train and validation | Gap | Test | Notes |
|---|---|---|---|---|
| T-A | up to 2021-12-31 | 2022 | 2023-01-01 to 2025-12-31 | test fully after baseline 04.00; one-year gap limits seasonal leakage |
| T-B | up to 2022-12-31 | 2023 | 2024-01-01 to 2025-12-31 | more training data; Sentinel-2C appears only in test |
| T-C | up to 2020-12-31 | 2021 | 2022-01-01 to 2025-12-31 | longest test window and most change events; least training data |

Pair rule to decide: a change pair `(t0, t1)` is test only if **both** dates are after the cutoff (strict), or if `t1` is after the cutoff and `t0` was never used in training (looser). `SplitContract.check_sample` checks one date; apply it to both.

## Label source candidates

Every candidate needs a licence check before it enters `configs/sources.yaml`.

| Candidate | Gives | Caveat |
|---|---|---|
| EA National Network of Regional Coastal Monitoring Programmes (beach profiles, lidar, aerial imagery) | shorelines and change in England | coverage and licence vary by region (*verify*) |
| Environment Agency National LIDAR Programme / DEFRA survey data | elevation change, cliff tops | epochs are irregular |
| Dynamic Coast (Scotland) | historic and recent shoreline change | method and licence (*verify*) |
| National Coastal Erosion Risk Mapping (NCERM) | erosion rates and baselines | modelled projections, not observed labels |
| Ordnance Survey OpenData (e.g. OS Open Rivers, Boundary-Line MHW line) | reference tide lines | low temporal resolution |
| ESA WorldCover 2020/2021 and Dynamic World | land-cover classes | model output, not ground truth; 2021 WorldCover touches option T-A's training cutoff |
| Hand labels on a ≤500-sample pilot | persistent change masks | required by the compute rule; needs a written protocol and double labelling |

"Persistent change" needs a definition: a change present across several post-event dates, after tidal-stage filtering, not a single-scene difference. Tide filtering needs a tide model (e.g. FES via pyTMD, or UKHO predictions); check licences.

## Leakage risks

- **Spatial autocorrelation at group edges.** Drop patches within a buffer (proposal: 5 km) of any train/test boundary. Record every patch's group, Sentinel-2 tile, and centroid.
- **Shared scenes.** One Sentinel-2 tile can cover two groups. Patches from one acquisition may not sit in both train and test. Normalisation statistics come from train only.
- **Temporal bleed.** No test image date on or before the cutoff. Composites must not span the cutoff. Cloud masks and harmonisation must not be fitted on test dates.
- **Label leakage.** Labels derived from post-cutoff surveys must not reach training groups. Model-derived labels (WorldCover, Dynamic World) may have been trained on the same imagery.
- **Pretraining overlap.** Prithvi-EO-2.0, Clay, TESSERA, SatMAE, and V-JEPA 2.1 were pretrained on large archives that may contain UK test scenes and dates. Record each model's pretraining sources and date range from its model card (*verify*) and report whether test dates fall after it.
- **Selection on test.** Test labels stay sealed: no threshold, calibration, checkpoint, or region choice from test numbers. The split file and its SHA are committed before the first non-dry-run job.
- **Tide and season confounds.** Low tide vs high tide and summer vs winter can mimic change. Match tidal stage and season within each pair.
- **Duplicate products.** The same acquisition can appear under several processing baselines or tile IDs. Deduplicate by acquisition time and orbit.

## Decisions needed

1. Grouping scheme (SMP2 / Dynamic Coast cells, or something else).
2. Assignment method: S1, S2, or S3.
3. Cutoff: T-A, T-B, or T-C, and the strict or loose pair rule.
4. Buffer distance.
5. Which label sources to licence-check first.
6. Who signs off, and the date. After that, `splits.py` gets the real IDs and `STATUS = "accepted"`.
