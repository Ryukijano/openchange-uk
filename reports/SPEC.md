# OpenChange-UK specification (provisional)

## Claim

Parameter-efficient adaptation of an open visual encoder improves coastal change detection under held-out UK regions and a future time cutoff, relative to a frozen encoder and a small from-scratch baseline.

## v1 task

Given two public Earth-observation observations of the same coastal location, predict a persistent change map and an uncertainty score. Class labels, if used, distinguish erosion, accretion, and other land-cover transition. Tide, cloud, and season are nuisance factors, not the label.

Flood extent, urban heat, and Earth-2 weather context are ablations or later tasks. Earth-2 is not the segmentation backbone.

## Not in scope

Pretraining a foundation model. A Nemotron chatbot. Clinical or robotic deployment. Random image splits. Redistributing imagery whose licence is unclear.

## Splits

Train regions and test regions are disjoint. The test window starts after a single cutoff. Both stay unset in code until this page is reviewed. `src/openchange/splits.py` rejects overlap.

## Pilot

100 to 500 hand-checked pairs. Two open encoders with public weights and aarch64-capable runtimes, candidates Prithvi and Clay, chosen after a licence check. V-JEPA 2.1, Tessera, and SatMAE wait until that harness runs.

## Metrics

Pixel IoU, pixel F1, boundary F1, per-class mean IoU, expected calibration error, correlation of uncertainty with error, and variance across held-out regions.

## Provenance

The reusable pattern from NV-Disruptron is temporal persistence, geospatial indexing, and evidence linked to a public source. The scientific project name is OpenChange-UK.

## Compute

Project u6xn, Isambard-AI Phase 2, 5000 NHR through 15 Mar 2027. Smoke is one GPU for 10 minutes. No multi-node work and no training job until the smoke log is clean.