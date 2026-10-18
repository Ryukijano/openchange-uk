# TESSERA compatibility pilot: exploratory, not an operational success

13 October 2026 run; archived for publication on 15 October. Read the
[current interpretation](../../../docs/PROGRESS.md#2-the-earlier-tessera-pilot),
[full output](evidence/results.json), [original report](evidence/report.md) and
[historical script](code/tessera_pilot.py).

348 usable crop parcels, 174 western parcels for fitting/alignment and 174 eastern
parcels held out. Same 2024 embedding year, `v1.0/vultr` → `v1.1/dclimate`.
This confounds model-version and inference-run changes. Ridge approximately
preserves the weak random-forest head (macro-F1 0.127 → 0.124; unchanged head on
new features 0.061), but does not establish useful crop accuracy or saved cost.
Procrustes fails that head. Logistic-regression results are also retained.

The JSON records package/source commits, label SHA-256, stores, split rule and
all routes. The prose report also includes bootstrap means; those must not be
substituted for point estimates. No original environment lock or full terminal
run log is available in this selected output directory; this archive does not
invent one.

Crop-label year/redistribution rights remain unresolved. Therefore the GeoJSON,
per-parcel manifest, labels and feature arrays are **not** published. No raw
satellite imagery or model weights are included. Replaying requires approved
external labels and feature access; publication is not approval to fetch them.
The archived script retains local source paths and is not a one-command clean
reproduction package. Aggregate output is evidence of an exploratory pilot,
not a releasable mapping model.
