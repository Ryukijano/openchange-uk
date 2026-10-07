# Problem brainstorm (2026-10-07)

Inputs: your two pasted briefs (OpenChange-UK and the JunctionLab traffic idea), the Isambard u6xn setup, and a web search done today. Anything marked "check" is a claim I haven't verified.

## What all your projects have in common

The ADC harness, OpenChange-UK and JunctionLab ask the same question: **does a pretrained model still work when the data moves away from what it was trained on, and does it know when it doesn't?** The evaluation pieces are the same each time: held-out splits, leakage traps, "confidently wrong" counts, Brier score and calibration. That is a strong identity for a research line, and it lets one evaluation harness serve several domains.

## What I found today

- PANGAEA (arXiv 2412.04204, now in IEEE) found that geospatial foundation models "do not consistently outperform" supervised UNet and ViT baselines. So "GFM beats UNet" is not a safe assumption. It is a question worth testing.
- TESSERA (arXiv 2506.20380, CVPR 2026) publishes precomputed annual 10 m embeddings worldwide, made from Sentinel-1 and Sentinel-2 time series, with 128 values per pixel. Few-label probing on them needs almost no GPU.
- V-JEPA 2.1 (arXiv 2603.14482, released 2026-03-16) is aimed at dense, temporally consistent features. That suits segmentation, and image time series can be treated as video.
- Cosmos-Predict2.5 has an official LoRA post-training recipe. The example uses 8 GPUs, which is 2 Isambard nodes, so memory on GH200 needs checking.
- Single-junction traffic-signal benchmarks on SUMO already exist: RESCO, and Traffic-Alpha/single-tsc-baselines, which has fixed-time, max-pressure, Webster and RL baselines over 12 intersections. JunctionLab should build on these, not reimplement them.
- The Environment Agency publishes "Remotely Sensed Flood Estimates" and "Recorded Flood Outlines" (licence: check). These could supply flood labels.
- The UKRI AIRR Gateway route gives up to 10,000 GPU hours to use within 3 months. I don't know u6xn's actual allocation.

## Candidates, ranked by novelty × feasibility

1. **Leakage-aware transfer benchmark for EO foundation models.** If a model's pretraining imagery already covers your test regions and dates, a "future, held-out region" test is not held out. Record each model's pretraining coverage and dates (check: Prithvi-EO-2.0, Clay, TESSERA, SatMAE). Build splits that are clean for each model, and measure how much score is lost once leakage is removed. This is the leakage-trap idea from the ADC work, applied to EO. Few people do it, and it needs little compute.
2. **Free embeddings vs paid fine-tuning.** On the same UK held-out splits, compare: a linear or small-MLP probe on precomputed TESSERA embeddings; LoRA on Prithvi and Clay; a UNet trained from scratch; and index-differencing baselines. Plot score against GPU-hours and trainable parameters. The answer is useful whichever way it comes out, and it is cheap.
3. **UK flood mapping from Sentinel-1, with calibration under shift.** Use EA outlines as labels and recent named storms as the future window (check which events have outlines). Test whether confidence stays meaningful on new terrain: urban areas, wet sand, radar shadow. It reuses the OpenChange harness unchanged.
4. **JunctionLab (your pasted plan).** An action-conditioned junction model in SUMO that knows when to hand back to the simulator. Sharpened question: at matched compute, does planning with the learned model lose decision quality compared with planning in SUMO itself? Mostly CPU; GPU only for batched rollouts.
5. **Grading video world models against simulator truth.** Post-train Cosmos (LoRA) or an action-conditioned V-JEPA 2.1 on SUMO renders, then score the generated futures on counts and queue lengths taken from SUMO state, not on how they look. This is a benchmark for "plausible but wrong" video. High novelty but the heaviest compute; do it after item 4 works.
6. **V-JEPA 2.1 on satellite time series.** Treat a year of Sentinel-2 images as a video and compare its dense features with EO-specific encoders on the item 1 and 2 splits. A clean cross-domain test, slotted in as one more backbone.
7. **Automatic failure search.** A scenario or region search that looks for cases where models are confidently wrong, then saves them as a fixed test suite. It works for EO (regions and dates) and for SUMO (demand and sensor faults). A tool that strengthens 1–5 rather than a standalone paper.
8. **Weather context as an ablation.** Does adding ERA5 or Earth-2 context (rain, surge, wind) improve change or flood detection? Small, and belongs inside 1 or 3.

## My suggestion

Make **1 + 2** the core of OpenChange-UK, with coastal change and flood (3) as two tasks on the same split rules. Keep **4** as the second track and add **5** only once 4 has results. Treat 6–8 as add-ons within those tracks, not projects of their own.

## What I need from you

- The u6xn GPU-hour allocation and end date.
- Which tracks the two of you actually want: EO, traffic, or both.
- Any deadline or venue.
- Whether you can hand-label a small test set.
