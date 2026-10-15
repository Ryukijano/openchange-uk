# A demo people can use, not a model showcase

Research date: 13 October2026. Draft — independent reviewer recommendations not yet incorporated. No deployment or new data approval is implied.

## Decision criteria
The best bet must have all five:
1. A user with a repeated job and a concrete output they can reuse.
2. Existing legal data/weights and a low-friction browser-only path.
3. A visible advantage over strong existing tools, not just a new interface.
4. Failure detection, geographic/time validation and cost accounting.
5. A route from a useful baseline to our representation/compatibility/observation-policy research.

## Established facts that constrain the choice
- [TEE](https://github.com/ucam-eo/TEE) already provides similarity search, habitat pins/polygons, classifier evaluation, spatial holdout, year comparison and GeoTIFF output. Its hosted service lets visitors explore existing viewports, but creating new ones requires an account; classifier evaluation needs a user-run compute server. Browser-only hosted training removes friction but is not a novel research algorithm.
- [GFW integrated disturbance alerts](https://www.globalnaturewatch.org/blog/data-and-tools/integrated-deforestation-alerts/) now combine four optical/radar systems, confidence levels and repeated vegetation disturbances globally. Generic radar+optical confidence mapping is an incumbent capability.
- [GFW Pro](https://pro-news.globalforestwatch.org/news-events/how-gfw-pro-supports-eudr-compliance) describes visual review with high-resolution imagery and sometimes further imagery/field investigation. That supports the existence of evidence-review work, not a quantified saving in analyst minutes.
- [TerraMind anomaly detection](https://github.com/edornd/terramind-ad) already offers patch-level temporal anomaly maps and a dashboard. An anomaly heatmap alone is not enough either.
- GeoTessera explicitly forbids mixing version/variant spaces. A validated task remaining reliable through migration is a plausible research target; it is not solved by plotting cosine similarity or retaining all old predictions.
- The exploratory Vienna crop pilot had weak absolute accuracy; ridge retained some predictions but did not establish safe migration, useful maps or statistical superiority over retraining. Keep it internal.

## Product shape to compare
A. Forest evidence desk: candidate areas -> dated evidence/support/gaps -> ranked review queue and evidence packet. Research: fewer observation/review actions at matched missed-event risk, preserved across sensor/model transitions. Risks: incumbent confidence fusion, unresolved forest data rights, sparse truth and no human review outcomes.
B. Persistent custom monitoring: trusted user polygons -> bounded map/task with honest spatial validation -> saved versioned task that refuses unsafe upgrades. Research: held-out decision safety and reduced label/rebuild cost. Risks: existing TEE mapping, weak starting classifier, less broad appeal than disaster maps.
C. Generic change/anomaly map: area+dates -> heatmap. Simple to understand but highly crowded, often uncalibrated and only weakly connected to our exact research. Do not select just because the interface is attractive.

## Shared engineering and experimental gates
See DEMO_BUILD_REQUIREMENTS.md. A published example gallery is historical replay, not worldwide live monitoring. The first usable release must either support genuinely new bounded inputs or be labelled a replay/prototype. Start with a pinned baseline; add our model only after it visibly improves the workflow.

## Source/hosting constraints
- TESSERA annual fused vectors are appropriate for annual mapping/version migration, not monthly event timing or SAR-only fallback.
- PASTIS-R is a paired-sensor crop control; not forest review truth. MultiEarth forest rights/sparse dates remain gates. New alternatives need independent exact source/licence checks.
- [CPU Basic](https://huggingface.co/docs/hub/spaces-overview) has no hourly charge but new Gradio/Docker hosting requires an eligible paid plan; Static is free and qualifying free personal accounts can host limited ZeroGPU Spaces. Don't promise a universally freeCPU service.
- [ZeroGPU](https://huggingface.co/docs/hub/spaces-zerogpu) is optional for bounded encoder inference, Gradio/PyTorch only, with visitor quotas and queues. Not a training backend or reliable always-on service.
- [Disk](https://huggingface.co/docs/hub/spaces-storage) is ephemeral. Export a task/evidence bundle; private uploads not persisted/shared by default. Paid hardware/storage require separate approval.

## Pending independent audits
User/product: a0d67b0691b34a9088f8060a042f83fc
Data/weights:4669a5c19d4849f1aa35103cce0bc456
Novelty/redteam:d9a0d08da3ba4d00a8bf3573de73eb00

## No research-result claim
This note records planning/source findings. No new imagery, model training, user interviews, timing study or deployed demo was run for this comparison.
