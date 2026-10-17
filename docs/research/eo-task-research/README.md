# EO task research: reading guide and archive

**Start with [What we have done so far](../../PROGRESS.md).** Updated 15 October
2026. It separates measurements, stopped experiments, proposals and unresolved
questions. The negative two-beach result supersedes earlier recommendations to
proceed with a metadata-only model.

## Visuals

Download an HTML file and open it locally for interaction; GitHub normally shows
HTML source rather than running it. No GitHub Pages site or Hugging Face Space
has been deployed. PDFs are static versions of the older design guides.

| Guide | HTML | PDF | How to read it |
|---|---|---|---|
| Latest coastal results | [Open/download](visuals/COASTAL_RESULTS_VISUAL.html) | — | Data gate, folds and failed performance target; wording corrected for publication |
| Coastal model and pretraining route | [HTML](visuals/COASTAL_MODEL_VISUAL.html) | [PDF](visuals/COASTAL_MODEL_VISUAL.pdf) | Earlier one-site screen and conditional training design; not the latest two-site result |
| Task-first plan | [HTML](visuals/TASK_FIRST_PLAN_VISUAL.html) | [PDF](visuals/TASK_FIRST_PLAN_VISUAL.pdf) | Explanatory drawings and overlap review; illustrative controls/mock-up values are not experimental evidence |
| Scientific problem comparison | [HTML](visuals/EO_RESEARCH_VISUAL.html) | [PDF](visuals/EO_RESEARCH_VISUAL.pdf) | Earlier shortlist; ratings are reviewer judgments, not measured impact |

## Research notes, in decision order

| Note | Role / present status |
|---|---|
| [Early task dossier](notes/topic_dossier.md) | Historical forest-evidence/continuity proposal; not the selected task |
| [Demo bets](notes/HF_DEMO_BETS.md) | Early draft before reviewers were incorporated |
| [Continuity demo recommendation](notes/HF_DEMO_RECOMMENDATION.md) | Historical recommendation; not evidence of adoption or savings |
| [Shared demo requirements](notes/DEMO_BUILD_REQUIREMENTS.md) | Useful design constraints; budgets and targets are proposals |
| [Higher-impact problem comparison](notes/HIGHER_IMPACT_SCIENTIFIC_PROBLEMS.md) | Five independent audits plus a red team; research recommendations |
| [Task-first training plan](notes/TASK_FIRST_TRAINING_PLAN.md) | Coastal small-head falsifier first; forecasting backup; no method commitment |
| [Coastal training audit](notes/coastal_change_training_plan.md) | Conditional small-head design before results |
| [Forecasting training audit](notes/forecast_observe_training_plan.md) | Backup; data/timestamp/licence gates are not a training authorization |
| [Event-search training audit](notes/event_search_training_plan.md) | No trained model justified for the assigned morphology-retrieval task |
| [Overlap paper review](notes/OVERLAP_PAPERS_REVIEW.md) | Coastal paper abstract-only; forecasting full text |
| [Coastal plan and first screen](notes/COASTAL_MODEL_PLAN_AND_SCREEN.md) | Measured one-site screen plus conditional pre/post-training recipe; predates the stopped and replacement experiments |

The notes are preserved byte-for-byte. Their dates, proposed next steps, statements
about permissions at that time and relative/absolute workspace paths are
historical. Use the current summary and experiment JSON for present status.
Reviewer suggestions are not verified results or legal advice. For example,
downstream licence obligations for EarthNet-derived weights/Space code need
separate legal review; an early review's speculation does not settle them.

## Independent reviewer reports

Original reports are in [reviews](reviews/). They record the supporting and
opposing arguments rather than only the final choice:

- Early scope: [workflow burden](reviews/early-costly-workflows.md),
  [novelty audit](reviews/early-novelty-audit.md).
- Demo comparison: [product workflows](reviews/demo-product-workflows.md),
  [data/model feasibility](reviews/demo-data-model-feasibility.md),
  [sceptical review](reviews/demo-sceptical-review.md).
- Impact comparison: [forecasting](reviews/impact-forecasting.md),
  [coastal](reviews/impact-coastal-change.md),
  [disaster transfer](reviews/impact-disaster-transfer.md),
  [restoration](reviews/impact-restoration.md),
  [event search](reviews/impact-event-search.md),
  [red team](reviews/impact-red-team.md).
- Broader coastal model: [pretraining red team](reviews/coastal-pretraining-red-team.md),
  [latest data-feasibility report](reviews/coastal-data-feasibility.md).

These are research snapshots, not endorsements of every claim. Later realised
data audits take precedence over earlier suggestions that an archive is usable.

## Experimental evidence and earlier engineering

- [Coastal experiment index](../../../reports/coastal/README.md): one-site screen,
  stopped three-site protocol and completed two-site comparison.
- [TESSERA migration pilot](../../../reports/tessera/2026-10-13/README.md): code
  and aggregate outputs; label/feature payloads excluded.
- [Publication scope and checksums](../../../reports/research-publication/2026-10-15/README.md).
- [Earlier programme archive](../airr-programme/README.md),
  [clean-JEPA design](../clean-jepa/README.md), and
  [separate implementation PR #2](https://github.com/Ryukijano/openchange-uk/pull/2).

No datasets, model weights, source-clone snapshots or third-party paper copies
are added by this archive. Data manifests and licence evidence are retained with
the experiments. This publication does not authorize new downloads or compute.
