# Research notes and experiments

**Current entry point: [progress summary](../PROGRESS.md)** and
[task-first research archive](eo-task-research/README.md). These cover problem
selection, independent reviews, visual guides, the stopped three-site audit and
the negative Narrabeen + Duck metadata-head experiment. A new foundation model
is not yet justified by these results.

The [clean-JEPA EO design](clean-jepa/README.md) contains the earlier selected
pilot design, exact formulation, implementation gates, sources and standalone
training diagram. The separate [implementation PR #2](https://github.com/Ryukijano/openchange-uk/pull/2)
contains later synthetic CPU/distributed validation and metadata audits; these
are engineering evidence, not coastal/EO task performance. Neither track changes
the benchmark's held-out specification or authorizes data/compute access.

The older predictive-model/planning explorations below are retained for provenance. They are superseded as the next project direction, and their budgets/claims are not implementation instructions. The wider AIRR programme planning (masterplan, execution plan, world-action-model bridge and their reviews) is archived in [airr-programme](airr-programme/).

## Earlier synthesis (superseded)
- `deep-think.md` — v2 synthesis: the loss-vs-decisions question, the planner-reachable-error reformulation, substrate verdicts (TORAX/FreeGSNKE first, LIBERO pilot, The Well out), pilots P1–P3, kill criteria, risks, gate verdicts.
- `deep-explorer-report.md`, `deep-pessimist-report.md`, `deep-verifier-report.md` — three independent child-session reports that fed v2 (formulations, red-team, claim checks).

## Earlier rounds
- `world-model-updates-oct2026.md` — what shipped in world models (Sept–Oct 2026).
- `video-world-models.md`, `science-testbeds.md` — video/robot and physics substrate surveys.
- `budget-design.md` — staged GPU-hour estimates (needs redo per the verifier's corrections).
- `uk-fit.md` — UK AI-for-Science/AIRR priority alignment.
- `scaling-laws-agent-review.md` + `agent-report-{explorer,pessimist,verifier}.md` — first agent round on the SUMO/LIBERO framing (SUMO since dropped).
- `problem-brainstorm-early.md`, `problem-choice-early.md`, `scaling-laws-plan-early.md` — earliest drafts, kept for provenance.

## Guardrails carried from this repo's rules
No dataset downloads without source approval, no Isambard jobs or SSH from the agent, no result claims without committed logs, splits and protocols frozen before confirmatory runs, honest reporting including negative results.
