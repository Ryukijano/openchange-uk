# Research notes: EO clean-JEPA study and earlier explorations

The current research direction is the [clean-JEPA EO pilot](clean-jepa/README.md): selected design, exact formulation, implementation gates, sources and a standalone training diagram. This is a gated pretraining/representation study beside the OpenChange-UK benchmark, not a change to its held-out specification. No JEPA training is implemented or run yet.

The older predictive-model/planning explorations below are retained for provenance. They are superseded as the next project direction, and their budgets/claims are not implementation instructions.

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
