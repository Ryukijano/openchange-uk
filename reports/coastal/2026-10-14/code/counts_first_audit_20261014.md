# Counts-first data-contract audit — 2026-10-14

Status: exploratory audit, NOT a registered experiment. One question: which
candidate survey-label sources can supply enough close-time whole-scene matches
to make a new, separately-registered experiment worth writing? No model fitted,
no fold run, nothing here retroactively changes the stopped
coastal-three-site-20261013 protocol, which stays failed at its pre-specified
gate and preserved unmodified.

## Method

- Same matching contract: nearest survey date per transect/scene, absolute gap,
  primary window ±48 h; fit <=2009, cal 2010-2016, test >=2017.
- Survey dates carry no time-of-day; matching uses midnight UTC, so Duck counts
  are an UPPER BOUND before contour-quality exclusions. Truc Vert's counts are
  realised (post-exclusion).
- Duck survey dates are catalog filenames only (metadata fetch); no .nc
  payloads downloaded. USACE CHL THREDDS public catalog
  (frf/geomorphology/elevationTransects/survey); licence terms still need a
  formal check before payload download or a sources.yaml entry.

## Results

| Site | Survey dates | Scenes avail | ±48 h rows/scenes | fit | cal | test | Gate? |
|---|---:|---:|---:|---|---|---|---|
| Narrabeen | raw profiles on disk | 891 | 661/144 realised | 138/31 | 194/45 | 329/68 | PASS |
| Truc Vert | 294 | 133 | 10/5 realised | 1/1 | 2/1 | 7/3 | FAIL |
| Duck | 1,203 catalog | 3,487 obs (4 fixed transects) | 816/212 upper bound | 304/81 | 90/23 | 422/108 | PASS (pre-exclusion) |
| Torrey Pines | payload unavailable (401/403) | ? | ? | ? | ? | ? | UNKNOWN |

Duck evidence: duck_potential_counts.json from count_potential_duck.py
(stdlib + catalog HTML, no pandas; restricted to the four fixed benchmark transects [-91,1,1006,1097]). Exclusions later can only reduce it.

## What a new contract could look like

- Viable: Narrabeen + Duck — fit on Duck's dense record, calibrate/test across
  the two sites. A three-site design needs a third source: Torrey (user upload)
  or a new approved source.
- Truc Vert is out in any role; adding profiles cannot fix a 5-scene shortfall.
- Any new design needs its own registered protocol BEFORE model code, with the
  counts gate verified on REALISED (post-exclusion) matches, not this bound.

## Next gated step

Realised Duck counts need the Duck .nc payloads or the benchmark's processed
Duck product — a new source approval. Bounded subset: only matched dates,
only the transects the protocol selects (~a few hundred MB worst case; exact
bytes after catalog size listing). Recommend approving a bounded Duck survey
acquisition only after a new protocol is agreed.
