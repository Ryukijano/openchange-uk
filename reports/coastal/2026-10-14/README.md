# Narrabeen+Duck registration and data gate — 2026-10-14

Exploratory counts-first audit, then a NEW registered protocol
(code/protocol_narrabeen_duck_20261014.md). The failed three-site protocol is
preserved unmodified under 2026-10-13.

## Gate outcome

| Site | ±48 h rows/scenes | fit | cal | test | Gate |
|---|---:|---:|---:|---:|---|
| Narrabeen | 661/144 realised | 138/31 | 194/45 | 329/68 | PASS |
| Duck | 679/204 realised | 252/77 | 79/22 | 348/105 | PASS |
| Truc Vert | 10/5 realised | — | — | — | FAIL |
| Torrey | payload unavailable | — | — | — | UNKNOWN |

Duck figures are REALISED (contour-quality exclusions applied): 561 accepted
contour rows, 115 of 676 profile-dates rejected. Survey dates are date-only
(midnight UTC contract). Caveats kept: mean off-axis 9.5 m, max 230.6 m;
CoastSat-SHOREX disagreement missing for 29% of Duck rows.

## Provenance

- Duck: USACE CHL THREDDS public catalog; licence verbatim from file metadata
  (unrestricted use, must acknowledge USACE FRF + supporting investigators).
  Only the 169 files matching scene dates ±48 h on transects [-91,1,1006,1097]
  were fetched: 157,193,250 bytes (corrects an initial 25 MB estimate).
  Per-file SHA-256 in evidence/duck_downloads.jsonl.
- sources.yaml: updated allowlist, appended duck-frf-surveys entry.
- No imagery, weights, GPU or cluster work. Payloads and derived CSVs are
  excluded from this archive.
