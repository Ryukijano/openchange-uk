# UK fit check for the scaling-law study (2026-10-07)

Primary sources read: AI for Science Strategy (GOV.UK, Nov 2025); AIRR AI for Science and AI Open Access calls (UKRI/GOV.UK); AIRR Gateway route and its assessment criteria (GOV.UK guidance); Isambard Access Terms; UKRI AI Strategy news (19 Feb 2026); UKRI "Enabling AI adoption across science and engineering" call (opened 29 Sep 2026, closes 1 Dec 2026); EPSRC themes; ARIA Robot Dexterity; UKAEA Fusion Computing Lab / SUNRISE.

## What the UK names as priorities
- AI for Science Strategy, five priority areas: engineering biology, fusion energy, materials science, quantum technologies, medical research. Chosen for "existing UK strength" and alignment with the Modern Industrial Strategy. Weather/flood and "AI-augmented plasma control" appear in the motivation text. The strategy also names "frontier capability in AI-driven science": general-purpose AI science tools and autonomous labs.
- AIRR AI for Science / Open Access calls: the same five plus "AI-driven research and scientific discovery", and they ask how projects serve the government's five missions (growth, NHS, clean energy, safer streets, opportunity).
- AIRR Gateway (10k GPU-h, 3 months, rolling): assessed on alignment with AIRR scope, clear AI use "including novel aspects", justified compute, outputs, wider benefit, and "does the project support the Industrial Strategy, missions or other relevant policy priorities". Scope text explicitly lists "scalability and adaptability: algorithms that enable AI models to apply knowledge from one domain to another" and "benchmarking of algorithms, code and workflows before applying for larger AIRR opportunities".
- UKRI AI Strategy (Feb 2026, £1.6bn 2026–30): names computing and "agentic AI" as historic UK strengths; maths/CS/engineering underpinning AI.
- UKRI Enabling AI adoption across science and engineering (closes 1 Dec 2026): domain-first; "projects that focus primarily on advancing AI methods, models or algorithms, without a clear link to accelerating progress within a research domain, are out of scope". Priority areas: one of the following priority areas: - engineering biology - advanced materials - quantum technologies - medical research - fusion energy Within the engineering biology priority area, we would particularly encourage applications in AI-enabled biotechnologies relevant to sustainable food systems, clean growth, health and environmental solutions, examples are: - identifying novel biological components, pathways and functional traits - modelling cellular and metabolic systems - predictive genome de
- EPSRC themes include "artificial intelligence and robotics", "advanced materials", "clean energy", "engineering", "healthcare technologies".
- ARIA Robot Dexterity (£57m): hardware and advanced simulation for manipulation; no open call now. A Robot Locomotion programme is being scoped (Apr 2026).
- UKAEA: Fusion Computing Lab, SUNRISE (£45m AI supercomputer, DESNZ-funded), UK–US SUNRISE–STELLAR-AI link (Sep 2026). Fusion AI is a growing, well-funded UK niche.
- Access Terms: u6xn use must stay within the Research Project it was granted for; data only with owner permission; results to be disseminated. BriCS cannot extend end dates or hours.

## Fit of each candidate
| Candidate | Named priority? | Mission hook | Comment |
|---|---|---|---|
| Video world models + robot planning (LIBERO) | Not in the AI-for-Science five. Fits EPSRC "AI and robotics", UKRI "agentic AI", Gateway "scalability and adaptability / domain transfer". | Growth (robotics), weak | Fundable as AI-methods research (Gateway/Open Access AI track), not as AI for Science. Out of scope for the Dec 2026 domain-first call. |
| Physics surrogates on The Well, control in HydroGym | Partly: fluid/MHD datasets touch fusion (plasma) and engineering; not a named area as "generic PDE". | Clean energy if framed via fusion/plasma; engineering | Strongest if the MHD/turbulence datasets and plasma-control framing are used. UKAEA ecosystem is active and UK-specific. |
| Materials (interatomic potentials, property scaling) | Yes, directly | Growth, clean energy | Named priority, but industrially crowded (UMA, Matbench). Novelty harder. |
| Fusion (FAIR-MAST, TokaMark + synthetic MHD) | Yes, directly | Clean energy | Best policy fit; data small; needs synthetic simulation to make a scaling law. |
| Weather/flood (ERA5) | Motivation text, Met Office/Turing own it (FastNet) | Resilience | Crowded on scaling laws. |
| QEC decoders | Quantum technologies, yes | — | Crowded; aarch64 builds. |
| OpenChange-UK coastal EO | Not named | Resilience | Finish small. |

## Reading
1. The study we want (do loss, shifted loss and decision quality scale together?) is an AI-methods question. UK AI-for-Science money is domain-first, and the newest EPSRC call explicitly excludes methods-only work. So the UK fit comes from the substrate, not the question.
2. Physics with a plasma/MHD or engineering flavour gives the best combination: a named priority (fusion energy), the clean-energy mission, exact simulator ground truth, and an active UK ecosystem (UKAEA, SUNRISE) for follow-on compute. The Well includes MHD datasets; HydroGym gives control. A fusion-adjacent framing ("predictive surrogates for plasma/fluid control under regime shift, and how they scale") hits the AI for Science Strategy's own words, "AI-augmented plasma control".
3. The video/robot track is fundable through the general AI routes (Gateway criteria name scalability and cross-domain transfer; UKRI names agentic AI), and ARIA's robotics programmes show appetite, but it will not score as AI for Science and is less likely to attract follow-on UK compute.
4. Doing both (one recipe, two substrates) is defensible: the physics side carries the UK/policy case, the video side carries the robotics/agentic case and the user's trajectory. The cost is the one argued in BUDGET_DESIGN.md.
5. Anything run on u6xn must stay inside the project u6xn was granted for. If u6xn's stated project is OpenChange-UK/EO, a physics or video scaling study may need the allocating body's agreement before it starts. This needs checking against the award letter.
