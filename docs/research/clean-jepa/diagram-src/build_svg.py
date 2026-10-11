
from html import escape
W=1440
COLX=[20+305*c for c in range(5)]
NW,NH=268,70
def node(c,r,title,sub,sub2,detail,y0=80,dim=False,x=None,y=None,w=NW):
    x = COLX[c]+14 if x is None else x
    y = y0+r*100 if y is None else y
    cls="a-node a-node--dim" if dim else "a-node"
    s=f'<g class="{cls}" data-a-node tabindex="0" role="button" data-a-title="{escape(title,quote=True)}" data-a-detail="{escape(detail,quote=True)}">'
    s+=f'<rect x="{x}" y="{y}" width="{w}" height="{NH}" rx="6"/>'
    s+=f'<text class="a-node__title" x="{x+12}" y="{y+24}">{escape(title)}</text>'
    s+=f'<text class="a-node__sub" x="{x+12}" y="{y+42}">{escape(sub)}</text>'
    if sub2: s+=f'<text class="a-node__sub" x="{x+12}" y="{y+58}">{escape(sub2)}</text>'
    return s+'</g>'
def group(x,y,w,h,label):
    return f'<g class="a-group"><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8"/><text x="{x+14}" y="{y+22}">{escape(label)}</text></g>'
def edge(d,label=None,lx=0,ly=0,dim=False):
    s=f'<path class="a-edge{" a-edge--dim" if dim else ""}" d="{d}" marker-end="url(#a-arrow)"/>'
    if label: s+=f'<text class="a-edge__label" x="{lx}" y="{ly}">{escape(label)}</text>'
    return s

parts=[]
# groups, top band
parts.append(group(COLX[0],40,290,420,"1 · open corpora"))
parts.append(group(COLX[1],40,290,420,"2 · curation and packing"))
parts.append(group(COLX[2],40,290,420,"3 · sampler and views"))
parts.append(group(COLX[3],40,290,420,"4 · encoder and projector"))
parts.append(group(COLX[4],40,290,420,"5 · loss and optimiser"))
# bottom band
parts.append(group(COLX[0],500,595,230,"6 · checkpoints and baselines"))
parts.append(group(COLX[2],500,900,230,"7 · evaluation: the three questions"))
parts.append(group(COLX[0],770,1510,110,"8 · gates (nothing on Isambard before these pass)"))

# col 0 corpora
parts.append(node(0,0,"Copernicus-Pretrain S1+S2","247,723 cells · 1.07M patches","~4 dates · CC-BY-4.0 · 6.3-6.9 TB",
 "Location pool. S1 GRD VV/VH, S2 L1C TOA 13 bands, 264 px at 10 m. S1 and S2 are aligned in space only: S2 dates are same-season across 2019-2022, S1 dates differ. 4.2M images per sensor; the 18.7M headline includes S3, S5P and DEM. Fact: dataset card and Copernicus-FM paper."))
parts.append(node(0,1,"SSL4EO-S12 v1.1","251k locations · time-matched S1/S2","median 3-day gap · SEnSeI masks",
 "Supplies the paired S1-S2 views, temporal positives and per-pixel cloud+shadow masks. CC-BY-4.0 reported; pin revision and record licence before any fetch. Already contained in Copernicus-Pretrain, so deduplicate. 80.9 percent of its S2 images have cloud fraction 0 (measured by the data subagent)."))
parts.append(node(0,2,"Major TOM (optional arm)","S2 23 TB · S1 16 TB · CC-BY-SA","only if the 100% point is data-limited",
 "Different licence (CC-BY-SA) and mostly single-date S1/S2 with mismatched semantics. Kept as a separately labelled arm, never mixed silently into the main corpus.",dim=True))
parts.append(node(0,3,"Benchmarks (held out)","Copernicus-Bench · PANGAEA","OpenChange-UK frozen splits",
 "Never enter pretraining. Their footprints and dates go on the exclusion list in step 2. EuroSAT, LCZ and BigEarthNet are city-centred in Europe like the corpus, so overlap must be measured and reported."))
# col 1 curation
parts.append(node(1,0,"Source gate","validate-sources · require_approved()","pin revision · checksums · licence",
 "OpenChange-UK rule: no fetch unless the source passes make validate-sources and require_approved(). Record dataset revision, licence text, SHA-256 of every shard."))
parts.append(node(1,1,"Dedupe · split · exclude","key = Major TOM 10 km cell","UK box 49-61.5N, 11W-2.5E removed",
 "Unique-location key is the Major TOM 10 km grid cell of the patch centre; footprints with IoU over 0.25 in one cell merge. All UK land plus buffer is removed from pretraining because OpenChange-UK regions and cutoff are not yet fixed. Data axis 1/10/100 percent counts cells, stratified by latitude band. Splits are geographic and temporal, never random."))
parts.append(node(1,2,"Availability strata m1-m4","c = SEnSeI cloud+shadow fraction","m2 upper bins and m3 are synthesised",
 "m1 S2 clear (c <= 0.01, ~84 percent). m2 partly cloudy: (0.01,0.1], (0.1,0.3], (0.3,0.7]. m3 S1-only (S2 absent or c > 0.7). m4 both sensors (c <= 0.3 and S1 within 7 days, ~72 percent). Because the corpus is cloud-filtered at collection, m2 upper bins and m3 are made by dropping S2 or pasting real SEnSeI masks from other scenes. The paper must say so."))
parts.append(node(1,3,"Pack shards","uint16 · 12 S2 bands (no B10) + 2 S1","WebDataset ~2 GB · ~0.6 MB per pair",
 "Native resolution uint16, S1 in dB. About 1,250 shards for 4.2M pairs, npy + json per sample, wids index. Crop 264 to 256 at load so 16 px patches tile exactly. Stored bands set bytes read; token dropping does not. Lives in PROJECTDIR (200 TiB), not scratch (5 TiB)."))
# col 2 views
parts.append(node(2,0,"Stratified sampler","global batch N = 1024, fixed","256 per stratum per step (4 × 256)",
 "Per-stratum quotas so every stratum has at least 256 embeddings each step (SIGReg needs enough samples against its finite-sample floor). Rare strata sampled with replacement. Global batch is fixed across the whole grid because SIGReg is multiplied by N."))
parts.append(node(2,1,"Read once → 8 views","2 global 224 px (S1+S2)","6 local 96 px · S1/S2/both .4/.4/.2",
 "Global crops scale 0.4-1.0, local 0.05-0.4. All views regress to the mean of the two global embeddings (the centre). Local views draw a sensor subset so the encoder sees S1-only, S2-only and joint inputs. Samples without clear S2 use the S1 global as target and sit in stratum m3."))
parts.append(node(2,2,"Augmentation","flips · 90° rotations · S2 gain ±5%","S1: ±1 dB offset only",
 "EO has no canonical orientation, so rotations are free invariances. Optical gain jitter stands in for atmospheric variation; S1 gets only a calibration-offset nuisance. Temporal positive (another date as an extra global) is an ablation row only: it may erase change signal."))
parts.append(node(2,3,"Token dropping ρ_g 0.9 / ρ_l 0.5","global 224: 196→20 (one) · 392→39 (both)","local 96: 36→18 · 72→36 · after pos-embed",
 "LeVJEPA-style uniform token dropping applied per view before the encoder; position embeddings kept for the surviving tokens. Cuts encoder FLOPs about 10x; bytes read are unchanged."))
# col 3 model
parts.append(node(3,0,"Per-sensor patch stems","S2: 12×16×16 → d · S1: 2×16×16 → d","+ modality embedding, shared 2-D pos",
 "One linear stem per sensor into one shared ViT. Accepts S1-only, S2-only or both by construction, no zero-filling. Rejected alternatives: one stacked 14-band stem, per-band hypernetwork stems (Copernicus-FM), per-band-group tokens."))
parts.append(node(3,1,"Shared ViT/16 encoder f","Ti 5.7M · S 22M · 40M · B 86M · L 303M","one pre-registered ~1B run",
 "Backbone output f = CLS token (or mean pool, pre-registered). Drop-path 0.1. No EMA teacher, no predictor, no mask tokens: the only collapse control is SIGReg. 40M model is width 512, 12 layers, added so a saturating law can be fitted."))
parts.append(node(3,2,"Projector g → h ∈ R^K","MLP d → 2048 → 2048 → K, BatchNorm","K = 128 fixed across sizes · K=64 abl.",
 "K must be identical for every model size so Q2 can compare projector spaces by Procrustes. The identifiability theorem and SIGReg both act on h, not on f. K=64 is pre-registered because stratified-SIGReg sensitivity to a low-dimensional gap decays about 1/K."))
parts.append(node(3,3,"Embeddings per step","z_{n,v} = g(f(x_{n,v})), n ≤ N, v ≤ 8","μ_n = mean of the 2 global z",
 "Shapes: N x V x K. Under DDP each rank holds N/R samples; the characteristic-function means are all-reduced so SIGReg sees the global batch."))
# col 4 loss
parts.append(node(4,0,"Invariance term L_inv","mean_{n,v,k} (μ_{n,k} − z_{n,v,k})²","no stop-grad · globals also pulled to local mean",
 "Exactly as in the official MINIMAL code: per-coordinate MSE of every view to the mean of the global views. This is the only term that aligns strata (S1-only local onto a full-observation centre)."))
parts.append(node(4,1,"Stratified SIGReg","Σ_m w_m · EP(z in stratum m)","w_m = 1/S · 1024 slices · fp32 · floor ≈ 1.05",
 "Epps-Pulley test of 1-D projections against N(0,1): random unit directions a_m resampled each step (seeded by global step so DDP ranks agree), characteristic function on t in linspace(0,3,17), trapezoid weights, Gaussian window, times N. Pooled SIGReg is nearly blind to a modality gap confined to few coordinates (measured: 1.05 vs 1.07 Gaussian at K=128). Stratifying by availability makes z Gaussian conditional on m, hence independent of m."))
parts.append(node(4,2,"Stratum-mean term T_μ (variant)","Σ_m w_m N_m ||mean_m z||² / K","λ_μ ∈ {0, λ, 10λ} · slice-free, K-free",
 "T_μ = Σ_m w_m N_m ||mean_m z||² / K, null expectation 1. Redundant with stratified SIGReg at the exact optimum but gives a slice-free, K-independent gradient direction on a between-stratum mean gap (measured SNR 3 vs 1.2 for stratified SIGReg at 256 slices). Ablation: none / pooled / stratified / stratified without paired-view invariance / stratified + mean term."))
parts.append(node(4,3,"Total loss and optimiser","L = (1−λ) L_inv + λ SIGReg, λ = 0.05","AdamW β(.9,.95) · wd .05 · clip 3 · WSD · bf16",
 "λ=0.05 from the paper (MINIMAL run uses 0.02); P1 sweeps {0.02,0.05,0.1}. Log L_inv and the unscaled EP separately, never compare loss totals across different N. Warmup-stable-decay so each run yields 5+ cooldown checkpoints for the law fit."))
# bottom band col0-1
parts.append(node(0,0,"LeJEPA grid","5 sizes × data 1/10/100% × 3 budgets","× seeds · cooldowns · ≥390 GPU-h, I/O-bound",
 "Token budgets 2B / 8B / 32B. Each (size, data, seed) run branches cooldown checkpoints. Unit of analysis for Q1 is one independently trained run at its final checkpoint; intermediate checkpoints are not independent points.",y0=540))
parts.append(node(0,1,"Population W sweeps","64 Sobol configs per objective","ViT-S · 2B tokens · ~40 GPU-h est.",
 "The LeJEPA claim (loss ranks checkpoints, rho about 0.85) was measured within one setting over hyperparameters. Sweep lr, wd, lambda in [0.01,0.2], local-view count, cooldown length; 8 configs x 3 seeds for seed noise. Repeat 24 configs at ViT-B/8B.",y0=540))
parts.append(node(1,0,"Matched Barlow Twins + MAE","same backbone, data, tokens, schedule","only the objective differs",
 "Controls for Q1 and Q2. TESSERA v2 (Barlow Twins, 395 runs) found pretraining loss useless for selection across scales; LeJEPA found the opposite within a setting. Both populations W and A are produced for every objective so the two claims are compared on equal terms.",y0=540))
parts.append(node(1,1,"Harness","Cop.-Bench · PANGAEA · OpenChange-UK","frozen linear + kNN + light decoder",
 "EuroSAT-S1/S2, BigEarthNet-S1/S2, DFC2020-S1/S2, Flood-S1, LCZ-S2, Cloud-S2; Sen1Floods11 and BioMassters from PANGAEA (Copernicus-FM baseline rerun, not copied); OpenChange-UK geographic and temporal holdout. GEO-Bench rows for comparability only. About 1.2-1.7 GPU-h per checkpoint on the fast suite (estimate).",y0=540))
# bottom col2-4
parts.append(node(2,0,"Q1 label-free selection","Spearman(proxy, probe) per task family","bootstrap 10k, cluster by config",
 "Proxies: raw loss, lambda-rescaled loss, SIGReg term, invariance term, RankMe, LiDAR, alpha-ReQ (reference unverified), 1k-label probe. Kill at pilot if the LeJEPA raw-loss Spearman upper CI bound is below 0.5 on all four in-distribution families. Success: rho >= 0.7 with lower bound >= 0.5 on 3 of 4 families at S and B; rho_LeJEPA minus rho_BT >= 0.3 with CI excluding 0; OOD lower bound >= 0.4.",y0=540))
parts.append(node(2,1,"Q1 population A","across sizes, data and compute","the TESSERA-style population",
 "Grid endpoints, about 63 runs per objective. Loss distributions shift with scale, so correlation can legitimately differ from population W. Report both; do not pool them.",y0=540))
parts.append(node(3,0,"Q2 theory test on h","orthogonal Procrustes: SVD of H_Aᵀ H_B","n_cal ∈ {500, 1k, 5k, 20k} images",
 "Prediction from arXiv:2605.26379 for two independently trained clean JEPAs at the optimum: h_B = Q h_A with Q orthogonal. Report normalised residual against general linear map and CCA for seed pairs, size pairs, data-scale pairs, per objective. Expected floor (K - n)/K if K exceeds the true latent dimension. S1/S2 views are different observation functions, which the theorem does not cover, so this is a hypothesis test.",y0=540))
parts.append(node(3,1,"Q2 applied test on f","r = acc(probe_A∘M on f_B)/acc(probe_B)","M ∈ {orthogonal, linear, CCA}",
 "The practical question: can a probe trained on model version A be carried to version B with a small calibration fit instead of re-embedding an archive. GeoTessera v1.0/v1.1/v2 pairs (Barlow-Twins family, documented as non-interoperable) are a baseline point, not a theory test.",y0=540))
parts.append(node(4,0,"Q3 sensor substitution","Δ = score(S2 input) − score(S1 input)","vs compute, per task family",
 "Paired S1/S2 tasks: EuroSAT, BigEarthNet, DFC2020; S1-only on Flood-S1 and Sen1Floods11. Theory side-test: variables both sensors see (DEM, water, built-up) should be linearly decodable from S1 embeddings; optical-only ones (NDVI, S5P NO2) should not; pre-stated MLP-minus-linear R-squared threshold.",y0=540))
parts.append(node(4,1,"M diagnostics","probe for m, c from f and h → chance","cross-stratum kNN of the same scene",
 "Pass only if the availability probe falls to chance, the downstream drop when S2 is removed at test time shrinks, and clean-input scores stay within seed noise. Failure mode: strata aligned by rotation rather than overlap (stratified SIGReg passes, cross-stratum retrieval fails).",y0=540))
# gates row
gx=[COLX[0]+14, COLX[0]+14+380, COLX[0]+14+760, COLX[0]+14+1140]
parts.append(node(0,0,"P0-P5 local, CPU only","500-location SSL4EO slice via source gate","tiny LeJEPA + stratified SIGReg, harnesses",
 "P0 data slice (UK box removed, 84 per latitude band, >=25 percent with cloud > 0.05, 2.64 km dedupe, selection by sha256 of sample id at a pinned revision). P1 5M/20M models, pooled vs stratified SIGReg, K in {64,128,256}, 3 seeds, lambda sweep. P2 Procrustes/linear/CCA on seed pairs. P3 proxy harness + two paired Copernicus-Bench loaders + one OpenChange-UK loader. P4 pack shards, bytes per sample, decode throughput. P5 pre-registration draft.",x=gx[0],y=800,w=350))
parts.append(node(0,0,"30-min 1-GPU smoke (human-run)","NGC JAX/PyTorch image · Apptainer --nv","measure MFU and Lustre read bandwidth",
 "Run by the user, never by the agent. Outputs a committed log. Decides whether ViT-S/B are I/O-bound (compute-side demand at 0.35 MFU: S ~50k samples/s/node, B ~13k, L ~3.7k).",x=gx[1],y=800,w=350))
parts.append(node(0,0,"Re-price and freeze protocol","grid cost from measured MFU and bandwidth","pre-registration locked",
 "Current estimate ~3,000 GPU-h = ~750 NHR, ~1,100 NHR with 50 percent contingency = 22 percent of 5,000 NHR (u6xn balance per repo docs, award letter unconfirmed). 35 percent MFU and H100-like peak are unverified assumptions.",x=gx[2],y=800,w=350))
parts.append(node(0,0,"Population-W pilot → main grid","kill criterion applied before scaling up","then matched BT/MAE and eval",
 "Only after the W pilot survives its kill criterion does the full LeJEPA grid, the matched baselines and the 1B run start.",x=gx[3],y=800,w=340))

# edges top band (horizontal between columns at row centres)
def hx(c): return COLX[c]+14+NW
def nx(c): return COLX[c]+14
def ry(r,y0=80): return y0+r*100+NH/2
E=[]
E.append(edge(f"M{hx(0)},{ry(0)} H{nx(1)}", "approved fetch", hx(0)+2, ry(0)-8))
E.append(edge(f"M{hx(0)},{ry(1)} H{nx(1)}"))
E.append(edge(f"M{hx(0)},{ry(2)} H{nx(1)}", dim=True))
E.append(edge(f"M{hx(0)},{ry(3)} H{nx(1)}", "exclusion list", hx(0)+2, ry(3)-8))
# vertical chain in col1
for r in range(3): E.append(edge(f"M{nx(1)+NW/2},{80+r*100+NH} V{80+(r+1)*100}"))
E.append(edge(f"M{hx(1)},{ry(3)} H{nx(2)+NW/2-60} V{80+NH}", "shards + strata", hx(1)+2, ry(3)-8))
for r in range(3): E.append(edge(f"M{nx(2)+NW/2},{80+r*100+NH} V{80+(r+1)*100}"))
E.append(edge(f"M{hx(2)},{ry(3)} H{nx(3)+NW/2} V{80+NH}", "~26-51 tokens/view", hx(2)+2, ry(3)-8))
for r in range(3): E.append(edge(f"M{nx(3)+NW/2},{80+r*100+NH} V{80+(r+1)*100}"))
E.append(edge(f"M{hx(3)},{ry(3)} H{nx(4)+NW/2} V{80+NH}", "z, μ, m", hx(3)+2, ry(3)-8))
for r in range(3): E.append(edge(f"M{nx(4)+NW/2},{80+r*100+NH} V{80+(r+1)*100}"))
# loss -> back to encoder (gradient)
E.append(edge(f"M{nx(4)},{ry(3,80)+20} H{hx(3)+10} ", "∇θ", nx(4)-40, ry(3)+36, dim=True))
# loss -> checkpoint zoo
E.append(edge(f"M{nx(4)+NW/2},{80+3*100+NH} V480 H{nx(0)+NW+24} V575 H{hx(0)+2}", "checkpoints", nx(2), 474))
# zoo -> Q
E.append(edge(f"M{hx(1)},{ry(0,540)} H{nx(2)}", "f, h", hx(1)+2, ry(0,540)-8))
E.append(edge(f"M{hx(1)},{ry(1,540)} H{nx(2)}", "probes", hx(1)+2, ry(1,540)-8))
E.append(edge(f"M{hx(2)},{ry(0,540)} H{nx(3)}"))
E.append(edge(f"M{hx(3)},{ry(0,540)} H{nx(4)}"))
E.append(edge(f"M{hx(3)},{ry(1,540)} H{nx(4)}"))
E.append(edge(f"M{nx(3)+NW/2},{540+NH} V640"))
E.append(edge(f"M{nx(4)+NW/2},{540+NH} V640"))
# gates chain
for i in range(3): E.append(edge(f"M{gx[i]+350},{835} H{gx[i+1]}"))

svg='\n'.join(parts)+'\n'+'\n'.join(E)
open('/home/ubuntu/airr-masterplan/diagram/pipeline_svg.html','w').write(svg)
print(len(svg))
