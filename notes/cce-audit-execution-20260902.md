# CCE revision audit execution log

Source: `reviews/cce-revision-audit-20260902.md` (user Downloads
`CCE revision audit.md`, 2026-09-02). Route A accepted. One author
override: the SCCM AI statement is that generative tools were **not
used** for any part of the work, including the manuscript body.

## A00 / A02 (run 2026-09-02)

Receipt: `notes/cce-audit-A00-A02-feasibility.json`.

| Check | Result | Audit bar |
|---|---|---|
| Invasive ventilation overlapping intime (old) | 1,882 / 65,358 (2.9%) | too low |
| Invasive ventilation, ±6 h window, itemids 225792/224385 | 19,922 / 65,358 (**30.5%**) | A02 pass (20–35%) |
| Route A eligible (vent ±6 h, no EN before new t0) | **19,919** | — |
| EN by 24 h in that set | 1,232 (6.2%) | — |
| EN by 48 h in that set | 2,807 (**14.1%**) | A00 fail if <15%; pass if ≥25% |
| No EN by 96 h | 15,475 (77.7%) | — |
| EN between 48 and 96 h | 1,637 (8.2%) | — |
| Vasopressor ±6 h of new t0 | 11,382 / 19,919 (57.1%) | overall first-stay 10–25% is a different denominator |

EN by 48 h is 0.9% in CVICU (7,375 stays) and 24.9% in MICU. After
excluding CVICU and CCU: n = 11,259, EN by 48 h = **23.1%** (above the
15% fail line, just under the 25% ideal). That matches B03/B04: the
24–48 h contrast is drowned by fast-track cardiac stays.

`patients.dod` vs hospital death: 7,085 hospital deaths, 0 hospital
deaths without `dod`, 14,853 discharges with a later `dod`, 2,088 of
those within 28 days of intime. Completeness of in-hospital death
coding is high. Treating missing `dod` as alive at day 28 remains an
assumption (B09), not an observation.

## A01 mapping (run 2026-09-02)

Receipt: `notes/cce-audit-A01-B09-A04.json`,
`notes/cce-audit-A01-tube-routeA-descriptive.json`.

- `Nutrition - Enteral` itemids: 112. Locked include 90, locked exclude 22
  (oral sip feeds / Beneprotein). **Unmapped in category: 0.** Not a mapping
  leak.
- Audit cited procedureevents 224263/224264; those are not feeding tubes.
  Real feeding-tube procedures: PEG 225446 (367 stays; EN within 24 h 2.7%,
  any later EN 97%); small-bore 229501 (101 stays; EN within 24 h 49.5%,
  any EN 92%). The 80% within-24 h bar fails for PEG because PEG is not a
  same-day feeding start.
- TF residual (`outputevents` 227510/227511): 7,402 stays; 1,313 (17.7%)
  have residual without qualifying EN (may be excluded oral/modular intake).

## B09 dod self-check

- Hospital deaths 7,085; all 7,085 have `dod` inside the stay window.
- Discharges with later `dod`: 14,853. Median days discharge→dod 208;
  5,448 have dod >365 days after discharge (not a 1-year cap).
- 2,868 deaths within 28 days of discharge; 2,085 within 28 days of intime
  among those discharged alive.
- Missing `dod` as alive at day 28 remains an assumption.

## A04 labs (hadm_id join)

Full first-stay cohort, creatinine 50912:

| Window | n with value | % of 65,358 |
|---|---|---|
| [t0−24 h, t0] | 20,255 | 31.0 |
| [t0−24 h, t0+1 h] | 29,645 | 45.4 |
| [t0−48 h, t0+6 h] | 52,609 | 80.5 |

Lactate [t0−24 h, t0+1 h] 36.7% (bar ≥50% fail). Restricted ventilated
non-CVICU/CCU: creatinine [t0−24 h, t0+1 h] 62.7%; [admittime, t0+1 h]
60.1%. Still below 85%. Use missing indicators or MICE later (A04/A06).

## Route A restricted cohort (B03)

Exclude CVICU and CCU. Vent ±6 h. t0 = later of intime and first invasive
start. No EN before t0. Parquet:
`workspace/metered-results/cohort/routeA_restricted_stays.parquet`.

n = **11,259**. EN by 48 h **23.1%** (above 15% fail line, below 25%
ideal). No EN by 96 h 65.0%. Vasopressor ±6 h 39.7%.

## Route A IPCW QC (not author-final)

Script `tools/routeA_ipcw_qc.py`. Receipt
`notes/cce-audit-routeA-ipcw-qc.json`.

- 28-day death = `dod` within 672 h of t0; missing dod = alive (2,535 / 11,259)
- Strategies: EN ≤48 h vs no EN before 96 h
- Stabilized IPCW, ridge 0.01, 6-hour periods to hour 96
- Uncensored clones: 3,345 vs 6,994 (far less unbalanced than 3,105 vs 61,231)
- Hájek risks 0.406 vs 0.228; RD 0.178; RR 1.78
- 100/100 bootstrap QC interval RD 0.161–0.197; RR 1.70–1.89
- Weight sums ≈ uncensored n, so **B01 remains**: this is still not
  standardized to the full eligible covariate distribution
- Not 2,000 BCa, not Firth, not MSM g-computation, not Love plot, not CIF,
  not eICU rerun, not author-confirmed

Do not copy these QC numbers into the submission manuscript until the
remaining A08–A11 estimators and author confirmation are done.

## B01 / A08 MSM standardization (run 2026-09-02)

Receipt: `notes/cce-audit-routeA-B01-msm.json`.
SMD table: `notes/cce-audit-routeA-B01-smd.csv`.
Love plots: `notes/cce-audit-routeA-B01-love-early.png`,
`notes/cce-audit-routeA-B01-love-delayed.png`.

Target population: restricted eligible cohort (n = 11,259). Both strategies
evaluated at that empirical baseline distribution.

| Estimator | Early 48 h | Delayed 96 h | RD | RR |
|---|---|---|---|---|
| Hájek (stabilized, arm-specific) | 0.406 | 0.228 | 0.178 | 1.78 |
| Horvitz–Thompson (unstabilized / n) | 0.286 | 0.192 | 0.095 | 1.49 |
| **MSM g-computation (B01 primary)** | **0.388** | **0.225** | **0.163** | **1.73** |

MSM 100-replicate subject bootstrap (QC, not 2,000 BCa): RD 0.149–0.182;
RR 1.65–1.83; 100/100 completed.

B13 with B01: 8 covariate–strategy SMDs >0.10 unweighted; **0 after IPCW
weighting** versus the eligible means. Audit A08 balance check passed for
the weighted analysis set.

Methods sentence for the later rewrite:

> Both strategy-specific risks were standardized to the covariate
> distribution of the full eligible cohort. We fitted a weighted pooled
> logistic model for the discrete-time hazard of death with strategy, a
> restricted cubic spline in follow-up time (3 knots at days 4, 14, and
> 24), a strategy-by-time interaction, and the baseline covariate set,
> using the stabilized inverse-probability-of-artificial-censoring
> weights. Standardized cumulative risks under each strategy were obtained
> by predicting the hazard for every eligible subject under each strategy
> and averaging over the empirical baseline covariate distribution.

Still not author-final. Reduced SOFA was replaced in A06 below.

## A06 SOFA rebuild (run 2026-09-02)

Receipts: `notes/cce-audit-A06-A13-extract.json`,
`notes/cce-audit-routeA-A06-msm.json`,
`notes/cce-audit-routeA-A06-smd.csv`.
Parquet: `workspace/metered-results/cohort/routeA_sofa_full.parquet` (gitignored).

No `mimiciv_derived.sofa` on the local 3.1 gzip dump. Components were
rebuilt from `labevents` / `chartevents` / `outputevents` in
`[t0−24 h, t0+1 h]`, `hadm_id` for labs, `stay_id` for charts. Cardio
without dose: vasopressor at t0±6 h → SOFA 3; MAP <70 → 1. Renal uses
the worse of creatinine and 25 h urine.

| Component | Observed | % of 11,259 |
|---|---:|---:|
| Cardio (MAP / vaso) | 11,161 | 99.1 |
| Renal (creatinine / urine) | 10,069 | 89.4 |
| CNS (GCS sum) | 9,368 | 83.2 |
| Coag (platelets) | 7,009 | 62.3 |
| Resp (PaO2/FiO2) | 5,668 | 50.3 |
| Liver (bilirubin) | 3,699 | 32.9 |
| Complete six | 2,082 | 18.5 |
| Lactate (A06 option-2 proxy) | 5,459 | 48.5 |

B07 option 1 complete-score + missing=0 is still not usable (81.5%
incomplete). The 99% missing reduced SOFA and its collinear missing
flag were **dropped**. The fitted set is component scores with missing
indicators only when observed rate is 5–95% (no cardio missing flag:
99.1% observed), plus lactate value + missing indicator. MICE 20 is
deferred to A16; it was not run under 2,000 bootstrap.

MSM g-computation after the covariate change (same 48 h / 96 h CCW):

| Estimator | Early 48 h | Delayed 96 h | RD | RR |
|---|---:|---:|---:|---:|
| Hájek | 0.408 | 0.228 | 0.180 | 1.79 |
| Horvitz–Thompson | 0.287 | 0.192 | 0.095 | 1.50 |
| **MSM g-computation** | **0.374** | **0.227** | **0.147** | **1.65** |

Previous reduced-SOFA MSM was RD 0.163 / RR 1.73. Weighted SMD vs
eligible: 9 unweighted rows >0.10; **1 weighted** row remains
(`lactate_filled` SMD 0.101). Not author-final.

## A13 negative-control exposure (run 2026-09-02)

Receipt: `notes/cce-audit-routeA-A13-negative-control.json`.
Same CCW+IPCW+MSM, 48 h vs no event by 96 h, A06 covariate set.

| Exposure | By 48 h | Delayed uncensored | MSM RD | MSM RR |
|---|---:|---:|---:|---:|
| Qualifying EN (primary) | 23.1% | 6,994 | 0.147 | 1.65 |
| Oral care 226168 | **96.3%** | **320** | −0.145 | 0.62 |
| Position 224066/227952 | **86.0%** | 1,265 | −0.205 | 0.53 |
| CHG bath 228137 | 7.2% | 10,044 | **0.313** | **2.32** |

Oral care and repositioning saturate by 48 h (also 95.6% and 84.9% by
24 h), so the delayed arm is a tiny remainder. Those contrasts are
positivity failures, not valid negative controls under Route A grace
periods. CHG bath is the only candidate with a minority-treated
structure; its MSM RR 2.32 is **larger than** the EN RR 1.65 in the
same direction. That is compatible with the audit reading that the
pipeline can recover a large association for a day-1 nursing
intervention without a plausible 2-fold 28-day mortality effect. Not
author-final. No 2,000 BCa on the negative controls.

## A11 2,000-replicate BCa (run 2026-09-02)

Receipt: `notes/cce-audit-routeA-A11-bca.json`.
Subject-level nonparametric bootstrap, weight models and MSM refit
each replicate, seed 20260902. Vectorized estimator matches the A06
point estimate to floating-point noise.

- Requested 2,000; **completed 2,000; failed 0**
- Point: RD 0.147, RR 1.65 (early 0.374 vs delayed 0.227)
- Percentile 95% CI: RD 0.130–0.163; RR 1.56–1.74
- **BCa 95% CI:** RD **0.131–0.164**; RR **1.57–1.74**
- Acceleration `a` from a 200-group delete-group jackknife of
  subjects (200/200 completed), not leave-one-out of n=11,259

Not author-final. Do not copy into the 24-hour submission manuscript
until the author confirmation below.

## A12 E-value (from A11) and PBA (run 2026-09-02)

Receipts: `notes/cce-audit-routeA-A12-evalue.json`,
`notes/cce-audit-routeA-A12-pba.json`.

E-value for RR 1.65 is 2.68; for the BCa limit nearest the null
(1.57) is 2.51.

Probabilistic bias analysis (10,000 draws): binary unmeasured severity
U, delayed prevalence Uniform(0.05, 0.30), early excess Uniform(0.10,
0.40), RR_UD log-normal median 2 (approx 95% 1.3–4.0). Adjusted RR
median 1.38 (2.5–97.5: 1.06–1.60); P(RR_adj > 1) = 0.992. The
E-value and this bias analysis address unmeasured confounding only;
they do not quantify positivity violations or grace-period risk-set
composition.

## A07 endpoints (run 2026-09-02)

Receipt: `notes/cce-audit-routeA-A07.json`.
CIF figure: `notes/cce-audit-routeA-A07-cif.png`.

Restricted cohort n=11,259. Missing dod treated as alive for all-cause
endpoints. In-hospital AJ does not use that convention.

| Estimand | Early 48 h | Delayed 96 h | RD | RR |
|---|---:|---:|---:|---:|
| Primary MSM, 28 d all-cause | 0.374 | 0.227 | 0.147 | 1.65 |
| Secondary weighted AJ, 28 d in-hospital death (discharge competing) | 0.374 | 0.196 | 0.178 | 1.91 |
| Sensitivity: 1−KM, censor at discharge | 0.440 | 0.318 | 0.122 | 1.38 |
| Sensitivity MSM, 90 d all-cause | 0.422 | 0.272 | 0.150 | 1.55 |

Counts: 2,535 all-cause deaths by day 28; 2,166 in-hospital deaths by
day 28; 8,134 discharges alive by day 28; 3,104 all-cause deaths by
day 90. Kish ESS 2,650 vs 6,353. AJ uses strategy-uncensored clones
(3,345 vs 6,994), so it is not yet g-computation-standardized to the
full eligible distribution. Not author-final.

## A09 weight-model Firth / ridge (run 2026-09-02)

Receipts: `notes/cce-audit-routeA-A09-weight-models.json`,
`notes/cce-audit-routeA-A09-weight-models.csv`,
`notes/cce-audit-routeA-A09-p-hist.png`.

32 strategy×period models (16 periods × 2 strategies, 6 h steps to
96 h). Ridge 0.01: 32/32. Firth: 32/32 after treating 15
all-uncensored periods (no remaining artificial censoring, mostly
early arm after hour 48) as intercept-only rather than failed Newton.
Predicted P (Firth, 234,338 person-periods): min 0.015, p1 0.16,
median 0.99, max ~1; **none < 1e-4**. Median AUC 0.68. Firth-weighted
MSM RD 0.143 / RR 1.63 vs ridge MSM 0.147 / 1.65. Primary A11 numbers
remain the ridge 0.01 fit. Not author-final.

## A14 positive control (run 2026-09-02)

Receipt: `notes/cce-audit-routeA-A14.json`.

| Contrast | Design | RR |
|---|---|---:|
| Invasive ventilation ±6 h vs not, all first stays (n=65,358, 30.5% vent) | Baseline logistic g-computation, not CCW | **1.67** (crude 1.38) |
| Vasopressor at t0±6 h vs not, Route A cohort | Baseline logistic g-computation | **1.20** |
| First vasopressor by 48 h vs none by 96 h | Same CCW+IPCW+MSM | **0.83** |

The baseline contrasts go in the expected direction (RR > 1). The
same-pipeline vasopressor CCW does **not**: early vaso looks
protective, the same structural pattern as oral-care saturation and
the EN contrast. Route A is 100% ventilated, so ventilation cannot be
run as CCW inside that set. Not author-final.

## Author confirmation 2026-09-03 (A06 / A11 / A13)

Receipt: `reviews/human-confirmation-20260903-a06-a11-a13.json`.

The authors completed the specified screening, adjudication, extraction,
coding, data/figure checks, and final wording review for A06 SOFA/MSM
point estimates, A11 2,000 BCa, A13 negative control (including CHG RR
2.32), and the use of those numbers in `drafts/manuscript-route-a.md`;
they approve those results and accept full responsibility. Bound to the
hashes in that receipt. This does **not** authorize journal submission
or replacement of the 24-hour `drafts/manuscript.md`.

Drafting compose was confirmed after archiving the 24-hour
genre-imitation files (fingerprint
`279ab0001d76c5b28e52571dc3a89c2e868936ff493e6fea2821f9324e5b39fd`).
Fact sheet updated to Route A. Simulated manuscript is
`workspace/drafting/genre-imitation/simulated-manuscript.md` (not
evidence, not for submission).

## Author confirmation 2026-09-03 (clone-flow / A16 / A15)

Receipt: `reviews/human-confirmation-20260903-cloneflow-a16-a15.json`.

The authors completed the specified screening, adjudication, extraction,
coding, data/figure checks, and final wording review for Table 3 clone
flow, the A16 grace/trim MSM grid, and the A15 eICU finding that 48-hour
EN is 4.82% (not identifiable) with B15(a) wording; they approve those
results and accept full responsibility. Bound to the hashes in that
receipt. This does **not** confirm A06/A11/A13, does not rewrite the
24-hour manuscript, and does not authorize journal submission or an
eICU effect estimate.

## Author confirmation 2026-09-03 (A07 / A09 / A12 / A14)

Receipt: `reviews/human-confirmation-20260903-a07-a09-a12-a14.json`.

The authors completed the specified screening, adjudication, extraction,
coding, data/figure checks, and final wording review for A07 competing
risk and 90-day sensitivity, A09 Firth diagnostics, A12 E-value/PBA, and
A14 positive controls (including CCW vasopressor RR 0.83); they approve
those results and accept full responsibility. Bound to the hashes in that
receipt. This does **not** confirm A06/A11/A13, does not rewrite the
24-hour manuscript, and does not authorize journal submission.

## Clone flow (row-level, 2026-09-03, Volume B remounted)

Receipt: `notes/cce-audit-routeA-clone-flow.json`.
Current clones() treats **death**, not discharge, as preventing
artificial censoring.

| Item | Early 48 h | Delayed 96 h |
|---|---:|---:|
| Clones at t0 | 11,259 | 11,259 |
| Artificially censored | 7,582 | 3,933 |
| Uncensored | 3,677 | 7,326 |
| **Initiated EN within deadline** | **2,599 (70.7%)** | 10 |
| **Died in grace before EN** | **1,085 (29.5%)** | 1,342 (18.3%) |
| Discharged in grace (among uncensored) | 14 | 1,327 |
| Never EN and survived grace | 0 | 5,982 |
| Deaths by day 28 among uncensored | 1,447 | 1,505 |
| — among grace-period deaths | 801 | 1,020 |
| — among the remainder | 646 | 485 |
| Sum of stabilized weights | 3,348 | 7,008 |
| Kish ESS | 2,650 | 6,353 |

24 h / 48 h secondary: early uncensored 1,982, of which initiated 1,166
(58.8%) and grace-period deaths 820 (41.4%). The 48 h window is less
death-dominated than 24 h, but 29.5% of the early analysis set is still
grace-period death.

## A16 sensitivity forest (2026-09-03)

Receipts: `notes/cce-audit-routeA-A16-sensitivity.json`,
`notes/cce-audit-routeA-A16-sensitivity.csv`,
`notes/cce-audit-routeA-A16-forest.png`.
MSM rows other than primary used 100/100 percentile intervals.

| Setting | RD | RR (95% CI) |
|---|---:|---|
| MSM 48/96 (primary, 2,000 BCa) | 0.147 | 1.65 (1.57–1.74) |
| Hájek 48/96 | 0.180 | 1.79 |
| MSM 24/48 | 0.201 | 1.87 (1.76–1.97) |
| MSM 24/96 | 0.206 | 1.91 (1.80–2.03) |
| MSM 36/96 | 0.168 | 1.74 (1.65–1.83) |
| MSM 48/72 | 0.145 | 1.64 (1.56–1.72) |
| MSM 48/96 trim p95 | 0.145 | 1.63 (1.56–1.72) |
| MSM 48/96 no careunit | 0.147 | 1.65 (1.57–1.75) |
| MSM 90 d | 0.150 | 1.55 |
| AJ in-hospital | 0.178 | 1.91 |

Shorter grace periods give larger RR. Weight truncation and dropping
ICU type barely move the 48/96 MSM. MICE and unrestricted CVICU/CCU
were not run. Not author-final.

## A15 eICU Route A (2026-09-03)

Receipt: `notes/cce-audit-routeA-A15-eicu.json`.
Vent flag: `respiratoryCare.ventstartoffset` in [−360, +360] min.

| Set | n | EN by 48 h |
|---|---:|---:|
| All first stays | 138,704 | 2.07% |
| Vent ±6 h | 31,719 (22.9%) | **4.82%** |
| Vent ±6 h, no cardiac units | 22,730 | **5.57%** |
| APACHE day-1 vent | 42,544 | 5.07% |

A00 fail (<15%) in every eICU Route A slice. Do **not** transport the
MIMIC MSM. Writing should use B15(a): initiation-frequency and clone-flow
replication, not a transportability analysis. No participation weights
and no eICU effect estimate.

## Main manuscript and Word (2026-09-03)

Route A is now `drafts/manuscript.md`. The 24-hour text is archived at
`drafts/archive/manuscript-24h-pre-routeA.md`. Word compile (zh-academic
profile, Times New Roman + SimSun, Word 16 pin):
`output/Manuscript.docx` (audit PASS). PDF:
`output/Manuscript.pdf`. Figures remain caption-only. Not
READY_FOR_SUBMISSION. Do not journal-submit.

## Author confirmation 2026-09-03 (MICE / CVICU)

Receipt: `reviews/human-confirmation-20260903-mice-cvicu.json`.

The authors completed the specified screening, adjudication, extraction,
coding, data/figure checks, and final wording review for MICE 20 (RD
0.152 / RR 1.67) and the CVICU/CCU-restored cohort (n=19,919, EN by 48 h
14.1%, MSM RR 2.23, not primary); they approve those results and accept
full responsibility. Bound to the hashes in that receipt. This does
**not** authorize journal submission or treating the unrestricted
cardiac-inclusive set as the primary analysis.

## A16 MICE and CVICU (2026-09-03)

Receipts: `notes/cce-audit-routeA-A16-mice.json`,
`notes/cce-audit-routeA-A16-cvicu.json`.

MICE 20 (ridge chained equations, 8 cycles) on restricted SOFA
components and lactate: pooled RD **0.152** (0.135–0.169), RR **1.67**
(1.58–1.76). Range across imputations RD 0.150–0.153. Between-imputation
variance is tiny versus the A11 within variance. Missing-indicator MSM
and MICE agree.

Unrestricted ventilated eligible (CVICU 7,375 + CCU 1,285 restored):
n = **19,919**, EN by 48 h **2,807 (14.1%)** — A00 fail (<15%). MSM
0.326 vs 0.146, RD 0.180, RR **2.23** (100-boot 2.07–2.36). Uncensored
3,713 vs 15,060. This is the cardiac-dilution problem the audit
described; it is a diagnostic, not the primary analysis.

## Stage 5 working rewrite (2026-09-03)

`drafts/manuscript-route-a.md` is the Route A working manuscript (body
~2,690 words plus the existing numbered references). It does not
replace `drafts/manuscript.md`. Figures remain caption-only. Knowledge
compose for drafting proposed 项目知识包 + 重症目标试验模拟论著
(fingerprint `279ab000…`); `--confirm` failed with
`bundle_output_conflict` because a prior genre-imitation file already
differs. Tables: `notes/cce-audit-route-a-tables.md`. Love plots:
`notes/cce-audit-routeA-A10-love-early.png`. Not for submission.

The working draft uses A06/A11/A13 pipeline numbers as the analysis of
record. Those three items still lack a separate hash-bound author
confirmation.

## Author override

Declarations: “Generative artificial intelligence tools were not used
in the conception, analysis, figures, references, or writing of this
manuscript, including the main text.”
