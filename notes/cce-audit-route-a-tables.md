# Route A tables (working; not for portal upload)

EN by 48 h in Table 2 is descriptive only. Analysis is clone-based.

## Table 1. Target trial and emulation

| Component | Target trial | Emulation |
| --- | --- | --- |
| Eligibility | Adults, first ICU stay, invasive ventilation, no EN yet | Adult first ICU stay; invasive ventilation itemids 225792/224385 within ±6 h of `intime`; no qualifying EN before t0; CVICU and CCU excluded |
| Time zero | Later of ICU admission and start of invasive ventilation | Same |
| Strategies | EN within 48 h vs no EN before 96 h | Cloning; artificial censoring at first incompatibility |
| Outcome | All-cause death by day 28 | `patients.dod` within 672 h of t0; missing dod treated as alive; secondary in-hospital death with discharge as competing event |
| Contrast | Per-protocol 28-day RD and RR | MSM g-computation standardized to the eligible cohort; 2,000-replicate BCa |

## Table 2. Baseline (full spec)

See notes/cce-audit-routeA-table2.md. Eligible n=11,259. EN by 48 h n=2,599; no EN by 48 h n=8,660. Confirmed 27-covariate IPCW: 9 of 27 unweighted and 1 of 27 weighted (lactate filled 0.101) above 0.10. Expanded A05 rows are descriptive; 6 of 54 tabulated SMDs remained above 0.10 after the existing weights. Sepsis-3 unavailable. Creatinine coverage 62.7%; lactate 48.5%.

## Table 3. Clone flow

See `notes/cce-audit-routeA-clone-flow.json`. Early uncensored 3,677: 2,599 initiated EN (70.7%), 1,085 died in grace before EN (29.5%). Delayed uncensored 7,326: 5,982 never EN and survived grace. Kish ESS 2,650 vs 6,353.

## Table 4. Contrasts

| Analysis | Early | Delayed | RD | RR |
| --- | ---: | ---: | ---: | ---: |
| Primary MSM 28-d death, 48/96 | 0.374 | 0.227 | 0.147 (BCa 0.131–0.164) | 1.65 (1.57–1.74) |
| AJ 28-d in-hospital death | 0.374 | 0.196 | 0.178 | 1.91 |
| MSM 90-d death | 0.422 | 0.272 | 0.150 | 1.55 |
| Hájek | 0.408 | 0.228 | 0.180 | 1.79 |
| MSM 24/48 (diagnostic) | 0.433 | 0.232 | 0.201 | 1.87 (1.76–1.97) |
| Negative control, CHG bath | 0.550 | 0.237 | 0.313 | 2.32 |
| Positive control, ventilation (first stays, not CCW) | 0.179 | 0.107 | 0.072 | 1.67 |
| MICE 20 SOFA/lactate (restricted) | — | — | 0.152 (0.135–0.169) | 1.67 (1.58–1.76) |
| MSM including CVICU+CCU (n=19,919; EN48 14.1%) | 0.326 | 0.146 | 0.180 (0.157–0.201) | 2.23 (2.07–2.36) |

E-value for RR 1.65: 2.68 (CI limit 2.51). E-value does not address positivity or grace-period composition.

## eTable. Vasopressor itemids

| itemid | Label |
| --- | --- |
| 221289 | Epinephrine |
| 221662 | Dopamine |
| 221749 | Phenylephrine |
| 221906 | Norepinephrine |
| 222315 | Vasopressin |
| 229630 | Phenylephrine (50/250) |
| 229632 | Phenylephrine (200/250) |
| 221653 | Dobutamine (not in locked seven; listed for SDC completeness) |
| 221986 | Milrinone (not in locked seven; listed for SDC completeness) |

Locked analysis set remains the seven infusion itemids used at t0±6 h.

## Figure files (not in body)

High-resolution (1200 dpi TIFF LZW + PDF) in `output/figures-route-a/`:

- Figure 2 Love plot
- Figure 3 Aalen-Johansen
- Figure 4 Predicted probability histogram (source PNG also copied)
- eFigure 1 Forest

Working PNGs remain in `notes/`.
