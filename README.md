# Route A clone-censor-weight analysis code (MIMIC-IV / eICU)

Analysis scripts and the locked item mapping for:

Timing of Enteral Nutrition Initiation in Mechanically Ventilated Adults:
A Target Trial Emulation in MIMIC-IV with Feasibility Replication in eICU-CRD.

Protocol: OSF DOI 10.17605/OSF.IO/QJMGX

## What is in this archive

- `tools/routeA_*.py` — cohort, IPCW, MSM, bootstrap, MICE, eICU feasibility, figures
- `workspace/itemid_mapping.csv` — locked Nutrition-Enteral and vasopressor mapping (SHA-256 65eb634a2066fa988cfa81a322e4b2b5fc67515aecd3a1c36cde56e5b885e590)
- `notes/` — aggregate JSON/CSV receipts (no patient-level rows)

## What is not in this archive

MIMIC-IV and eICU-CRD patient-level extracts are **not** included. Those databases are available to credentialed PhysioNet users under a data use agreement. Redistribution of the underlying records is not permitted.

## Reproduce

1. Obtain MIMIC-IV v3.1 and eICU-CRD v2.0 from PhysioNet.
2. Point the scripts at local gzip CSV paths (see each `routeA_*.py` header).
3. Python 3.11+, numpy, pandas, duckdb, matplotlib.

Primary confirmed contrast (restricted ventilated non-CVICU/CCU cohort, n=11,259):
MSM 28-day death 0.374 vs 0.227; RD 0.147 (BCa 0.131–0.164); RR 1.65 (1.57–1.74).
