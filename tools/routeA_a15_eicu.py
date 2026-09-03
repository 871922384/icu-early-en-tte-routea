#!/usr/bin/env python3
"""A15 eICU Route A feasibility: invasive vent ±6 h, 48/96 EN, clone flow.

Pipeline QC. Not author-final. Does not write manuscript placeholders.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EICU = Path("/Volumes/B/litflow-cache/physionet/physionet.org/files/eicu-crd/2.0")
IDX = ROOT / "workspace/metered-results/eicu-cohort/index_stays.parquet"
OUT = ROOT / "notes/cce-audit-routeA-A15-eicu.json"
CSV = ROOT / "notes/cce-audit-routeA-A15-eicu-units.csv"


def now():
    from datetime import datetime, timedelta, timezone

    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def main() -> None:
    con = duckdb.connect()
    idx = IDX.as_posix()
    rc = (EICU / "respiratoryCare.csv.gz").as_posix()
    ap = (EICU / "apachePredVar.csv.gz").as_posix()
    pt = (EICU / "patient.csv.gz").as_posix()
    df = con.execute(
        f"""
        WITH vent_rc AS (
          SELECT patientunitstayid AS stay_id,
                 MIN(ventstartoffset)/60.0 AS vent_start_h
          FROM read_csv_auto('{rc}', header=true, compression='gzip')
          WHERE ventstartoffset BETWEEN -360 AND 360
          GROUP BY 1
        ),
        vent_ap AS (
          SELECT patientunitstayid AS stay_id, ventday1, oobventday1, oobintubday1
          FROM read_csv_auto('{ap}', header=true, compression='gzip')
        )
        SELECT
          i.stay_id, i.subject_id, i.age_years, i.gender,
          i.en_start_hour, i.death_hour, i.hospital_death,
          i.vasopressor_at_t0, i.lactate, i.creatinine, i.bilirubin,
          p.unittype,
          v.vent_start_h,
          CASE WHEN v.stay_id IS NOT NULL THEN 1 ELSE 0 END AS vent_pm6,
          CASE WHEN a.ventday1=1 OR a.oobventday1=1 THEN 1 ELSE 0 END AS apache_vent_d1
        FROM read_parquet('{idx}') i
        LEFT JOIN vent_rc v ON v.stay_id=i.stay_id
        LEFT JOIN vent_ap a ON a.stay_id=i.stay_id
        LEFT JOIN read_csv_auto('{pt}', header=true, compression='gzip') p
          ON p.patientunitstayid=i.stay_id
        """
    ).fetchdf()
    n = len(df)
    vent = df[df["vent_pm6"] == 1].copy()
    nv = len(vent)
    cardiac = {"CCU-CTICU", "Cardiac ICU", "CSICU", "CTICU"}
    vent_nc = vent[~vent["unittype"].isin(cardiac)].copy()

    def rates(sub, label):
        m = len(sub)
        en24 = int(((sub["en_start_hour"].notna()) & (sub["en_start_hour"] <= 24)).sum())
        en48 = int(((sub["en_start_hour"].notna()) & (sub["en_start_hour"] <= 48)).sum())
        no96 = int((sub["en_start_hour"].isna() | (sub["en_start_hour"] > 96)).sum())
        dead = sub["death_hour"]
        dead28 = int(((dead.notna()) & (dead >= 0) & (dead <= 28 * 24)).sum())
        return {
            "label": label,
            "n": m,
            "en24": en24,
            "en24_pct": round(100 * en24 / m, 2) if m else None,
            "en48": en48,
            "en48_pct": round(100 * en48 / m, 2) if m else None,
            "no_en_by_96": no96,
            "no_en_by_96_pct": round(100 * no96 / m, 2) if m else None,
            "dead28_if_death_hour": dead28,
            "a00_en48_ge_15": bool(m and 100 * en48 / m >= 15),
            "a00_en48_ge_25": bool(m and 100 * en48 / m >= 25),
        }

    unit_rows = []
    for unit, sub in vent.groupby(vent["unittype"].fillna("missing")):
        unit_rows.append(rates(sub, str(unit)))
    import csv

    with CSV.open("w", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=list(unit_rows[0].keys()))
        w.writeheader()
        w.writerows(unit_rows)

    def clone_flow(sub, early_h, delay_h):
        en = sub["en_start_hour"].to_numpy(dtype=float)
        death = sub["death_hour"].to_numpy(dtype=float)
        en_ok = np.isfinite(en)
        death_ok = np.isfinite(death)
        keep_early = (en_ok & (en <= early_h)) | (death_ok & (death <= early_h))
        c_early = (~keep_early).sum()
        delay_cens = en_ok & (en < delay_h) & (~death_ok | (death > en))
        initiated = int((en_ok & (en <= early_h)).sum())
        died_g = int((death_ok & (death <= early_h) & (~en_ok | (death <= en))).sum())
        never = int((~en_ok | (en >= delay_h)).sum() - int((death_ok & (death < delay_h) & en_ok & (en < delay_h)).sum()))
        nsub = len(sub)
        return {
            "n": nsub,
            "early_censored": int(c_early),
            "early_uncensored": nsub - int(c_early),
            "initiated_en_by_early": initiated,
            "died_grace_before_en_early": died_g,
            "delayed_censored": int(delay_cens.sum()),
            "delayed_uncensored": nsub - int(delay_cens.sum()),
            "never_en_by_delay_approx": int((~en_ok | (en >= delay_h)).sum()),
        }

    vent_flow = clone_flow(vent, 48, 96)
    overall = rates(df, "all_first_stays")
    vent_r = rates(vent, "vent_pm6_respiratoryCare")
    vent_nc_r = rates(vent_nc, "vent_pm6_excluding_cardiac_units")
    apache = df[df["apache_vent_d1"] == 1]
    apache_r = rates(apache, "apache_vent_or_oobvent_day1")
    identifiable = vent_r["a00_en48_ge_15"] or vent_nc_r["a00_en48_ge_15"]
    payload = {
        "generated_at": now(),
        "status": "pipeline_qc_not_author_final",
        "volume_B_mounted": True,
        "n_index": n,
        "vent_definition": "respiratoryCare.ventstartoffset in [-360, +360] minutes from unit admit (Route A analogue of ±6 h). apachePredVar ventday1/oobventday1 is a day-1 flag, not ±6 h.",
        "overall_first_stays": overall,
        "routeA_vent_pm6": vent_r,
        "routeA_vent_pm6_no_cardiac": vent_nc_r,
        "apache_day1_vent": apache_r,
        "clone_flow_48_96_vent_pm6": vent_flow,
        "unit_csv": str(CSV.relative_to(ROOT)),
        "identifiable_48_96": identifiable,
        "decision": (
            "Route A 48/96 is not identifiable in eICU under either the ±6 h respiratoryCare vent flag "
            "or the cardiac-unit exclusion: EN-by-48h remains ~5%. Do not transport the MIMIC MSM. "
            "Writing should use B15(a): feasibility replication of initiation frequency and clone flow, "
            "not a transportability analysis and not an eICU effect estimate."
            if not identifiable
            else "EN-by-48h cleared 15%; MSM not auto-run in this script."
        ),
        "prior_24h_qc": {"en_by_24_all_first_stays_pct": overall["en24_pct"]},
        "note": "eICU has no state death registry equivalent to patients.dod. death_hour is hospital death timing. Not author-final.",
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({k: payload[k] for k in ("routeA_vent_pm6", "routeA_vent_pm6_no_cardiac", "identifiable_48_96", "decision")}, indent=2))


if __name__ == "__main__":
    main()
