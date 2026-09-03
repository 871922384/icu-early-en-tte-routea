#!/usr/bin/env python3
"""Route A Table 2: EN-by-48h descriptives, missing %, IPCW SMDs.

Uses confirmed primary IPCW weights. Does not refit MSM on the A05 set.
EN-by-48h grouping is descriptive; analysis remains cloned.
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from routeA_a06_a13_a11 import choose_keys, overlay_a06, set_keys  # noqa: E402
from routeA_a11_fast import censor_times, estimate_msm, matrices  # noqa: E402
from routeA_ipcw_qc import KEYS as IPCW_KEYS  # noqa: E402
from routeA_ipcw_qc import assemble  # noqa: E402

COHORT = ROOT / "workspace/metered-results/cohort"
A05 = COHORT / "routeA_a05_covariates.parquet"
STAYS = COHORT / "routeA_restricted_stays.parquet"
SOFA = COHORT / "routeA_sofa_full.parquet"
BASE = COHORT / "baseline_admissions.parquet"
EXTRACT = ROOT / "notes/cce-audit-A06-A13-extract.json"
OUT_CSV = ROOT / "notes/cce-audit-routeA-table2.csv"
OUT_MD = ROOT / "notes/cce-audit-routeA-table2.md"
OUT_JSON = ROOT / "notes/cce-audit-routeA-table2.json"


def now() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def finite(a: np.ndarray) -> np.ndarray:
    return a[np.isfinite(a)]


def fmt_mean_sd(a: np.ndarray, digits: int = 1) -> str:
    x = finite(a)
    if not len(x):
        return "—"
    return f"{x.mean():.{digits}f} ({x.std(ddof=1):.{digits}f})"


def fmt_median_iqr(a: np.ndarray, digits: int = 1) -> str:
    x = finite(a)
    if not len(x):
        return "—"
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    return f"{med:.{digits}f} [{q1:.{digits}f}–{q3:.{digits}f}]"


def fmt_n_pct(mask: np.ndarray) -> str:
    n = int(np.sum(mask))
    return f"{n:,} ({100.0 * n / len(mask):.1f})"


def miss_pct(a: np.ndarray) -> str:
    if not len(a):
        return "—"
    return f"{100.0 * float(np.mean(~np.isfinite(a))):.1f}"


def smd_pair(elig: np.ndarray, uncens: np.ndarray, w: np.ndarray) -> tuple[float | None, float | None]:
    x0 = finite(elig)
    if len(x0) < 2:
        return None, None
    sd = float(x0.std(ddof=1) or 1.0)
    m0 = float(x0.mean())
    xu = finite(uncens)
    m_u = float(xu.mean()) if len(xu) else None
    ww = np.where(np.isfinite(w) & np.isfinite(uncens), w, 0.0)
    xx = np.where(np.isfinite(uncens), uncens, 0.0)
    sw = float(ww.sum())
    m_w = float(np.sum(ww * xx) / sw) if sw > 0 else None
    smd_u = None if m_u is None else (m_u - m0) / sd
    smd_w = None if m_w is None else (m_w - m0) / sd
    return smd_u, smd_w


def fmt_smd(v: float | None) -> str:
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "—"
    return f"{v:.3f}"


def careunit_group(label: str | None) -> str:
    s = label or ""
    if "MICU/SICU" in s or "Medical/Surgical" in s:
        return "MICU/SICU"
    if "Trauma SICU" in s:
        return "Trauma SICU"
    if "Neuro SICU" in s:
        return "Neuro SICU"
    if s.startswith("Neuro"):
        return "Neuro other"
    if "MICU" in s and "SICU" not in s:
        return "MICU"
    if "SICU" in s:
        return "SICU"
    return "Other"


def main() -> None:
    if not A05.exists():
        raise SystemExit(f"missing {A05}; run routeA_a05_extract.py first")
    extract_meta = json.loads(EXTRACT.read_text())
    decision = choose_keys(extract_meta["sofa_component_coverage"], float(extract_meta["lactate_observed_pct"]))
    set_keys(decision["keys"])
    stays = overlay_a06(assemble())
    x, en, death, dead28, subj = matrices(stays, list(IPCW_KEYS))
    point = estimate_msm(x, en, death, dead28)
    c_e = point["c_early"]
    w_e = point["w_early"]
    uncens_e = ~np.isfinite(c_e)
    stay_ids = np.array([int(s["stay_id"]) for s in stays], dtype=int)
    en48 = np.array([s["en_h"] is not None and s["en_h"] <= 48.0 for s in stays], dtype=bool)

    con = duckdb.connect()
    raw = con.execute(
        f"""
        SELECT e.stay_id, e.first_careunit, e.admission_location, e.admission_type,
               e.icu_los_days, e.gender, e.anchor_age, e.dischtime, e.t0, e.dod
        FROM read_parquet('{STAYS.as_posix()}') e
        """
    ).fetchdf().set_index("stay_id")
    a05 = con.execute(f"SELECT * FROM read_parquet('{A05.as_posix()}')").fetchdf().set_index("stay_id")
    sofa = con.execute(f"SELECT * FROM read_parquet('{SOFA.as_posix()}')").fetchdf().set_index("stay_id")
    base = con.execute(f"SELECT * FROM read_parquet('{BASE.as_posix()}')").fetchdf().set_index("stay_id")

    by_id = {int(s["stay_id"]): s for s in stays}

    def col_from_stays(key: str) -> np.ndarray:
        return np.array([float(s[key]) if s.get(key) is not None else np.nan for s in stays], dtype=float)

    def col_a05(key: str) -> np.ndarray:
        out = np.full(len(stay_ids), np.nan)
        for i, sid in enumerate(stay_ids):
            if sid in a05.index:
                v = a05.loc[sid, key] if key in a05.columns else np.nan
                if v is not None and not (isinstance(v, float) and math.isnan(v)) and not pd.isna(v):
                    out[i] = float(v)
        return out

    def col_sofa(key: str) -> np.ndarray:
        out = np.full(len(stay_ids), np.nan)
        for i, sid in enumerate(stay_ids):
            if sid in sofa.index and key in sofa.columns:
                v = sofa.loc[sid, key]
                if v is not None and not (isinstance(v, float) and math.isnan(v)) and not pd.isna(v):
                    out[i] = float(v)
        return out

    def col_raw_group(fn) -> np.ndarray:
        out = np.empty(len(stay_ids), dtype=object)
        for i, sid in enumerate(stay_ids):
            out[i] = fn(raw.loc[sid] if sid in raw.index else None)
        return out

    icu_los = np.array(
        [float(raw.loc[sid, "icu_los_days"]) if sid in raw.index and not pd.isna(raw.loc[sid, "icu_los_days"]) else np.nan for sid in stay_ids]
    )
    hosp_los = col_a05("hospital_los_days")
    fu = np.full(len(stay_ids), 28.0)
    for i, s in enumerate(stays):
        dh = s["death_h"]
        if dh is not None and 0 <= dh <= 28 * 24:
            fu[i] = dh / 24.0

    race_white = 1.0 - (
        col_from_stays("race_black")
        + col_from_stays("race_hispanic")
        + col_from_stays("race_asian")
        + col_from_stays("race_other")
    )

    rows_spec: list[tuple] = [
        ("Demographics", None, None, None),
        ("Age, years", col_from_stays("anchor_age"), "cont", 1),
        ("Female", col_from_stays("female"), "bin", None),
        ("Race/ethnicity, White", race_white, "bin", None),
        ("Black", col_from_stays("race_black"), "bin", None),
        ("Hispanic", col_from_stays("race_hispanic"), "bin", None),
        ("Asian", col_from_stays("race_asian"), "bin", None),
        ("Other or unknown", col_from_stays("race_other"), "bin", None),
        ("Admission context", None, None, None),
        ("ICU, MICU", col_from_stays("unit_micu"), "bin", None),
        ("ICU, any SICU", col_from_stays("unit_sicu"), "bin", None),
        ("ICU, neuro", col_from_stays("unit_neuro"), "bin", None),
        ("Admission location, ED", col_a05("loc_ed"), "bin", None),
        ("Transfer", col_a05("loc_transfer"), "bin", None),
        ("Referral / clinic", col_a05("loc_referral"), "bin", None),
        ("OR / PACU / procedure", col_a05("loc_or"), "bin", None),
        ("First service, medical", col_a05("svc_medical"), "bin", None),
        ("Surgical", col_a05("svc_surgical"), "bin", None),
        ("Neuro", col_a05("svc_neuro"), "bin", None),
        ("Cardiac surgery", col_a05("svc_cardiac"), "bin", None),
        ("Admission type, emergency", col_from_stays("admission_emergency"), "bin", None),
        ("Elective", col_from_stays("admission_elective"), "bin", None),
        ("Era 2008–2010", col_a05("era_2008_2010"), "bin", None),
        ("2011–2013", col_a05("era_2011_2013"), "bin", None),
        ("2014–2016", col_a05("era_2014_2016"), "bin", None),
        ("2017–2019", col_a05("era_2017_2019"), "bin", None),
        ("2020–2022", col_a05("era_2020_2022"), "bin", None),
        ("Prior hospitalisation within 30 d", col_a05("prior_hosp_30d"), "bin", None),
        ("Severity at time zero", None, None, None),
        ("Vasopressor at t0 ±6 h", col_from_stays("vaso_t0"), "bin", None),
        ("SOFA respiratory", col_sofa("sofa_resp"), "cont", 2),
        ("SOFA coagulation", col_sofa("sofa_coag"), "cont", 2),
        ("SOFA liver", col_sofa("sofa_liver"), "cont", 2),
        ("SOFA cardiovascular", col_sofa("sofa_cardio"), "cont", 2),
        ("SOFA CNS", col_sofa("sofa_cns"), "cont", 2),
        ("SOFA renal", col_sofa("sofa_renal"), "cont", 2),
        ("GCS", col_a05("gcs_sum"), "cont", 1),
        ("Admission weight, kg", col_a05("weight_kg"), "cont", 1),
        ("PEEP, cmH2O", col_a05("peep"), "cont", 1),
        ("FiO2", col_a05("fio2"), "cont", 1),
        ("Tidal volume, mL", col_a05("vt_ml"), "cont", 0),
        ("Sepsis-3", None, "unavailable", None),
        ("Laboratories at time zero", None, None, None),
        ("Lactate, mmol/L", col_a05("lactate"), "cont", 1),
        ("Creatinine, mg/dL", col_a05("creatinine"), "cont", 2),
        ("Bilirubin, mg/dL", col_a05("bilirubin"), "cont", 2),
        ("Platelets, ×10^9/L", col_a05("platelet"), "cont", 0),
        ("PaO2, mmHg", col_a05("po2"), "cont", 0),
        ("Hemoglobin, g/dL", col_a05("hb"), "cont", 1),
        ("WBC, ×10^9/L", col_a05("wbc"), "cont", 1),
        ("Sodium, mEq/L", col_a05("sodium"), "cont", 1),
        ("Potassium, mEq/L", col_a05("potassium"), "cont", 1),
        ("Glucose, mg/dL", col_a05("glucose"), "cont", 0),
        ("Albumin, g/dL", col_a05("albumin"), "cont", 1),
        ("pH", col_a05("ph"), "cont", 2),
        ("INR", col_a05("inr"), "cont", 2),
        ("Comorbidity", None, None, None),
        ("Quan-Charlson conditions", col_from_stays("charlson_conditions"), "cont", 1),
        ("Elixhauser unweighted sum", col_a05("elixhauser_sum"), "cont", 1),
        ("Primary digestive ICD", col_from_stays("dx_digestive"), "bin", None),
        ("Follow-up descriptors", None, None, None),
        ("ICU length of stay, days", icu_los, "footer", 1),
        ("Hospital length of stay, days", hosp_los, "footer", 1),
        ("Follow-up to day 28, days", fu, "footer", 1),
    ]

    # Careunit full counts for the note
    cu = [careunit_group(None if sid not in raw.index else str(raw.loc[sid, "first_careunit"])) for sid in stay_ids]
    cu_counts = pd.Series(cu).value_counts().to_dict()

    out_rows = []
    md_lines = [
        "# Table 2. Baseline characteristics at time zero, by observed EN initiation, with balance",
        "",
        "EN by 48 h is descriptive. Analysis uses cloned strategies. SMD columns are early-arm uncensored clones versus the eligible cohort, before and after the confirmed primary IPCW (the weight model was not refit on the A05 expansion).",
        "",
        f"Eligible n={len(stays):,}. Observed EN by 48 h n={int(en48.sum()):,}; no EN by 48 h n={int((~en48).sum()):,}.",
        "",
        "| Characteristic | Eligible (N="
        + f"{len(stays):,}"
        + ") | EN by 48 h (n="
        + f"{int(en48.sum()):,}"
        + ") | No EN by 48 h (n="
        + f"{int((~en48).sum()):,}"
        + ") | Missing, % | SMD unweighted | SMD weighted |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    n_smd_u = 0
    n_smd_w = 0
    n_smd = 0
    for label, arr, kind, digits in rows_spec:
        if kind is None:
            md_lines.append(f"| *{label}* |  |  |  |  |  |  |")
            continue
        if kind == "unavailable":
            md_lines.append("| Sepsis-3 | Unavailable (no mimiciv_derived.sepsis3 on this dump) | — | — | 100 | — | — |")
            out_rows.append(
                {
                    "characteristic": "Sepsis-3",
                    "kind": "unavailable",
                    "eligible": None,
                    "en48": None,
                    "no_en48": None,
                    "missing_pct": 100.0,
                    "smd_unweighted": None,
                    "smd_weighted": None,
                }
            )
            continue
        assert arr is not None
        miss = miss_pct(arr) if kind in {"cont", "footer"} else "0.0"
        if kind == "bin":
            miss = "0.0" if np.all(np.isfinite(arr)) else miss_pct(arr)
            elig_s = fmt_n_pct(arr >= 0.5)
            en_s = fmt_n_pct(arr[en48] >= 0.5)
            no_s = fmt_n_pct(arr[~en48] >= 0.5)
            smd_u, smd_w = smd_pair(arr, arr[uncens_e], w_e[uncens_e])
        else:
            d = 1 if digits is None else digits
            elig_s = f"{fmt_mean_sd(arr, d)}; {fmt_median_iqr(arr, d)}"
            en_s = f"{fmt_mean_sd(arr[en48], d)}; {fmt_median_iqr(arr[en48], d)}"
            no_s = f"{fmt_mean_sd(arr[~en48], d)}; {fmt_median_iqr(arr[~en48], d)}"
            if kind == "footer":
                smd_u, smd_w = None, None
            else:
                filled = np.where(np.isfinite(arr), arr, 0.0)
                smd_u, smd_w = smd_pair(filled, filled[uncens_e], w_e[uncens_e])
        if smd_u is not None:
            n_smd += 1
            if abs(smd_u) > 0.10:
                n_smd_u += 1
            if smd_w is not None and abs(smd_w) > 0.10:
                n_smd_w += 1
        md_lines.append(
            f"| {label} | {elig_s} | {en_s} | {no_s} | {miss} | {fmt_smd(smd_u)} | {fmt_smd(smd_w)} |"
        )
        out_rows.append(
            {
                "characteristic": label,
                "kind": kind,
                "eligible": elig_s,
                "en48": en_s,
                "no_en48": no_s,
                "missing_pct": None if miss == "—" else float(miss),
                "smd_unweighted": smd_u,
                "smd_weighted": smd_w,
            }
        )

    md_lines.extend(
        [
            "",
            "Values for continuous variables are mean (SD); median [IQR]. Binary variables are n (%). Missingness is the percent of eligible stays without a value in the analysis window. SMD is early-strategy uncensored clones versus the eligible cohort. Weighted SMD uses the confirmed primary IPCW; A05 variables were not in that weight model.",
            "",
            f"ICU type (full labels): {', '.join(f'{k} {v}' for k, v in sorted(cu_counts.items(), key=lambda kv: -kv[1]))}.",
            "",
            f"Among tabulated SMD rows, {n_smd_u} of {n_smd} had |SMD| > 0.10 unweighted and {n_smd_w} of {n_smd} after weighting.",
            "",
        ]
    )
    OUT_MD.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    pd.DataFrame(out_rows).to_csv(OUT_CSV, index=False)
    meta = {
        "created_at": now(),
        "n_eligible": int(len(stays)),
        "n_en48": int(en48.sum()),
        "n_no_en48": int((~en48).sum()),
        "n_uncensored_early": int(uncens_e.sum()),
        "n_smd_rows": n_smd,
        "n_smd_unweighted_gt_0_10": n_smd_u,
        "n_smd_weighted_gt_0_10": n_smd_w,
        "careunit_full": {str(k): int(v) for k, v in cu_counts.items()},
        "msm_point_check": {
            "early_48": point["early_48"],
            "delayed_96": point["delayed_96"],
            "rd": point["rd"],
            "rr": point["rr"],
        },
        "primary_msm_unchanged": True,
        "csv": str(OUT_CSV.relative_to(ROOT)),
        "md": str(OUT_MD.relative_to(ROOT)),
    }
    OUT_JSON.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
