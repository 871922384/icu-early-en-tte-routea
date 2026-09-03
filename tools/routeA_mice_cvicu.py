#!/usr/bin/env python3
"""A16 sensitivities: MICE 20 on restricted SOFA, and MSM with CVICU/CCU restored.

Pipeline QC. Not author-final.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import duckdb
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from routeA_a06_a13_a11 import BASE_KEYS, choose_keys, now, overlay_a06, set_keys  # noqa: E402
from routeA_a11_fast import estimate_msm, matrices  # noqa: E402
from routeA_a16_cloneflow import boot_rd_rr, estimate_variant  # noqa: E402
from routeA_ipcw_qc import KEYS as IPCW_KEYS  # noqa: E402
from routeA_ipcw_qc import assemble  # noqa: E402

COHORT = ROOT / "workspace/metered-results/cohort"
MIMIC = Path("/Volumes/B/litflow-cache/physionet/physionet.org/files/mimiciv/3.1")
UNREST = COHORT / "routeA_unrestricted_stays.parquet"
SOFA_U = COHORT / "routeA_sofa_full_unrestricted.parquet"
OUT_MICE = ROOT / "notes/cce-audit-routeA-A16-mice.json"
OUT_CVICU = ROOT / "notes/cce-audit-routeA-A16-cvicu.json"
SEED = 20260902
IMPUTE = ("sofa_resp", "sofa_coag", "sofa_liver", "sofa_cardio", "sofa_cns", "sofa_renal", "lactate")


def build_unrestricted() -> int:
    con = duckdb.connect()
    idx = (COHORT / "index_stays.parquet").as_posix()
    enp = (COHORT / "qualifying_en.parquet").as_posix()
    icu = (MIMIC / "icu/icustays.csv.gz").as_posix()
    proc = (MIMIC / "icu/procedureevents.csv.gz").as_posix()
    patients = (MIMIC / "hosp/patients.csv.gz").as_posix()
    adm = (MIMIC / "hosp/admissions.csv.gz").as_posix()
    con.execute(
        f"""
        COPY (
          WITH vent AS (
            SELECT i.stay_id, i.intime,
                   MIN(p.starttime) AS first_inv_start
            FROM read_parquet('{idx}') i
            JOIN read_csv_auto('{proc}', header=true, compression='gzip') p
              ON p.stay_id=i.stay_id
             AND p.itemid IN (225792, 224385)
             AND p.starttime <= i.intime + INTERVAL 6 HOUR
             AND (p.endtime IS NULL OR p.endtime >= i.intime - INTERVAL 6 HOUR)
            GROUP BY i.stay_id, i.intime
          ),
          base AS (
            SELECT i.stay_id, i.subject_id, i.hadm_id, i.intime, i.outtime,
                   v.first_inv_start,
                   CASE WHEN i.intime >= v.first_inv_start THEN i.intime ELSE v.first_inv_start END AS t0,
                   icu.first_careunit, i.anchor_age, i.gender
            FROM read_parquet('{idx}') i
            JOIN vent v USING (stay_id)
            JOIN read_csv_auto('{icu}', header=true, compression='gzip') icu
              ON icu.stay_id=i.stay_id
          ),
          pre_en AS (
            SELECT DISTINCT b.stay_id
            FROM base b
            JOIN read_parquet('{enp}') e ON e.stay_id=b.stay_id
            WHERE e.starttime < b.t0
          ),
          first_en AS (
            SELECT b.stay_id, MIN(e.starttime) AS first_en
            FROM base b
            JOIN read_parquet('{enp}') e ON e.stay_id=b.stay_id AND e.starttime >= b.t0
            GROUP BY b.stay_id
          )
          SELECT
            b.stay_id, b.subject_id, b.hadm_id, b.intime, b.t0, b.first_inv_start,
            b.first_careunit, a.admission_location, a.admission_type,
            b.anchor_age, b.gender,
            CASE WHEN f.first_en IS NULL THEN NULL
                 ELSE date_diff('epoch', b.t0, f.first_en)/3600.0 END AS en_h,
            f.first_en,
            pt.dod, a.dischtime, a.hospital_expire_flag,
            date_diff('epoch', b.intime, b.outtime)/86400.0 AS icu_los_days
          FROM base b
          LEFT JOIN pre_en pe USING (stay_id)
          LEFT JOIN first_en f USING (stay_id)
          LEFT JOIN read_csv_auto('{patients}', header=true, compression='gzip') pt
            ON pt.subject_id=b.subject_id
          LEFT JOIN read_csv_auto('{adm}', header=true, compression='gzip') a
            ON a.hadm_id=b.hadm_id
          WHERE pe.stay_id IS NULL
        ) TO '{UNREST.as_posix()}' (FORMAT PARQUET)
        """
    )
    n = con.execute(f"SELECT COUNT(*) FROM read_parquet('{UNREST.as_posix()}')").fetchone()[0]
    en48 = con.execute(
        f"SELECT SUM(CASE WHEN en_h IS NOT NULL AND en_h<=48 THEN 1 ELSE 0 END) FROM read_parquet('{UNREST.as_posix()}')"
    ).fetchone()[0]
    print("unrestricted n", n, "en48", en48, "pct", round(100 * en48 / n, 2) if n else None, flush=True)
    return int(n)


def mice_impute(obs: np.ndarray, rng: np.random.Generator, n_imp: int = 20, cycles: int = 8) -> list[np.ndarray]:
    """Chained equations with ridge-linear draws. obs has np.nan for missing."""
    n, p = obs.shape
    out = []
    for _ in range(n_imp):
        x = obs.copy()
        for j in range(p):
            miss = np.isnan(x[:, j])
            if miss.any() and (~miss).any():
                x[miss, j] = float(np.nanmean(x[:, j]))
        for _c in range(cycles):
            for j in range(p):
                miss = np.isnan(obs[:, j])
                seen = ~np.isnan(obs[:, j])
                if not miss.any() or seen.sum() < 20:
                    continue
                pred = [k for k in range(p) if k != j]
                z = np.column_stack([np.ones(int(seen.sum())), x[seen][:, pred]])
                y = x[seen, j]
                xtx = z.T @ z
                xtx.flat[:: z.shape[1] + 1] += 1e-2
                try:
                    beta = np.linalg.solve(xtx, z.T @ y)
                except np.linalg.LinAlgError:
                    continue
                fitted = z @ beta
                resid = y - fitted
                sigma = float(np.std(resid, ddof=1) or 1e-3)
                z_m = np.column_stack([np.ones(int(miss.sum())), x[miss][:, pred]])
                draw = z_m @ beta + rng.normal(0.0, sigma, size=int(miss.sum()))
                x[miss, j] = draw
        # clip SOFA 0-4 (first 6 cols) and lactate >=0
        x[:, :6] = np.clip(np.round(x[:, :6]), 0, 4)
        x[:, 6] = np.clip(x[:, 6], 0, None)
        out.append(x)
    return out


def run_mice(stays: list[dict], x_base_keys: list[str]) -> dict:
    n = len(stays)
    raw = np.full((n, 7), np.nan)
    for i, s in enumerate(stays):
        for j, name in enumerate(IMPUTE):
            if name == "lactate":
                if s.get("lactate_missing", 1) == 0:
                    raw[i, j] = float(s["lactate_filled"])
            else:
                if s.get(f"{name}_miss", 1) == 0:
                    raw[i, j] = float(s[name])
    rng = np.random.default_rng(SEED)
    imps = mice_impute(raw, rng, n_imp=20, cycles=8)
    keys_mice = list(x_base_keys) + list(IMPUTE)
    # drop miss flags from design
    keys_mice = [k for k in keys_mice if not k.endswith("_miss") and k != "lactate_missing"]
    if "lactate_filled" in keys_mice:
        keys_mice.remove("lactate_filled")
    rds, rrs = [], []
    last = None
    en, death, dead28 = None, None, None
    x0, en, death, dead28, subj = matrices(stays, list(IPCW_KEYS))
    for k, imp in enumerate(imps):
        filled = []
        for i, s in enumerate(stays):
            row = dict(s)
            for j, name in enumerate(IMPUTE):
                row[name] = float(imp[i, j])
                row[f"{name}_miss"] = 0.0
            row["lactate_filled"] = float(imp[i, 6])
            row["lactate_missing"] = 0.0
            filled.append(row)
        set_keys(keys_mice)
        xf, en, death, dead28, subj = matrices(filled, keys_mice)
        est = estimate_msm(xf, en, death, dead28, days=28)
        rds.append(float(est["rd"]))
        rrs.append(float(est["rr"]))
        last = est
        print(f"mice {k+1}/20 RD={est['rd']:.4f} RR={est['rr']:.4f}", flush=True)
    rds = np.array(rds)
    rrs = np.array(rrs)
    # Rubin using A11 complete-case bootstrap SE as within-imputation W
    se_rd = (0.16399620924044 - 0.13096838492458115) / 3.92
    se_rr = (1.7403166942947013 - 1.5660946648660876) / 3.92
    m = 20.0

    def rubin(qs, se_w):
        qbar = float(qs.mean())
        B = float(qs.var(ddof=1))
        W = float(se_w ** 2)
        T = W + (1.0 + 1.0 / m) * B
        return {
            "mean": qbar,
            "between_var": B,
            "within_var_from_A11": W,
            "total_var": T,
            "ci95": [qbar - 1.96 * math.sqrt(T), qbar + 1.96 * math.sqrt(T)],
            "min": float(qs.min()),
            "max": float(qs.max()),
        }

    payload = {
        "generated_at": now(),
        "status": "pipeline_qc_not_author_final",
        "n_stays": n,
        "n_imputations": 20,
        "cycles": 8,
        "imputed": list(IMPUTE),
        "keys": keys_mice,
        "rd": rubin(rds, se_rd),
        "rr": rubin(rrs, se_rr),
        "last_msm": {
            "early_48": last["early_48"],
            "delayed_96": last["delayed_96"],
            "rd": last["rd"],
            "rr": last["rr"],
        },
        "note": "Missing indicators dropped after imputation. Within-imputation variance taken from the A11 complete-case BCa width, not a nested bootstrap. Not author-final.",
    }
    OUT_MICE.write_text(json.dumps(payload, indent=2) + "\n")
    print("MICE RD", payload["rd"]["mean"], payload["rd"]["ci95"], "RR", payload["rr"]["mean"], payload["rr"]["ci95"], flush=True)
    return payload


def assemble_from(parquet: Path) -> list[dict]:
    # reuse assemble SQL by temporarily... just duplicate join pattern
    from routeA_ipcw_qc import COHORT as C

    con = duckdb.connect()
    path = lambda n: (C / n).as_posix()
    df = con.execute(
        f"""
        SELECT
          e.stay_id, e.subject_id, e.anchor_age,
          CASE WHEN e.gender='F' THEN 1 ELSE 0 END AS female,
          e.en_h, e.t0, e.dod, e.dischtime, e.hospital_expire_flag,
          e.first_careunit,
          CASE WHEN e.first_careunit LIKE '%MICU%' AND e.first_careunit NOT LIKE '%SICU%' THEN 1 ELSE 0 END AS unit_micu,
          CASE WHEN e.first_careunit LIKE '%SICU%' THEN 1 ELSE 0 END AS unit_sicu,
          CASE WHEN e.first_careunit LIKE '%Neuro%' THEN 1 ELSE 0 END AS unit_neuro,
          CASE WHEN e.first_careunit LIKE '%CVICU%' OR e.first_careunit LIKE '%Cardiac Vascular%' THEN 1 ELSE 0 END AS unit_cvicu,
          CASE WHEN e.first_careunit LIKE '%Coronary Care%' THEN 1 ELSE 0 END AS unit_ccu,
          COALESCE(a.race_black,0) race_black,
          COALESCE(a.race_hispanic,0) race_hispanic,
          COALESCE(a.race_asian,0) race_asian,
          COALESCE(a.race_other,0) race_other,
          COALESCE(a.admission_elective,0) admission_elective,
          COALESCE(a.admission_emergency,0) admission_emergency,
          COALESCE(a.charlson_conditions,0) charlson_conditions,
          COALESCE(a.dx_digestive,0) dx_digestive
        FROM read_parquet('{parquet.as_posix()}') e
        LEFT JOIN read_parquet('{path("baseline_admissions.parquet")}') a USING (stay_id)
        """
    ).fetchdf()
    vaso = con.execute(
        f"""
        SELECT DISTINCT e.stay_id
        FROM read_parquet('{parquet.as_posix()}') e
        JOIN read_parquet('{path("vasopressor.parquet")}') v USING (stay_id)
        WHERE v.starttime <= e.t0 + INTERVAL 6 HOUR
          AND (v.endtime IS NULL OR v.endtime >= e.t0 - INTERVAL 6 HOUR)
        """
    ).fetchdf()
    vaso_set = set(vaso["stay_id"].tolist()) if len(vaso) else set()
    rows = []
    for rec in df.to_dict("records"):
        dod = rec["dod"]
        t0 = rec["t0"]
        death_h = None
        if dod is not None and t0 is not None:
            try:
                death_h = float((np.datetime64(dod) - np.datetime64(t0)) / np.timedelta64(1, "h"))
            except Exception:
                death_h = None
        dead28 = 1.0 if (death_h is not None and 0 <= death_h <= 28 * 24) else 0.0
        en = rec["en_h"]
        if en is not None and isinstance(en, float) and math.isnan(en):
            en = None
        rows.append(
            {
                "stay_id": int(rec["stay_id"]),
                "subject_id": int(rec["subject_id"]),
                "anchor_age": float(rec["anchor_age"] or 0),
                "female": float(rec["female"]),
                "en_h": None if en is None else float(en),
                "death_h": death_h,
                "dead28": dead28,
                "vaso_t0": 1.0 if rec["stay_id"] in vaso_set else 0.0,
                "unit_micu": float(rec["unit_micu"]),
                "unit_sicu": float(rec["unit_sicu"]),
                "unit_neuro": float(rec["unit_neuro"]),
                "unit_cvicu": float(rec["unit_cvicu"]),
                "unit_ccu": float(rec["unit_ccu"]),
                "race_black": float(rec["race_black"]),
                "race_hispanic": float(rec["race_hispanic"]),
                "race_asian": float(rec["race_asian"]),
                "race_other": float(rec["race_other"]),
                "admission_elective": float(rec["admission_elective"]),
                "admission_emergency": float(rec["admission_emergency"]),
                "charlson_conditions": float(rec["charlson_conditions"]),
                "dx_digestive": float(rec["dx_digestive"]),
            }
        )
    return rows


def overlay_sofa(stays: list[dict], sofa_path: Path) -> list[dict]:
    con = duckdb.connect()
    sofa = con.execute(f"SELECT * FROM read_parquet('{sofa_path.as_posix()}')").fetchdf().set_index("stay_id")
    out = []
    for s in stays:
        row = dict(s)
        if s["stay_id"] not in sofa.index:
            for name in ("resp", "coag", "liver", "cardio", "cns", "renal"):
                row[f"sofa_{name}"] = 0.0
                row[f"sofa_{name}_miss"] = 1.0
            row["lactate_filled"] = 0.0
            row["lactate_missing"] = 1.0
            out.append(row)
            continue
        r = sofa.loc[s["stay_id"]]
        import pandas as pd

        for name in ("resp", "coag", "liver", "cardio", "cns", "renal"):
            val = r[f"sofa_{name}"]
            missing = bool(pd.isna(val))
            row[f"sofa_{name}"] = 0.0 if missing else float(val)
            row[f"sofa_{name}_miss"] = 1.0 if missing else 0.0
        lac = r["lactate"]
        row["lactate_filled"] = 0.0 if pd.isna(lac) else float(lac)
        row["lactate_missing"] = 1.0 if pd.isna(lac) else 0.0
        out.append(row)
    return out


def run_cvicu() -> dict:
    stays = overlay_sofa(assemble_from(UNREST), SOFA_U)
    n = len(stays)
    en48 = sum(1 for s in stays if s["en_h"] is not None and s["en_h"] <= 48)
    cvicu = sum(s["unit_cvicu"] for s in stays)
    ccu = sum(s["unit_ccu"] for s in stays)
    keys = list(BASE_KEYS) + [
        "unit_cvicu",
        "unit_ccu",
        "sofa_resp",
        "sofa_resp_miss",
        "sofa_coag",
        "sofa_coag_miss",
        "sofa_liver",
        "sofa_liver_miss",
        "sofa_cardio",
        "sofa_cns",
        "sofa_cns_miss",
        "sofa_renal",
        "sofa_renal_miss",
        "lactate_filled",
        "lactate_missing",
    ]
    # BASE_KEYS already has unit_micu/sicu/neuro
    set_keys(keys)
    x, en, death, dead28, subj = matrices(stays, keys)
    est = estimate_variant(x, en, death, dead28, 48.0, 96.0, days=28, trim_q=None)
    boot = boot_rd_rr(x, en, death, dead28, subj, 48.0, 96.0, 28, None, 100)
    payload = {
        "generated_at": now(),
        "status": "pipeline_qc_not_author_final",
        "n_stays": n,
        "en48": en48,
        "en48_pct": round(100 * en48 / n, 2) if n else None,
        "n_cvicu": int(cvicu),
        "n_ccu": int(ccu),
        "a00_en48_ge_15": bool(n and 100 * en48 / n >= 15),
        "msm": {
            "early_48": est["early_48"],
            "delayed_96": est["delayed_96"],
            "rd": est["rd"],
            "rr": est["rr"],
            "n_uncensored_early": est["n_uncensored_early"],
            "n_uncensored_delayed": est["n_uncensored_delayed"],
        },
        "hajek": {"rd": est["hajek_rd"], "rr": est["hajek_rr"], "early": est["hajek_early"], "delayed": est["hajek_delayed"]},
        "bootstrap_100": boot,
        "keys": keys,
        "note": "Vent ±6 h, no EN before t0, CVICU and CCU included. Missing dod as alive. Not 2,000 BCa. Not author-final.",
    }
    OUT_CVICU.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({k: payload[k] for k in ("n_stays", "en48_pct", "msm", "bootstrap_100", "a00_en48_ge_15")}, indent=2), flush=True)
    return payload


def main() -> None:
    extract_meta = json.loads((ROOT / "notes/cce-audit-A06-A13-extract.json").read_text())
    decision = choose_keys(extract_meta["sofa_component_coverage"], float(extract_meta["lactate_observed_pct"]))
    set_keys(decision["keys"])
    stays = overlay_a06(assemble())
    print("MICE on restricted", len(stays), flush=True)
    run_mice(stays, decision["keys"])
    print("build unrestricted", flush=True)
    build_unrestricted()
    print("extract SOFA unrestricted (external script)", flush=True)
    # caller runs extract; if sofa exists, continue
    if not SOFA_U.exists():
        print("MISSING", SOFA_U, "run extract then re-invoke with --cvicu-only", flush=True)
        return
    run_cvicu()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--cvicu-only":
        run_cvicu()
    elif len(sys.argv) > 1 and sys.argv[1] == "--mice-only":
        extract_meta = json.loads((ROOT / "notes/cce-audit-A06-A13-extract.json").read_text())
        decision = choose_keys(extract_meta["sofa_component_coverage"], float(extract_meta["lactate_observed_pct"]))
        set_keys(decision["keys"])
        run_mice(overlay_a06(assemble()), decision["keys"])
    else:
        main()
