#!/usr/bin/env python3
"""Route A clone-censor-weight QC: 48h vs 96h EN, 28-day dod convention.

Pipeline QC for the CCE audit. Not author-final. Not a 2,000-replicate BCa
interval. Restricted cohort excludes CVICU and CCU.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import duckdb
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
COHORT = ROOT / "workspace/metered-results/cohort"
OUT = ROOT / "notes/cce-audit-routeA-ipcw-qc.json"
PERIOD_STARTS = tuple(range(0, 96, 6))
RIDGE = 1e-2
BOOT = 100
SEED = 20260902


def sigmoid(z: np.ndarray) -> np.ndarray:
    zc = np.clip(z, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-zc))


def fit_logit(x: np.ndarray, y: np.ndarray, ridge: float = RIDGE, steps: int = 50) -> np.ndarray:
    n, p = x.shape
    beta = np.zeros(p)
    if n == 0:
        return beta
    s = float(y.sum())
    if s <= 0:
        beta[0] = -20.0
        return beta
    if s >= n:
        beta[0] = 20.0
        return beta
    for _ in range(steps):
        mu = sigmoid(x @ beta)
        w = mu * (1.0 - mu)
        grad = x.T @ (y - mu) - ridge * beta
        hess = x.T @ (x * w[:, None])
        hess.flat[:: p + 1] += ridge
        try:
            delta = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            break
        beta = beta + delta
        if float(np.max(np.abs(delta))) < 1e-8:
            break
    return beta


def design(rows: list[dict], keys: list[str]) -> np.ndarray:
    n = len(rows)
    x = np.ones((n, 1 + len(keys)))
    for j, key in enumerate(keys):
        x[:, j + 1] = [0.0 if r.get(key) is None else float(r[key]) for r in rows]
    return x


def assemble() -> list[dict]:
    con = duckdb.connect()
    path = lambda n: (COHORT / n).as_posix()
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
          COALESCE(a.race_black,0) race_black,
          COALESCE(a.race_hispanic,0) race_hispanic,
          COALESCE(a.race_asian,0) race_asian,
          COALESCE(a.race_other,0) race_other,
          COALESCE(a.admission_elective,0) admission_elective,
          COALESCE(a.admission_emergency,0) admission_emergency,
          COALESCE(a.charlson_conditions,0) charlson_conditions,
          COALESCE(a.dx_digestive,0) dx_digestive,
          COALESCE(s.sofa_t0,0) sofa_t0,
          COALESCE(s.sofa_missing,1) sofa_missing
        FROM read_parquet('{path("routeA_restricted_stays.parquet")}') e
        LEFT JOIN read_parquet('{path("baseline_admissions.parquet")}') a USING (stay_id)
        LEFT JOIN read_parquet('{path("sofa_t0.parquet")}') s USING (stay_id)
        """
    ).fetchdf()
    vaso = con.execute(
        f"""
        SELECT DISTINCT e.stay_id
        FROM read_parquet('{path("routeA_restricted_stays.parquet")}') e
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
                delta = (np.datetime64(dod) - np.datetime64(t0)) / np.timedelta64(1, "h")
                death_h = float(delta)
            except Exception:
                death_h = None
        dead28 = 1.0 if (death_h is not None and 0 <= death_h <= 28 * 24) else 0.0
        rows.append(
            {
                "stay_id": int(rec["stay_id"]),
                "subject_id": int(rec["subject_id"]),
                "anchor_age": float(rec["anchor_age"] or 0),
                "female": float(rec["female"]),
                "en_h": None if rec["en_h"] is None or (isinstance(rec["en_h"], float) and math.isnan(rec["en_h"])) else float(rec["en_h"]),
                "death_h": death_h,
                "dead28": dead28,
                "vaso_t0": 1.0 if rec["stay_id"] in vaso_set else 0.0,
                "unit_micu": float(rec["unit_micu"]),
                "unit_sicu": float(rec["unit_sicu"]),
                "unit_neuro": float(rec["unit_neuro"]),
                "race_black": float(rec["race_black"]),
                "race_hispanic": float(rec["race_hispanic"]),
                "race_asian": float(rec["race_asian"]),
                "race_other": float(rec["race_other"]),
                "admission_elective": float(rec["admission_elective"]),
                "admission_emergency": float(rec["admission_emergency"]),
                "charlson_conditions": float(rec["charlson_conditions"]),
                "dx_digestive": float(rec["dx_digestive"]),
                "sofa_t0": float(rec["sofa_t0"]),
                "sofa_missing": float(rec["sofa_missing"]),
            }
        )
    return rows


KEYS = [
    "anchor_age",
    "female",
    "vaso_t0",
    "unit_micu",
    "unit_sicu",
    "unit_neuro",
    "race_black",
    "race_hispanic",
    "race_asian",
    "race_other",
    "admission_elective",
    "admission_emergency",
    "charlson_conditions",
    "dx_digestive",
    "sofa_t0",
    "sofa_missing",
]


def clones(stays: list[dict]) -> list[dict]:
    out = []
    for s in stays:
        en = s["en_h"]
        death = s["death_h"]
        # early 48h: censor at 48 if no EN by 48 and no death by 48
        if en is not None and en <= 48:
            c_early = None
        elif death is not None and death <= 48:
            c_early = None
        else:
            c_early = 48.0
        # delayed 96h: first EN before 96 censors unless death first
        if en is not None and en < 96 and (death is None or death > en):
            c_delay = float(en)
        else:
            c_delay = None
        for strat, cen in (("early_48", c_early), ("delayed_96", c_delay)):
            out.append({**s, "strategy": strat, "censor_h": cen})
    return out


def person_periods(clones_rows: list[dict]) -> list[dict]:
    rows = []
    for c in clones_rows:
        death = c["death_h"]
        cen = c["censor_h"]
        for start in PERIOD_STARTS:
            if death is not None and death <= start:
                continue
            if cen is not None and cen <= start:
                continue
            end = start + 6
            uncensored = 1.0
            if cen is not None and start < cen <= end:
                uncensored = 0.0
            rows.append({**c, "period": start, "uncensored": uncensored})
            if uncensored == 0.0:
                break
    return rows


def weights(pp: list[dict]) -> dict[tuple, float]:
    w_cum: dict[tuple, float] = {}
    by = {}
    for r in pp:
        by.setdefault((r["strategy"], r["period"]), []).append(r)
    period_w: dict[tuple, float] = {}
    for key, rows in by.items():
        y = np.array([r["uncensored"] for r in rows])
        x_num = np.ones((len(rows), 1))
        b_num = fit_logit(x_num, y)
        p_num = sigmoid(x_num @ b_num)
        x_den = design(rows, KEYS)
        b_den = fit_logit(x_den, y)
        p_den = np.clip(sigmoid(x_den @ b_den), 1e-4, 1 - 1e-4)
        pw = p_num / p_den
        for r, wi in zip(rows, pw):
            period_w[(r["stay_id"], r["strategy"], r["period"])] = float(wi)
    for r in pp:
        if r["uncensored"] != 1.0:
            continue
        prev = w_cum.get((r["stay_id"], r["strategy"]), 1.0)
        w_cum[(r["stay_id"], r["strategy"])] = prev * period_w[(r["stay_id"], r["strategy"], r["period"])]
    return w_cum


def hajek(clones_rows: list[dict], w_cum: dict[tuple, float]) -> dict:
    out = {}
    for strat in ("early_48", "delayed_96"):
        num = den = 0.0
        n = 0
        ws = []
        for c in clones_rows:
            if c["strategy"] != strat:
                continue
            if c["censor_h"] is not None:
                continue
            w = w_cum.get((c["stay_id"], strat))
            if w is None or not math.isfinite(w):
                continue
            num += w * c["dead28"]
            den += w
            n += 1
            ws.append(w)
        risk = num / den if den else None
        out[strat] = {
            "n_uncensored": n,
            "weight_sum": den,
            "risk": risk,
            "w_min": min(ws) if ws else None,
            "w_max": max(ws) if ws else None,
            "w_median": float(np.median(ws)) if ws else None,
        }
    r0 = out["early_48"]["risk"]
    r1 = out["delayed_96"]["risk"]
    out["rd"] = None if r0 is None or r1 is None else r0 - r1
    out["rr"] = None if r0 is None or r1 is None or r1 == 0 else r0 / r1
    return out


def bootstrap(stays: list[dict], n_boot: int) -> dict:
    rng = np.random.default_rng(SEED)
    rds, rrs = [], []
    failed = 0
    by_subj: dict[int, list[dict]] = {}
    for s in stays:
        by_subj.setdefault(s["subject_id"], []).append(s)
    subjects = list(by_subj)
    for _ in range(n_boot):
        draw = rng.choice(subjects, size=len(subjects), replace=True)
        sample = []
        for i, sid in enumerate(draw):
            for row in by_subj[int(sid)]:
                sample.append({**row, "stay_id": row["stay_id"] * 100000 + i})
        try:
            cl = clones(sample)
            pp = person_periods(cl)
            w = weights(pp)
            est = hajek(cl, w)
            if est["rd"] is None or est["rr"] is None:
                failed += 1
                continue
            rds.append(est["rd"])
            rrs.append(est["rr"])
        except Exception:
            failed += 1
    def ci(xs):
        if len(xs) < 20:
            return None
        return [float(np.percentile(xs, 2.5)), float(np.percentile(xs, 97.5))]
    return {"n_requested": n_boot, "n_ok": len(rds), "n_failed": failed, "rd_ci": ci(rds), "rr_ci": ci(rrs)}


def main() -> None:
    stays = assemble()
    cl = clones(stays)
    pp = person_periods(cl)
    w = weights(pp)
    est = hajek(cl, w)
    boot = bootstrap(stays, BOOT)
    n_early_c = sum(1 for c in cl if c["strategy"] == "early_48" and c["censor_h"] is not None)
    n_delay_c = sum(1 for c in cl if c["strategy"] == "delayed_96" and c["censor_h"] is not None)
    payload = {
        "generated_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "status": "pipeline_qc_not_author_final",
        "n_stays": len(stays),
        "dead28": int(sum(s["dead28"] for s in stays)),
        "clones": len(cl),
        "early_censored": n_early_c,
        "delayed_censored": n_delay_c,
        "person_periods": len(pp),
        "ridge": RIDGE,
        "bootstrap_replicates_qc": BOOT,
        "estimate": est,
        "bootstrap": boot,
        "note": "28-day death treats missing dod as alive. Restricted non-CVICU/CCU ventilated cohort. Not 2000 BCa. Not MSM g-computation.",
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
