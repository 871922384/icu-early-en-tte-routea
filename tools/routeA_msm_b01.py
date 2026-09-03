#!/usr/bin/env python3
"""B01/A08: standardize both strategy risks to the eligible covariate distribution.

Weighted pooled logistic MSM + g-computation, plus Hajek and Horvitz-Thompson,
plus weighted SMD vs the eligible cohort (B13 required with B01).

Pipeline QC. Not author-final. Not 2,000 BCa.
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from routeA_ipcw_qc import (  # noqa: E402
    KEYS,
    assemble,
    clones,
    design,
    fit_logit,
    hajek,
    person_periods,
    sigmoid,
    weights,
)

OUT = ROOT / "notes/cce-audit-routeA-B01-msm.json"
SMD_CSV = ROOT / "notes/cce-audit-routeA-B01-smd.csv"
DAYS = 28
RIDGE = 1e-2
KNOTS = (4.0, 14.0, 24.0)


def rcs_terms(t: np.ndarray, knots=KNOTS) -> tuple[np.ndarray, np.ndarray]:
    k1, k2, k3 = knots

    def p(x: np.ndarray) -> np.ndarray:
        return np.clip(x, 0.0, None) ** 3

    z1 = t
    z2 = p(t - k1) - ((k3 - k1) / (k3 - k2)) * p(t - k2) + ((k2 - k1) / (k3 - k2)) * p(t - k3)
    return z1, z2


def unstabilized_weights(pp: list[dict]) -> dict[tuple, float]:
    by: dict[tuple, list] = {}
    for r in pp:
        by.setdefault((r["strategy"], r["period"]), []).append(r)
    period_w: dict[tuple, float] = {}
    for rows in by.values():
        y = np.array([r["uncensored"] for r in rows])
        x_den = design(rows, KEYS)
        b_den = fit_logit(x_den, y)
        p_den = np.clip(sigmoid(x_den @ b_den), 1e-4, 1 - 1e-4)
        for r, p in zip(rows, p_den):
            period_w[(r["stay_id"], r["strategy"], r["period"])] = float(1.0 / p)
    w_cum: dict[tuple, float] = {}
    for r in pp:
        if r["uncensored"] != 1.0:
            continue
        prev = w_cum.get((r["stay_id"], r["strategy"]), 1.0)
        w_cum[(r["stay_id"], r["strategy"])] = prev * period_w[(r["stay_id"], r["strategy"], r["period"])]
    return w_cum


def horvitz_thompson(clones_rows: list[dict], w_u: dict[tuple, float], n_elig: int) -> dict:
    out = {}
    for strat in ("early_48", "delayed_96"):
        total = 0.0
        n_used = 0
        for c in clones_rows:
            if c["strategy"] != strat or c["censor_h"] is not None:
                continue
            w = w_u.get((c["stay_id"], strat))
            if w is None or not math.isfinite(w):
                continue
            total += w * c["dead28"]
            n_used += 1
        risk = total / n_elig if n_elig else None
        out[strat] = {"n_uncensored": n_used, "weighted_event_sum": total, "risk": risk}
    r0 = out["early_48"]["risk"]
    r1 = out["delayed_96"]["risk"]
    out["rd"] = None if r0 is None or r1 is None else r0 - r1
    out["rr"] = None if r0 is None or r1 is None or r1 == 0 else r0 / r1
    return out


def outcome_days(clones_rows: list[dict], w_s: dict[tuple, float]) -> list[dict]:
    rows = []
    for c in clones_rows:
        if c["censor_h"] is not None:
            continue
        w = w_s.get((c["stay_id"], c["strategy"]))
        if w is None or not math.isfinite(w):
            continue
        death_day = None if c["death_h"] is None else int(c["death_h"] // 24)
        last = DAYS - 1
        if death_day is not None and death_day < last:
            last = death_day
        for day in range(last + 1):
            y = 1.0 if death_day is not None and day == death_day and c["dead28"] == 1.0 else 0.0
            rows.append({**c, "day": float(day), "y_death": y, "w": w})
            if y == 1.0:
                break
    return rows


def msm_design(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(rows)
    t = np.array([r["day"] for r in rows], dtype=float)
    z1, z2 = rcs_terms(t)
    early = np.array([1.0 if r["strategy"] == "early_48" else 0.0 for r in rows])
    x_base = design(rows, KEYS)
    x = np.column_stack(
        [
            x_base,
            early,
            z1,
            z2,
            early * z1,
            early * z2,
        ]
    )
    y = np.array([r["y_death"] for r in rows], dtype=float)
    w = np.array([r["w"] for r in rows], dtype=float)
    return x, y, w


def fit_logit_weighted(x: np.ndarray, y: np.ndarray, w: np.ndarray, ridge: float = RIDGE, steps: int = 80) -> np.ndarray:
    n, p = x.shape
    beta = np.zeros(p)
    if n == 0:
        return beta
    for _ in range(steps):
        mu = sigmoid(x @ beta)
        var = mu * (1.0 - mu)
        resid = y - mu
        grad = x.T @ (w * resid) - ridge * beta
        hess = x.T @ (x * (w * var)[:, None])
        hess.flat[:: p + 1] += ridge
        try:
            delta = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            break
        beta = beta + delta
        if float(np.max(np.abs(delta))) < 1e-8:
            break
    return beta


def g_compute(stays: list[dict], beta: np.ndarray) -> dict:
    n = len(stays)
    x_base = design(stays, KEYS)
    days = np.arange(DAYS, dtype=float)
    z1, z2 = rcs_terms(days)
    risks = {}
    for strat, early in (("early_48", 1.0), ("delayed_96", 0.0)):
        f28 = np.zeros(n)
        surv = np.ones(n)
        for d in range(DAYS):
            extra = np.column_stack(
                [
                    np.full(n, early),
                    np.full(n, z1[d]),
                    np.full(n, z2[d]),
                    np.full(n, early * z1[d]),
                    np.full(n, early * z2[d]),
                ]
            )
            x = np.column_stack([x_base, extra])
            h = np.clip(sigmoid(x @ beta), 1e-12, 1 - 1e-12)
            f28 += surv * h
            surv *= 1.0 - h
        risks[strat] = float(np.mean(f28))
        risks[f"{strat}_p50"] = float(np.median(f28))
        risks[f"{strat}_p05"] = float(np.percentile(f28, 5))
        risks[f"{strat}_p95"] = float(np.percentile(f28, 95))
    r0, r1 = risks["early_48"], risks["delayed_96"]
    risks["rd"] = r0 - r1
    risks["rr"] = None if r1 == 0 else r0 / r1
    return risks


def smd_table(stays: list[dict], clones_rows: list[dict], w_s: dict[tuple, float]) -> list[dict]:
    n = len(stays)
    pop = {k: np.array([s[k] for s in stays], dtype=float) for k in KEYS}
    pop_mean = {k: float(v.mean()) for k, v in pop.items()}
    pop_sd = {k: float(v.std(ddof=1) or 1.0) for k, v in pop.items()}
    rows = []
    for strat in ("early_48", "delayed_96"):
        uncens = [c for c in clones_rows if c["strategy"] == strat and c["censor_h"] is None]
        ww = np.array([w_s.get((c["stay_id"], strat), 0.0) for c in uncens], dtype=float)
        ww = np.where(np.isfinite(ww), ww, 0.0)
        for k in KEYS:
            x = np.array([c[k] for c in uncens], dtype=float)
            m_u = float(x.mean()) if len(x) else None
            m_w = float(np.sum(ww * x) / np.sum(ww)) if np.sum(ww) else None
            smd_u = None if m_u is None else (m_u - pop_mean[k]) / pop_sd[k]
            smd_w = None if m_w is None else (m_w - pop_mean[k]) / pop_sd[k]
            rows.append(
                {
                    "strategy": strat,
                    "covariate": k,
                    "eligible_mean": pop_mean[k],
                    "uncensored_unweighted_mean": m_u,
                    "uncensored_weighted_mean": m_w,
                    "smd_unweighted_vs_eligible": smd_u,
                    "smd_weighted_vs_eligible": smd_w,
                    "n_uncensored": len(uncens),
                    "n_eligible": n,
                }
            )
    return rows


def estimate_bundle(stays: list[dict]) -> dict:
    cl = clones(stays)
    pp = person_periods(cl)
    w_s = weights(pp)
    w_u = unstabilized_weights(pp)
    haj = hajek(cl, w_s)
    ht = horvitz_thompson(cl, w_u, len(stays))
    od = outcome_days(cl, w_s)
    x, y, w = msm_design(od)
    beta = fit_logit_weighted(x, y, w)
    msm = g_compute(stays, beta)
    return {"hajek": haj, "ht": ht, "msm": msm, "clones": cl, "w_s": w_s, "events": int(y.sum()), "days": len(od)}


def bootstrap_msm(stays: list[dict], n_boot: int, seed: int = 20260902) -> dict:
    rng = np.random.default_rng(seed)
    by_subj: dict[int, list[dict]] = {}
    for s in stays:
        by_subj.setdefault(s["subject_id"], []).append(s)
    subjects = list(by_subj)
    rds, rrs = [], []
    failed = 0
    for _ in range(n_boot):
        draw = rng.choice(subjects, size=len(subjects), replace=True)
        sample = []
        for i, sid in enumerate(draw):
            for row in by_subj[int(sid)]:
                sample.append({**row, "stay_id": int(row["stay_id"]) * 100000 + i})
        try:
            est = estimate_bundle(sample)["msm"]
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

    return {
        "n_requested": n_boot,
        "n_ok": len(rds),
        "n_failed": failed,
        "rd_ci": ci(rds),
        "rr_ci": ci(rrs),
    }


def main() -> None:
    stays = assemble()
    bundled = estimate_bundle(stays)
    haj = bundled["hajek"]
    ht = bundled["ht"]
    msm = bundled["msm"]
    cl = bundled["clones"]
    w_s = bundled["w_s"]
    smd = smd_table(stays, cl, w_s)
    n_gt_01_u = sum(1 for r in smd if r["smd_unweighted_vs_eligible"] is not None and abs(r["smd_unweighted_vs_eligible"]) > 0.10)
    n_gt_01_w = sum(1 for r in smd if r["smd_weighted_vs_eligible"] is not None and abs(r["smd_weighted_vs_eligible"]) > 0.10)
    with SMD_CSV.open("w") as handle:
        cols = list(smd[0].keys())
        handle.write(",".join(cols) + "\n")
        for r in smd:
            handle.write(",".join("" if r[c] is None else str(r[c]) for c in cols) + "\n")
    n_boot = 0
    if len(sys.argv) > 1:
        n_boot = int(sys.argv[1])
    boot = bootstrap_msm(stays, n_boot) if n_boot else {"n_requested": 0, "n_ok": 0, "n_failed": 0, "rd_ci": None, "rr_ci": None}
    payload = {
        "generated_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "status": "pipeline_qc_not_author_final",
        "b01": "MSM g-computation standardizes both strategies to the empirical baseline distribution of the restricted eligible cohort.",
        "n_stays": len(stays),
        "outcome_person_days": bundled["days"],
        "msm_events": bundled["events"],
        "rcs_knots_days": list(KNOTS),
        "hajek_stabilized": haj,
        "horvitz_thompson_unstabilized": ht,
        "msm_gcomputation": msm,
        "smd": {
            "n_covariate_strategy_rows": len(smd),
            "n_smd_unweighted_gt_0_10": n_gt_01_u,
            "n_smd_weighted_gt_0_10": n_gt_01_w,
            "csv": str(SMD_CSV.relative_to(ROOT)),
        },
        "bootstrap_msm": boot,
        "note": "Target population is the restricted ventilated non-CVICU/CCU eligible cohort. Missing dod treated as alive at day 28. Not Firth. Reduced SOFA still used.",
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
