#!/usr/bin/env python3
"""A07 competing-risk AJ, A09 Firth weight diagnostics, A12 PBA, A14 positive control.

Pipeline QC. Not author-final. Does not write manuscript placeholders.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import duckdb
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from routeA_a06_a13_a11 import choose_keys, now, overlay_a06, set_keys  # noqa: E402
from routeA_a11_fast import (  # noqa: E402
    DAYS,
    PERIODS,
    censor_times,
    estimate_msm,
    ipc_weights,
    matrices,
)
from routeA_ipcw_qc import KEYS as IPCW_KEYS  # noqa: E402
from routeA_ipcw_qc import RIDGE, assemble, fit_logit, sigmoid  # noqa: E402

COHORT = ROOT / "workspace/metered-results/cohort"
MIMIC = Path("/Volumes/B/litflow-cache/physionet/physionet.org/files/mimiciv/3.1")
OUT_A07 = ROOT / "notes/cce-audit-routeA-A07.json"
OUT_A09 = ROOT / "notes/cce-audit-routeA-A09-weight-models.json"
CSV_A09 = ROOT / "notes/cce-audit-routeA-A09-weight-models.csv"
PNG_A09 = ROOT / "notes/cce-audit-routeA-A09-p-hist.png"
PNG_A07 = ROOT / "notes/cce-audit-routeA-A07-cif.png"
OUT_A12 = ROOT / "notes/cce-audit-routeA-A12-pba.json"
OUT_A14 = ROOT / "notes/cce-audit-routeA-A14.json"
SEED = 20260902
HORIZON_H = 28 * 24


def _num(v):
    if v is None:
        return None
    try:
        if v != v:
            return None
    except Exception:
        return None
    try:
        return float(v)
    except Exception:
        return None


def overlay_times(stays: list[dict]) -> list[dict]:
    con = duckdb.connect()
    path = (COHORT / "routeA_restricted_stays.parquet").as_posix()
    df = con.execute(
        f"""
        SELECT stay_id, t0, dod, dischtime, hospital_expire_flag
        FROM read_parquet('{path}')
        """
    ).fetchdf()
    by = {int(r["stay_id"]): r for r in df.to_dict("records")}
    out = []
    for s in stays:
        r = by[int(s["stay_id"])]
        t0 = r["t0"]
        disch_h = None
        if r["dischtime"] is not None and t0 is not None:
            try:
                disch_h = float((np.datetime64(r["dischtime"]) - np.datetime64(t0)) / np.timedelta64(1, "h"))
            except Exception:
                disch_h = None
        hosp = int(r["hospital_expire_flag"] or 0)
        death_h = s["death_h"]
        inhosp_h = None
        if hosp == 1 and death_h is not None and death_h >= 0:
            if disch_h is None or death_h <= disch_h + 24.0:
                inhosp_h = float(death_h)
        row = dict(s)
        row["discharge_h"] = disch_h
        row["hospital_expire_flag"] = float(hosp)
        row["inhosp_death_h"] = inhosp_h
        row["dead90"] = 1.0 if (death_h is not None and 0 <= death_h <= 90 * 24) else 0.0
        out.append(row)
    return out


def competing_times(stays: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """event 1=in-hospital death, 2=discharge alive, 0=still followed at day 28."""
    n = len(stays)
    time_h = np.full(n, float(HORIZON_H))
    event = np.zeros(n, dtype=int)
    for i, s in enumerate(stays):
        ih = s["inhosp_death_h"]
        dh = s["discharge_h"]
        if ih is not None and 0 <= ih <= HORIZON_H:
            time_h[i] = ih
            event[i] = 1
        elif dh is not None and 0 <= dh <= HORIZON_H:
            time_h[i] = dh
            event[i] = 2
    return time_h, event


def weighted_aj(time_h: np.ndarray, event: np.ndarray, w: np.ndarray, use: np.ndarray) -> dict:
    t = np.floor(time_h / 24.0)
    ev = event
    ww = w
    ok = use & np.isfinite(ww) & np.isfinite(t)
    cif1 = np.zeros(28)
    cif2 = np.zeros(28)
    surv = 1.0
    n_risk = []
    for d in range(28):
        at = ok & (t >= d)
        r = float(ww[at].sum())
        n_risk.append(int(at.sum()))
        d1 = float(ww[ok & (t == d) & (ev == 1)].sum())
        d2 = float(ww[ok & (t == d) & (ev == 2)].sum())
        haz1 = d1 / r if r > 0 else 0.0
        haz2 = d2 / r if r > 0 else 0.0
        cif1[d] = cif1[d - 1] + surv * haz1 if d else surv * haz1
        cif2[d] = cif2[d - 1] + surv * haz2 if d else surv * haz2
        surv = max(0.0, surv * (1.0 - haz1 - haz2))
    return {
        "cif_inhosp_death_28": float(cif1[-1]),
        "cif_discharge_28": float(cif2[-1]),
        "cif_death_curve": [float(x) for x in cif1],
        "cif_discharge_curve": [float(x) for x in cif2],
        "n_used": int(ok.sum()),
        "n_risk_day0": n_risk[0] if n_risk else 0,
        "weight_sum": float(ww[ok].sum()) if ok.any() else 0.0,
    }


def weighted_km_censor_discharge(stays: list[dict], w: np.ndarray, use: np.ndarray) -> dict:
    n = len(stays)
    time_h = np.full(n, float(HORIZON_H))
    event = np.zeros(n, dtype=int)
    for i, s in enumerate(stays):
        ih = s["inhosp_death_h"]
        dh = s["discharge_h"]
        if ih is not None and 0 <= ih <= HORIZON_H and (dh is None or ih <= dh):
            time_h[i] = ih
            event[i] = 1
        elif dh is not None and 0 <= dh <= HORIZON_H:
            time_h[i] = dh
            event[i] = 0  # censored at discharge
    t = np.floor(time_h / 24.0)
    ww = w
    ok = use & np.isfinite(ww)
    cif = np.zeros(28)
    surv = 1.0
    for d in range(28):
        at = ok & (t >= d)
        r = float(ww[at].sum())
        d1 = float(ww[ok & (t == d) & (event == 1)].sum())
        haz = d1 / r if r > 0 else 0.0
        cif[d] = cif[d - 1] + surv * haz if d else surv * haz
        surv = max(0.0, surv * (1.0 - haz))
    return {"risk_28": float(cif[-1]), "curve": [float(x) for x in cif], "n_used": int(ok.sum())}


def compact_msm(msm: dict) -> dict:
    keep = (
        "early_48",
        "delayed_96",
        "rd",
        "rr",
        "events",
        "days",
        "n_uncensored_early",
        "n_uncensored_delayed",
        "horizon_days",
        "early_48_p50",
        "delayed_96_p50",
    )
    return {k: msm[k] for k in keep if k in msm}


def fit_logit_firth(x: np.ndarray, y: np.ndarray, steps: int = 80) -> tuple[np.ndarray, bool, int, float]:
    n, p = x.shape
    beta = np.zeros(p)
    if n == 0:
        return beta, False, 0, None
    last_delta = None
    for it in range(steps):
        mu = sigmoid(x @ beta)
        w = mu * (1.0 - mu)
        hess = x.T @ (x * w[:, None])
        hess.flat[:: p + 1] += 1e-8
        try:
            inv = np.linalg.inv(hess)
        except np.linalg.LinAlgError:
            return beta, False, it + 1, last_delta
        h = np.sum((x @ inv) * x, axis=1) * w
        resid = y - mu + h * (0.5 - mu)
        grad = x.T @ resid
        try:
            delta = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            return beta, False, it + 1, last_delta
        beta = beta + delta
        last_delta = float(np.max(np.abs(delta)))
        if last_delta < 1e-8:
            return beta, True, it + 1, last_delta
    return beta, False, steps, last_delta


def auc_score(y: np.ndarray, p: np.ndarray) -> float | None:
    n1 = float((y == 1).sum())
    n0 = float((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return None
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), dtype=float)
    ranks[order] = np.arange(1, len(p) + 1, dtype=float)
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def calib_logit(y: np.ndarray, p: np.ndarray) -> tuple[float | None, float | None]:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    lp = np.log(p / (1.0 - p))
    x = np.column_stack([np.ones(len(p)), lp])
    b = fit_logit(x, y)
    return float(b[0]), float(b[1])


def period_diagnostics(x: np.ndarray, death: np.ndarray, censor: np.ndarray, strategy: str, method: str) -> list[dict]:
    death_ok = np.isfinite(death)
    rows = []
    for start in PERIODS:
        at = (~death_ok | (death > start)) & (~np.isfinite(censor) | (censor > start))
        if not np.any(at):
            rows.append(
                {
                    "strategy": strategy,
                    "period_start_h": start,
                    "method": method,
                    "n": 0,
                    "n_uncensored": 0,
                    "n_covariates": int(x.shape[1]),
                    "converged": False,
                    "iterations": 0,
                    "max_abs_delta": None,
                    "auc": None,
                    "calib_intercept": None,
                    "calib_slope": None,
                    "p_min": None,
                    "p_p01": None,
                    "p_p50": None,
                    "p_p99": None,
                    "p_max": None,
                    "p_lt_1e4": None,
                }
            )
            continue
        xr = x[at]
        y = np.ones(xr.shape[0])
        cen = censor[at]
        hit = np.isfinite(cen) & (cen > start) & (cen <= start + 6.0)
        y[hit] = 0.0
        n_y = float(y.sum())
        if n_y == 0.0 or n_y == len(y):
            p = np.full(len(y), n_y / len(y) if len(y) else 1.0)
            conv, it, md = True, 0, 0.0
            method_note = "all_uncensored" if n_y == len(y) else "all_censored"
        elif method == "firth":
            beta, conv, it, md = fit_logit_firth(xr, y)
            p = sigmoid(xr @ beta)
            method_note = None
        else:
            beta = fit_logit(xr, y)
            conv, it, md = True, None, None
            p = sigmoid(xr @ beta)
            method_note = None
        degenerate = n_y == 0.0 or n_y == len(y)
        pclip = np.clip(p, 1e-12, 1 - 1e-12)
        ci = cs = None
        if not degenerate:
            ci, cs = calib_logit(y, pclip)
        rows.append(
            {
                "strategy": strategy,
                "period_start_h": start,
                "method": method,
                "n": int(xr.shape[0]),
                "n_uncensored": int(y.sum()),
                "n_covariates": int(xr.shape[1]),
                "converged": bool(conv),
                "iterations": it,
                "max_abs_delta": md,
                "auc": None if degenerate else auc_score(y, pclip),
                "calib_intercept": ci,
                "calib_slope": cs,
                "p_min": float(p.min()),
                "p_p01": float(np.percentile(p, 1)),
                "p_p50": float(np.percentile(p, 50)),
                "p_p99": float(np.percentile(p, 99)),
                "p_max": float(p.max()),
                "p_lt_1e4": int((p < 1e-4).sum()),
                "degenerate": method_note,
                "p": p,
            }
        )
    return rows


def kish_ess(w: np.ndarray) -> float | None:
    w = w[np.isfinite(w)]
    if w.size == 0:
        return None
    s1 = float(w.sum())
    s2 = float(np.square(w).sum())
    return None if s2 == 0 else s1 * s1 / s2


def run_a07(stays, x, en, death, dead28) -> dict:
    msm = estimate_msm(x, en, death, dead28, days=28)
    time_h, event = competing_times(stays)
    c_e, c_d = msm["c_early"], msm["c_delay"]
    w_e, w_d = msm["w_early"], msm["w_delayed"]
    use_e = ~np.isfinite(c_e) & np.isfinite(w_e)
    use_d = ~np.isfinite(c_d) & np.isfinite(w_d)
    aj_e = weighted_aj(time_h, event, w_e, use_e)
    aj_d = weighted_aj(time_h, event, w_d, use_d)
    km_e = weighted_km_censor_discharge(stays, w_e, use_e)
    km_d = weighted_km_censor_discharge(stays, w_d, use_d)
    dead90 = np.array([s["dead90"] for s in stays], dtype=float)
    msm90 = estimate_msm(x, en, death, dead90, days=90)
    n_inhosp = int(sum(1 for s in stays if s["inhosp_death_h"] is not None and s["inhosp_death_h"] <= HORIZON_H))
    n_disch = int(sum(1 for s in stays if s["inhosp_death_h"] is None and s["discharge_h"] is not None and s["discharge_h"] <= HORIZON_H))
    n_dead90 = int(dead90.sum())
    rd_aj = aj_e["cif_inhosp_death_28"] - aj_d["cif_inhosp_death_28"]
    rr_aj = None if aj_d["cif_inhosp_death_28"] == 0 else aj_e["cif_inhosp_death_28"] / aj_d["cif_inhosp_death_28"]
    rd_km = km_e["risk_28"] - km_d["risk_28"]
    rr_km = None if km_d["risk_28"] == 0 else km_e["risk_28"] / km_d["risk_28"]
    payload = {
        "generated_at": now(),
        "status": "pipeline_qc_not_author_final",
        "n_stays": len(stays),
        "primary_28d_allcause_msm": compact_msm(msm),
        "counts": {
            "dead28_allcause": int(sum(s["dead28"] for s in stays)),
            "inhosp_death_by_28d": n_inhosp,
            "discharge_alive_by_28d": n_disch,
            "dead90_allcause": n_dead90,
        },
        "secondary_aj_28d_inhospital": {
            "early_48": aj_e["cif_inhosp_death_28"],
            "delayed_96": aj_d["cif_inhosp_death_28"],
            "rd": rd_aj,
            "rr": rr_aj,
            "cif_discharge_early": aj_e["cif_discharge_28"],
            "cif_discharge_delayed": aj_d["cif_discharge_28"],
            "n_used_early": aj_e["n_used"],
            "n_used_delayed": aj_d["n_used"],
            "note": "Weighted Aalen-Johansen on strategy-uncensored clones. Discharge alive is a competing event. Missing dod is not used.",
        },
        "sensitivity_censor_at_discharge": {
            "early_48": km_e["risk_28"],
            "delayed_96": km_d["risk_28"],
            "rd": rd_km,
            "rr": rr_km,
            "note": "Weighted 1-KM treating discharge alive as independent censoring. Not the primary competing-risk estimand.",
        },
        "sensitivity_90d_allcause_msm": compact_msm(msm90),
        "ess": {
            "early_kish": kish_ess(w_e[use_e]),
            "delayed_kish": kish_ess(w_d[use_d]),
        },
        "curves": {
            "aj_death_early": aj_e["cif_death_curve"],
            "aj_death_delayed": aj_d["cif_death_curve"],
            "aj_discharge_early": aj_e["cif_discharge_curve"],
            "aj_discharge_delayed": aj_d["cif_discharge_curve"],
        },
        "note": "Not author-final. Not copied into the 24h manuscript.",
    }
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        days = list(range(1, 29))
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        ax.step(days, aj_e["cif_death_curve"], where="post", label="Early 48 h EN, in-hospital death")
        ax.step(days, aj_d["cif_death_curve"], where="post", label="No EN by 96 h, in-hospital death")
        ax.step(days, aj_e["cif_discharge_curve"], where="post", linestyle="--", label="Early, discharge alive")
        ax.step(days, aj_d["cif_discharge_curve"], where="post", linestyle="--", label="Delayed, discharge alive")
        ax.set_xlabel("Days from time zero")
        ax.set_ylabel("Weighted cumulative incidence")
        ax.set_xlim(1, 28)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(PNG_A07, dpi=150)
        plt.close(fig)
        payload["cif_png"] = str(PNG_A07.relative_to(ROOT))
    except Exception as exc:
        payload["cif_png_error"] = str(exc)
    OUT_A07.write_text(json.dumps(payload, indent=2) + "\n")
    print("A07", json.dumps({k: payload[k] for k in ("primary_28d_allcause_msm", "secondary_aj_28d_inhospital", "sensitivity_90d_allcause_msm", "counts")}, indent=2), flush=True)
    return payload


def run_a09(x, en, death) -> dict:
    c_e, c_d = censor_times(en, death)
    diags = []
    all_p = []
    for method in ("ridge_0.01", "firth"):
        for strat, cen in (("early_48", c_e), ("delayed_96", c_d)):
            rows = period_diagnostics(x, death, cen, strat, method)
            for r in rows:
                p = r.pop("p", None)
                if p is not None and method == "firth":
                    all_p.append(p)
                diags.append(r)
    n_conv = sum(1 for r in diags if r["method"] == "firth" and r["converged"])
    n_firth = sum(1 for r in diags if r["method"] == "firth")
    n_degen = sum(1 for r in diags if r["method"] == "firth" and r.get("degenerate"))
    with CSV_A09.open("w", newline="") as handle:
        cols = [k for k in diags[0].keys()]
        w = csv.DictWriter(handle, fieldnames=cols)
        w.writeheader()
        for r in diags:
            w.writerow(r)
    # Firth IPCW then MSM point
    def firth_weights(censor):
        n = x.shape[0]
        w_cum = np.ones(n)
        seen = np.zeros(n, dtype=bool)
        death_ok = np.isfinite(death)
        for start in PERIODS:
            at = (~death_ok | (death > start)) & (~np.isfinite(censor) | (censor > start))
            if not np.any(at):
                continue
            seen[at] = True
            xr = x[at]
            y = np.ones(xr.shape[0])
            cen = censor[at]
            hit = np.isfinite(cen) & (cen > start) & (cen <= start + 6.0)
            y[hit] = 0.0
            b_num = fit_logit(np.ones((xr.shape[0], 1)), y)
            p_num = sigmoid(np.full(xr.shape[0], b_num[0]))
            b_den, _, _, _ = fit_logit_firth(xr, y)
            p_den = np.clip(sigmoid(xr @ b_den), 1e-4, 1 - 1e-4)
            pw = p_num / p_den
            idx = np.flatnonzero(at)
            keep = y == 1.0
            w_cum[idx[keep]] *= pw[keep]
        w_cum[~seen] = np.nan
        return w_cum

    from routeA_a11_fast import g_compute, outcome_stack
    from routeA_msm_b01 import fit_logit_weighted

    w_e = firth_weights(c_e)
    w_d = firth_weights(c_d)
    u_e = ~np.isfinite(c_e)
    u_d = ~np.isfinite(c_d)
    dead = (np.isfinite(death) & (death >= 0) & (death <= HORIZON_H)).astype(float)
    xe, ye, we = outcome_stack(x, death, dead, u_e, w_e, 1.0)
    xd, yd, wd = outcome_stack(x, death, dead, u_d, w_d, 0.0)
    beta = fit_logit_weighted(np.vstack((xe, xd)), np.concatenate((ye, yd)), np.concatenate((we, wd)))
    msm_firth = g_compute(x, beta)
    payload = {
        "generated_at": now(),
        "status": "pipeline_qc_not_author_final",
        "n_models_firth": n_firth,
        "n_converged_firth": n_conv,
        "n_degenerate_no_censoring_events": n_degen,
        "ridge": RIDGE,
        "csv": str(CSV_A09.relative_to(ROOT)),
        "firth_msm_gcomputation": {
            "early_48": msm_firth["early_48"],
            "delayed_96": msm_firth["delayed_96"],
            "rd": msm_firth["rd"],
            "rr": msm_firth["rr"],
        },
        "note": "Primary A11/A06 weights used ridge 0.01. Firth is the A09 alternative. Predicted probabilities are untruncated in the diagnostic table; IPCW still clips denominator p to 1e-4 as in the Route A QC pipeline.",
    }
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        pooled = np.concatenate(all_p) if all_p else np.array([])
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        ax.hist(pooled, bins=40, range=(0, 1))
        ax.set_xlabel("Predicted probability of remaining uncensored (Firth)")
        ax.set_ylabel("Person-periods")
        fig.tight_layout()
        fig.savefig(PNG_A09, dpi=150)
        plt.close(fig)
        payload["histogram"] = str(PNG_A09.relative_to(ROOT))
        if pooled.size:
            payload["p_distribution_firth"] = {
                "n": int(pooled.size),
                "min": float(pooled.min()),
                "p01": float(np.percentile(pooled, 1)),
                "p25": float(np.percentile(pooled, 25)),
                "p50": float(np.percentile(pooled, 50)),
                "p75": float(np.percentile(pooled, 75)),
                "p99": float(np.percentile(pooled, 99)),
                "max": float(pooled.max()),
                "n_lt_1e4": int((pooled < 1e-4).sum()),
            }
    except Exception as exc:
        payload["histogram_error"] = str(exc)
    OUT_A09.write_text(json.dumps(payload, indent=2) + "\n")
    print("A09", json.dumps({k: payload[k] for k in payload if k != "csv"}, indent=2)[:2000], flush=True)
    return payload


def run_a12(rr_obs: float, rd_obs: float, delayed_risk: float) -> dict:
    rng = np.random.default_rng(SEED)
    n = 10000
    # U = unmeasured pre-t0 severity / comfort-care trajectory
    p0 = rng.uniform(0.05, 0.30, size=n)  # prevalence in delayed strategy
    delta = rng.uniform(0.10, 0.40, size=n)
    p1 = np.clip(p0 + delta, 0.0, 0.95)
    log_rr_ud = rng.normal(math.log(2.0), 0.287, size=n)
    rr_ud = np.exp(log_rr_ud)
    bf = (rr_ud * p1 + (1.0 - p1)) / (rr_ud * p0 + (1.0 - p0))
    rr_adj = rr_obs / bf
    rd_adj = delayed_risk * (rr_adj - 1.0)
    payload = {
        "generated_at": now(),
        "status": "pipeline_qc_not_author_final",
        "observed_rr": rr_obs,
        "observed_rd": rd_obs,
        "n_draws": n,
        "priors": {
            "U": "unmeasured baseline severity / comfort-care trajectory, binary",
            "p_delayed": "Uniform(0.05, 0.30)",
            "prevalence_difference_early_minus_delayed": "Uniform(0.10, 0.40)",
            "RR_UD": "LogNormal(log 2.0, 0.287); median 2, approx 95% 1.3-4.0",
            "bias_model": "multiplicative RR bias factor (RR_UD*p1+(1-p1))/(RR_UD*p0+(1-p0))",
        },
        "adjusted_rr": {
            "p50": float(np.median(rr_adj)),
            "p025": float(np.percentile(rr_adj, 2.5)),
            "p975": float(np.percentile(rr_adj, 97.5)),
            "prob_rr_gt_1": float(np.mean(rr_adj > 1.0)),
            "prob_rr_gt_obs": float(np.mean(rr_adj > rr_obs)),
        },
        "adjusted_rd_if_delayed_risk_held": {
            "p50": float(np.median(rd_adj)),
            "p025": float(np.percentile(rd_adj, 2.5)),
            "p975": float(np.percentile(rd_adj, 97.5)),
        },
        "bias_factor": {
            "p50": float(np.median(bf)),
            "p025": float(np.percentile(bf, 2.5)),
            "p975": float(np.percentile(bf, 97.5)),
        },
        "required_sentence": "The E-value and this bias analysis address unmeasured confounding only; they do not quantify positivity violations or the structural composition of grace-period risk sets, which remain the dominant threats in this contrast.",
        "note": "Not author-final. Not copied into the 24h manuscript.",
    }
    OUT_A12.write_text(json.dumps(payload, indent=2) + "\n")
    print("A12", json.dumps(payload["adjusted_rr"], indent=2), flush=True)
    return payload


def run_a14_vaso(stays, x, death, dead28) -> dict:
    con = duckdb.connect()
    stays_p = (COHORT / "routeA_restricted_stays.parquet").as_posix()
    vaso_p = (COHORT / "vasopressor.parquet").as_posix()
    df = con.execute(
        f"""
        SELECT e.stay_id,
               MIN(date_diff('epoch', e.t0, v.starttime)/3600.0) AS vaso_h
        FROM read_parquet('{stays_p}') e
        LEFT JOIN read_parquet('{vaso_p}') v ON v.stay_id=e.stay_id
         AND v.starttime >= e.t0 - INTERVAL 6 HOUR
        GROUP BY e.stay_id
        """
    ).fetchdf()
    by = {int(r["stay_id"]): _num(r["vaso_h"]) for r in df.to_dict("records")}
    vaso_h = np.array([np.nan if by.get(s["stay_id"]) is None else max(0.0, by[s["stay_id"]]) for s in stays])
    n = len(stays)
    by48 = int(np.sum(np.isfinite(vaso_h) & (vaso_h <= 48)))
    msm = estimate_msm(x, vaso_h, death, dead28, days=28)
    vaso_t0 = np.array([float(s["vaso_t0"]) for s in stays])
    y = dead28
    keys_no_vaso = [k for k in IPCW_KEYS if k != "vaso_t0"]
    xb = np.ones((n, 1 + len(keys_no_vaso) + 1))
    xb[:, 1] = vaso_t0
    for j, key in enumerate(keys_no_vaso):
        xb[:, j + 2] = x[:, 1 + list(IPCW_KEYS).index(key)]
    beta = fit_logit(xb, y)
    x1 = xb.copy()
    x1[:, 1] = 1.0
    x0 = xb.copy()
    x0[:, 1] = 0.0
    r1 = float(np.mean(sigmoid(x1 @ beta)))
    r0 = float(np.mean(sigmoid(x0 @ beta)))
    return {
        "ccw": {
            "exposure": "first_vasopressor_start_relative_to_t0",
            "n": n,
            "by_48h": by48,
            "by_48h_pct": round(100 * by48 / n, 2),
            "msm": compact_msm(msm),
            "note": "Same 48h vs 96h CCW+IPCW+MSM. Expected direction if the pipeline recovered a harmful exposure: RR > 1.",
        },
        "baseline_vaso_t0": {
            "exposure": "vasopressor_overlapping_t0_plus_minus_6h",
            "n": n,
            "n_vaso": int(vaso_t0.sum()),
            "crude_vaso": float(y[vaso_t0 == 1].mean()) if (vaso_t0 == 1).any() else None,
            "crude_no_vaso": float(y[vaso_t0 == 0].mean()) if (vaso_t0 == 0).any() else None,
            "gcomputation": {"vaso": r1, "no_vaso": r0, "rd": r1 - r0, "rr": None if r0 == 0 else r1 / r0},
            "expected_direction": "RR > 1",
        },
    }


def run_a14_vent() -> dict:
    con = duckdb.connect()
    idx = (COHORT / "index_stays.parquet").as_posix()
    adm = (COHORT / "baseline_admissions.parquet").as_posix()
    proc = (MIMIC / "icu/procedureevents.csv.gz").as_posix()
    patients = (MIMIC / "hosp/patients.csv.gz").as_posix()
    df = con.execute(
        f"""
        WITH vent AS (
          SELECT DISTINCT i.stay_id
          FROM read_parquet('{idx}') i
          JOIN read_csv_auto('{proc}', header=true, compression='gzip') p
            ON p.stay_id=i.stay_id
           AND p.itemid IN (225792, 224385)
           AND p.starttime <= i.intime + INTERVAL 6 HOUR
           AND (p.endtime IS NULL OR p.endtime >= i.intime - INTERVAL 6 HOUR)
        )
        SELECT
          i.stay_id, i.subject_id, i.anchor_age,
          CASE WHEN i.gender='F' THEN 1 ELSE 0 END AS female,
          CASE WHEN v.stay_id IS NOT NULL THEN 1 ELSE 0 END AS vent,
          COALESCE(a.race_black,0) race_black,
          COALESCE(a.race_hispanic,0) race_hispanic,
          COALESCE(a.race_asian,0) race_asian,
          COALESCE(a.race_other,0) race_other,
          COALESCE(a.admission_elective,0) admission_elective,
          COALESCE(a.admission_emergency,0) admission_emergency,
          COALESCE(a.charlson_conditions,0) charlson_conditions,
          COALESCE(a.dx_digestive,0) dx_digestive,
          CASE WHEN pt.dod IS NOT NULL
                AND date_diff('hour', i.intime, CAST(pt.dod AS TIMESTAMP)) BETWEEN 0 AND {HORIZON_H}
               THEN 1 ELSE 0 END AS dead28
        FROM read_parquet('{idx}') i
        LEFT JOIN vent v USING (stay_id)
        LEFT JOIN read_parquet('{adm}') a USING (stay_id)
        LEFT JOIN read_csv_auto('{patients}', header=true, compression='gzip') pt
          ON pt.subject_id=i.subject_id
        """
    ).fetchdf()
    keys = [
        "anchor_age",
        "female",
        "race_black",
        "race_hispanic",
        "race_asian",
        "race_other",
        "admission_elective",
        "admission_emergency",
        "charlson_conditions",
        "dx_digestive",
    ]
    y = df["dead28"].to_numpy(dtype=float)
    a = df["vent"].to_numpy(dtype=float)
    n = len(df)
    x = np.ones((n, 1 + len(keys) + 1))
    x[:, 1] = a
    for j, k in enumerate(keys):
        x[:, j + 2] = df[k].to_numpy(dtype=float)
    beta = fit_logit(x, y)
    # g-comp: everyone vent=1 vs 0
    x1 = x.copy()
    x1[:, 1] = 1.0
    x0 = x.copy()
    x0[:, 1] = 0.0
    r1 = float(np.mean(sigmoid(x1 @ beta)))
    r0 = float(np.mean(sigmoid(x0 @ beta)))
    crude1 = float(y[a == 1].mean()) if (a == 1).any() else None
    crude0 = float(y[a == 0].mean()) if (a == 0).any() else None
    return {
        "exposure": "invasive_ventilation_plus_minus_6h_of_intime",
        "cohort": "all_first_icu_stays",
        "n": int(n),
        "n_vent": int(a.sum()),
        "vent_pct": round(100 * float(a.mean()), 2),
        "crude": {"vent": crude1, "no_vent": crude0, "rd": None if crude1 is None or crude0 is None else crude1 - crude0, "rr": None if not crude0 else crude1 / crude0},
        "gcomputation": {
            "vent": r1,
            "no_vent": r0,
            "rd": r1 - r0,
            "rr": None if r0 == 0 else r1 / r0,
        },
        "expected_direction": "RR > 1 (invasive ventilation associated with higher 28-day death)",
        "note": "Not CCW. Baseline binary exposure on the unrestricted first-stay cohort because the Route A analysis set is 100% ventilated. Missing dod treated as alive at day 28.",
    }


def main() -> None:
    extract_meta = json.loads((ROOT / "notes/cce-audit-A06-A13-extract.json").read_text())
    decision = choose_keys(extract_meta["sofa_component_coverage"], float(extract_meta["lactate_observed_pct"]))
    set_keys(decision["keys"])
    stays = overlay_times(overlay_a06(assemble()))
    x, en, death, dead28, subj = matrices(stays, list(IPCW_KEYS))
    print("loaded", len(stays), "keys", len(IPCW_KEYS), flush=True)
    a07 = run_a07(stays, x, en, death, dead28)
    run_a09(x, en, death)
    run_a12(float(a07["primary_28d_allcause_msm"]["rr"]), float(a07["primary_28d_allcause_msm"]["rd"]), float(a07["primary_28d_allcause_msm"]["delayed_96"]))
    vaso = run_a14_vaso(stays, x, death, dead28)
    print("A14 vaso", json.dumps(vaso, indent=2), flush=True)
    vent = run_a14_vent()
    print("A14 vent", json.dumps(vent, indent=2), flush=True)
    OUT_A14.write_text(
        json.dumps(
            {
                "generated_at": now(),
                "status": "pipeline_qc_not_author_final",
                "vasopressor": vaso,
                "baseline_invasive_ventilation": vent,
                "note": "Not author-final. Not copied into the 24h manuscript.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
