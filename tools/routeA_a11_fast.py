#!/usr/bin/env python3
"""Vectorized Route A MSM + 2,000 BCa. Matches A06 point estimate, not author-final."""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from routeA_a06_a13_a11 import (  # noqa: E402
    BOOT_RD,
    BOOT_RR,
    OUT_A06,
    OUT_A11,
    SEED,
    SOFA,
    bca_interval,
    choose_keys,
    now,
    overlay_a06,
    set_keys,
)
from routeA_ipcw_qc import KEYS as IPCW_KEYS  # noqa: E402
from routeA_ipcw_qc import assemble  # noqa: E402
from routeA_ipcw_qc import fit_logit, sigmoid  # noqa: E402
from routeA_msm_b01 import KNOTS, RIDGE, fit_logit_weighted, rcs_terms  # noqa: E402

DAYS = 28
PERIODS = tuple(range(0, 96, 6))


def matrices(stays: list[dict], keys: list[str]):
    n = len(stays)
    p = len(keys)
    x = np.ones((n, 1 + p))
    for j, key in enumerate(keys):
        x[:, j + 1] = [0.0 if s.get(key) is None else float(s[key]) for s in stays]
    en = np.array([np.nan if s["en_h"] is None else float(s["en_h"]) for s in stays])
    death = np.array([np.nan if s["death_h"] is None else float(s["death_h"]) for s in stays])
    dead28 = np.array([float(s["dead28"]) for s in stays])
    subj = np.array([int(s["subject_id"]) for s in stays], dtype=int)
    return x, en, death, dead28, subj


def censor_times(en: np.ndarray, death: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = len(en)
    en_ok = np.isfinite(en)
    death_ok = np.isfinite(death)
    c_early = np.full(n, np.nan)
    keep_early = (en_ok & (en <= 48.0)) | (death_ok & (death <= 48.0))
    c_early[~keep_early] = 48.0
    c_delay = np.full(n, np.nan)
    delay_cens = en_ok & (en < 96.0) & (~death_ok | (death > en))
    c_delay[delay_cens] = en[delay_cens]
    return c_early, c_delay


def ipc_weights(x: np.ndarray, death: np.ndarray, censor: np.ndarray) -> np.ndarray:
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
        b_den = fit_logit(xr, y)
        p_den = np.clip(sigmoid(xr @ b_den), 1e-4, 1 - 1e-4)
        pw = p_num / p_den
        idx = np.flatnonzero(at)
        keep = y == 1.0
        w_cum[idx[keep]] *= pw[keep]
    w_cum[~seen] = np.nan
    return w_cum


def outcome_stack(
    x: np.ndarray,
    death: np.ndarray,
    dead: np.ndarray,
    uncens: np.ndarray,
    w: np.ndarray,
    early: float,
    days: int = DAYS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    death_ok = np.isfinite(death)
    death_day = np.full(len(death), np.nan)
    death_day[death_ok] = np.floor(death[death_ok] / 24.0)
    last = np.full(len(death), days - 1, dtype=int)
    cut = np.isfinite(death_day) & (death_day < (days - 1))
    last[cut] = death_day[cut].astype(int)
    use = uncens & np.isfinite(w) & (last >= 0)
    idx = np.flatnonzero(use)
    counts = last[idx] + 1
    total = int(counts.sum()) if len(counts) else 0
    p = x.shape[1]
    xo = np.empty((total, p + 5))
    yo = np.zeros(total)
    wo = np.empty(total)
    z1_all, z2_all = rcs_terms(np.arange(days, dtype=float))
    row = 0
    for i in idx:
        L = int(last[i]) + 1
        sl = slice(row, row + L)
        xo[sl, :p] = x[i]
        xo[sl, p] = early
        xo[sl, p + 1] = z1_all[:L]
        xo[sl, p + 2] = z2_all[:L]
        xo[sl, p + 3] = early * z1_all[:L]
        xo[sl, p + 4] = early * z2_all[:L]
        wo[sl] = w[i]
        dd = death_day[i]
        if np.isfinite(dd) and int(dd) < L and dead[i] == 1.0:
            yo[row + int(dd)] = 1.0
        row += L
    return xo, yo, wo


def g_compute(x: np.ndarray, beta: np.ndarray, days: int = DAYS) -> dict:
    n = x.shape[0]
    t = np.arange(days, dtype=float)
    z1, z2 = rcs_terms(t)
    risks = {}
    for name, early in (("early_48", 1.0), ("delayed_96", 0.0)):
        f28 = np.zeros(n)
        surv = np.ones(n)
        extra_base = np.empty((n, 5))
        for d in range(days):
            extra_base[:, 0] = early
            extra_base[:, 1] = z1[d]
            extra_base[:, 2] = z2[d]
            extra_base[:, 3] = early * z1[d]
            extra_base[:, 4] = early * z2[d]
            h = np.clip(sigmoid(np.hstack((x, extra_base)) @ beta), 1e-12, 1 - 1e-12)
            f28 += surv * h
            surv *= 1.0 - h
        risks[name] = float(np.mean(f28))
        risks[f"{name}_p50"] = float(np.median(f28))
        risks[f"{name}_p05"] = float(np.percentile(f28, 5))
        risks[f"{name}_p95"] = float(np.percentile(f28, 95))
    r0, r1 = risks["early_48"], risks["delayed_96"]
    risks["rd"] = r0 - r1
    risks["rr"] = None if r1 == 0 else r0 / r1
    return risks


def estimate_msm(
    x: np.ndarray,
    en: np.ndarray,
    death: np.ndarray,
    dead: np.ndarray,
    days: int = DAYS,
) -> dict:
    c_early, c_delay = censor_times(en, death)
    w_e = ipc_weights(x, death, c_early)
    w_d = ipc_weights(x, death, c_delay)
    u_e = ~np.isfinite(c_early)
    u_d = ~np.isfinite(c_delay)
    xe, ye, we = outcome_stack(x, death, dead, u_e, w_e, 1.0, days=days)
    xd, yd, wd = outcome_stack(x, death, dead, u_d, w_d, 0.0, days=days)
    xo = np.vstack((xe, xd))
    yo = np.concatenate((ye, yd))
    wo = np.concatenate((we, wd))
    beta = fit_logit_weighted(xo, yo, wo)
    msm = g_compute(x, beta, days=days)
    msm["events"] = int(yo.sum())
    msm["days"] = int(len(yo))
    msm["n_uncensored_early"] = int(u_e.sum())
    msm["n_uncensored_delayed"] = int(u_d.sum())
    msm["horizon_days"] = days
    msm["w_early"] = w_e
    msm["w_delayed"] = w_d
    msm["c_early"] = c_early
    msm["c_delay"] = c_delay
    return msm


def subject_index(subj: np.ndarray) -> dict[int, np.ndarray]:
    by: dict[int, list[int]] = {}
    for i, sid in enumerate(subj.tolist()):
        by.setdefault(int(sid), []).append(i)
    return {k: np.array(v, dtype=int) for k, v in by.items()}


def resample(x, en, death, dead28, subj_index, subjects, rng):
    draw = rng.choice(subjects, size=len(subjects), replace=True)
    parts = [subj_index[int(s)] for s in draw]
    idx = np.concatenate(parts) if parts else np.array([], dtype=int)
    return x[idx], en[idx], death[idx], dead28[idx]


def main() -> None:
    n_boot = 2000 if len(sys.argv) < 2 else int(sys.argv[1])
    n_jack = 200 if len(sys.argv) < 3 else int(sys.argv[2])
    extract_meta = json.loads((ROOT / "notes/cce-audit-A06-A13-extract.json").read_text())
    decision = choose_keys(extract_meta["sofa_component_coverage"], float(extract_meta["lactate_observed_pct"]))
    set_keys(decision["keys"])
    stays = overlay_a06(assemble())
    x, en, death, dead28, subj = matrices(stays, list(IPCW_KEYS))
    t0 = datetime.now()
    point = estimate_msm(x, en, death, dead28)
    print("point", json.dumps({k: point[k] for k in ("early_48", "delayed_96", "rd", "rr", "events", "days")}), flush=True)
    print("elapsed_point", (datetime.now() - t0).total_seconds(), flush=True)

    rng = np.random.default_rng(SEED)
    by = subject_index(subj)
    subjects = np.array(list(by), dtype=int)
    rds, rrs = [], []
    failed = 0
    fail_reasons: dict[str, int] = {}
    t_boot = datetime.now()
    for i in range(n_boot):
        xs, ens, ds, d28 = resample(x, en, death, dead28, by, subjects, rng)
        try:
            est = estimate_msm(xs, ens, ds, d28)
            if est["rd"] is None or est["rr"] is None or not math.isfinite(est["rd"]) or not math.isfinite(est["rr"]):
                failed += 1
                fail_reasons["nonfinite"] = fail_reasons.get("nonfinite", 0) + 1
                continue
            rds.append(float(est["rd"]))
            rrs.append(float(est["rr"]))
        except Exception as exc:
            failed += 1
            name = type(exc).__name__
            fail_reasons[name] = fail_reasons.get(name, 0) + 1
        if (i + 1) % 25 == 0 or i + 1 == n_boot:
            np.save(BOOT_RD, np.array(rds, dtype=float))
            np.save(BOOT_RR, np.array(rrs, dtype=float))
            elapsed = (datetime.now() - t_boot).total_seconds()
            print(f"boot {i + 1}/{n_boot} ok={len(rds)} failed={failed} elapsed_s={elapsed:.1f}", flush=True)

    # grouped jackknife
    rng_j = np.random.default_rng(SEED + 1)
    order = subjects.copy()
    rng_j.shuffle(order)
    groups = np.array_split(order, n_jack)
    jack_rd, jack_rr = [], []
    jack_fail = 0
    for i, g in enumerate(groups):
        hold = set(int(v) for v in g)
        keep = np.array([j for sid, idx in by.items() if sid not in hold for j in idx], dtype=int)
        try:
            est = estimate_msm(x[keep], en[keep], death[keep], dead28[keep])
            if est["rd"] is None or est["rr"] is None:
                jack_fail += 1
                continue
            jack_rd.append(float(est["rd"]))
            jack_rr.append(float(est["rr"]))
        except Exception:
            jack_fail += 1
        if (i + 1) % 20 == 0:
            print(f"jackknife {i + 1}/{n_jack} ok={len(jack_rd)}", flush=True)

    bca_rd = bca_interval(float(point["rd"]), rds, jack_rd)
    bca_rr = bca_interval(float(point["rr"]), rrs, jack_rr)
    payload = {
        "generated_at": now(),
        "status": "pipeline_qc_not_author_final",
        "estimator": "vectorized_msm_gcomputation_same_ridge_rcs_ipcw_as_A06",
        "n_stays": int(x.shape[0]),
        "keys": list(IPCW_KEYS),
        "point": {
            "early_48": point["early_48"],
            "delayed_96": point["delayed_96"],
            "rd": point["rd"],
            "rr": point["rr"],
            "events": point["events"],
            "days": point["days"],
        },
        "bootstrap": {
            "n_requested": n_boot,
            "n_ok": len(rds),
            "n_failed": failed,
            "fail_reasons": fail_reasons,
            "percentile_rd": None if len(rds) < 20 else [float(np.percentile(rds, 2.5)), float(np.percentile(rds, 97.5))],
            "percentile_rr": None if len(rrs) < 20 else [float(np.percentile(rrs, 2.5)), float(np.percentile(rrs, 97.5))],
        },
        "jackknife": {"n_groups": n_jack, "n_ok": len(jack_rd), "n_failed": jack_fail},
        "bca_rd": bca_rd,
        "bca_rr": bca_rr,
        "acceleration_note": "Acceleration a uses a 200-group delete-group jackknife of subjects, not a full n=11,259 leave-one-out jackknife.",
        "note": "Not author-final. Do not copy into the 24h manuscript without author confirmation.",
    }
    OUT_A11.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({k: payload[k] for k in ("point", "bootstrap", "bca_rd", "bca_rr", "jackknife")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
