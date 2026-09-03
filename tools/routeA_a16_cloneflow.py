#!/usr/bin/env python3
"""Clone-flow Table 3, A16 sensitivity forest, A15 eICU probe.

Pipeline QC. Not author-final. Does not write manuscript placeholders.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from routeA_a06_a13_a11 import choose_keys, now, overlay_a06, set_keys  # noqa: E402
from routeA_a07_a09_a12_a14 import HORIZON_H, kish_ess, overlay_times  # noqa: E402
from routeA_a11_fast import (  # noqa: E402
    DAYS,
    g_compute,
    ipc_weights,
    matrices,
    outcome_stack,
    resample,
    subject_index,
)
from routeA_ipcw_qc import KEYS as IPCW_KEYS  # noqa: E402
from routeA_ipcw_qc import assemble, fit_logit  # noqa: E402
from routeA_msm_b01 import fit_logit_weighted  # noqa: E402

OUT_FLOW = ROOT / "notes/cce-audit-routeA-clone-flow.json"
CSV_FLOW = ROOT / "notes/cce-audit-routeA-clone-flow.csv"
OUT_A16 = ROOT / "notes/cce-audit-routeA-A16-sensitivity.json"
CSV_A16 = ROOT / "notes/cce-audit-routeA-A16-sensitivity.csv"
PNG_A16 = ROOT / "notes/cce-audit-routeA-A16-forest.png"
OUT_A15 = ROOT / "notes/cce-audit-routeA-A15-eicu.json"
SEED = 20260902


def censor_times_grace(en: np.ndarray, death: np.ndarray, early_h: float, delay_h: float):
    n = len(en)
    en_ok = np.isfinite(en)
    death_ok = np.isfinite(death)
    c_early = np.full(n, np.nan)
    keep_early = (en_ok & (en <= early_h)) | (death_ok & (death <= early_h))
    c_early[~keep_early] = float(early_h)
    c_delay = np.full(n, np.nan)
    delay_cens = en_ok & (en < delay_h) & (~death_ok | (death > en))
    c_delay[delay_cens] = en[delay_cens]
    return c_early, c_delay


def ipc_weights_h(x, death, censor, horizon_h: float):
    # temporarily use global PERIODS by calling ipc_weights after monkeypatch is messy;
    # copy loop with local periods.
    from routeA_ipcw_qc import sigmoid

    n = x.shape[0]
    w_cum = np.ones(n)
    seen = np.zeros(n, dtype=bool)
    death_ok = np.isfinite(death)
    periods = tuple(range(0, int(horizon_h), 6))
    for start in periods:
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
        p_num = 1.0 / (1.0 + np.exp(-np.clip(np.full(xr.shape[0], b_num[0]), -60, 60)))
        b_den = fit_logit(xr, y)
        p_den = np.clip(sigmoid(xr @ b_den), 1e-4, 1 - 1e-4)
        pw = p_num / p_den
        idx = np.flatnonzero(at)
        keep = y == 1.0
        w_cum[idx[keep]] *= pw[keep]
    w_cum[~seen] = np.nan
    return w_cum


def trim_weights(w: np.ndarray, use: np.ndarray, q: float | None) -> np.ndarray:
    if q is None:
        return w
    ww = w.copy()
    finite = use & np.isfinite(ww)
    if not np.any(finite):
        return ww
    cap = float(np.quantile(ww[finite], q))
    ww[np.isfinite(ww)] = np.minimum(ww[np.isfinite(ww)], cap)
    return ww


def estimate_variant(
    x,
    en,
    death,
    dead,
    early_h: float,
    delay_h: float,
    days: int = 28,
    trim_q: float | None = None,
) -> dict:
    c_e, c_d = censor_times_grace(en, death, early_h, delay_h)
    w_e = ipc_weights_h(x, death, c_e, delay_h)
    w_d = ipc_weights_h(x, death, c_d, delay_h)
    u_e = ~np.isfinite(c_e)
    u_d = ~np.isfinite(c_d)
    w_e = trim_weights(w_e, u_e, trim_q)
    w_d = trim_weights(w_d, u_d, trim_q)
    xe, ye, we = outcome_stack(x, death, dead, u_e, w_e, 1.0, days=days)
    xd, yd, wd = outcome_stack(x, death, dead, u_d, w_d, 0.0, days=days)
    beta = fit_logit_weighted(np.vstack((xe, xd)), np.concatenate((ye, yd)), np.concatenate((we, wd)))
    msm = g_compute(x, beta, days=days)
    # Hajek
    def hajek(use, w):
        ok = use & np.isfinite(w)
        den = float(w[ok].sum())
        if den == 0:
            return None, None, None
        risk = float((w[ok] * dead[ok]).sum() / den)
        return risk, int(ok.sum()), den

    h0, n0, s0 = hajek(u_e, w_e)
    h1, n1, s1 = hajek(u_d, w_d)
    n = x.shape[0]
    # unstabilized HT approximated with same stabilized clip path is not HT;
    # report Hajek + MSM only except when trim is None we still have Hajek.
    msm["hajek_early"] = h0
    msm["hajek_delayed"] = h1
    msm["hajek_rd"] = None if h0 is None or h1 is None else h0 - h1
    msm["hajek_rr"] = None if h0 is None or h1 is None or h1 == 0 else h0 / h1
    msm["n_uncensored_early"] = n0
    msm["n_uncensored_delayed"] = n1
    msm["weight_sum_early"] = s0
    msm["weight_sum_delayed"] = s1
    msm["ess_early"] = kish_ess(w_e[u_e])
    msm["ess_delayed"] = kish_ess(w_d[u_d])
    msm["c_early"] = c_e
    msm["c_delay"] = c_d
    msm["w_early"] = w_e
    msm["w_delayed"] = w_d
    msm["n"] = n
    return msm


def boot_rd_rr(x, en, death, dead, subj, early_h, delay_h, days, trim_q, n_boot, seed=SEED):
    rng = np.random.default_rng(seed)
    by = subject_index(subj)
    subjects = np.array(list(by), dtype=int)
    rds, rrs = [], []
    failed = 0
    for _ in range(n_boot):
        xs, ens, ds, d28 = resample(x, en, death, dead, by, subjects, rng)
        try:
            est = estimate_variant(xs, ens, ds, d28, early_h, delay_h, days=days, trim_q=trim_q)
            if est["rd"] is None or est["rr"] is None or not math.isfinite(est["rd"]):
                failed += 1
                continue
            rds.append(float(est["rd"]))
            rrs.append(float(est["rr"]))
        except Exception:
            failed += 1
    def ci(xs):
        if len(xs) < 20:
            return None
        return [float(np.percentile(xs, 2.5)), float(np.percentile(xs, 97.5))]
    return {"n_ok": len(rds), "n_failed": failed, "rd_ci": ci(rds), "rr_ci": ci(rrs)}


def pct(num, den):
    return None if not den else round(100.0 * num / den, 2)


def arm_flow(stays, en, death, disch, dead28, censor, w, grace_h, role: str) -> dict:
    n = len(stays)
    en_ok = np.isfinite(en)
    death_ok = np.isfinite(death)
    disch_ok = np.isfinite(disch)
    uncens = ~np.isfinite(censor)
    cens = np.isfinite(censor)
    initiated = en_ok & (en <= grace_h)
    died_grace = death_ok & (death <= grace_h)
    died_before_en = died_grace & (~en_ok | (death <= en))
    disch_grace = disch_ok & (disch <= grace_h) & ~died_grace
    cens_final = cens & (censor > grace_h - 6.0) & (censor <= grace_h)
    dead_u = uncens & (dead28 == 1.0)
    ww = w
    okw = uncens & np.isfinite(ww)
    return {
        "role": role,
        "grace_h": grace_h,
        "clones_at_t0": n,
        "artificially_censored": int(cens.sum()),
        "censored_in_final_grace_window": int(cens_final.sum()),
        "uncensored": int(uncens.sum()),
        "initiated_en_within_deadline": int(initiated.sum()),
        "initiated_en_among_uncensored": int((uncens & initiated).sum()),
        "died_during_grace": int(died_grace.sum()),
        "died_during_grace_before_en": int(died_before_en.sum()),
        "died_during_grace_among_uncensored": int((uncens & died_before_en).sum()),
        "discharged_during_grace": int(disch_grace.sum()),
        "discharged_during_grace_among_uncensored": int((uncens & disch_grace).sum()),
        "never_en_and_survived_grace": int((uncens & ~initiated & ~died_grace).sum()),
        "deaths_day28_among_uncensored": int(dead_u.sum()),
        "deaths_day28_among_grace_period_deaths": int((dead_u & died_grace).sum()),
        "deaths_day28_among_remainder": int((dead_u & ~died_grace).sum()),
        "sum_stabilized_weights": float(ww[okw].sum()) if okw.any() else None,
        "kish_ess": kish_ess(ww[okw]),
        "pct_initiated_among_uncensored": pct(int((uncens & initiated).sum()), int(uncens.sum())),
        "pct_grace_death_among_uncensored": pct(int((uncens & died_before_en).sum()), int(uncens.sum())),
        "note": "Current clones() treats death, not discharge, as preventing artificial censoring. Discharge-during-grace is counted separately.",
    }


def run_clone_flow(stays, en, death, dead28, msm48) -> dict:
    disch = np.array([np.nan if s.get("discharge_h") is None else float(s["discharge_h"]) for s in stays])
    early = arm_flow(stays, en, death, disch, dead28, msm48["c_early"], msm48["w_early"], 48.0, "early_48")
    delayed = arm_flow(stays, en, death, disch, dead28, msm48["c_delay"], msm48["w_delayed"], 96.0, "delayed_96")
    # 24/48 secondary
    c24, c48 = censor_times_grace(en, death, 24.0, 48.0)
    w24 = ipc_weights_h(np.ones((len(stays), 1)), death, c24, 48.0)  # weights only for flow ESS optional
    w48 = ipc_weights_h(np.ones((len(stays), 1)), death, c48, 48.0)
    # Use primary A06 covariates weights for 24/48 flow via a real estimate
    # cheap intercept-only ESS is not the analysis weights; compute real 24/48 weights with x later in A16.
    payload = {
        "generated_at": now(),
        "status": "pipeline_qc_not_author_final",
        "n_stays": len(stays),
        "primary_48_96": {"early_en": early, "no_en_before_96": delayed},
        "note": "Clone counts are not mutually exclusive across arms. Grace-period deaths are compatible with both strategies.",
    }
    with CSV_FLOW.open("w", newline="") as handle:
        rows = [early, delayed]
        cols = [k for k in early.keys() if k != "note"]
        w = csv.DictWriter(handle, fieldnames=["contrast"] + cols)
        w.writeheader()
        for r, label in ((early, "48h_vs_96h_early"), (delayed, "48h_vs_96h_delayed")):
            out = {"contrast": label}
            out.update({k: r[k] for k in cols})
            w.writerow(out)
    OUT_FLOW.write_text(json.dumps(payload, indent=2) + "\n")
    print("clone-flow early uncensored", early["uncensored"], "initiated", early["initiated_en_among_uncensored"], "grace death", early["died_during_grace_among_uncensored"], flush=True)
    print("clone-flow delayed uncensored", delayed["uncensored"], "never EN survived", delayed["never_en_and_survived_grace"], flush=True)
    return payload


def run_a16(stays, x, en, death, dead28, subj, a11: dict, a07: dict) -> dict:
    care_cols = [i for i, k in enumerate(IPCW_KEYS) if k.startswith("unit_")]
    # x has intercept at 0, keys at 1:
    drop = [1 + i for i in care_cols]
    keep = [j for j in range(x.shape[1]) if j not in drop]
    x_nocare = x[:, keep]
    scenarios = [
        {"id": "msm_48_96_primary", "early": 48, "delay": 96, "trim": None, "days": 28, "x": "full", "boot": 0, "use_a11": True},
        {"id": "hajek_48_96", "early": 48, "delay": 96, "trim": None, "days": 28, "x": "full", "boot": 0, "estimator": "hajek"},
        {"id": "msm_24_48", "early": 24, "delay": 48, "trim": None, "days": 28, "x": "full", "boot": 100},
        {"id": "msm_24_72", "early": 24, "delay": 72, "trim": None, "days": 28, "x": "full", "boot": 100},
        {"id": "msm_24_96", "early": 24, "delay": 96, "trim": None, "days": 28, "x": "full", "boot": 100},
        {"id": "msm_36_72", "early": 36, "delay": 72, "trim": None, "days": 28, "x": "full", "boot": 100},
        {"id": "msm_36_96", "early": 36, "delay": 96, "trim": None, "days": 28, "x": "full", "boot": 100},
        {"id": "msm_48_72", "early": 48, "delay": 72, "trim": None, "days": 28, "x": "full", "boot": 100},
        {"id": "msm_48_96_trim_p95", "early": 48, "delay": 96, "trim": 0.95, "days": 28, "x": "full", "boot": 100},
        {"id": "msm_48_96_trim_p99", "early": 48, "delay": 96, "trim": 0.99, "days": 28, "x": "full", "boot": 100},
        {"id": "msm_48_96_trim_p995", "early": 48, "delay": 96, "trim": 0.995, "days": 28, "x": "full", "boot": 100},
        {"id": "msm_48_96_no_careunit", "early": 48, "delay": 96, "trim": None, "days": 28, "x": "nocare", "boot": 100},
        {"id": "msm_90d_48_96", "early": 48, "delay": 96, "trim": None, "days": 90, "x": "full", "boot": 0, "use_a07_90": True},
        {"id": "aj_28d_inhospital", "early": 48, "delay": 96, "trim": None, "days": 28, "x": "full", "boot": 0, "use_a07_aj": True},
    ]
    dead90 = np.array([float(s["dead90"]) for s in stays])
    rows = []
    for sc in scenarios:
        print("A16", sc["id"], flush=True)
        xx = x if sc["x"] == "full" else x_nocare
        if sc.get("use_a11"):
            p = a11["point"]
            row = {
                "id": sc["id"],
                "estimator": "msm",
                "early_h": 48,
                "delay_h": 96,
                "early_risk": p["early_48"],
                "delayed_risk": p["delayed_96"],
                "rd": p["rd"],
                "rr": p["rr"],
                "rd_ci_lo": a11["bca_rd"]["interval"][0],
                "rd_ci_hi": a11["bca_rd"]["interval"][1],
                "rr_ci_lo": a11["bca_rr"]["interval"][0],
                "rr_ci_hi": a11["bca_rr"]["interval"][1],
                "n_boot_ok": a11["bootstrap"]["n_ok"],
                "n_boot_failed": a11["bootstrap"]["n_failed"],
                "ci": "bca_2000",
            }
            rows.append(row)
            continue
        if sc.get("use_a07_90"):
            p = a07["sensitivity_90d_allcause_msm"]
            rows.append(
                {
                    "id": sc["id"],
                    "estimator": "msm",
                    "early_h": 48,
                    "delay_h": 96,
                    "early_risk": p["early_48"],
                    "delayed_risk": p["delayed_96"],
                    "rd": p["rd"],
                    "rr": p["rr"],
                    "rd_ci_lo": None,
                    "rd_ci_hi": None,
                    "rr_ci_lo": None,
                    "rr_ci_hi": None,
                    "n_boot_ok": 0,
                    "n_boot_failed": 0,
                    "ci": "point_only",
                }
            )
            continue
        if sc.get("use_a07_aj"):
            p = a07["secondary_aj_28d_inhospital"]
            rows.append(
                {
                    "id": sc["id"],
                    "estimator": "aj",
                    "early_h": 48,
                    "delay_h": 96,
                    "early_risk": p["early_48"],
                    "delayed_risk": p["delayed_96"],
                    "rd": p["rd"],
                    "rr": p["rr"],
                    "rd_ci_lo": None,
                    "rd_ci_hi": None,
                    "rr_ci_lo": None,
                    "rr_ci_hi": None,
                    "n_boot_ok": 0,
                    "n_boot_failed": 0,
                    "ci": "point_only",
                }
            )
            continue
        dead = dead28 if sc["days"] == 28 else dead90
        est = estimate_variant(xx, en, death, dead, sc["early"], sc["delay"], days=sc["days"], trim_q=sc["trim"])
        estimator = sc.get("estimator", "msm")
        if estimator == "hajek":
            rd, rr = est["hajek_rd"], est["hajek_rr"]
            e0, e1 = est["hajek_early"], est["hajek_delayed"]
        else:
            rd, rr = est["rd"], est["rr"]
            e0, e1 = est["early_48"], est["delayed_96"]
        boot = {"rd_ci": None, "rr_ci": None, "n_ok": 0, "n_failed": 0}
        if sc["boot"]:
            boot = boot_rd_rr(xx, en, death, dead, subj, sc["early"], sc["delay"], sc["days"], sc["trim"], sc["boot"])
        rows.append(
            {
                "id": sc["id"],
                "estimator": estimator,
                "early_h": sc["early"],
                "delay_h": sc["delay"],
                "trim": sc["trim"],
                "early_risk": e0,
                "delayed_risk": e1,
                "rd": rd,
                "rr": rr,
                "rd_ci_lo": None if not boot["rd_ci"] else boot["rd_ci"][0],
                "rd_ci_hi": None if not boot["rd_ci"] else boot["rd_ci"][1],
                "rr_ci_lo": None if not boot["rr_ci"] else boot["rr_ci"][0],
                "rr_ci_hi": None if not boot["rr_ci"] else boot["rr_ci"][1],
                "n_boot_ok": boot["n_ok"],
                "n_boot_failed": boot["n_failed"],
                "ci": None if not sc["boot"] else f"percentile_{sc['boot']}",
                "n_uncensored_early": est["n_uncensored_early"],
                "n_uncensored_delayed": est["n_uncensored_delayed"],
            }
        )
        print(" ", sc["id"], "RD", rd, "RR", rr, flush=True)
    with CSV_A16.open("w", newline="") as handle:
        cols = list(rows[0].keys())
        for r in rows:
            for k in r:
                if k not in cols:
                    cols.append(k)
        w = csv.DictWriter(handle, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    payload = {
        "generated_at": now(),
        "status": "pipeline_qc_not_author_final",
        "n_stays": len(stays),
        "rows": rows,
        "not_run": [
            "MICE_20 vs missing_indicators (deferred; A06 uses missing indicators)",
            "unrestricted_first_stays including CVICU/CCU (no 19,919 stay parquet rebuilt this run)",
        ],
        "note": "Primary row uses A11 2,000 BCa. Other MSM rows use 100-replicate percentile intervals. Not author-final.",
    }
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plot_rows = [r for r in rows if r["rd"] is not None]
        fig, ax = plt.subplots(figsize=(7.2, 6.4))
        y = np.arange(len(plot_rows))
        rd = np.array([r["rd"] for r in plot_rows], dtype=float)
        lo = np.array([r["rd"] if r["rd_ci_lo"] is None else r["rd_ci_lo"] for r in plot_rows], dtype=float)
        hi = np.array([r["rd"] if r["rd_ci_hi"] is None else r["rd_ci_hi"] for r in plot_rows], dtype=float)
        ax.errorbar(rd, y, xerr=[rd - lo, hi - rd], fmt="o", capsize=2)
        ax.axvline(0.0, color="black", linewidth=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels([r["id"] for r in plot_rows], fontsize=8)
        ax.set_xlabel("Risk difference (early minus delayed)")
        ax.invert_yaxis()
        fig.tight_layout()
        fig.savefig(PNG_A16, dpi=150)
        plt.close(fig)
        payload["forest_png"] = str(PNG_A16.relative_to(ROOT))
    except Exception as exc:
        payload["forest_error"] = str(exc)
    OUT_A16.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def run_a15() -> dict:
    candidates = [
        Path("/Volumes/B/litflow-cache/physionet/physionet.org/files/eicu-crd"),
        Path("/Volumes/B/litflow-cache/physionet/physionet.org/files/eicu"),
        Path("/Volumes/B/litflow-cache/eicu-crd"),
        ROOT / "workspace/metered-results/eicu",
    ]
    files_root = Path("/Volumes/B/litflow-cache/physionet/physionet.org/files")
    hits = [str(p) for p in candidates if p.exists()]
    if files_root.exists():
        hits.extend(str(p) for p in files_root.iterdir() if "eicu" in p.name.lower())
    payload = {
        "generated_at": now(),
        "status": "not_run_environment" if not hits else "files_found_not_analysed",
        "hits": hits,
        "decision": "B15(a) wording downgrade remains the writing-path default until eICU-CRD is available locally for a full Route A rerun or odds-of-participation transport.",
        "note": "Did not fit participation weights or eICU MSM this run.",
    }
    OUT_A15.write_text(json.dumps(payload, indent=2) + "\n")
    print("A15 hits", len(hits), flush=True)
    return payload


def main() -> None:
    extract_meta = json.loads((ROOT / "notes/cce-audit-A06-A13-extract.json").read_text())
    a11 = json.loads((ROOT / "notes/cce-audit-routeA-A11-bca.json").read_text())
    a07 = json.loads((ROOT / "notes/cce-audit-routeA-A07.json").read_text())
    decision = choose_keys(extract_meta["sofa_component_coverage"], float(extract_meta["lactate_observed_pct"]))
    set_keys(decision["keys"])
    stays = overlay_times(overlay_a06(assemble()))
    x, en, death, dead28, subj = matrices(stays, list(IPCW_KEYS))
    print("loaded", len(stays), flush=True)
    msm48 = estimate_variant(x, en, death, dead28, 48.0, 96.0, days=28, trim_q=None)
    run_clone_flow(stays, en, death, dead28, msm48)
    # 24/48 clone flow add-on
    msm24 = estimate_variant(x, en, death, dead28, 24.0, 48.0, days=28, trim_q=None)
    flow = json.loads(OUT_FLOW.read_text())
    disch = np.array([np.nan if s.get("discharge_h") is None else float(s["discharge_h"]) for s in stays])
    flow["secondary_24_48"] = {
        "early_en": arm_flow(stays, en, death, disch, dead28, msm24["c_early"], msm24["w_early"], 24.0, "early_24"),
        "no_en_before_48": arm_flow(stays, en, death, disch, dead28, msm24["c_delay"], msm24["w_delayed"], 48.0, "delayed_48"),
    }
    OUT_FLOW.write_text(json.dumps(flow, indent=2) + "\n")
    with CSV_FLOW.open("a", newline="") as handle:
        early = flow["secondary_24_48"]["early_en"]
        delayed = flow["secondary_24_48"]["no_en_before_48"]
        cols = [k for k in early.keys() if k != "note"]
        w = csv.DictWriter(handle, fieldnames=["contrast"] + cols)
        for r, label in ((early, "24h_vs_48h_early"), (delayed, "24h_vs_48h_delayed")):
            out = {"contrast": label}
            out.update({k: r[k] for k in cols})
            w.writerow(out)
    run_a16(stays, x, en, death, dead28, subj, a11, a07)
    run_a15()


if __name__ == "__main__":
    main()
