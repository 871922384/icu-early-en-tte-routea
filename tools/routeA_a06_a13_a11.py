#!/usr/bin/env python3
"""A06 covariate rebuild, A13 negative-control MSM, A11 2,000-replicate BCa.

Pipeline QC. Not author-final. Does not write manuscript placeholders.
"""

from __future__ import annotations

import json
import math
import sys
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from routeA_ipcw_qc import KEYS as IPCW_KEYS  # noqa: E402
from routeA_ipcw_qc import assemble, clones, design  # noqa: E402
from routeA_msm_b01 import (  # noqa: E402
    estimate_bundle,
    g_compute,
    smd_table,
)

COHORT = ROOT / "workspace/metered-results/cohort"
SOFA = COHORT / "routeA_sofa_full.parquet"
NC = COHORT / "routeA_negative_control_events.parquet"
OUT_A06 = ROOT / "notes/cce-audit-routeA-A06-msm.json"
OUT_A13 = ROOT / "notes/cce-audit-routeA-A13-negative-control.json"
OUT_A11 = ROOT / "notes/cce-audit-routeA-A11-bca.json"
SMD_CSV = ROOT / "notes/cce-audit-routeA-A06-smd.csv"
BOOT_RD = ROOT / "notes/cce-audit-routeA-A11-boot-rd.npy"
BOOT_RR = ROOT / "notes/cce-audit-routeA-A11-boot-rr.npy"
SEED = 20260902
COMPONENTS = ("resp", "coag", "liver", "cardio", "cns", "renal")
BASE_KEYS = [
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
]


def now() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def set_keys(keys: list[str]) -> None:
    IPCW_KEYS.clear()
    IPCW_KEYS.extend(keys)


def choose_keys(coverage: dict, lactate_pct: float) -> dict:
    """Drop 99% missing reduced SOFA. Keep components with usable variance."""
    keys = list(BASE_KEYS)
    dropped = ["sofa_t0", "sofa_missing"]
    kept_components = []
    miss_flags = []
    option = "component_scores_with_missing_indicators"
    for name in COMPONENTS:
        obs = coverage[name]["observed_pct"]
        keys.append(f"sofa_{name}")
        kept_components.append(name)
        if 5.0 <= obs <= 95.0:
            keys.append(f"sofa_{name}_miss")
            miss_flags.append(name)
        elif obs < 5.0:
            # near-zero variance if filled with 0; still keep the score (almost all 0)
            # but do not add a missing flag that is ~1.0 (B07 collinearity)
            dropped.append(f"sofa_{name}_miss")
    if 5.0 <= lactate_pct <= 95.0:
        keys.extend(["lactate_filled", "lactate_missing"])
    elif lactate_pct > 95.0:
        keys.append("lactate_filled")
        dropped.append("lactate_missing")
    else:
        keys.append("lactate_missing")
        dropped.append("lactate_filled")
    complete_pct = coverage.get("complete_six_pct", 0)
    if complete_pct < 15:
        option = "b07_option2_plus_usable_components"
    return {
        "keys": keys,
        "dropped": dropped,
        "kept_components": kept_components,
        "miss_flags": miss_flags,
        "option": option,
        "complete_six_pct": complete_pct,
        "lactate_observed_pct": lactate_pct,
        "mice20": False,
        "mice20_reason": "MICE 20 x 2,000 bootstrap is deferred to A16; A06 uses component scores plus missing indicators with variance, never the 99% missing reduced SOFA.",
    }


def overlay_a06(stays: list[dict]) -> list[dict]:
    con = duckdb.connect()
    sofa = con.execute(f"SELECT * FROM read_parquet('{SOFA.as_posix()}')").fetchdf().set_index("stay_id")
    nc = con.execute(f"SELECT * FROM read_parquet('{NC.as_posix()}')").fetchdf().set_index("stay_id")
    out = []
    for s in stays:
        sid = s["stay_id"]
        r = sofa.loc[sid]
        ncr = nc.loc[sid]
        row = dict(s)
        for name in COMPONENTS:
            val = r[f"sofa_{name}"]
            missing = bool(pd.isna(val))
            row[f"sofa_{name}"] = 0.0 if missing else float(val)
            row[f"sofa_{name}_miss"] = 1.0 if missing else 0.0
        lac = r["lactate"]
        row["lactate_filled"] = 0.0 if pd.isna(lac) else float(lac)
        row["lactate_missing"] = 1.0 if pd.isna(lac) else 0.0
        row["sofa_n_observed"] = 0.0 if pd.isna(r["sofa_n_observed"]) else float(r["sofa_n_observed"])
        row["sofa_sum_observed"] = 0.0 if pd.isna(r["sofa_sum_observed"]) else float(r["sofa_sum_observed"])
        for key in ("oral_care_h", "position_h", "chg_h"):
            val = ncr[key] if key in ncr.index else None
            if val is None or (isinstance(val, float) and math.isnan(val)) or pd.isna(val):
                row[key] = None
            else:
                row[key] = float(val)
        out.append(row)
    return out


def clones_exposure(stays: list[dict], exposure_key: str) -> list[dict]:
    patched = []
    for s in stays:
        row = dict(s)
        row["en_h"] = s.get(exposure_key)
        patched.append(row)
    return clones(patched)


def estimate_exposure(stays: list[dict], exposure_key: str = "en_h") -> dict:
    if exposure_key == "en_h":
        return estimate_bundle(stays)
    patched = []
    for s in stays:
        row = dict(s)
        row["en_h"] = s.get(exposure_key)
        patched.append(row)
    return estimate_bundle(patched)


def exposure_prevalence(stays: list[dict], key: str) -> dict:
    n = len(stays)
    hours = [s[key] for s in stays]
    def within(hmax):
        return sum(1 for h in hours if h is not None and 0 <= h <= hmax)
    return {
        "n": n,
        "any": sum(1 for h in hours if h is not None),
        "by_48h": within(48),
        "by_48h_pct": round(100 * within(48) / n, 2) if n else None,
        "by_96h": within(96),
        "by_96h_pct": round(100 * within(96) / n, 2) if n else None,
        "none_by_96h": sum(1 for h in hours if h is None or h > 96),
    }


def compact_msm(bundled: dict) -> dict:
    msm = bundled["msm"]
    haj = bundled["hajek"]
    ht = bundled["ht"]
    return {
        "n_stays_in_bundle": None,
        "outcome_person_days": bundled["days"],
        "msm_events": bundled["events"],
        "hajek": {
            "early_48": haj["early_48"]["risk"],
            "delayed_96": haj["delayed_96"]["risk"],
            "rd": haj["rd"],
            "rr": haj["rr"],
            "n_uncensored_early": haj["early_48"]["n_uncensored"],
            "n_uncensored_delayed": haj["delayed_96"]["n_uncensored"],
        },
        "ht": {
            "early_48": ht["early_48"]["risk"],
            "delayed_96": ht["delayed_96"]["risk"],
            "rd": ht["rd"],
            "rr": ht["rr"],
        },
        "msm_gcomputation": msm,
    }


def bootstrap_msm_checkpoint(
    stays: list[dict],
    n_boot: int,
    exposure_key: str = "en_h",
    seed: int = SEED,
    resume: bool = True,
) -> dict:
    rng = np.random.default_rng(seed)
    by_subj: dict[int, list[dict]] = {}
    for s in stays:
        by_subj.setdefault(s["subject_id"], []).append(s)
    subjects = np.array(list(by_subj), dtype=int)
    rds: list[float] = []
    rrs: list[float] = []
    if resume and BOOT_RD.exists() and BOOT_RR.exists() and exposure_key == "en_h":
        rds = [float(x) for x in np.load(BOOT_RD)]
        rrs = [float(x) for x in np.load(BOOT_RR)]
        print(f"resume bootstrap at {len(rds)}", flush=True)
    failed = 0
    fail_reasons: dict[str, int] = {}
    start_i = len(rds)
    for i in range(n_boot):
        draw = rng.choice(subjects, size=len(subjects), replace=True)
        if i < start_i:
            continue
        sample = []
        for j, sid in enumerate(draw):
            for row in by_subj[int(sid)]:
                sample.append({**row, "stay_id": int(row["stay_id"]) * 100000 + j})
        try:
            est = estimate_exposure(sample, exposure_key)["msm"]
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
        if exposure_key == "en_h" and (len(rds) % 25 == 0 or i + 1 == n_boot):
            np.save(BOOT_RD, np.array(rds, dtype=float))
            np.save(BOOT_RR, np.array(rrs, dtype=float))
            print(f"boot {i + 1}/{n_boot} ok={len(rds)} failed={failed}", flush=True)
    return {
        "n_requested": n_boot,
        "n_ok": len(rds),
        "n_failed": failed,
        "fail_reasons": fail_reasons,
        "rds": rds,
        "rrs": rrs,
        "percentile_rd": None if len(rds) < 20 else [float(np.percentile(rds, 2.5)), float(np.percentile(rds, 97.5))],
        "percentile_rr": None if len(rrs) < 20 else [float(np.percentile(rrs, 2.5)), float(np.percentile(rrs, 97.5))],
    }


def bca_interval(theta_hat: float, boots: list[float], jack: list[float], alpha: float = 0.05) -> dict:
    from math import erf, sqrt

    def Phi(z: float) -> float:
        return 0.5 * (1.0 + erf(z / sqrt(2.0)))

    def Phi_inv(p: float) -> float:
        # Acklam approximation
        if p <= 0.0:
            return -8.0
        if p >= 1.0:
            return 8.0
        a = [0, -3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02, 1.383577459334128e02, -3.066479806614716e01, 2.506628277459239e00]
        b = [0, -5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02, 6.680131188771972e01, -1.328068071618818e01]
        c = [0, -7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00, -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
        d = [0, 7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00]
        plow = 0.02425
        phigh = 1 - plow
        if p < plow:
            q = math.sqrt(-2 * math.log(p))
            return (((((c[1] * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) * q + c[6]) / ((((d[1] * q + d[2]) * q + d[3]) * q + d[4]) * q + 1)
        if p > phigh:
            q = math.sqrt(-2 * math.log(1 - p))
            return -(((((c[1] * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) * q + c[6]) / ((((d[1] * q + d[2]) * q + d[3]) * q + d[4]) * q + 1)
        q = p - 0.5
        r = q * q
        return (((((a[1] * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * r + a[6]) * q / (((((b[1] * r + b[2]) * r + b[3]) * r + b[4]) * r + b[5]) * r + 1)

    arr = np.array(boots, dtype=float)
    z0 = Phi_inv(float(np.mean(arr < theta_hat)))
    jack_arr = np.array(jack, dtype=float)
    theta_dot = float(jack_arr.mean()) if len(jack_arr) else theta_hat
    d = theta_dot - jack_arr
    den = float(np.sum(d ** 2))
    if den <= 0:
        a_hat = 0.0
    else:
        a_hat = float(np.sum(d ** 3)) / (6.0 * (den ** 1.5))
    z_lo = Phi_inv(alpha / 2.0)
    z_hi = Phi_inv(1.0 - alpha / 2.0)

    def adj(z):
        num = z0 + z
        return Phi(z0 + num / (1.0 - a_hat * num))

    p_lo = adj(z_lo)
    p_hi = adj(z_hi)
    p_lo = min(max(p_lo, 0.0), 1.0)
    p_hi = min(max(p_hi, 0.0), 1.0)
    return {
        "z0": z0,
        "a": a_hat,
        "p_lo": p_lo,
        "p_hi": p_hi,
        "interval": [float(np.quantile(arr, p_lo)), float(np.quantile(arr, p_hi))],
        "n_boot": len(arr),
        "n_jack_groups": len(jack_arr),
        "n_below_theta": int(np.sum(arr < theta_hat)),
    }


def grouped_jackknife(stays: list[dict], n_groups: int = 200, exposure_key: str = "en_h", seed: int = SEED) -> dict:
    rng = np.random.default_rng(seed + 1)
    by_subj: dict[int, list[dict]] = {}
    for s in stays:
        by_subj.setdefault(s["subject_id"], []).append(s)
    subjects = np.array(list(by_subj), dtype=int)
    rng.shuffle(subjects)
    groups = np.array_split(subjects, n_groups)
    rds, rrs = [], []
    failed = 0
    for i, g in enumerate(groups):
        hold = set(int(x) for x in g)
        sample = [s for s in stays if s["subject_id"] not in hold]
        try:
            est = estimate_exposure(sample, exposure_key)["msm"]
            if est["rd"] is None or est["rr"] is None:
                failed += 1
                continue
            rds.append(float(est["rd"]))
            rrs.append(float(est["rr"]))
        except Exception:
            failed += 1
        if (i + 1) % 20 == 0:
            print(f"jackknife {i + 1}/{n_groups} ok={len(rds)}", flush=True)
    return {"rd": rds, "rr": rrs, "n_groups": n_groups, "n_ok": len(rds), "n_failed": failed}


def run_a06(stays: list[dict], decision: dict) -> dict:
    bundled = estimate_bundle(stays)
    smd = smd_table(stays, bundled["clones"], bundled["w_s"])
    n_gt_01_u = sum(1 for r in smd if r["smd_unweighted_vs_eligible"] is not None and abs(r["smd_unweighted_vs_eligible"]) > 0.10)
    n_gt_01_w = sum(1 for r in smd if r["smd_weighted_vs_eligible"] is not None and abs(r["smd_weighted_vs_eligible"]) > 0.10)
    with SMD_CSV.open("w") as handle:
        cols = list(smd[0].keys())
        handle.write(",".join(cols) + "\n")
        for r in smd:
            handle.write(",".join("" if r[c] is None else str(r[c]) for c in cols) + "\n")
    payload = {
        "generated_at": now(),
        "status": "pipeline_qc_not_author_final",
        "b07_decision": decision,
        "n_stays": len(stays),
        "keys": list(IPCW_KEYS),
        **compact_msm(bundled),
        "smd": {
            "n_covariate_strategy_rows": len(smd),
            "n_smd_unweighted_gt_0_10": n_gt_01_u,
            "n_smd_weighted_gt_0_10": n_gt_01_w,
            "csv": str(SMD_CSV.relative_to(ROOT)),
        },
        "note": "Reduced SOFA (99% missing, missing=0) removed. Component scores from [t0-24h, t0+1h]. Missing dod treated as alive at day 28. Not Firth. Not author-final.",
    }
    OUT_A06.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({k: payload[k] for k in ("generated_at", "n_stays", "msm_gcomputation", "smd", "b07_decision") if k in payload}, indent=2), flush=True)
    return payload


def run_a13(stays: list[dict]) -> dict:
    results = {}
    for key, label in (
        ("oral_care_h", "oral_care_226168"),
        ("position_h", "position_224066_227952"),
        ("chg_h", "chg_bath_228137"),
    ):
        prev = exposure_prevalence(stays, key)
        print(f"A13 {label} prevalence", prev, flush=True)
        try:
            bundled = estimate_exposure(stays, key)
            results[label] = {
                "exposure_key": key,
                "prevalence": prev,
                **compact_msm(bundled),
            }
            msm = bundled["msm"]
            print(f"A13 {label} MSM RD={msm['rd']} RR={msm['rr']}", flush=True)
        except Exception as exc:
            results[label] = {
                "exposure_key": key,
                "prevalence": prev,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
            print(f"A13 {label} FAILED {exc}", flush=True)
    payload = {
        "generated_at": now(),
        "status": "pipeline_qc_not_author_final",
        "grace": "48h vs no event by 96h, same CCW+IPCW+MSM as Route A EN",
        "keys": list(IPCW_KEYS),
        "results": results,
        "interpretation_rule": "If a negative-control contrast is similar in magnitude and direction to the EN contrast, the pipeline is mainly measuring the severity that prompts a day-1 intervention.",
        "note": "Not author-final. Not copied into the 24h manuscript.",
    }
    OUT_A13.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def run_a11(stays: list[dict], n_boot: int, n_jack: int, a06_msm: dict) -> dict:
    boot = bootstrap_msm_checkpoint(stays, n_boot, exposure_key="en_h", resume=True)
    print("jackknife groups", n_jack, flush=True)
    jack = grouped_jackknife(stays, n_groups=n_jack, exposure_key="en_h")
    rd_hat = float(a06_msm["rd"])
    rr_hat = float(a06_msm["rr"])
    bca_rd = bca_interval(rd_hat, boot["rds"], jack["rd"])
    bca_rr = bca_interval(rr_hat, boot["rrs"], jack["rr"])
    payload = {
        "generated_at": now(),
        "status": "pipeline_qc_not_author_final",
        "n_stays": len(stays),
        "keys": list(IPCW_KEYS),
        "point": {"rd": rd_hat, "rr": rr_hat, "early_48": a06_msm["early_48"], "delayed_96": a06_msm["delayed_96"]},
        "bootstrap": {
            "n_requested": boot["n_requested"],
            "n_ok": boot["n_ok"],
            "n_failed": boot["n_failed"],
            "fail_reasons": boot["fail_reasons"],
            "percentile_rd": boot["percentile_rd"],
            "percentile_rr": boot["percentile_rr"],
        },
        "jackknife": {"n_groups": jack["n_groups"], "n_ok": jack["n_ok"], "n_failed": jack["n_failed"]},
        "bca_rd": bca_rd,
        "bca_rr": bca_rr,
        "acceleration_note": "Acceleration a uses a 200-group delete-group jackknife of subjects, not a full n=11,259 leave-one-out jackknife.",
        "note": "Not author-final. Do not copy into the 24h manuscript without author confirmation.",
    }
    OUT_A11.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({k: payload[k] for k in ("point", "bootstrap", "bca_rd", "bca_rr", "jackknife")}, indent=2), flush=True)
    return payload


def main() -> None:
    if not SOFA.exists() or not NC.exists():
        raise SystemExit(f"missing extract parquets: {SOFA} {NC}")
    extract_meta = json.loads((ROOT / "notes/cce-audit-A06-A13-extract.json").read_text())
    n_boot = 2000
    n_jack = 200
    skip_boot = False
    if len(sys.argv) > 1:
        if sys.argv[1] in {"--no-boot", "0"}:
            skip_boot = True
            n_boot = 0
        else:
            n_boot = int(sys.argv[1])
    if len(sys.argv) > 2:
        n_jack = int(sys.argv[2])
    coverage = extract_meta["sofa_component_coverage"]
    decision = choose_keys(coverage, float(extract_meta["lactate_observed_pct"]))
    set_keys(decision["keys"])
    print("KEYS", IPCW_KEYS, flush=True)
    print("A06 decision", decision["option"], flush=True)
    stays = overlay_a06(assemble())
    a06 = run_a06(stays, decision)
    run_a13(stays)
    if not skip_boot and n_boot:
        run_a11(stays, n_boot=n_boot, n_jack=n_jack, a06_msm=a06["msm_gcomputation"])


if __name__ == "__main__":
    main()
