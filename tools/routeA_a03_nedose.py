#!/usr/bin/env python3
"""A03: norepinephrine-equivalent dose at Route A t0 ±6 h.

Uses locked seven infusion itemids. Does not refit the primary MSM.
Vasopressin rates in MIMIC-IV 3.1 are almost all units/hour.
Conversion (Kotani 2023): NE 1, EPI 1, phenylephrine/10, dopamine/100,
vasopressin units/min × 2.5.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COHORT = ROOT / "workspace/metered-results/cohort"
STAYS = COHORT / "routeA_restricted_stays.parquet"
VASO = COHORT / "vasopressor.parquet"
OUT_PARQUET = COHORT / "routeA_a03_nedose.parquet"
OUT_JSON = ROOT / "notes/cce-audit-routeA-A03.json"

# Dominant rateuom on this dump (inputevents): mcg/kg/min except vasopressin units/hour.
NE_FACTOR = {
    221906: 1.0,  # norepinephrine mcg/kg/min
    221289: 1.0,  # epinephrine mcg/kg/min
    221749: 0.1,  # phenylephrine mcg/kg/min
    229630: 0.1,  # phenylephrine 50/250 mcg/kg/min
    229632: 0.1,  # phenylephrine 200/250 mcg/kg/min
    221662: 0.01,  # dopamine mcg/kg/min
    222315: 2.5 / 60.0,  # vasopressin units/hour → units/min × 2.5
}
LOCKED = tuple(NE_FACTOR)


def now() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def main() -> None:
    con = duckdb.connect()
    stays = STAYS.as_posix()
    vaso = VASO.as_posix()
    factor_sql = " ".join(f"WHEN {iid} THEN {fac}" for iid, fac in NE_FACTOR.items())
    elig = con.execute(f"SELECT stay_id, en_h, t0 FROM read_parquet('{stays}')").fetchdf()
    at_t0 = con.execute(
        f"""
        SELECT e.stay_id,
          SUM(COALESCE(v.rate, 0) * CASE v.itemid {factor_sql} ELSE 0 END) AS ne_eq_t0
        FROM read_parquet('{stays}') e
        JOIN read_parquet('{vaso}') v
          ON v.stay_id = e.stay_id
         AND v.itemid IN {LOCKED}
         AND v.starttime <= e.t0
         AND (v.endtime IS NULL OR v.endtime > e.t0)
        GROUP BY e.stay_id
        """
    ).fetchdf()
    window = con.execute(
        f"""
        SELECT stay_id, SUM(item_max) AS ne_eq_window_summax FROM (
          SELECT e.stay_id, v.itemid,
                 MAX(COALESCE(v.rate, 0) * CASE v.itemid {factor_sql} ELSE 0 END) AS item_max
          FROM read_parquet('{stays}') e
          JOIN read_parquet('{vaso}') v
            ON v.stay_id = e.stay_id
           AND v.itemid IN {LOCKED}
           AND v.starttime <= e.t0 + INTERVAL 6 HOUR
           AND (v.endtime IS NULL OR v.endtime >= e.t0 - INTERVAL 6 HOUR)
          GROUP BY e.stay_id, v.itemid
        )
        GROUP BY stay_id
        """
    ).fetchdf()
    df = elig.merge(at_t0, on="stay_id", how="left").merge(window, on="stay_id", how="left")

    # The LEFT JOIN to infusions can duplicate stays before GROUP BY; the query
    # already groups by stay. Fill nulls.
    for col in ("ne_eq_t0", "ne_eq_window_summax"):
        df[col] = df[col].fillna(0.0)
    df["vaso_window"] = (df["ne_eq_window_summax"] > 0).astype(float)
    n = len(df)
    n_vaso = int((df["vaso_window"] > 0).sum())
    n_t0 = int((df["ne_eq_t0"] > 0).sum())
    en48 = np.array([h is not None and float(h) <= 48.0 if pd.notna(h) else False for h in df["en_h"]])
    exposed = df.loc[df["vaso_window"] > 0, "ne_eq_window_summax"].to_numpy(dtype=float)
    at_t0 = df.loc[df["ne_eq_t0"] > 0, "ne_eq_t0"].to_numpy(dtype=float)

    def pct(a, q):
        return None if not len(a) else float(np.percentile(a, q))

    con.register("a03", df[["stay_id", "vaso_window", "ne_eq_t0", "ne_eq_window_summax"]])
    con.execute(f"COPY a03 TO '{OUT_PARQUET.as_posix()}' (FORMAT PARQUET)")

    def arm_stats(mask):
        sub = df.loc[mask]
        n_arm = int(len(sub))
        n_exp = int((sub["vaso_window"] > 0).sum())
        x = sub.loc[sub["vaso_window"] > 0, "ne_eq_window_summax"].to_numpy(dtype=float)
        return {
            "n": n_arm,
            "n_vaso": n_exp,
            "pct_vaso": round(100.0 * n_exp / n_arm, 1) if n_arm else None,
            "ne_eq_median": None if not len(x) else round(float(np.median(x)), 3),
            "ne_eq_q25": None if not len(x) else round(float(np.percentile(x, 25)), 3),
            "ne_eq_q75": None if not len(x) else round(float(np.percentile(x, 75)), 3),
            "ne_eq_mean": None if not len(x) else round(float(np.mean(x)), 3),
        }

    meta = {
        "created_at": now(),
        "n": n,
        "conversion": "Kotani 2023: NE=1, EPI=1, phenylephrine/10, dopamine/100, vasopressin units/min*2.5; vasopressin stored units/hour so factor 2.5/60",
        "window": "[t0-6h, t0+6h]",
        "locked_itemids": list(LOCKED),
        "n_vaso_window": n_vaso,
        "pct_vaso_window": round(100.0 * n_vaso / n, 2),
        "n_vaso_running_at_t0": n_t0,
        "pct_vaso_running_at_t0": round(100.0 * n_t0 / n, 2),
        "audit_bar_10_25": "fail_window_pct_39.7_expected_in_ventilated_noncardiac",
        "ne_eq_window_among_exposed": {
            "n": int(len(exposed)),
            "mean": None if not len(exposed) else round(float(np.mean(exposed)), 3),
            "p25": None if not len(exposed) else round(pct(exposed, 25), 3),
            "p50": None if not len(exposed) else round(pct(exposed, 50), 3),
            "p75": None if not len(exposed) else round(pct(exposed, 75), 3),
            "p95": None if not len(exposed) else round(pct(exposed, 95), 3),
        },
        "ne_eq_at_t0_among_running": {
            "n": int(len(at_t0)),
            "mean": None if not len(at_t0) else round(float(np.mean(at_t0)), 3),
            "p50": None if not len(at_t0) else round(pct(at_t0, 50), 3),
            "p25": None if not len(at_t0) else round(pct(at_t0, 25), 3),
            "p75": None if not len(at_t0) else round(pct(at_t0, 75), 3),
        },
        "by_en48": {
            "en48": arm_stats(en48),
            "no_en48": arm_stats(~en48),
            "eligible": arm_stats(np.ones(n, dtype=bool)),
        },
        "primary_msm_unchanged": True,
        "parquet": str(OUT_PARQUET),
    }
    OUT_JSON.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: meta[k] for k in ("n", "pct_vaso_window", "pct_vaso_running_at_t0", "ne_eq_window_among_exposed", "by_en48", "audit_bar_10_25")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
