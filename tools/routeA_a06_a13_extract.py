#!/usr/bin/env python3
"""A06 full SOFA in [t0-24h, t0+1h] and A13 first oral-care / position-change times.

Restricted Route A cohort only. Patient-level parquet is gitignored.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
MIMIC = Path("/Volumes/B/litflow-cache/physionet/physionet.org/files/mimiciv/3.1")
COHORT = ROOT / "workspace/metered-results/cohort"
OUT_SOFA = COHORT / "routeA_sofa_full.parquet"
OUT_NC = COHORT / "routeA_negative_control_events.parquet"
OUT_JSON = ROOT / "notes/cce-audit-A06-A13-extract.json"

LAB_BILI = 50885
LAB_CR = 50912
LAB_PLT = 51265
LAB_PO2 = 50821
LAB_LAC = 50813
CHART_MAP = (220052, 220181, 225312)
CHART_GCS = (220739, 223900, 223901)
CHART_FIO2 = 223835
ORAL = 226168
POSITION = (224066, 227952)
CHG = 228137
URINE = (226559, 226560, 226561, 226584, 226563, 226564, 226565, 226567, 226557, 226558)


def sofa_resp(pf: float | None, ventilated: bool = True) -> int | None:
    if pf is None:
        return None
    if ventilated:
        if pf < 100:
            return 4
        if pf < 200:
            return 3
        if pf < 300:
            return 2
        if pf < 400:
            return 1
        return 0
    if pf < 400:
        return 1
    return 0


def sofa_coag(plt: float | None) -> int | None:
    if plt is None:
        return None
    if plt < 20:
        return 4
    if plt < 50:
        return 3
    if plt < 100:
        return 2
    if plt < 150:
        return 1
    return 0


def sofa_liver(bili: float | None) -> int | None:
    if bili is None:
        return None
    if bili >= 12:
        return 4
    if bili >= 6:
        return 3
    if bili >= 2:
        return 2
    if bili >= 1.2:
        return 1
    return 0


def sofa_cns(gcs: float | None) -> int | None:
    if gcs is None:
        return None
    if gcs < 6:
        return 4
    if gcs <= 9:
        return 3
    if gcs <= 12:
        return 2
    if gcs <= 14:
        return 1
    return 0


def sofa_renal(cr: float | None, urine_ml: float | None) -> int | None:
    cr_score = None
    if cr is not None:
        if cr >= 5:
            cr_score = 4
        elif cr >= 3.5:
            cr_score = 3
        elif cr >= 2:
            cr_score = 2
        elif cr >= 1.2:
            cr_score = 1
        else:
            cr_score = 0
    u_score = None
    if urine_ml is not None:
        if urine_ml < 200:
            u_score = 4
        elif urine_ml < 500:
            u_score = 3
        else:
            u_score = 0
    if cr_score is None and u_score is None:
        return None
    if cr_score is None:
        return u_score
    if u_score is None:
        return cr_score
    return max(cr_score, u_score)


def sofa_cardio(map_min: float | None, vaso: int) -> int | None:
    # vaso 0/1 at t0±6h; without dose, MAP<70 -> 1, vaso -> 3
    if vaso:
        return 3
    if map_min is None:
        return None
    if map_min < 70:
        return 1
    return 0


def main() -> None:
    import sys

    global OUT_SOFA, OUT_NC, OUT_JSON
    stays_name = "routeA_restricted_stays.parquet"
    if len(sys.argv) > 1:
        stays_name = sys.argv[1]
    if len(sys.argv) > 2:
        OUT_SOFA = COHORT / sys.argv[2]
    if len(sys.argv) > 3:
        OUT_NC = COHORT / sys.argv[3]
    if len(sys.argv) > 4:
        OUT_JSON = ROOT / "notes" / sys.argv[4]
    con = duckdb.connect()
    stays = (COHORT / stays_name).as_posix()
    vaso = (COHORT / "vasopressor.parquet").as_posix()
    con.execute(f"CREATE VIEW elig AS SELECT * FROM read_parquet('{stays}')")
    n = con.execute("SELECT COUNT(*) FROM elig").fetchone()[0]
    print("elig", n)

    print("labs...")
    con.execute(
        f"""
        CREATE VIEW labs AS
        SELECT hadm_id, itemid, charttime, valuenum
        FROM read_csv_auto('{(MIMIC / "hosp/labevents.csv.gz").as_posix()}',
                           header=true, compression='gzip', ignore_errors=true)
        WHERE itemid IN ({LAB_BILI}, {LAB_CR}, {LAB_PLT}, {LAB_PO2}, {LAB_LAC})
          AND valuenum IS NOT NULL
          AND hadm_id IN (SELECT hadm_id FROM elig)
        """
    )
    con.execute(
        """
        CREATE TEMP TABLE lab_win AS
        SELECT e.stay_id,
          MIN(CASE WHEN l.itemid=50885 THEN l.valuenum END) AS bili_min,
          MAX(CASE WHEN l.itemid=50885 THEN l.valuenum END) AS bili_max,
          MIN(CASE WHEN l.itemid=50912 THEN l.valuenum END) AS cr_min,
          MAX(CASE WHEN l.itemid=50912 THEN l.valuenum END) AS cr_max,
          MIN(CASE WHEN l.itemid=51265 THEN l.valuenum END) AS plt_min,
          MIN(CASE WHEN l.itemid=50821 THEN l.valuenum END) AS po2_min,
          MAX(CASE WHEN l.itemid=50813 THEN l.valuenum END) AS lac_max
        FROM elig e
        LEFT JOIN labs l
          ON l.hadm_id=e.hadm_id
         AND l.charttime BETWEEN e.t0 - INTERVAL 24 HOUR AND e.t0 + INTERVAL 1 HOUR
        GROUP BY e.stay_id
        """
    )
    print("lab_win", con.execute("SELECT COUNT(*) FROM lab_win").fetchone())

    print("chartevents (MAP, GCS, FiO2, oral care, position)...")
    itemids = ",".join(str(i) for i in (*CHART_MAP, *CHART_GCS, CHART_FIO2, ORAL, *POSITION, CHG))
    con.execute(
        f"""
        CREATE VIEW charts AS
        SELECT stay_id, itemid, charttime, valuenum, value
        FROM read_csv_auto('{(MIMIC / "icu/chartevents.csv.gz").as_posix()}',
                           header=true, compression='gzip', ignore_errors=true)
        WHERE itemid IN ({itemids})
          AND stay_id IN (SELECT stay_id FROM elig)
        """
    )
    con.execute(
        """
        CREATE TEMP TABLE chart_win AS
        SELECT e.stay_id,
          MIN(CASE WHEN c.itemid IN (220052,220181,225312) AND c.valuenum IS NOT NULL
                    AND c.charttime BETWEEN e.t0 - INTERVAL 24 HOUR AND e.t0 + INTERVAL 1 HOUR
                   THEN c.valuenum END) AS map_min,
          MIN(CASE WHEN c.itemid=220739 AND c.valuenum IS NOT NULL
                    AND c.charttime BETWEEN e.t0 - INTERVAL 24 HOUR AND e.t0 + INTERVAL 1 HOUR
                   THEN c.valuenum END) AS gcs_eye,
          MIN(CASE WHEN c.itemid=223900 AND c.valuenum IS NOT NULL
                    AND c.charttime BETWEEN e.t0 - INTERVAL 24 HOUR AND e.t0 + INTERVAL 1 HOUR
                   THEN c.valuenum END) AS gcs_verb,
          MIN(CASE WHEN c.itemid=223901 AND c.valuenum IS NOT NULL
                    AND c.charttime BETWEEN e.t0 - INTERVAL 24 HOUR AND e.t0 + INTERVAL 1 HOUR
                   THEN c.valuenum END) AS gcs_mot,
          MAX(CASE WHEN c.itemid=223835 AND c.valuenum IS NOT NULL
                    AND c.charttime BETWEEN e.t0 - INTERVAL 24 HOUR AND e.t0 + INTERVAL 1 HOUR
                   THEN c.valuenum END) AS fio2_max,
          MIN(CASE WHEN c.itemid=226168 AND c.charttime >= e.t0 THEN c.charttime END) AS oral_care_time,
          MIN(CASE WHEN c.itemid IN (224066,227952) AND c.charttime >= e.t0 THEN c.charttime END) AS position_time,
          MIN(CASE WHEN c.itemid=228137 AND c.charttime >= e.t0 THEN c.charttime END) AS chg_time
        FROM elig e
        LEFT JOIN charts c ON c.stay_id=e.stay_id
        GROUP BY e.stay_id
        """
    )
    print("chart_win", con.execute("SELECT COUNT(*) FROM chart_win").fetchone())

    print("urine...")
    urine_ids = ",".join(str(i) for i in URINE)
    con.execute(
        f"""
        CREATE VIEW uo AS
        SELECT stay_id, itemid, charttime, value
        FROM read_csv_auto('{(MIMIC / "icu/outputevents.csv.gz").as_posix()}',
                           header=true, compression='gzip')
        WHERE itemid IN ({urine_ids})
          AND stay_id IN (SELECT stay_id FROM elig)
        """
    )
    con.execute(
        """
        CREATE TEMP TABLE uo_win AS
        SELECT e.stay_id, SUM(TRY_CAST(u.value AS DOUBLE)) AS urine_ml
        FROM elig e
        LEFT JOIN uo u
          ON u.stay_id=e.stay_id
         AND u.charttime BETWEEN e.t0 - INTERVAL 24 HOUR AND e.t0 + INTERVAL 1 HOUR
        GROUP BY e.stay_id
        """
    )

    con.execute(
        f"""
        CREATE TEMP TABLE vaso_t0 AS
        SELECT DISTINCT e.stay_id
        FROM elig e
        JOIN read_parquet('{vaso}') v USING (stay_id)
        WHERE v.starttime <= e.t0 + INTERVAL 6 HOUR
          AND (v.endtime IS NULL OR v.endtime >= e.t0 - INTERVAL 6 HOUR)
        """
    )

    df = con.execute(
        """
        SELECT e.stay_id, e.t0,
          l.bili_max, l.cr_max, l.plt_min, l.po2_min, l.lac_max,
          c.map_min, c.gcs_eye, c.gcs_verb, c.gcs_mot, c.fio2_max,
          c.oral_care_time, c.position_time, c.chg_time,
          u.urine_ml,
          CASE WHEN v.stay_id IS NOT NULL THEN 1 ELSE 0 END AS vaso
        FROM elig e
        LEFT JOIN lab_win l USING (stay_id)
        LEFT JOIN chart_win c USING (stay_id)
        LEFT JOIN uo_win u USING (stay_id)
        LEFT JOIN vaso_t0 v USING (stay_id)
        """
    ).fetchdf()

    recs = []
    nc = []
    miss = {
        "resp": 0,
        "coag": 0,
        "liver": 0,
        "cardio": 0,
        "cns": 0,
        "renal": 0,
        "any": 0,
        "complete": 0,
    }

    def num(v):
        if v is None:
            return None
        try:
            if v != v:  # NaN / NaT
                return None
        except Exception:
            return None
        try:
            return float(v)
        except Exception:
            return None

    for r in df.to_dict("records"):
        po2 = num(r["po2_min"])
        fio2 = num(r["fio2_max"])
        pf = None
        if po2 is not None and fio2 not in (None, 0):
            fi = float(fio2)
            if fi > 1:
                fi = fi / 100.0
            if fi > 0:
                pf = float(po2) / fi
        gcs = None
        ge, gv, gm = num(r["gcs_eye"]), num(r["gcs_verb"]), num(r["gcs_mot"])
        if ge is not None and gv is not None and gm is not None:
            gcs = ge + gv + gm
        resp = sofa_resp(pf)
        coag = sofa_coag(num(r["plt_min"]))
        liver = sofa_liver(num(r["bili_max"]))
        cardio = sofa_cardio(num(r["map_min"]), int(r["vaso"] or 0))
        cns = sofa_cns(gcs)
        urine = num(r["urine_ml"])
        renal = sofa_renal(num(r["cr_max"]), urine)
        lac = num(r["lac_max"])
        comps = [resp, coag, liver, cardio, cns, renal]
        names = ["resp", "coag", "liver", "cardio", "cns", "renal"]
        any_m = False
        total = 0
        n_obs = 0
        for name, val in zip(names, comps):
            if val is None:
                miss[name] += 1
                any_m = True
            else:
                total += val
                n_obs += 1
        if any_m:
            miss["any"] += 1
        else:
            miss["complete"] += 1
        recs.append(
            {
                "stay_id": int(r["stay_id"]),
                "sofa_resp": resp,
                "sofa_coag": coag,
                "sofa_liver": liver,
                "sofa_cardio": cardio,
                "sofa_cns": cns,
                "sofa_renal": renal,
                "sofa_sum_observed": total,
                "sofa_n_observed": n_obs,
                "sofa_any_missing": int(any_m),
                "sofa_complete": int(not any_m),
                "pf_ratio": pf,
                "gcs_sum": gcs,
                "lactate": lac,
                "lactate_missing": int(lac is None),
                "urine_ml": urine,
                "map_min": num(r["map_min"]),
            }
        )
        t0 = r["t0"]
        oral_h = pos_h = chg_h = None
        if r["oral_care_time"] is not None and t0 is not None and str(r["oral_care_time"]) != "NaT":
            try:
                oral_h = (r["oral_care_time"] - t0).total_seconds() / 3600.0
            except Exception:
                oral_h = None
        if r["position_time"] is not None and t0 is not None and str(r["position_time"]) != "NaT":
            try:
                pos_h = (r["position_time"] - t0).total_seconds() / 3600.0
            except Exception:
                pos_h = None
        if r["chg_time"] is not None and t0 is not None and str(r["chg_time"]) != "NaT":
            try:
                chg_h = (r["chg_time"] - t0).total_seconds() / 3600.0
            except Exception:
                chg_h = None
        nc.append(
            {
                "stay_id": int(r["stay_id"]),
                "oral_care_h": oral_h,
                "position_h": pos_h,
                "chg_h": chg_h,
            }
        )

    import pandas as pd

    sofa_df = pd.DataFrame(recs)
    nc_df = pd.DataFrame(nc)
    con.register("sofa_df", sofa_df)
    con.register("nc_df", nc_df)
    con.execute(f"COPY sofa_df TO '{OUT_SOFA.as_posix()}' (FORMAT PARQUET)")
    con.execute(f"COPY nc_df TO '{OUT_NC.as_posix()}' (FORMAT PARQUET)")
    n = len(recs)
    coverage = {k: {"missing": v, "observed": n - v, "observed_pct": round(100 * (n - v) / n, 2)} for k, v in miss.items() if k not in ("any", "complete")}
    coverage["any_component_missing"] = miss["any"]
    coverage["complete_six"] = miss["complete"]
    coverage["complete_six_pct"] = round(100 * miss["complete"] / n, 2)
    def by_h(key, h):
        return sum(1 for x in nc if x[key] is not None and 0 <= x[key] <= h)

    oral_n = sum(1 for x in nc if x["oral_care_h"] is not None)
    pos_n = sum(1 for x in nc if x["position_h"] is not None)
    chg_n = sum(1 for x in nc if x["chg_h"] is not None)
    lac_n = sum(1 for x in recs if x["lactate"] is not None)
    out = {
        "generated_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "n": n,
        "window": "[t0-24h, t0+1h]",
        "sofa_component_coverage": coverage,
        "lactate_observed": lac_n,
        "lactate_observed_pct": round(100 * lac_n / n, 2),
        "oral_care_any": oral_n,
        "oral_care_by_48h": by_h("oral_care_h", 48),
        "oral_care_by_48h_pct": round(100 * by_h("oral_care_h", 48) / n, 2),
        "oral_care_by_96h": by_h("oral_care_h", 96),
        "position_any": pos_n,
        "position_by_48h": by_h("position_h", 48),
        "position_by_48h_pct": round(100 * by_h("position_h", 48) / n, 2),
        "position_by_96h": by_h("position_h", 96),
        "chg_any": chg_n,
        "chg_by_48h": by_h("chg_h", 48),
        "chg_by_48h_pct": round(100 * by_h("chg_h", 48) / n, 2),
        "chg_by_96h": by_h("chg_h", 96),
        "parquet_sofa": str(OUT_SOFA),
        "parquet_nc": str(OUT_NC),
        "note": "No mimiciv_derived.sofa on the 3.1 gzip dump; components rebuilt from labevents/chartevents/outputevents. Urine window is the same 25 h SOFA window. Negative-control hours are first event at or after t0.",
    }
    OUT_JSON.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
