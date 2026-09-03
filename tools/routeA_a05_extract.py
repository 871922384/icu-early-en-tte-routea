#!/usr/bin/env python3
"""A05 adjustment-set extract for the Route A restricted cohort.

Adds services, era, admission-location class, admission weight, GCS copy,
prior 30-day hospitalization, Quan Elixhauser, and closest-to-t0 labs.
Does not refit the confirmed primary MSM. Sepsis-3 is unavailable on this
MIMIC-IV 3.1 gzip dump (no mimiciv_derived.sepsis3). Patient parquet is
gitignored via the metered-results symlink.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MIMIC = Path("/Volumes/B/litflow-cache/physionet/physionet.org/files/mimiciv/3.1")
COHORT = ROOT / "workspace/metered-results/cohort"
STAYS = COHORT / "routeA_restricted_stays.parquet"
SOFA = COHORT / "routeA_sofa_full.parquet"
OUT_PARQUET = COHORT / "routeA_a05_covariates.parquet"
OUT_JSON = ROOT / "notes/cce-audit-routeA-A05.json"

LAB_ITEMS = {
    51222: "hb",
    51301: "wbc",
    50983: "sodium",
    50971: "potassium",
    50931: "glucose",
    50862: "albumin",
    50820: "ph",
    51237: "inr",
    50813: "lactate",
    50912: "creatinine",
    50885: "bilirubin",
    51265: "platelet",
    50821: "po2",
}
CHART_WEIGHT_KG = 226512
CHART_WEIGHT_LB = 226531
CHART_PEEP = (220339, 224700)
CHART_FIO2 = 223835
CHART_VT = (224684, 224685)

# Quan 2005 Elixhauser prefixes (dots stripped). Hierarchy applied after match.
ELIX_ICD10 = {
    "chf": ("I099", "I110", "I130", "I132", "I255", "I420", "I425", "I426", "I427", "I428", "I429", "I43", "I50", "P290"),
    "arrhythmia": ("I441", "I442", "I443", "I456", "I459", "I47", "I48", "I49", "R000", "R001", "R008", "T821", "Z450", "Z950"),
    "valvular": ("A520", "I05", "I06", "I07", "I08", "I091", "I098", "I34", "I35", "I36", "I37", "I38", "I39", "Q230", "Q231", "Q232", "Q233", "Z952", "Z953"),
    "pcd": ("I26", "I27", "I280", "I288", "I289"),
    "pvd": ("I70", "I71", "I731", "I738", "I739", "I771", "I790", "I792", "K551", "K558", "K559", "Z958", "Z959"),
    "htn_uncomp": ("I10",),
    "htn_comp": ("I11", "I12", "I13", "I15"),
    "paralysis": ("G041", "G114", "G801", "G802", "G81", "G82", "G830", "G831", "G832", "G833", "G834", "G839"),
    "neuro": ("G10", "G11", "G12", "G13", "G20", "G21", "G22", "G254", "G255", "G312", "G318", "G319", "G32", "G35", "G36", "G37", "G40", "G41", "G931", "G934", "R470", "R56"),
    "copd": ("I278", "I279", "J40", "J41", "J42", "J43", "J44", "J45", "J46", "J47", "J60", "J61", "J62", "J63", "J64", "J65", "J66", "J67", "J684", "J701", "J703"),
    "dm_uncomp": ("E100", "E101", "E109", "E110", "E111", "E119", "E120", "E121", "E129", "E130", "E131", "E139", "E140", "E141", "E149"),
    "dm_comp": ("E102", "E103", "E104", "E105", "E106", "E107", "E108", "E112", "E113", "E114", "E115", "E116", "E117", "E118", "E122", "E123", "E124", "E125", "E126", "E127", "E128", "E132", "E133", "E134", "E135", "E136", "E137", "E138", "E142", "E143", "E144", "E145", "E146", "E147", "E148"),
    "hypothyroid": ("E00", "E01", "E02", "E03", "E890"),
    "renal": ("I120", "I131", "N18", "N19", "N250", "Z490", "Z491", "Z492", "Z940", "Z992"),
    "liver": ("B18", "I85", "I864", "I982", "K70", "K711", "K713", "K714", "K715", "K717", "K72", "K73", "K74", "K760", "K762", "K763", "K764", "K765", "K766", "K767", "K768", "K769", "Z944"),
    "pud": ("K257", "K259", "K267", "K269", "K277", "K279", "K287", "K289"),
    "hiv": ("B20", "B21", "B22", "B24"),
    "lymphoma": ("C81", "C82", "C83", "C84", "C85", "C88", "C900", "C902"),
    "metastatic": ("C77", "C78", "C79", "C80"),
    "tumor": ("C00", "C01", "C02", "C03", "C04", "C05", "C06", "C07", "C08", "C09", "C10", "C11", "C12", "C13", "C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21", "C22", "C23", "C24", "C25", "C26", "C30", "C31", "C32", "C33", "C34", "C37", "C38", "C39", "C40", "C41", "C43", "C45", "C46", "C47", "C48", "C49", "C50", "C51", "C52", "C53", "C54", "C55", "C56", "C57", "C58", "C60", "C61", "C62", "C63", "C64", "C65", "C66", "C67", "C68", "C69", "C70", "C71", "C72", "C73", "C74", "C75", "C76", "C97"),
    "rheum": ("L940", "L941", "L943", "M05", "M06", "M08", "M120", "M123", "M30", "M310", "M311", "M312", "M313", "M32", "M33", "M34", "M35", "M45", "M461", "M468", "M469"),
    "coag": ("D65", "D66", "D67", "D68", "D691", "D693", "D694", "D695", "D696"),
    "obesity": ("E66",),
    "weight_loss": ("E40", "E41", "E42", "E43", "E44", "E45", "E46", "R634", "R64"),
    "fluid": ("E222", "E86", "E87"),
    "blood_loss": ("D500",),
    "anemia": ("D508", "D509", "D51", "D52", "D53"),
    "alcohol": ("F10", "E52", "G621", "I426", "K292", "K700", "K703", "K709", "T51", "Z502", "Z714", "Z721"),
    "drugs": ("F11", "F12", "F13", "F14", "F15", "F16", "F18", "F19", "Z715", "Z722"),
    "psychoses": ("F20", "F22", "F23", "F24", "F25", "F28", "F29", "F302", "F312", "F315"),
    "depression": ("F204", "F313", "F314", "F32", "F33", "F341", "F412", "F432"),
}
ELIX_ICD9 = {
    "chf": ("39891", "40201", "40211", "40291", "40401", "40403", "40411", "40413", "40491", "40493", "4254", "4255", "4257", "4258", "4259", "428"),
    "arrhythmia": ("4260", "42613", "4267", "4269", "42610", "42612", "4270", "4271", "4272", "4273", "4274", "4276", "4277", "4278", "4279", "7850", "V450", "V533"),
    "valvular": ("0932", "394", "395", "396", "397", "424", "7463", "7464", "7465", "7466", "V422", "V433"),
    "pcd": ("4150", "4151", "416", "4170", "4178", "4179"),
    "pvd": ("0930", "4373", "440", "441", "4431", "4432", "4438", "4439", "4471", "5571", "5579", "V434"),
    "htn_uncomp": ("401",),
    "htn_comp": ("402", "403", "404", "405"),
    "paralysis": ("3341", "342", "343", "3440", "3441", "3442", "3443", "3444", "3445", "3446", "3449"),
    "neuro": ("3319", "3320", "3321", "3334", "3335", "33392", "334", "335", "3362", "340", "341", "345", "3481", "3483", "7803", "7843"),
    "copd": ("4168", "4169", "490", "491", "492", "493", "494", "495", "496", "500", "501", "502", "503", "504", "505", "5064", "5081", "5088"),
    "dm_uncomp": ("2500", "2501", "2502", "2503"),
    "dm_comp": ("2504", "2505", "2506", "2507", "2508", "2509"),
    "hypothyroid": ("243", "244"),
    "renal": ("40301", "40311", "40391", "40402", "40403", "40412", "40413", "40492", "40493", "582", "5830", "5831", "5832", "5834", "5836", "5837", "585", "586", "5880", "V420", "V451", "V56"),
    "liver": ("07022", "07023", "07032", "07033", "07044", "07054", "0706", "0709", "4560", "4561", "4562", "570", "571", "5722", "5723", "5724", "5728", "5733", "5734", "5738", "5739", "V427"),
    "pud": ("5317", "5319", "5327", "5329", "5337", "5339", "5347", "5349"),
    "hiv": ("042", "043", "044"),
    "lymphoma": ("200", "201", "202", "2030", "2386"),
    "metastatic": ("196", "197", "198", "199"),
    "tumor": ("140", "141", "142", "143", "144", "145", "146", "147", "148", "149", "150", "151", "152", "153", "154", "155", "156", "157", "158", "159", "160", "161", "162", "163", "164", "165", "166", "167", "168", "169", "170", "171", "172", "174", "175", "176", "177", "178", "179", "180", "181", "182", "183", "184", "185", "186", "187", "188", "189", "190", "191", "192", "193", "194", "195"),
    "rheum": ("446", "7010", "7100", "7101", "7102", "7103", "7104", "7108", "7109", "7112", "714", "7193", "720", "725", "7285", "72889", "72930"),
    "coag": ("286", "2871", "2873", "2874", "2875"),
    "obesity": ("2780",),
    "weight_loss": ("260", "261", "262", "263", "7832", "7994"),
    "fluid": ("276",),
    "blood_loss": ("2800",),
    "anemia": ("2801", "2808", "2809", "281"),
    "alcohol": ("2652", "2911", "2912", "2913", "2915", "2918", "2919", "3030", "3039", "3050", "3575", "4255", "5353", "5710", "5711", "5712", "5713", "980", "V113"),
    "drugs": ("292", "304", "3052", "3053", "3054", "3055", "3056", "3057", "3058", "3059", "V6542"),
    "psychoses": ("2938", "295", "29604", "29614", "29644", "29654", "297", "298"),
    "depression": ("2962", "2963", "2965", "3004", "309", "311"),
}
ELIX_ORDER = list(ELIX_ICD10.keys())
VW = {
    "chf": 7, "arrhythmia": 5, "valvular": -1, "pcd": 4, "pvd": 2,
    "htn_uncomp": 0, "htn_comp": 0, "paralysis": 7, "neuro": 6, "copd": 3,
    "dm_uncomp": 0, "dm_comp": 0, "hypothyroid": 0, "renal": 5, "liver": 11,
    "pud": 0, "hiv": 0, "lymphoma": 9, "metastatic": 12, "tumor": 4,
    "rheum": 0, "coag": 3, "obesity": -4, "weight_loss": 6, "fluid": 5,
    "blood_loss": -2, "anemia": -2, "alcohol": 0, "drugs": -7, "psychoses": 0,
    "depression": -3,
}

SERVICE_GROUP = {
    "CSURG": "cardiac_surgery",
    "NSURG": "neuro",
    "NMED": "neuro",
    "SURG": "surgical",
    "TSURG": "surgical",
    "VSURG": "surgical",
    "PSURG": "surgical",
    "ORTHO": "surgical",
    "ENT": "surgical",
    "GU": "surgical",
    "GYN": "surgical",
    "TRAUM": "surgical",
    "OBS": "surgical",
    "MED": "medical",
    "CMED": "medical",
    "OMED": "medical",
}


def now() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def loc_group(label: str | None) -> str:
    s = (label or "").upper()
    if s in {"EMERGENCY ROOM", "WALK-IN/SELF REFERRAL"}:
        return "ed"
    if "TRANSFER" in s or s.startswith("INTERNAL TRANSFER"):
        return "transfer"
    if s in {"PHYSICIAN REFERRAL", "CLINIC REFERRAL", "AMBULATORY SURGERY TRANSFER"}:
        return "referral"
    if s in {"PROCEDURE SITE", "PACU"}:
        return "or"
    return "unknown"


def match_elix(code: str, version: int) -> set[str]:
    raw = (code or "").replace(".", "").strip().upper()
    if not raw:
        return set()
    table = ELIX_ICD10 if int(version) == 10 else ELIX_ICD9
    hit: set[str] = set()
    for name, prefixes in table.items():
        for p in prefixes:
            if raw.startswith(p):
                hit.add(name)
                break
    return hit


def apply_hierarchy(flags: set[str]) -> set[str]:
    out = set(flags)
    if "htn_comp" in out:
        out.discard("htn_uncomp")
    if "dm_comp" in out:
        out.discard("dm_uncomp")
    if "metastatic" in out:
        out.discard("tumor")
    return out


def main() -> None:
    con = duckdb.connect()
    stays_path = STAYS.as_posix()
    sofa_path = SOFA.as_posix()
    con.execute(f"CREATE VIEW elig AS SELECT * FROM read_parquet('{stays_path}')")
    n = con.execute("SELECT COUNT(*) FROM elig").fetchone()[0]
    print("elig", n, flush=True)

    print("patients era...", flush=True)
    patients = (MIMIC / "hosp/patients.csv.gz").as_posix()
    era = con.execute(
        f"""
        SELECT e.stay_id, p.anchor_year_group
        FROM elig e
        LEFT JOIN read_csv_auto('{patients}', header=true, compression='gzip') p
          USING (subject_id)
        """
    ).fetchdf()

    print("first service...", flush=True)
    services = (MIMIC / "hosp/services.csv.gz").as_posix()
    svc = con.execute(
        f"""
        SELECT stay_id, curr_service AS first_service FROM (
          SELECT e.stay_id, s.curr_service,
                 ROW_NUMBER() OVER (PARTITION BY e.stay_id ORDER BY s.transfertime, s.curr_service) AS rn
          FROM elig e
          LEFT JOIN read_csv_auto('{services}', header=true, compression='gzip') s
            ON s.hadm_id = e.hadm_id
        ) WHERE rn = 1
        """
    ).fetchdf()

    print("admissions (LOS, prior 30d)...", flush=True)
    adm = (MIMIC / "hosp/admissions.csv.gz").as_posix()
    los = con.execute(
        f"""
        SELECT e.stay_id, a.admittime, a.dischtime AS hosp_dischtime,
               date_diff('second', a.admittime, a.dischtime) / 86400.0 AS hospital_los_days
        FROM elig e
        LEFT JOIN read_csv_auto('{adm}', header=true, compression='gzip') a
          ON a.hadm_id = e.hadm_id
        """
    ).fetchdf()
    prior = con.execute(
        f"""
        SELECT e.stay_id,
               COUNT(p.hadm_id) AS prior_hosp_30d_n
        FROM elig e
        JOIN read_csv_auto('{adm}', header=true, compression='gzip') idx
          ON idx.hadm_id = e.hadm_id
        LEFT JOIN read_csv_auto('{adm}', header=true, compression='gzip') p
          ON p.subject_id = e.subject_id
         AND p.hadm_id <> e.hadm_id
         AND p.admittime < idx.admittime
         AND p.admittime >= idx.admittime - INTERVAL 30 DAY
        GROUP BY e.stay_id
        """
    ).fetchdf()

    print("diagnoses for Elixhauser...", flush=True)
    dxp = (MIMIC / "hosp/diagnoses_icd.csv.gz").as_posix()
    dx = con.execute(
        f"""
        SELECT e.stay_id, d.icd_code, d.icd_version
        FROM elig e
        JOIN read_csv_auto('{dxp}', header=true, compression='gzip') d
          ON d.hadm_id = e.hadm_id
        """
    ).fetchdf()
    print("dx rows", len(dx), flush=True)

    print("labs closest to t0...", flush=True)
    lab_ids = ",".join(str(i) for i in LAB_ITEMS)
    labp = (MIMIC / "hosp/labevents.csv.gz").as_posix()
    con.execute(
        f"""
        CREATE VIEW labs AS
        SELECT hadm_id, itemid, charttime, valuenum
        FROM read_csv_auto('{labp}', header=true, compression='gzip', ignore_errors=true)
        WHERE itemid IN ({lab_ids})
          AND valuenum IS NOT NULL
          AND hadm_id IN (SELECT hadm_id FROM elig)
        """
    )
    lab_near = con.execute(
        """
        SELECT stay_id, itemid, valuenum FROM (
          SELECT e.stay_id, l.itemid, l.valuenum,
                 ROW_NUMBER() OVER (
                   PARTITION BY e.stay_id, l.itemid
                   ORDER BY abs(date_diff('second', e.t0, l.charttime))
                 ) AS rn
          FROM elig e
          JOIN labs l
            ON l.hadm_id = e.hadm_id
           AND l.charttime BETWEEN e.t0 - INTERVAL 24 HOUR AND e.t0 + INTERVAL 1 HOUR
        ) WHERE rn = 1
        """
    ).fetchdf()
    print("lab_near", len(lab_near), flush=True)

    print("chartevents weight and vent settings...", flush=True)
    chart_ids = ",".join(
        str(i) for i in (CHART_WEIGHT_KG, CHART_WEIGHT_LB, CHART_FIO2, *CHART_PEEP, *CHART_VT)
    )
    chartp = (MIMIC / "icu/chartevents.csv.gz").as_posix()
    con.execute(
        f"""
        CREATE VIEW charts AS
        SELECT stay_id, itemid, charttime, valuenum
        FROM read_csv_auto('{chartp}', header=true, compression='gzip', ignore_errors=true)
        WHERE itemid IN ({chart_ids})
          AND valuenum IS NOT NULL
          AND stay_id IN (SELECT stay_id FROM elig)
        """
    )
    chart_near = con.execute(
        """
        SELECT stay_id, itemid, valuenum FROM (
          SELECT e.stay_id, c.itemid, c.valuenum,
                 ROW_NUMBER() OVER (
                   PARTITION BY e.stay_id, c.itemid
                   ORDER BY abs(date_diff('second', e.t0, c.charttime))
                 ) AS rn
          FROM elig e
          JOIN charts c
            ON c.stay_id = e.stay_id
           AND c.charttime BETWEEN e.t0 - INTERVAL 24 HOUR AND e.t0 + INTERVAL 6 HOUR
        ) WHERE rn = 1
        """
    ).fetchdf()
    print("chart_near", len(chart_near), flush=True)

    sofa = con.execute(
        f"SELECT stay_id, gcs_sum, pf_ratio FROM read_parquet('{sofa_path}')"
    ).fetchdf()
    elig = con.execute(
        "SELECT stay_id, subject_id, hadm_id, admission_location, first_careunit FROM elig"
    ).fetchdf()

    print("assemble...", flush=True)
    labs_wide = (
        lab_near.pivot_table(index="stay_id", columns="itemid", values="valuenum", aggfunc="first")
        if len(lab_near)
        else pd.DataFrame()
    )
    charts_wide = (
        chart_near.pivot_table(index="stay_id", columns="itemid", values="valuenum", aggfunc="first")
        if len(chart_near)
        else pd.DataFrame()
    )

    def first_map(frame: pd.DataFrame, key: str, val: str) -> dict:
        out = {}
        if not len(frame):
            return out
        for rec in frame.to_dict("records"):
            v = rec.get(val)
            out[int(rec[key])] = None if v is None or (isinstance(v, float) and np.isnan(v)) or pd.isna(v) else v
        return out

    svc_map = first_map(svc, "stay_id", "first_service")
    era_map = first_map(era, "stay_id", "anchor_year_group")
    prior_map = first_map(prior, "stay_id", "prior_hosp_30d_n")
    los_map = first_map(los, "stay_id", "hospital_los_days")
    gcs_map = first_map(sofa, "stay_id", "gcs_sum")
    pf_map = first_map(sofa, "stay_id", "pf_ratio")

    elix_sum = {}
    elix_vw = {}
    if len(dx):
        grouped = dx.groupby("stay_id")
        for sid, g in grouped:
            flags: set[str] = set()
            for rec in g.itertuples(index=False):
                flags |= match_elix(str(rec.icd_code), int(rec.icd_version))
            flags = apply_hierarchy(flags)
            elix_sum[int(sid)] = float(len(flags))
            elix_vw[int(sid)] = float(sum(VW[k] for k in flags))

    rows = []
    for rec in elig.to_dict("records"):
        sid = int(rec["stay_id"])
        loc = loc_group(rec.get("admission_location"))
        first_svc = svc_map.get(sid)
        if first_svc is not None:
            first_svc = str(first_svc)
        svc_g = SERVICE_GROUP.get(first_svc or "", "other")
        era_g = era_map.get(sid)
        if era_g is not None:
            era_g = str(era_g)
        prior_n = float(prior_map.get(sid) or 0.0)
        hosp_los = los_map.get(sid)
        if hosp_los is not None:
            hosp_los = float(hosp_los)
        wkg = None
        peep = None
        fio2 = None
        vt = None
        if sid in charts_wide.index:
            rowc = charts_wide.loc[sid]
            if CHART_WEIGHT_KG in charts_wide.columns and not pd.isna(rowc.get(CHART_WEIGHT_KG, np.nan)):
                wkg = float(rowc[CHART_WEIGHT_KG])
            elif CHART_WEIGHT_LB in charts_wide.columns and not pd.isna(rowc.get(CHART_WEIGHT_LB, np.nan)):
                wkg = float(rowc[CHART_WEIGHT_LB]) * 0.453592
            for iid in CHART_PEEP:
                if iid in charts_wide.columns and not pd.isna(rowc.get(iid, np.nan)):
                    peep = float(rowc[iid])
                    break
            if CHART_FIO2 in charts_wide.columns and not pd.isna(rowc.get(CHART_FIO2, np.nan)):
                fio2 = float(rowc[CHART_FIO2])
            for iid in CHART_VT:
                if iid in charts_wide.columns and not pd.isna(rowc.get(iid, np.nan)):
                    vt = float(rowc[iid])
                    break
        gcs = gcs_map.get(sid)
        if gcs is not None:
            gcs = float(gcs)
        pf = pf_map.get(sid)
        if pf is not None:
            pf = float(pf)
        lab_vals = {}
        if sid in labs_wide.index:
            rowl = labs_wide.loc[sid]
            for iid, name in LAB_ITEMS.items():
                if iid in labs_wide.columns and not pd.isna(rowl.get(iid, np.nan)):
                    lab_vals[name] = float(rowl[iid])
        row = {
            "stay_id": sid,
            "subject_id": int(rec["subject_id"]),
            "hadm_id": int(rec["hadm_id"]),
            "anchor_year_group": era_g,
            "first_service": first_svc,
            "service_group": svc_g,
            "loc_group": loc,
            "prior_hosp_30d": 1.0 if prior_n > 0 else 0.0,
            "prior_hosp_30d_n": prior_n,
            "weight_kg": wkg,
            "gcs_sum": gcs,
            "pf_ratio": pf,
            "peep": peep,
            "fio2": fio2,
            "vt_ml": vt,
            "elixhauser_sum": elix_sum.get(sid, 0.0),
            "elixhauser_vw": elix_vw.get(sid, 0.0),
            "sepsis3": None,
            "hospital_los_days": hosp_los,
            "svc_medical": 1.0 if svc_g == "medical" else 0.0,
            "svc_surgical": 1.0 if svc_g == "surgical" else 0.0,
            "svc_neuro": 1.0 if svc_g == "neuro" else 0.0,
            "svc_cardiac": 1.0 if svc_g == "cardiac_surgery" else 0.0,
            "loc_ed": 1.0 if loc == "ed" else 0.0,
            "loc_transfer": 1.0 if loc == "transfer" else 0.0,
            "loc_referral": 1.0 if loc == "referral" else 0.0,
            "loc_or": 1.0 if loc == "or" else 0.0,
            "era_2008_2010": 1.0 if era_g == "2008 - 2010" else 0.0,
            "era_2011_2013": 1.0 if era_g == "2011 - 2013" else 0.0,
            "era_2014_2016": 1.0 if era_g == "2014 - 2016" else 0.0,
            "era_2017_2019": 1.0 if era_g == "2017 - 2019" else 0.0,
            "era_2020_2022": 1.0 if era_g == "2020 - 2022" else 0.0,
        }
        for name in LAB_ITEMS.values():
            row[name] = lab_vals.get(name)
        rows.append(row)

    df = pd.DataFrame(rows)
    con.register("a05", df)
    con.execute(f"COPY a05 TO '{OUT_PARQUET.as_posix()}' (FORMAT PARQUET)")

    def cov(col: str) -> dict:
        s = df[col]
        obs = int(s.notna().sum())
        return {"n": int(len(s)), "observed": obs, "observed_pct": round(100.0 * obs / len(s), 2) if len(s) else None}

    coverage = {c: cov(c) for c in ["weight_kg", "gcs_sum", "peep", "fio2", "vt_ml", "hospital_los_days"] + list(LAB_ITEMS.values())}
    svc_counts = df["service_group"].value_counts(dropna=False).to_dict()
    loc_counts = df["loc_group"].value_counts(dropna=False).to_dict()
    era_counts = df["anchor_year_group"].value_counts(dropna=False).to_dict()
    meta = {
        "created_at": now(),
        "n": int(len(df)),
        "parquet": str(OUT_PARQUET),
        "sepsis3": "unavailable_no_mimiciv_derived_sepsis3",
        "elixhauser": "Quan 2005 ICD-9/ICD-10 prefixes with AHRQ hierarchy; discharge-coded",
        "weight_itemid": CHART_WEIGHT_KG,
        "lab_window": "[t0-24h, t0+1h] closest to t0 by hadm_id",
        "chart_window": "[t0-24h, t0+6h] closest to t0 by stay_id",
        "primary_msm_unchanged": True,
        "coverage": coverage,
        "service_group": {str(k): int(v) for k, v in svc_counts.items()},
        "loc_group": {str(k): int(v) for k, v in loc_counts.items()},
        "era": {str(k): int(v) for k, v in era_counts.items()},
        "prior_hosp_30d_n": int(df["prior_hosp_30d"].sum()),
        "prior_hosp_30d_pct": round(100.0 * float(df["prior_hosp_30d"].mean()), 2),
        "elixhauser_sum_mean": round(float(df["elixhauser_sum"].mean()), 3),
        "elixhauser_vw_mean": round(float(df["elixhauser_vw"].mean()), 3),
    }
    OUT_JSON.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: meta[k] for k in ("n", "coverage", "service_group", "loc_group", "era", "prior_hosp_30d_pct", "elixhauser_sum_mean")}, indent=2), flush=True)
    print("wrote", OUT_PARQUET, OUT_JSON, flush=True)


if __name__ == "__main__":
    main()
