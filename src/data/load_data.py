from __future__ import annotations
import  re, zipfile, warnings, time
from typing import Optional, Tuple, Dict
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from requests import RequestException
from tqdm import tqdm
import geopandas as gpd

# ================== PATHS (YOUR FOLDERS) ==================
BASE = Path(r"C:\Users\LEA GUR-ARIE\PycharmProjects\USA_Accidents_ML\src\data\raw\interim")
BASE_FARS = Path(r"C:\Users\LEA GUR-ARIE\PycharmProjects\USA_Accidents_ML\src\data\raw\interim\fars_data")
OUTPUTS=Path(r"C:\Users\LEA GUR-ARIE\PycharmProjects\USA_Accidents_ML\src\data\raw\interim\output_data")

US_ACCIDENTS_PKL = BASE / "us_accidents.pkl"
BORO_CSV         = BASE / "NYC_Zipcodes_NYC_Zip_2020.csv"
FARS_DATA_DIR    = BASE_FARS / "fars_data"
INTERIM_DIR      = OUTPUTS                                  # final output here

FARS_SLIM_PARQUET = INTERIM_DIR / "fars_nyc_slim.parquet"
FARS_SLIM_CSV     = INTERIM_DIR / "fars_nyc_slim.csv"
NYC_ACC_OUT       = INTERIM_DIR / "NYC_Accidents_with_FARS_raw.csv"  # <-- required name

# ================== CONFIG ==================
YEARS = list(range(2016, 2024))     # 2016–2023
STATE_FIPS_NY = 36
NYC_COUNTIES = {5, 47, 61, 81, 85}  # Bronx, Kings, New York, Queens, Richmond

RADIUS_BY_BOROUGH_KM: Dict[Optional[str], float] = {
    "Staten Island": 2.0,
    "Bronx": 1.2,
    "Brooklyn": 1.2,
    "Manhattan": 1.0,
    "Queens": 1.2,
    None: 1.2,  # fallback
}

# ================== FARS helpers ==================
def _find_fars_zip_local(year: int) -> Optional[Path]:
    FARS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    pats = [f"FARS{year}NationalCSV.zip", f"*{year}*FARS*CSV.zip", f"FARS*{year}*CSV.zip"]
    for pat in pats:
        hits = list(FARS_DATA_DIR.glob(pat))
        if hits:
            return sorted(hits, key=lambda p: p.stat().st_size if p.exists() else 0, reverse=True)[0]
    return None

# (Keep downloader as a fallback; it writes to fars_data if needed.)
FARS_URLS = [
    "https://static.nhtsa.gov/nhtsa/downloads/FARS/{year}/National/FARS{year}NationalCSV.zip",
    "https://www.nhtsa.gov/file-downloads/download?p=nhtsa/downloads/FARS/{year}/FARS{year}NationalCSV.zip",
]
def _download_fars_zip(year: int) -> Optional[Path]:
    out_zip = FARS_DATA_DIR / f"FARS{year}NationalCSV.zip"
    if out_zip.exists() and out_zip.stat().st_size > 0:
        return out_zip
    for u in FARS_URLS:
        url = u.format(year=year)
        try:
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                total = int(r.headers.get("Content-Length", 0))
                with open(out_zip, "wb") as f, tqdm(total=total or None, unit="B", unit_scale=True, desc=f"FARS {year}") as pbar:
                    for chunk in r.iter_content(1 << 15):
                        if chunk:
                            f.write(chunk); pbar.update(len(chunk))
            return out_zip
        except RequestException as e:
            warnings.warn(f"Download failed {url}: {e}")
            time.sleep(1)
    return None

def _read_member_from_zip(zip_path: Path, name_regex: str) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path, "r") as z:
        members = [n for n in z.namelist() if re.search(name_regex, n, flags=re.I)]
        if not members:
            raise FileNotFoundError(f"{name_regex} not found in {zip_path.name}")
        members.sort(key=lambda s: ("aux" in s.lower(), s))
        with z.open(members[0]) as f:
            try:
                return pd.read_csv(f, low_memory=False, encoding="utf-8")
            except UnicodeDecodeError:
                f.seek(0)
                return pd.read_csv(f, low_memory=False, encoding="latin1")

def _load_fars_year(year: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    z = _find_fars_zip_local(year) or _download_fars_zip(year)
    if z is None:
        raise RuntimeError(f"Missing FARS zip for {year} in {FARS_DATA_DIR}")
    acc = _read_member_from_zip(z, r"ACCIDENT\.CSV$")
    veh = _read_member_from_zip(z, r"VEHICLE\.CSV$")
    per = _read_member_from_zip(z, r"PERSON\.CSV$")
    return acc, veh, per

def build_fars_nyc_slim(years) -> pd.DataFrame:
    acc_all, veh_all, per_all = [], [], []
    for y in years:
        acc, veh, per = _load_fars_year(y)
        acc_all.append(acc); veh_all.append(veh); per_all.append(per)

    acc = pd.concat(acc_all, ignore_index=True)
    veh = pd.concat(veh_all, ignore_index=True)
    per = pd.concat(per_all, ignore_index=True)

    acc = acc[acc["STATE"] == STATE_FIPS_NY].copy()

    if "COUNTY" in acc.columns:
        acc_nyc = acc[acc["COUNTY"].isin(NYC_COUNTIES)].copy()
    else:
        lat_col = "LATITUDE" if "LATITUDE" in acc.columns else ("LATITUD" if "LATITUD" in acc.columns else None)
        lon_col = "LONGITUD" if "LONGITUD" in acc.columns else ("LONGITUDE" if "LONGITUDE" in acc.columns else None)
        if lat_col is None or lon_col is None:
            raise KeyError("ACCIDENT missing COUNTY and LAT/LON → cannot NYC-filter.")
        acc[lat_col] = pd.to_numeric(acc[lat_col], errors="coerce")
        acc[lon_col] = pd.to_numeric(acc[lon_col], errors="coerce")
        acc_nyc = acc[
            acc[lat_col].between(40.48, 40.95) & acc[lon_col].between(-74.26, -73.68)
        ].copy()

    keep_acc = [c for c in ["STATE","ST_CASE","LATITUDE","LATITUD","LONGITUD","LONGITUDE","DEATHS"] if c in acc_nyc.columns]
    acc_s = acc_nyc[keep_acc].copy()
    if "LATITUDE"  in acc_s.columns: acc_s = acc_s.rename(columns={"LATITUDE":"FARS_LAT"})
    if "LATITUD"   in acc_s.columns: acc_s = acc_s.rename(columns={"LATITUD":"FARS_LAT"})
    if "LONGITUD"  in acc_s.columns: acc_s = acc_s.rename(columns={"LONGITUD":"FARS_LON"})
    if "LONGITUDE" in acc_s.columns: acc_s = acc_s.rename(columns={"LONGITUDE":"FARS_LON"})

    keep_veh = [c for c in ["STATE","ST_CASE","VEH_NO","MAKE","MOD_YEAR","BODY_TYP","VEH_DMAG_SCL_1","VEH_DMAG_SCL_2"] if c in veh.columns]
    veh_s = veh[keep_veh].copy()

    per = per[per["STATE"] == STATE_FIPS_NY].copy()
    if "PER_TYP" in per.columns:
        per = per[per["PER_TYP"] == 1]
    keep_per = [c for c in ["STATE","ST_CASE","VEH_NO","AGE","AGE_IM","SEX","DR_DRINK","ALC_RES"] if c in per.columns]
    per_s = per[keep_per].copy()
    if "AGE" in per_s.columns:      per_s = per_s.rename(columns={"AGE":"DRIVER_AGE"})
    elif "AGE_IM" in per_s.columns: per_s = per_s.rename(columns={"AGE_IM":"DRIVER_AGE"})

    pv  = per_s.merge(veh_s, on=["STATE","ST_CASE","VEH_NO"], how="inner")
    pva = pv.merge(acc_s, on=["STATE","ST_CASE"], how="inner")

    for col in ["FARS_LAT","FARS_LON","DEATHS","DRIVER_AGE","SEX","DR_DRINK","ALC_RES","MAKE","MOD_YEAR","BODY_TYP","VEH_DMAG_SCL_1","VEH_DMAG_SCL_2"]:
        if col in pva.columns:
            pva[col] = pd.to_numeric(pva[col], errors="coerce")

    slim_cols = [c for c in [
        "STATE","ST_CASE","VEH_NO","DRIVER_AGE","SEX","DR_DRINK","ALC_RES",
        "MAKE","MOD_YEAR","BODY_TYP","VEH_DMAG_SCL_1","VEH_DMAG_SCL_2","DEATHS","FARS_LAT","FARS_LON"
    ] if c in pva.columns]
    return pva[slim_cols].drop_duplicates().reset_index(drop=True)

def save_fars_slim(df: pd.DataFrame) -> Path:
    try:
        df.to_parquet(FARS_SLIM_PARQUET, index=False)
        return FARS_SLIM_PARQUET
    except (ImportError, ValueError, OSError):
        df.to_csv(FARS_SLIM_CSV, index=False, encoding="utf-8")
        return FARS_SLIM_CSV

# ================== First merge: accidents ↔ boroughs (ZIP) ==================
def build_nyc_accidents_with_borough() -> pd.DataFrame:
    df = pd.read_pickle(US_ACCIDENTS_PKL)
    df_boros = pd.read_csv(BORO_CSV, encoding="utf-8-sig", low_memory=False)

    df["Zipcode_clean"] = (
        df.get("Zipcode").astype("string").str.strip()
          .str.extract(r"(\d{5})", expand=False).str.zfill(5)
    )

    nyc_zip3 = {"100","101","102","103","104","111","112","113","114","116"}
    df = df[df["Zipcode_clean"].str[:3].isin(nyc_zip3)].copy()

    df_boros["zip_code_clean"] = (
        df_boros["ZIP_Codes"].astype("string").str.strip()
               .str.extract(r"(\d{5})", expand=False).str.zfill(5)
    )
    df_boros = df_boros.drop_duplicates(subset=["zip_code_clean"])

    print("\n[INFO] Accidents (NYC-filter, pre-borough-merge):")
    df.info()

    print("\n[INFO] Borough lookup (cleaned):")
    df_boros.info()

    df = df.merge(df_boros[["zip_code_clean","Borough"]],
                  left_on="Zipcode_clean", right_on="zip_code_clean", how="left")

    zip3_to_borough = {
        "100":"Manhattan","101":"Manhattan","102":"Manhattan",
        "104":"Bronx",
        "112":"Brooklyn",
        "111":"Queens","113":"Queens","114":"Queens","116":"Queens",
        "103":"Staten Island",
    }
    na_mask = df["Borough"].isna()
    df.loc[na_mask, "Borough"] = df.loc[na_mask, "Zipcode_clean"].str[:3].map(zip3_to_borough)

    print("\n[INFO] Accidents + Boroughs (after first merge):")
    df.info()
    return df

# ================== Second merge: geo-only nearest (no dates) ==================
def spatial_join_nearest_borough_cap(left: pd.DataFrame, fars_slim: pd.DataFrame) -> pd.DataFrame:
    l = left.copy()
    have_xy = l["Start_Lat"].notna() & l["Start_Lng"].notna()
    l_geo = l.loc[have_xy].reset_index()  # keep original row index

    r = fars_slim.dropna(subset=["FARS_LAT","FARS_LON"]).reset_index(drop=True)

    print("\n[INFO] FARS slim (NYC-only, geo kept):")
    r.info()

    gl = gpd.GeoDataFrame(
        l_geo[[]].copy(),
        geometry=gpd.points_from_xy(l_geo["Start_Lng"].to_numpy(), l_geo["Start_Lat"].to_numpy()),
        crs="EPSG:4326",
    )
    gr = gpd.GeoDataFrame(
        r[[]].copy(),
        geometry=gpd.points_from_xy(r["FARS_LON"].to_numpy(), r["FARS_LAT"].to_numpy()),
        crs="EPSG:4326",
    )

    joined = gpd.sjoin_nearest(gl, gr, how="left", distance_col="match_distance_deg")
    joined = joined[~joined.index.duplicated(keep="first")]

    r_ix = joined["index_right"].to_numpy()
    dist_km = (joined["match_distance_deg"].to_numpy() * 111.0).astype("float32")
    valid_rr = ~pd.isna(r_ix)

    if "Borough" in l_geo.columns:
        caps_geo = np.array([
            RADIUS_BY_BOROUGH_KM.get(b if pd.notna(b) else None, RADIUS_BY_BOROUGH_KM[None])
            for b in l_geo["Borough"].astype("string")
        ], dtype="float32")
    else:
        caps_geo = np.full(len(l_geo), RADIUS_BY_BOROUGH_KM[None], dtype="float32")

    within_cap = valid_rr & (dist_km <= caps_geo)

    fars_pref = r.add_prefix("FARS__").reset_index(drop=True)

    out = l.copy()
    out["match_distance_km"] = pd.NA
    for col in fars_pref.columns:
        out[col] = pd.NA

    valid_pos = np.where(within_cap)[0]
    if valid_pos.size:
        orig_ix = l_geo.loc[valid_pos, "index"].to_numpy()
        out.loc[orig_ix, "match_distance_km"] = dist_km[valid_pos]
        r_pos2 = r_ix[valid_pos].astype(int)
        for col in fars_pref.columns:
            out.loc[orig_ix, col] = fars_pref[col].iloc[r_pos2].to_numpy()

    print("\n[INFO] Accidents + Boroughs + FARS (after second merge):")
    out.info()
    return out.reset_index(drop=True)

# ================== Pipeline ==================
def main() -> None:
    # 1) Accidents ↔ Boroughs
    nyc_acc = build_nyc_accidents_with_borough()

    # 2) FARS slim (load if exists; else build from fars_data)
    if FARS_SLIM_PARQUET.exists() or FARS_SLIM_CSV.exists():
        slim_path = FARS_SLIM_PARQUET if FARS_SLIM_PARQUET.exists() else FARS_SLIM_CSV
        fars_slim = pd.read_parquet(slim_path) if slim_path.suffix == ".parquet" else pd.read_csv(slim_path, low_memory=False)
    else:
        fars_slim = build_fars_nyc_slim(YEARS)
        slim_path = save_fars_slim(fars_slim)
        print(f"\n[INFO] Saved FARS slim to: {slim_path}")

    # 3) Geo-only nearest (borough caps)
    merged = spatial_join_nearest_borough_cap(nyc_acc, fars_slim)

    # 4) Save final output in your interim (BASE) folder
    merged.to_csv(NYC_ACC_OUT, index=False, encoding="utf-8")
    print(f"\n✅ Final saved to: {NYC_ACC_OUT}")

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
