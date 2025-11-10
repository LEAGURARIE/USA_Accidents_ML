
import pandas as pd
import numpy as np
import re
import os



# ---------------- I/O ----------------
in_path  = r"C:\Users\LEA GUR-ARIE\PycharmProjects\USA_Accidents_ML\src\data\raw\interim\output_data\NYC_Accidents_with_FARS_raw.csv"
out_path = r"C:\Users\LEA GUR-ARIE\PycharmProjects\USA_Accidents_ML\src\data\raw\interim\output_data\df_prepared.csv"

df = pd.read_csv(in_path, low_memory=False)
print(f"Loaded: {in_path}")
print(df.info())


# ---------- helpers ----------
def have(cols):
    missing = [c for c in cols if c not in df.columns]
    return len(missing) == 0, missing

_MISS_TOKENS = {
    "", "na", "n/a", "null", "none", "missing", "unk", "unknown",
    "<na>", "<nan>", "nan"
}

# =========================================================
# 1) Source vs Severity (crosstabs)
# =========================================================
ok, miss = have(["Source", "Severity"])
if ok:
    ct_counts = pd.crosstab(df["Source"], df["Severity"], dropna=False)
    print("\n=== Counts ===")
    print(ct_counts)

    ct_props = pd.crosstab(df["Source"], df["Severity"], normalize="index", dropna=False)
    print("\n=== Row Proportions ===")
    print(ct_props.round(3))
else:
    print(f"\n[Skip] Crosstabs: missing columns: {miss}")

# =========================================================
# 2) Street_is_highway flag
# =========================================================
ok, miss = have(["Street"])
if ok:
    s = df["Street"].astype("string").str.upper().str.strip()

    kw_regex = (
        r"(?:\b(?:EXPRESSWAY|EXPWY|EXPY|PARKWAY|PKWY|HIGHWAY|HWY)\b)"
        r"|(?:\bI-\s*\d+\b)"
        r"|(?:\bUS-\s*\d+\b)"
        r"|(?:\bNY-\s*\d+\b|\bNY\s*ROUTE\s*\d+\b|\bROUTE\s*\d+\b)"
    )
    kw_re = re.compile(kw_regex)

    name_re = re.compile("|".join(map(re.escape, [
        "BQE","BROOKLYN QUEENS EXPY","LIE","LONG ISLAND EXPY","GCP","GRAND CENTRAL PKWY",
        "FDR","FDR DR","HARLEM RIVER DR","CROSS BRONX","CROSS BRONX EXPY","MAJOR DEEGAN","DEEGAN",
        "VAN WYCK","VAN WYCK EXPY","BELT PKWY","BELT PARKWAY","JACKIE ROBINSON PKWY","INTERBORO PKWY",
        "HENRY HUDSON PKWY","WEST SIDE HWY","CLEARVIEW EXPY","BRONX RIVER PKWY",
        "WHITESTONE EXPWY","THROGS NECK EXPY","SIE","STATEN ISLAND EXPWY"
    ])), re.IGNORECASE)

    highway_mask = s.str.contains(kw_re, na=False) | s.str.contains(name_re, na=False)
    df["Street_is_highway"] = highway_mask.astype("boolean")

    print("\n[Street_is_highway] created.")
    print(df["Street_is_highway"].dtype)
    print(df["Street_is_highway"].value_counts(dropna=False))
    print(df[["Street","Street_is_highway"]].head(10))
else:
    print(f"\n[Skip] Street_is_highway: missing column {miss}")

# =========================================================
# 3) Airport_Code frequency table
# =========================================================
if "Airport_Code" in df.columns:
    df_freq = df.copy()
    df_freq["Airport_Code"] = df_freq["Airport_Code"].astype("string").fillna("Missing")
    counts = df_freq["Airport_Code"].value_counts(dropna=False).rename("Count")
    percents = (counts / counts.sum() * 100).round(2).rename("Percent")
    freq_table = pd.concat([counts, percents], axis=1).reset_index()
    freq_table.columns = ["Airport_Code", "Count", "Percent"]
    print("\n[Airport_Code] frequency:")
    print(freq_table.head(20))
else:
    print("\n[Skip] Airport_Code frequency: column missing")

# =========================================================
# 4) Wind_Direction -> Wind_Direction_grp
# =========================================================
if "Wind_Direction" in df.columns:
    dir_groups = {
        "N":"NORTH","NNE":"NORTH","NNW":"NORTH","NORTH":"NORTH",
        "E":"EAST","ENE":"EAST","ESE":"EAST","NE":"EAST","EAST":"EAST",
        "S":"SOUTH","SSE":"SOUTH","SE":"SOUTH","SSW":"SOUTH","SOUTH":"SOUTH",
        "W":"WEST","WNW":"WEST","WSW":"WEST","SW":"WEST","NW":"WEST","WEST":"WEST",
        "VAR":"OTHER","VARIABLE":"OTHER","CALM":"OTHER",
    }
    wind_cats = ["NORTH","EAST","SOUTH","WEST","OTHER","MISSING"]

    raw = df["Wind_Direction"].astype("string").str.strip()
    is_missing = raw.isna() | raw.str.casefold().isin(_MISS_TOKENS)

    df["Wind_Direction_grp"] = pd.Categorical(
        np.where(~is_missing, raw.str.upper().map(dir_groups).fillna("OTHER"), "MISSING"),
        categories=wind_cats, ordered=False
    )
    print("\n[Wind_Direction_grp] created.")
    print(df["Wind_Direction_grp"].value_counts(dropna=False))
else:
    print("\n[Skip] Wind_Direction_grp: Wind_Direction column missing")

# =========================================================
# 5) Weather_Condition -> Weather_ConditionGroup
# =========================================================
if "Weather_Condition" in df.columns:
    s = df["Weather_Condition"].astype("string").str.strip()
    is_missing = s.isna() | s.str.casefold().isin(_MISS_TOKENS)

    weather_map = {
        # CLEAR
        "Fair": "CLEAR", "Clear": "CLEAR", "Mostly Clear": "CLEAR",
        # CLOUDY
        "Cloudy": "CLOUDY", "Mostly Cloudy": "CLOUDY", "Overcast": "CLOUDY", "Partly Cloudy": "CLOUDY",
        # RAIN
        "Rain": "RAIN", "Light Rain": "RAIN", "Showers": "RAIN", "Drizzle": "RAIN",
        "Heavy Rain": "RAIN", "Rain Shower": "RAIN",
        # SNOW / ICE
        "Snow": "SNOW", "Light Snow": "SNOW", "Sleet": "SNOW", "Ice": "SNOW", "Hail": "SNOW",
        # FOG
        "Fog": "FOG", "Mist": "FOG", "Haze": "FOG", "Smoke": "FOG",
        # STORMS
        "Thunderstorms": "STORM", "T-Storm": "STORM", "Storm": "STORM", "Heavy Thunderstorms": "STORM", "Thunder": "STORM",
    }

    s_norm = s.str.title()
    mapped = s_norm.map(weather_map)
    grp = np.where(~is_missing, mapped.fillna("OTHER"), "MISSING")
    weather_cats = ["CLEAR", "CLOUDY", "RAIN", "SNOW", "FOG", "STORM", "OTHER", "MISSING"]
    df["Weather_ConditionGroup"] = pd.Categorical(grp, categories=weather_cats, ordered=False)

    print("\n[Weather_ConditionGroup] created.")
    print(df["Weather_ConditionGroup"].value_counts(dropna=False))
else:
    print("\n[Skip] Weather_ConditionGroup: Weather_Condition column missing")

# =========================================================
# 6) Distance_group (fixed cutpoints with safe fallback)
# =========================================================
col_dist = "Distance(mi)"
if col_dist in df.columns:
    dist = pd.to_numeric(df[col_dist], errors="coerce")
    # your fixed cutpoints:
    bins = [-np.inf, 0.134, 0.757, np.inf]
    labels = ["≤0.13 mi (Short)", "0.13–0.76 mi (Medium)", ">0.76 mi (Long)"]

    # sanity check: if fixed bins collapse due to data scale, fallback to Q2/Q3
    if not np.isfinite(dist).any():
        print("\n[Skip] Distance_group: all Distance(mi) are non-numeric/NaN")
    else:
        try:
            df["Distance_group"] = pd.cut(dist, bins=bins, labels=labels, include_lowest=True)
        except ValueError:
            q2 = dist.quantile(0.50)
            q3 = dist.quantile(0.75)
            bins_fallback = [-np.inf, q2, q3, np.inf]
            df["Distance_group"] = pd.cut(dist, bins=bins_fallback, labels=labels, include_lowest=True)
            print("[Info] Distance bins fell back to dataset quantiles (median/Q3).")

        print("\n[Distance_group] distribution (proportion):")
        print(df["Distance_group"].value_counts(normalize=True, dropna=False))
else:
    print("\n[Skip] Distance_group: Distance(mi) column missing")




# ---------------- save ----------------
os.makedirs(os.path.dirname(out_path), exist_ok=True)
df.to_csv(out_path, index=False, encoding="utf-8")
print(f"\n✅ Saved CSV to: {out_path}")

print("\n[Final df.info()]")
print(df.info())

