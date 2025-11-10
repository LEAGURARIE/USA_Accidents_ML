# ==========================================================
# TRAIN FE → SAVE ARTIFACTS (incl. Random-Weighted pools)
# → APPLY TO TRAIN/VALID/TEST → EXPORT CSV/Parquet (+info)
# ==========================================================

from __future__ import annotations

# --- Standard libs ---
import os
import glob as pyglob
import json
import platform
from typing import Dict, List, Any, Tuple, Sequence, Optional
from collections import Counter

# --- Third-party ---
import joblib
import numpy as np
import pandas as pd
from pandas.api.types import (
    CategoricalDtype,
    pandas_dtype,
    is_integer_dtype,
    is_float_dtype,
    is_bool_dtype,
    is_string_dtype,
    is_datetime64_any_dtype,
)

# Imputers (enable IterativeImputer)
import sklearn.experimental.enable_iterative_imputer  # noqa: F401
from sklearn.impute import KNNImputer
from sklearn.impute import IterativeImputer as SkIterativeImputer  # type: ignore[attr-defined]

# =============================
# Configuration (adjust paths)
# =============================

SPLIT_DIR  = r"C:\Users\LEA GUR-ARIE\PycharmProjects\USA_Accidents_ML\src\data\raw\interim\output_data\split"

TRAIN_PATH = os.path.join(SPLIT_DIR, "train.csv")
VALID_PATH = os.path.join(SPLIT_DIR, "val.csv")
TEST_PATH  = os.path.join(SPLIT_DIR, "test.csv")

ARTIFACT_DIR = r"C:\Users\LEA GUR-ARIE\PycharmProjects\USA_Accidents_ML\src\data\raw\interim\output_data\split\_fe_artifacts"
OUT_DIR      = r"C:\Users\LEA GUR-ARIE\PycharmProjects\USA_Accidents_ML\src\data\raw\interim\output_data\split\_fe_outputs"
os.makedirs(ARTIFACT_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# FE constants (must match TRAIN pipeline)
HARD_CAP_MINUTES = 30 * 24 * 60  # 43,200 minutes cap
KNN_COLS: List[str]  = ["Pressure(in)", "Visibility(mi)", "Temperature(F)", "Humidity(%)"]
MICE_COLS: List[str] = ["FARS__DRIVER_AGE", "FARS__MAKE", "FARS__ALC_RES", "FARS__BODY_TYP", "FARS__SEX"]
WR_GROUP_COLS: List[str] = ["Event_Year", "Event_Month", "Borough"]

# Columns to drop AFTER all imputations/derivations
DROP_AFTER_IMPUTE: List[str] = [
    "End_Lat", "End_Lng", "Visibility_is_low",
    "Wind_Speed_zero_flag", "duration_is_outlier_30d",
    "Start_Time", "End_Time", "Start_Time_parsed", "End_Time_parsed",
    "Event_TS", "Event_Date",
    "Source", "Airport_Code",
    "Severity_2cat", "Severity_4cat",
    "zip_code_clean", "Zipcode_clean", "Zip_code_clean",
    "Severity",
    "Severity_bin",
    "Distance(mi)",
    "Precipitation(in)",
    "Weather_ConditionGroup",
    "Distance_group",
    "Visibility_2cat",
    "FARS__SEX",
    "FARS__SEX_label",
    "FARS__BODY_TYP",
]

# RNG for random-weighted fills
SEED = 42
RNG = np.random.default_rng(SEED)

# =============================
# Small helpers
# =============================
def _as_tuple_key(key: Any) -> Tuple[Any, ...]:
    if isinstance(key, tuple):
        return key
    if isinstance(key, (list, tuple, np.ndarray, pd.Index, pd.Series, Sequence)) and not isinstance(key, (str, bytes)):
        return tuple(key)
    return (key,)

def _as_category_list(obj: Any) -> List[str]:
    if obj is None:
        return []
    if isinstance(obj, (list, tuple, pd.Index, np.ndarray)):
        return [str(x) for x in obj]
    return [str(obj)]

def info_preview(df: pd.DataFrame, tag: str) -> None:
    print(f"\n🧾 INFO PREVIEW — {tag}")
    print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} cols")
    df.info()

# =============================
# IO utilities
# =============================
def load_dataframe_any(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path not found: {path}")
    if os.path.isdir(path):
        preferred = pyglob.glob(os.path.join(path, "train.csv"))
        csv_candidates = preferred + (pyglob.glob(os.path.join(path, "*.csv")) if not preferred else [])
        if not csv_candidates:
            raise FileNotFoundError(f"No CSV files in: {path}")
        df_loaded = pd.read_csv(csv_candidates[0])
    else:
        if not path.lower().endswith(".csv"):
            raise ValueError(f"Expected .csv, got: {path}")
        df_loaded = pd.read_csv(path)
    df_loaded.columns = df_loaded.columns.str.strip()
    return df_loaded

def export_dataset(df: pd.DataFrame, stem: str, tag: str) -> None:
    info_preview(df, tag)
    csv_path = os.path.join(OUT_DIR, f"{stem}.csv")
    pq_path  = os.path.join(OUT_DIR, f"{stem}.parquet")
    df.to_csv(csv_path, index=False)
    try:
        df.to_parquet(pq_path, index=False)
    except (ImportError, ValueError, TypeError, OSError) as e:
        print(f"⚠️ Parquet not written ({e}). CSV still saved.")
    print(f"💾 Saved: {csv_path}")
    if os.path.exists(pq_path):
        print(f"💾 Saved: {pq_path}")

# =============================
# Datetime cleaning & parsing
# =============================
_CONTROL_CHARS_RE = r"[\x00-\x1F\x7F]"
_ZERO_WIDTH_RE    = r"[\u200B-\u200D\u2060\uFEFF]"
_MULTI_SPACE_RE   = r"\s+"
_ZERO_TIME_RE     = r"^\s*(?:0{1,2}:?0{1,2}(?::?0{1,2})?|0+)\s*$"

def clean_datetime_like(series_in: pd.Series) -> pd.Series:
    out = series_in.astype("string", copy=False)
    out = out.str.replace(_ZERO_WIDTH_RE, "", regex=True)
    out = out.str.replace(_CONTROL_CHARS_RE, " ", regex=True)
    out = out.str.normalize("NFKC").str.strip()
    out = out.str.replace(_MULTI_SPACE_RE, " ", regex=True)
    out = out.mask(out.str.fullmatch(_ZERO_TIME_RE), np.nan)
    out = out.mask(out == "", np.nan)
    return out

def parse_mixed_datetime(series_in: pd.Series) -> pd.Series:
    return pd.to_datetime(series_in, errors="coerce", utc=False, format="mixed")

# =============================
# Core FE pieces
# =============================
def to_numeric_safe(series_in: pd.Series) -> pd.Series:
    return pd.to_numeric(series_in, errors="coerce")

def build_bins_and_levels(train_df: pd.DataFrame) -> Dict:
    dist_series = to_numeric_safe(train_df.get("Distance(mi)", pd.Series(dtype=float)))
    max_dist = float(dist_series.max()) if dist_series.size else 0.0
    last_edge = max(1.892, max_dist)
    return {
        "Distance_group": {
            "bins":   [0.0, 0.757, 1.892, last_edge],
            "labels": ["Short", "Medium", "Long"],
        },
        "Precipitation_cat": {
            "bins":   [-1e-12, 0.0, 0.1, 0.3, 1.0, float("inf")],
            "labels": ["Dry", "Light", "Moderate", "Heavy", "Extreme"],
        },
        "Weather_ConditionGroup_order": ["CLEAR", "CLOUDY", "FOG", "RAIN", "SNOW", "STORM", "OTHER", "MISSING"],
        "Visibility_2cat_order": ["Normal", "Low", "Missing"],
    }

# --- Driver age normalization (new) ---
def normalize_driver_age(series_in: pd.Series,
                         min_age: int = 14,
                         max_age: int = 100,
                         invalid_codes: Tuple[int, ...] = (0, 997, 998, 999)) -> pd.Series:
    s = pd.to_numeric(series_in, errors="coerce")
    if s.empty:
        return s.astype("float32")
    invalid_mask = s.isin(invalid_codes)
    out_of_range = (s < min_age) | (s > max_age)
    s = s.mask(invalid_mask | out_of_range, np.nan)
    return s.astype("float32")

def apply_core_fe_inplace(df_work: pd.DataFrame, fe_levels: Dict) -> None:
    # Date/Time normalization and extraction
    if "Start_Time" in df_work.columns:
        df_work["Start_Time_parsed"] = parse_mixed_datetime(clean_datetime_like(df_work["Start_Time"]))
    else:
        df_work["Start_Time_parsed"] = pd.NaT

    if "End_Time" in df_work.columns:
        df_work["End_Time_parsed"] = parse_mixed_datetime(clean_datetime_like(df_work["End_Time"]))
    else:
        df_work["End_Time_parsed"] = pd.NaT

    df_work["Event_TS"]    = df_work["Start_Time_parsed"].combine_first(df_work["End_Time_parsed"])
    df_work["Event_Date"]  = df_work["Event_TS"].dt.normalize()
    df_work["Event_Year"]  = df_work["Event_TS"].dt.year.astype("Int64")
    df_work["Event_Month"] = df_work["Event_TS"].dt.month.astype("Int64")
    df_work["Event_Day"]   = df_work["Event_TS"].dt.day.astype("Int64")
    df_work["Event_Hour"]  = df_work["Event_TS"].dt.hour.astype("float64")
    df_work["Event_DOW"]   = df_work["Event_TS"].dt.weekday.astype("float64")
    df_work["Is_Weekend"]  = df_work["Event_DOW"].isin([4, 5])  # Friday/Saturday

    # Day/Night three-way label
    day_mask = df_work["Event_TS"].notna() & df_work["Event_TS"].dt.hour.between(6, 21, inclusive="both")
    dn_arr = np.select([df_work["Event_TS"].isna(), day_mask], ["Missing", "Day"], default="Night")
    df_work["DayNight_3cat"] = pd.Categorical(dn_arr, categories=["Day", "Night", "Missing"], ordered=False)

    # Distance features
    dcfg = fe_levels["Distance_group"]
    dist_vals = to_numeric_safe(df_work.get("Distance(mi)", pd.Series(np.nan, index=df_work.index))).clip(lower=0)
    df_work["Distance(mi)"] = dist_vals
    df_work["Distance(mi)_log"] = np.log1p(dist_vals.fillna(0))
    try:
        df_work["Distance_group"] = pd.cut(
            dist_vals.fillna(-1),
            bins=dcfg["bins"],
            labels=dcfg["labels"],
            include_lowest=True
        )
        if isinstance(df_work["Distance_group"].dtype, CategoricalDtype):
            df_work["Distance_group"] = df_work["Distance_group"].cat.set_categories(
                _as_category_list(dcfg["labels"]), ordered=True
            )
    except (ValueError, TypeError):
        df_work["Distance_group"] = pd.Categorical(
            ["Short"] * len(df_work),
            categories=_as_category_list(dcfg["labels"]),
            ordered=True
        )

    # Precipitation features (imputation comes later)
    if "Precipitation(in)" in df_work.columns:
        df_work["Precipitation(in)"] = to_numeric_safe(df_work["Precipitation(in)"]).clip(lower=0).astype("float32")
    else:
        df_work["Precipitation(in)"] = np.float32(0.0)
    df_work["Precipitation_log"] = np.log1p(df_work["Precipitation(in)"]).astype("float32")
    pcfg = fe_levels["Precipitation_cat"]
    try:
        df_work["Precipitation_cat"] = pd.cut(
            df_work["Precipitation(in)"], bins=pcfg["bins"], labels=pcfg["labels"],
            include_lowest=True, right=True
        )
        if isinstance(df_work["Precipitation_cat"].dtype, CategoricalDtype):
            df_work["Precipitation_cat"] = df_work["Precipitation_cat"].cat.set_categories(
                _as_category_list(pcfg["labels"]), ordered=True
            )
    except (ValueError, TypeError):
        df_work["Precipitation_cat"] = pd.Categorical(
            ["Dry"] * len(df_work),
            categories=_as_category_list(pcfg["labels"]),
            ordered=True
        )

    # Wind numeric casts (group stats will fill)
    df_work["Wind_Speed(mph)"] = to_numeric_safe(df_work.get("Wind_Speed(mph)", pd.Series(np.nan, index=df_work.index))).astype("float32")
    df_work["Wind_Chill(F)"]   = to_numeric_safe(df_work.get("Wind_Chill(F)",   pd.Series(np.nan, index=df_work.index))).astype("float32")

    # Ordered categoricals matching TRAIN order
    if "Weather_ConditionGroup" in df_work.columns:
        w_order = fe_levels["Weather_ConditionGroup_order"]
        w_str = df_work["Weather_ConditionGroup"].astype("string")
        uniq_levels = list(pd.Series(w_str.dropna().unique()))
        ordered_levels = [lvl for lvl in w_order if lvl in uniq_levels] + [lvl for lvl in uniq_levels if lvl not in w_order]
        df_work["Weather_ConditionGroup"] = pd.Categorical(w_str, categories=ordered_levels, ordered=True)
        df_work["Weather_ConditionGroup_ord"] = df_work["Weather_ConditionGroup"].cat.codes.astype("int16")

    if "Distance_group" in df_work.columns:
        dcfg_labels = _as_category_list(fe_levels["Distance_group"]["labels"])
        if isinstance(df_work["Distance_group"].dtype, CategoricalDtype):
            cats_now = list(df_work["Distance_group"].cat.categories)
            final_order = [x for x in dcfg_labels if x in cats_now] + [x for x in cats_now if x not in dcfg_labels]
            df_work["Distance_group"] = df_work["Distance_group"].cat.reorder_categories(final_order, ordered=True)
        else:
            df_work["Distance_group"] = pd.Categorical(
                df_work["Distance_group"].astype("string"),
                categories=dcfg_labels,
                ordered=True
            )
        df_work["Distance_group_ord"] = df_work["Distance_group"].cat.codes.astype("int16")

    if "Visibility_2cat" in df_work.columns:
        v_order = _as_category_list(fe_levels["Visibility_2cat_order"])
        v_str = df_work["Visibility_2cat"].astype("string")
        uniq_v = list(pd.Series(v_str.dropna().unique()))
        ord_v = [lvl for lvl in v_order if lvl in uniq_v] + [lvl for lvl in uniq_v if lvl not in v_order]
        df_work["Visibility_2cat"] = pd.Categorical(v_str, categories=ord_v, ordered=True)
        df_work["Visibility_2cat_ord"] = df_work["Visibility_2cat"].cat.codes.astype("int16")

    if "Severity_bin" in df_work.columns:
        df_work["Severity_bin"] = pd.Categorical(
            df_work["Severity_bin"].astype("Int64"),
            categories=[0, 1],
            ordered=True
        )
        df_work["Severity_bin_ord"] = df_work["Severity_bin"].cat.codes.astype("int16")

# =============================
# Group stats & Random-Weighted pools
# =============================
def build_group_stats(train_df: pd.DataFrame) -> Dict:
    defaults_map = {"Precipitation(in)": 0.0, "Wind_Speed(mph)": 0.0, "Wind_Chill(F)": 0.0}
    stats: Dict[str, Dict] = {}
    for feature_name, base_default in defaults_map.items():
        if feature_name not in train_df.columns:
            continue
        if set(WR_GROUP_COLS).issubset(train_df.columns):
            med_by_group = (
                train_df.groupby(WR_GROUP_COLS, observed=True)[feature_name]
                        .median(numeric_only=True)
            )
            group_map = {str(_as_tuple_key(key)): float(val) for key, val in med_by_group.items()}
        else:
            group_map = {}
        stats[feature_name] = {
            "default": base_default,
            "global_median": float(pd.to_numeric(train_df[feature_name], errors="coerce").median()),
            "by_group": group_map,
        }
    return stats

def build_rw_pools(train_df: pd.DataFrame) -> Dict[str, Dict[str, np.ndarray]]:
    pools: Dict[str, Dict[str, np.ndarray]] = {}
    targets = ["Precipitation(in)", "Wind_Speed(mph)", "Wind_Chill(F)"]
    for feature in targets:
        if feature not in train_df.columns:
            continue
        feature_vals = pd.to_numeric(train_df[feature], errors="coerce")
        df_tmp = train_df.copy()
        df_tmp[feature] = feature_vals

        group_dict: Dict[str, np.ndarray] = {}
        glob = feature_vals.dropna().to_numpy()
        if glob.size > 0:
            group_dict["__GLOBAL__"] = glob.astype(np.float64, copy=False)

        if set(WR_GROUP_COLS).issubset(df_tmp.columns):
            df_g = df_tmp.dropna(subset=WR_GROUP_COLS)
            for key, g in df_g.groupby(WR_GROUP_COLS, observed=True):
                key_str = str(_as_tuple_key(key))
                v = pd.to_numeric(g[feature], errors="coerce").dropna().to_numpy()
                if v.size > 0:
                    group_dict[key_str] = v.astype(np.float64, copy=False)

        pools[feature] = group_dict
    return pools

def apply_group_distributions_inplace(
    df_work: pd.DataFrame,
    rw_pools_map: Dict[str, Dict[str, np.ndarray]],
    gstats: Dict[str, Dict],
    rng: np.random.Generator
) -> None:
    targets = list(rw_pools_map.keys())
    if not targets:
        for feature_name, meta in gstats.items():
            if feature_name in df_work.columns:
                s = pd.to_numeric(df_work[feature_name], errors="coerce")
                df_work[feature_name] = s.fillna(meta.get("global_median", np.nan)).astype("float32")
        return

    have_groups = set(WR_GROUP_COLS).issubset(df_work.columns)
    if have_groups:
        keys_series = list(zip(df_work["Event_Year"], df_work["Event_Month"], df_work["Borough"].astype("string")))
    else:
        keys_series = [("__NOGRP__",)] * len(df_work)

    for feature_name, pools_map in rw_pools_map.items():
        if feature_name not in df_work.columns:
            df_work[feature_name] = np.nan
        s = pd.to_numeric(df_work[feature_name], errors="coerce")
        na_mask = s.isna()
        if not na_mask.any():
            continue

        fill_vals = s.copy()
        global_med = gstats.get(feature_name, {}).get("global_median", np.nan)
        idx_na = np.where(na_mask.to_numpy())[0]

        for i in idx_na:
            if have_groups:
                k_str = str(tuple(keys_series[i]))
            else:
                k_str = "__NOGRP__"

            pool = pools_map.get(k_str)
            if pool is not None and pool.size > 0:
                fill_vals.iat[i] = rng.choice(pool, replace=True)
            else:
                glob_pool = pools_map.get("__GLOBAL__")
                if glob_pool is not None and glob_pool.size > 0:
                    fill_vals.iat[i] = rng.choice(glob_pool, replace=True)
                else:
                    fill_vals.iat[i] = global_med

        df_work[feature_name] = fill_vals.astype("float32")

# =============================
# Schema & enforcement
# =============================
def build_schema_from_df(df_in: pd.DataFrame) -> Dict:
    schema: Dict[str, Dict] = {}
    for column_name in df_in.columns:
        dtype_str = str(df_in.dtypes[column_name])
        entry: Dict[str, Any] = {"dtype": dtype_str}
        if isinstance(df_in[column_name].dtype, CategoricalDtype):
            entry["ordered"] = bool(df_in[column_name].cat.ordered)
            entry["categories"] = df_in[column_name].cat.categories.astype("string").tolist()
        schema[column_name] = entry
    return schema

def _cast_to_dtype(series_in: pd.Series, target_dtype: object) -> pd.Series:
    try:
        dtype_obj = pandas_dtype(target_dtype)
    except (TypeError, ValueError):
        return series_in
    if is_datetime64_any_dtype(dtype_obj):
        return pd.to_datetime(series_in, errors="coerce")
    if is_integer_dtype(dtype_obj) or is_float_dtype(dtype_obj):
        numeric = pd.to_numeric(series_in, errors="coerce")
        try:
            return numeric.astype(dtype_obj)
        except (TypeError, ValueError):
            return numeric
    if is_bool_dtype(dtype_obj):
        try:
            return series_in.astype(dtype_obj)
        except (TypeError, ValueError):
            return series_in.astype("boolean")
    if is_string_dtype(dtype_obj):
        try:
            return series_in.astype(dtype_obj)
        except (TypeError, ValueError):
            return series_in.astype("string")
    if isinstance(dtype_obj, CategoricalDtype):
        return series_in
    try:
        return series_in.astype(dtype_obj)
    except (TypeError, ValueError):
        return series_in

def enforce_schema(df_in: pd.DataFrame, schema: dict) -> pd.DataFrame:
    aligned = df_in.copy()
    for column_name, spec in schema.items():
        if column_name not in aligned.columns:
            aligned[column_name] = pd.Series([np.nan] * len(aligned))
        if "categories" in spec:
            cats = _as_category_list(spec.get("categories"))
            aligned[column_name] = pd.Categorical(
                aligned[column_name].astype("string"),
                categories=cats,
                ordered=bool(spec.get("ordered", False)),
            )
        else:
            aligned[column_name] = _cast_to_dtype(aligned[column_name], spec["dtype"])
    extras = [name for name in aligned.columns if name not in schema]
    if extras:
        aligned.drop(columns=extras, inplace=True, errors="ignore")
    return aligned[list(schema.keys())]

# =============================
# Artifacts save/load
# =============================
def save_artifacts(
    train_final_df: pd.DataFrame,
    fe_levels: Dict,
    gstats: Dict,
    rw_pools_map: Dict[str, Dict[str, np.ndarray]],
    knn_imputer: Optional[KNNImputer],
    mice_imputer: Optional[SkIterativeImputer],
    used_knn_cols: List[str],
    used_mice_cols: List[str],
) -> None:
    joblib.dump({"model": knn_imputer, "cols": used_knn_cols},
                os.path.join(ARTIFACT_DIR, "knn_imputer.joblib"))
    joblib.dump({"model": mice_imputer, "cols": used_mice_cols},
                os.path.join(ARTIFACT_DIR, "mice_imputer.joblib"))
    joblib.dump(rw_pools_map, os.path.join(ARTIFACT_DIR, "rw_pools.joblib"))

    with open(os.path.join(ARTIFACT_DIR, "bins_levels.json"), "w", encoding="utf-8") as f_out:
        json.dump(fe_levels, f_out, ensure_ascii=False, indent=2)
    with open(os.path.join(ARTIFACT_DIR, "group_stats.json"), "w", encoding="utf-8") as f_out:
        json.dump(gstats, f_out, ensure_ascii=False, indent=2)
    with open(os.path.join(ARTIFACT_DIR, "drops.json"), "w", encoding="utf-8") as f_out:
        json.dump({"drop_after_impute": DROP_AFTER_IMPUTE}, f_out, ensure_ascii=False, indent=2)

    schema_dict = build_schema_from_df(train_final_df)
    with open(os.path.join(ARTIFACT_DIR, "schema.json"), "w", encoding="utf-8") as f_out:
        json.dump(schema_dict, f_out, ensure_ascii=False, indent=2)

    manifest = {
        "seed": SEED,
        "rows_train": int(train_final_df.shape[0]),
        "pandas": pd.__version__,
        "python": platform.python_version(),

        "notes": "Artifacts for applying TRAIN FE to VALID/TEST (no refit).",
    }

    with open(os.path.join(ARTIFACT_DIR, "manifest.json"), "w", encoding="utf-8") as f_out:
        json.dump(manifest, f_out, ensure_ascii=False, indent=2)

def load_artifacts():
    with open(os.path.join(ARTIFACT_DIR, "bins_levels.json"), "r", encoding="utf-8") as f_in:
        fe_levels = json.load(f_in)
    with open(os.path.join(ARTIFACT_DIR, "group_stats.json"), "r", encoding="utf-8") as f_in:
        gstats = json.load(f_in)
    with open(os.path.join(ARTIFACT_DIR, "drops.json"), "r", encoding="utf-8") as f_in:
        drops = json.load(f_in)["drop_after_impute"]
    with open(os.path.join(ARTIFACT_DIR, "schema.json"), "r", encoding="utf-8") as f_in:
        schema = json.load(f_in)
    knn_obj = joblib.load(os.path.join(ARTIFACT_DIR, "knn_imputer.joblib"))
    mice_obj = joblib.load(os.path.join(ARTIFACT_DIR, "mice_imputer.joblib"))
    rw_pools_map = joblib.load(os.path.join(ARTIFACT_DIR, "rw_pools.joblib"))
    return fe_levels, gstats, drops, schema, knn_obj, mice_obj, rw_pools_map

# =============================
# Apply artifacts to NEW data
# =============================
def apply_artifacts_to_new(
    df_new: pd.DataFrame,
    fe_levels: Dict,
    gstats: Dict,
    rw_pools_map: Dict[str, Dict[str, np.ndarray]],
    drops: List[str],
    schema: Dict,
    knn_obj: Dict,
    mice_obj: Dict
) -> pd.DataFrame:
    df_work = df_new.copy()
    df_work.columns = df_work.columns.str.strip()

    # Core FE (no refit)
    apply_core_fe_inplace(df_work, fe_levels)

    # Driver age normalization
    if "FARS__DRIVER_AGE" in df_work.columns:
        df_work["FARS__DRIVER_AGE"] = normalize_driver_age(df_work["FARS__DRIVER_AGE"])

    # Random-weighted per-group fills
    apply_group_distributions_inplace(df_work, rw_pools_map, gstats, RNG)

    # Derivations that depend on imputed values
    df_work["Precipitation_log"] = np.log1p(df_work["Precipitation(in)"]).astype("float32")
    pcfg = fe_levels["Precipitation_cat"]
    try:
        df_work["Precipitation_cat"] = pd.cut(
            df_work["Precipitation(in)"], bins=pcfg["bins"], labels=pcfg["labels"],
            include_lowest=True, right=True
        )
        if isinstance(df_work["Precipitation_cat"].dtype, CategoricalDtype):
            df_work["Precipitation_cat"] = df_work["Precipitation_cat"].cat.set_categories(
                _as_category_list(pcfg["labels"]), ordered=True
            )
    except (ValueError, TypeError):
        df_work["Precipitation_cat"] = pd.Categorical(
            ["Dry"] * len(df_work),
            categories=_as_category_list(pcfg["labels"]),
            ordered=True
        )

    # KNN/MICE (transform only; same column order used during fit)
    knn_cols_art: List[str] = knn_obj.get("cols", [])
    mice_cols_art: List[str] = mice_obj.get("cols", [])
    knn_model: Optional[KNNImputer] = knn_obj.get("model")
    mice_model: Optional[SkIterativeImputer] = mice_obj.get("model")

    if knn_model is not None and len(knn_cols_art) > 0 and all(name in df_work.columns for name in knn_cols_art):
        df_work[knn_cols_art] = knn_model.transform(df_work[knn_cols_art].astype("float64"))

    if mice_model is not None and len(mice_cols_art) >= 2 and all(name in df_work.columns for name in mice_cols_art):
        df_work[mice_cols_art] = mice_model.transform(df_work[mice_cols_art].astype("float64"))

    # Drops (same as TRAIN)
    cols_exist = [name for name in drops if name in df_work.columns]
    if cols_exist:
        df_work.drop(columns=cols_exist, inplace=True, errors="ignore")

    # Convert remaining object columns to category
    for obj_col in df_work.select_dtypes(include="object").columns:
        df_work[obj_col] = df_work[obj_col].astype("category")

    # Align to TRAIN schema
    df_aligned = enforce_schema(df_work, schema)
    return df_aligned

# =============================
# Diagnostics (optional, print)
# =============================
def print_final_summary(tag: str, df_in: pd.DataFrame) -> None:
    print(f"\n=== .info() AFTER FE ({tag}) ===")
    df_in.info()
    dtype_labels = [str(t) for t in df_in.dtypes]
    dtype_counts = Counter(dtype_labels)
    print("\nType distribution:")
    for dtype_name, count_val in dtype_counts.items():
        print(f"  {dtype_name:20} : {count_val:3} columns")

# =============================
# MAIN
# =============================
if __name__ == "__main__":
    # 1) Load TRAIN raw
    train_raw_df = load_dataframe_any(TRAIN_PATH)

    # 2) Build TRAIN levels and apply core FE (no refit components here)
    train_levels = build_bins_and_levels(train_raw_df)
    apply_core_fe_inplace(train_raw_df, train_levels)

    # 2b) Normalize driver age on TRAIN before fitting imputers
    if "FARS__DRIVER_AGE" in train_raw_df.columns:
        train_raw_df["FARS__DRIVER_AGE"] = normalize_driver_age(train_raw_df["FARS__DRIVER_AGE"])

    # 3) Build Random-Weighted pools + group stats; apply RW to TRAIN
    train_group_stats = build_group_stats(train_raw_df)
    rw_pools_train = build_rw_pools(train_raw_df)
    apply_group_distributions_inplace(train_raw_df, rw_pools_train, train_group_stats, RNG)

    # 4) Post-imputation derivations (mirror TRAIN)
    train_raw_df["Precipitation_log"] = np.log1p(train_raw_df["Precipitation(in)"]).astype("float32")
    train_precip_cfg = build_bins_and_levels(train_raw_df)["Precipitation_cat"]
    try:
        train_raw_df["Precipitation_cat"] = pd.cut(
            train_raw_df["Precipitation(in)"],
            bins=train_precip_cfg["bins"],
            labels=train_precip_cfg["labels"],
            include_lowest=True, right=True
        )
        if isinstance(train_raw_df["Precipitation_cat"].dtype, CategoricalDtype):
            train_raw_df["Precipitation_cat"] = train_raw_df["Precipitation_cat"].cat.set_categories(
                _as_category_list(train_precip_cfg["labels"]), ordered=True
            )
    except (ValueError, TypeError):
        train_raw_df["Precipitation_cat"] = pd.Categorical(
            ["Dry"] * len(train_raw_df),
            categories=_as_category_list(train_precip_cfg["labels"]),
            ordered=True
        )

    # 5) Fit imputers on TRAIN — using only columns that actually exist
    used_knn_cols_fit = [c for c in KNN_COLS if c in train_raw_df.columns]
    knn_imputer_model: Optional[KNNImputer] = None
    if len(used_knn_cols_fit) > 0:
        knn_imputer_model = KNNImputer(n_neighbors=5, weights="distance")
        knn_imputer_model.fit(train_raw_df[used_knn_cols_fit].astype("float64"))
    else:
        print("⚠️ KNNImputer skipped: no KNN columns available in TRAIN.")

    used_mice_cols_fit = [c for c in MICE_COLS if c in train_raw_df.columns]
    mice_imputer_model: Optional[SkIterativeImputer] = None
    if len(used_mice_cols_fit) >= 2:
        mice_imputer_model = SkIterativeImputer(
            max_iter=10, random_state=SEED, initial_strategy="median",
            sample_posterior=False, skip_complete=True
        )
        mice_imputer_model.fit(train_raw_df[used_mice_cols_fit].astype("float64"))
    else:
        print(f"⚠️ IterativeImputer (MICE) skipped: need ≥2 cols, have {len(used_mice_cols_fit)} — {used_mice_cols_fit}")

    # 6) TRAIN drops (exact list), then convert object→category
    existing_to_drop_train = [name for name in DROP_AFTER_IMPUTE if name in train_raw_df.columns]
    if existing_to_drop_train:
        train_raw_df.drop(columns=existing_to_drop_train, inplace=True, errors="ignore")
    for object_col_train in train_raw_df.select_dtypes(include="object").columns:
        train_raw_df[object_col_train] = train_raw_df[object_col_train].astype("category")

    # 7) Save artifacts from TRAIN (incl. RW pools + actual col lists)
    save_artifacts(
        train_final_df=train_raw_df,
        fe_levels=train_levels,
        gstats=train_group_stats,
        rw_pools_map=rw_pools_train,
        knn_imputer=knn_imputer_model,
        mice_imputer=mice_imputer_model,
        used_knn_cols=used_knn_cols_fit,
        used_mice_cols=used_mice_cols_fit,
    )

    # 8) Load artifacts (single source of truth for application)
    fe_levels_art, gstats_art, drops_art, schema_art, knn_art, mice_art, rw_pools_art = load_artifacts()

    # 9a) TRAIN: re-load raw → apply artifacts (same as VALID/TEST) → export
    train_raw_df_for_export = load_dataframe_any(TRAIN_PATH)
    train_fe_df = apply_artifacts_to_new(
        df_new=train_raw_df_for_export,
        fe_levels=fe_levels_art,
        gstats=gstats_art,
        rw_pools_map=rw_pools_art,
        drops=drops_art,
        schema=schema_art,
        knn_obj=knn_art,
        mice_obj=mice_art,
    )
    export_dataset(train_fe_df, "train_fe", tag="TRAIN")

    # 9b) VALID: load → apply artifacts → export
    valid_raw_df = load_dataframe_any(VALID_PATH)
    valid_fe_df = apply_artifacts_to_new(
        df_new=valid_raw_df,
        fe_levels=fe_levels_art,
        gstats=gstats_art,
        rw_pools_map=rw_pools_art,
        drops=drops_art,
        schema=schema_art,
        knn_obj=knn_art,
        mice_obj=mice_art,
    )
    export_dataset(valid_fe_df, "valid_fe", tag="VALID")

    # 9c) TEST: load → apply artifacts → export
    test_raw_df = load_dataframe_any(TEST_PATH)
    test_fe_df = apply_artifacts_to_new(
        df_new=test_raw_df,
        fe_levels=fe_levels_art,
        gstats=gstats_art,
        rw_pools_map=rw_pools_art,
        drops=drops_art,
        schema=schema_art,
        knn_obj=knn_art,
        mice_obj=mice_art,
    )
    export_dataset(test_fe_df, "test_fe", tag="TEST")

    print("\n✅ Done. Artifacts:", ARTIFACT_DIR)
    print("✅ Outputs:", OUT_DIR)
