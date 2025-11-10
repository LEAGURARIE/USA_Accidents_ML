# ==========================================
# Data Cleaning + EDA + Stats + Train/Val/Test Split
# ==========================================
from __future__ import annotations

# --- Core imports ---
import os
import math
import re
from datetime import datetime
from typing import Optional, cast, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle

# sklearn
from sklearn.model_selection import train_test_split

# pandas dtype helpers
from pandas.api.types import (
    CategoricalDtype,
    is_numeric_dtype,
    is_bool_dtype,
    is_object_dtype,
    is_string_dtype,
)

# ==============================
# Config: IO paths (ALL OUTPUTS)
# ==============================
CSV_PATH = r"C:\Users\LEA GUR-ARIE\PycharmProjects\USA_Accidents_ML\src\data\raw\interim\output_data\post_eda.csv"
OUTPUT_DIR = r"C:\Users\LEA GUR-ARIE\PycharmProjects\USA_Accidents_ML\src\data\raw\interim\EDA1_plot"
OUTPUT_DIR_SPLIT = r"C:\Users\LEA GUR-ARIE\PycharmProjects\USA_Accidents_ML\src\data\raw\interim\output_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# EDA image subfolders
HISTS_DIR = os.path.join(OUTPUT_DIR, "histograms_numeric")
BOXES_DIR = os.path.join(OUTPUT_DIR, "boxplots_numeric")
BARS_DIR = os.path.join(OUTPUT_DIR, "barplots_categorical")
for _subdir in (HISTS_DIR, BOXES_DIR, BARS_DIR):
    os.makedirs(_subdir, exist_ok=True)

# EDA tables main Excel
EDA_XLSX_NAME = "analysis_output.xlsx"
EDA_XLSX_PATH = os.path.join(OUTPUT_DIR, EDA_XLSX_NAME)

# Image outputs
IMG_HEAT_SPEARMAN = os.path.join(OUTPUT_DIR, "spearman_heatmap.png")
IMG_HEAT_CRAMERS = os.path.join(OUTPUT_DIR, "categorical_cramersV_heatmap.png")
IMG_ALL_NUM_BOX = os.path.join(OUTPUT_DIR, "boxplot_all_numeric_grid.png")
IMG_PAIRPLOT_NUM = os.path.join(OUTPUT_DIR, "pairplot_numeric_grid.png")

# Statistical results
TESTS_XLSX = os.path.join(OUTPUT_DIR, "test_results.xlsx")
TESTS_CSV_SUMMARY = os.path.join(OUTPUT_DIR, "test_results_summary.csv")
TESTS_CSV_SIGNIF = os.path.join(OUTPUT_DIR, "test_results_significant.csv")

# Full post-cleaning CSV
POST_CLEAN_CSV = os.path.join(OUTPUT_DIR, "post_cleaning.csv")

# Split outputs
SPLIT_DIR = os.path.join(OUTPUT_DIR_SPLIT, "split")
os.makedirs(SPLIT_DIR, exist_ok=True)
TRAIN_OUT_PATH = os.path.join(SPLIT_DIR, "train.csv")
VAL_OUT_PATH = os.path.join(SPLIT_DIR, "val.csv")
TEST_OUT_PATH = os.path.join(SPLIT_DIR, "test.csv")

# ==================
# Plot/group settings
# ==================
TOP_K = 15
MAX_LEVELS_CRAMERS = 30
INCLUDE_BOOLS_IN_SPEARMAN = True
MAX_PAIRPLOT_VARS = 12

# =========
# RNG setup
# =========
SEED = 42
RNG_GLOBAL = np.random.default_rng(SEED)

# =========
# Split ratios
# =========
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15
assert abs((TRAIN_FRAC + VAL_FRAC + TEST_FRAC) - 1.0) < 1e-9, "Fractions must sum to 1.0"
RANDOM_STATE = 42

# =========
# Utilities
# =========
def ensure_fig_saved_close(path: str, dpi: int = 150) -> None:
    try:
        plt.tight_layout()
    except (RuntimeError, ValueError):
        pass
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()

def _safe_name(name: str) -> str:
    cleaned = re.sub(r"[^\w\-]+", "_", str(name))
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned[:120]

def summarize_numeric(df_num: pd.DataFrame) -> pd.DataFrame:
    rows_out: list[dict[str, Any]] = []
    for colname_num in map(str, df_num.columns):
        s_vals = pd.to_numeric(df_num[colname_num], errors="coerce")
        n_total = int(s_vals.size)
        n_nonnull = int(s_vals.notna().sum())
        n_missing = n_total - n_nonnull
        pct_missing = (n_missing / n_total) * 100.0 if n_total else 0.0
        mean_val = float(s_vals.mean()) if n_nonnull else np.nan
        std_val = float(s_vals.std()) if n_nonnull else np.nan
        se_mean = float(std_val / np.sqrt(max(1, n_nonnull))) if (n_nonnull > 0 and np.isfinite(std_val)) else np.nan
        p01 = p05 = q1 = med = q3 = p95 = p99 = np.nan
        iqr_val = lower = upper = np.nan
        n_out = 0
        pct_out_iqr = np.nan
        if n_nonnull > 0:
            q = s_vals.quantile([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
            p01 = float(q.get(0.01, np.nan)); p05 = float(q.get(0.05, np.nan))
            q1 = float(q.get(0.25, np.nan));  med = float(q.get(0.50, np.nan)); q3 = float(q.get(0.75, np.nan))
            p95 = float(q.get(0.95, np.nan)); p99 = float(q.get(0.99, np.nan))
            if np.isfinite(q1) and np.isfinite(q3):
                iqr_val = q3 - q1; lower = q1 - 1.5 * iqr_val; upper = q3 + 1.5 * iqr_val
                if np.isfinite(lower) and np.isfinite(upper):
                    is_out = (s_vals < lower) | (s_vals > upper)
                    n_out = int(is_out.sum()); pct_out_iqr = (n_out / n_nonnull) * 100.0
        rows_out.append({
            "column": colname_num, "dtype": str(df_num[colname_num].dtype),
            "count": n_total, "non_null": n_nonnull, "missing": n_missing, "missing_pct": pct_missing,
            "mean": mean_val, "std": std_val, "se_mean": se_mean,
            "min": float(s_vals.min()) if n_nonnull else np.nan,
            "p01": p01, "p05": p05, "q1": q1, "median": med, "q3": q3, "p95": p95, "p99": p99,
            "max": float(s_vals.max()) if n_nonnull else np.nan,
            "iqr": iqr_val, "lower_fence": lower, "upper_fence": upper,
            "n_outliers_iqr": n_out, "pct_outliers_iqr": pct_out_iqr,
        })
    out_df = pd.DataFrame(rows_out).set_index("column").sort_index()
    for c_name in ["missing_pct","mean","std","min","p01","p05","q1","median","q3","p95","p99","max","iqr","lower_fence","upper_fence","pct_outliers_iqr"]:
        if c_name in out_df.columns:
            out_df[c_name] = pd.to_numeric(out_df[c_name], errors="coerce").round(3)
    if "se_mean" in out_df.columns:
        out_df["se_mean"] = pd.to_numeric(out_df["se_mean"], errors="coerce").round(6)
    return out_df

def write_eda_excel_safely(
    xlsx_output_path: str,
    numeric_detailed_tbl: pd.DataFrame,
    cat_dist_tbl: pd.DataFrame,
    cat_missing_tbl: pd.DataFrame,
    corr_spear_tbl: pd.DataFrame,
    cat_heat_matrix_tbl: Optional[pd.DataFrame],
) -> None:
    def _ts_copy_path(base_dir: str, stem: str = "analysis_output") -> str:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S");  return os.path.join(base_dir, f"{stem}_{ts}.xlsx")
    def _do_write(xlsx_file_path: str) -> None:
        with pd.ExcelWriter(xlsx_file_path, engine="xlsxwriter") as xlsx_writer:
            numeric_detailed_tbl.to_excel(xlsx_writer, sheet_name="Numeric Detailed")
            if not cat_dist_tbl.empty:
                cat_dist_tbl.sort_values(["variable","percent"], ascending=[True,False]).to_excel(xlsx_writer, sheet_name="Categorical Dist (%)", index=False)
            if not cat_missing_tbl.empty:
                cat_missing_tbl.to_excel(xlsx_writer, sheet_name="Categorical Missing", index=False)
            if not corr_spear_tbl.empty:
                corr_spear_tbl.to_excel(xlsx_writer, sheet_name="Spearman Corr")
            if cat_heat_matrix_tbl is not None and not cat_heat_matrix_tbl.empty:
                cat_heat_matrix_tbl.to_excel(xlsx_writer, sheet_name="CramersV (Cat)")
    try:
        _do_write(xlsx_output_path); print("✅ EDA Excel saved to:", xlsx_output_path)
    except PermissionError:
        alt = _ts_copy_path(OUTPUT_DIR, stem=os.path.splitext(os.path.basename(xlsx_output_path))[0]); _do_write(alt)
        print("⚠️ Excel was locked. Wrote a timestamped copy:", alt)
    except (OSError, ValueError) as e_xlsx:
        alt = _ts_copy_path(OUTPUT_DIR, stem="analysis_output_fallback")
        try:
            _do_write(alt); print(f"⚠️ Excel write issue ({e_xlsx}). Wrote fallback:", alt)
        except (IOError, OSError, PermissionError) as e2:
            print("❌ Failed to write Excel outputs:", e2)

def print_bin_dist(title_name: str, series_in: pd.Series) -> None:
    vc_abs = series_in.value_counts(dropna=False).sort_index()
    vc_pct = (series_in.value_counts(dropna=False, normalize=True).sort_index() * 100).round(2)
    joined = pd.concat([vc_abs.rename("count"), vc_pct.rename("percent")], axis=1)
    print(f"{title_name:<6} | Severity_bin distribution:\n{joined}\n")

def check_stratify_okay(series_in: pd.Series, min_per_class: int = 2) -> None:
    vc = series_in.value_counts(dropna=False)
    if vc.size < 2:
        raise ValueError("Severity_bin contains single class — stratified split not possible.")
    if (vc < min_per_class).any():
        raise ValueError(f"One class has < {min_per_class} samples. Counts: {vc.to_dict()}")

def save_multi_boxplot_grid(df_num_grid: pd.DataFrame, out_file_path: str, ncols: int = 4) -> None:
    cols_present = [c for c in df_num_grid.columns if pd.to_numeric(df_num_grid[c], errors="coerce").dropna().size > 0]
    if not cols_present: return
    n_vars = len(cols_present); nrows = max(1, math.ceil(n_vars / ncols))
    fig_grid, axes_arr = plt.subplots(nrows, ncols, figsize=(ncols * 5.0, nrows * 4.0), squeeze=False)
    axes_flat: list[Axes] = [cast(Axes, ax) for ax in axes_arr.flatten()]
    for idx, varname in enumerate(cols_present):
        ax = axes_flat[idx]
        try:
            sns.boxplot(y=pd.to_numeric(df_num_grid[varname], errors="coerce").dropna(), ax=ax, orient="v", showfliers=True)
            ax.set_title(str(varname), fontsize=10); ax.set_xlabel(""); ax.set_ylabel(""); ax.grid(True, axis="y", alpha=0.3)
        except (ValueError, TypeError):
            ax.axis("off")
    for idx in range(len(cols_present), len(axes_flat)): axes_flat[idx].axis("off")
    try: fig_grid.tight_layout()
    except (RuntimeError, ValueError): pass
    fig_grid.savefig(out_file_path, dpi=160, bbox_inches="tight"); plt.close(fig_grid)

# SciPy availability
try:
    from scipy.stats import (
        chi2_contingency as chi2_contingency_stats,
        fisher_exact, spearmanr, kendalltau, mannwhitneyu, kruskal
    )
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False
    chi2_contingency_stats = None  # type: ignore
    fisher_exact = None  # type: ignore
    spearmanr = None  # type: ignore
    kendalltau = None  # type: ignore
    mannwhitneyu = None  # type: ignore
    kruskal = None  # type: ignore
    print("⚠️ SciPy not found — categorical tests and Cramér's V may be limited.")

# =================
# 1) Load the data
# =================
if not os.path.exists(CSV_PATH):
    raise IOError(f"Input CSV not found: {CSV_PATH}")
df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.strip()
print("✅ Data loaded:", CSV_PATH)

# =====================================
# 2) Drop unwanted columns (if present)
# =====================================
drop_candidates = [
    "ID","Street","Zipcode","City","County","State","Country","Timezone",
    "Turning_Loop","Roundabout","Zipcode_clean","Zip_code_clean","Bamp",
    "FARS__DRUGRES1","FARS__DRUGRES2","FARS__DRUGRES3",
    "FARS__FARS_LAT","FARS__FARS_LON",
    "FARS__MONTH","FARS__HOUR","FARS__MINUTE","FARS__STATE","FARS__DAY","FARS__YEAR",
    "Weather_Timestamp","match_distance_km",
    "FARS__ST_CASE","FARS__VEH_NO","FARS__MOD_YEAR",
    "Wind_Direction","Weather_Condition","Description",
]
cols_found = [c for c in drop_candidates if c in df.columns]
if cols_found:
    df.drop(columns=cols_found, inplace=True, errors="ignore")
    print(f"🧹 Dropped columns: {cols_found}")

# ===========================================
# 3) Type casts + feature engineering (clean)
#    (clean driver age & alcohol BEFORE categorization)
# ===========================================
TIME_COLS = [c for c in ("Start_Time","End_Time") if c in df.columns]
obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
obj_cols = [c for c in obj_cols if c not in TIME_COLS]
if obj_cols:
    try: df[obj_cols] = df[obj_cols].astype("category")
    except (TypeError, ValueError) as e: print(f"Warning: Could not convert object columns to category: {e}")

# Driver age
if "FARS__DRIVER_AGE" in df.columns:
    age = pd.to_numeric(df["FARS__DRIVER_AGE"], errors="coerce")
    INVALID_AGE_CODES = {0, 997, 998, 999}
    age = age.mask(age.isin(INVALID_AGE_CODES) | (age < 14) | (age > 100), np.nan)
    df["FARS__DRIVER_AGE"] = age.astype("float32")

# Visibility features
if "Visibility(mi)" in df.columns:
    vis_vals = pd.to_numeric(df["Visibility(mi)"], errors="coerce")
    df["Visibility_is_low"] = pd.Series(np.where(vis_vals.isna(), pd.NA, vis_vals <= 3.0), dtype="boolean")
    vis_cat = pd.Series("Normal", index=df.index, dtype="object")
    vis_cat[vis_vals <= 3.0] = "Low"; vis_cat[vis_vals.isna()] = "Missing"
    df["Visibility_2cat"] = pd.Categorical(vis_cat, categories=["Low","Normal","Missing"], ordered=False)

# Boolean casts
_raw_bool_vars = ["Amenity","Bump","Crossing","Give_Way","Junction","No_Exit","Railway","Station",
                  "Stop","Traffic_Calming","Traffic_Signal","Turning_Loop","Street_is_highway",
                  "Start_Time_isna","End_Time_isna","Roundabout"]
for b in [v.strip() for v in _raw_bool_vars]:
    if b in df.columns:
        try: df[b] = df[b].astype("boolean")
        except (TypeError, ValueError) as e:
            try: df[b] = pd.Series(df[b]).astype("boolean")
            except (TypeError, ValueError): print(f"Warning: Could not convert {b} to boolean: {e}")

# Twilight → boolean
for tw in ["Sunrise_Sunset","Civil_Twilight","Nautical_Twilight","Astronomical_Twilight"]:
    if tw in df.columns:
        df[tw] = df[tw].astype("string").str.strip().map({"Day": True, "Night": False}).astype("boolean")

# Airport_Code: fill median category
if "Airport_Code" in df.columns:
    ac = "Airport_Code"
    df[ac] = df[ac].astype("category")
    freq = df[ac].value_counts(dropna=True, normalize=True).sort_values(ascending=False)
    csum = freq.cumsum()
    if len(csum[csum >= 0.5]) > 0:
        median_cat = csum.index[(csum >= 0.5)][0]
        if median_cat not in df[ac].cat.categories:
            df[ac] = df[ac].cat.add_categories([median_cat])
        df[ac] = df[ac].fillna(median_cat)
        print("Filled Airport_Code with median category:", median_cat)

# FARS__BODY_TYP → grouped labels
if "FARS__BODY_TYP" in df.columns:
    bt = pd.to_numeric(df["FARS__BODY_TYP"], errors="coerce").astype("Int64")
    passenger_car = {1,2,3,4,5,6,7,8,9,17}
    suv_cuv = {14,15,16,19}
    vans = {20,21,22,28,29,41,42,48}
    pickups_lt = {10,11,30,31,32,33,34,39,40,45,49}
    buses = {50,51,52,55,58,59,60}
    mhd_trucks = {61,62,63,64,65,66,67,71,72,73,78,79}
    motorcycles = {80,81,82,83,84,85,86,87}
    other_spec = {90,91,92,93,94,95,96,97}
    unknown_codes = {98,99}
    code_to_label = {}
    code_to_label.update({k:"Passenger Car" for k in passenger_car})
    code_to_label.update({k:"SUV/CUV" for k in suv_cuv})
    code_to_label.update({k:"Van" for k in vans})
    code_to_label.update({k:"Pickup/Light Truck" for k in pickups_lt})
    code_to_label.update({k:"Bus" for k in buses})
    code_to_label.update({k:"Medium/Heavy Truck" for k in mhd_trucks})
    code_to_label.update({k:"Motorcycle/3-Wheel" for k in motorcycles})
    code_to_label.update({k:"Other Special Vehicle" for k in other_spec})
    code_to_label.update({k:"Missing" for k in unknown_codes})
    bt_labels = bt.map(code_to_label).where(bt.notna(), "Missing").fillna("Missing")
    bt_cats = ["Passenger Car","SUV/CUV","Van","Pickup/Light Truck","Bus","Medium/Heavy Truck","Motorcycle/3-Wheel","Other Special Vehicle","Missing"]
    df["FARS__BODY_TYP_Group"] = pd.Categorical(bt_labels, categories=bt_cats, ordered=False)

# FARS__SEX → 3cat (clean)
if "FARS__SEX" in df.columns:
    sex_num = pd.to_numeric(df["FARS__SEX"], errors="coerce")
    VALID_SEX_CODES = {1,2}
    sex_num.loc[~sex_num.isin(VALID_SEX_CODES)] = np.nan
    df["FARS__SEX_3cat"] = sex_num.map({1:1, 2:2}).fillna(3).astype("Int64")
    df["FARS__SEX_label"] = df["FARS__SEX_3cat"].map({1:"Male", 2:"Female", 3:"Missing"})
    print("\nFARS__SEX distribution (no IQR removal):")
    print(df["FARS__SEX_label"].value_counts(dropna=False))

# FARS__ALC_RES → clean then 3 categories, then DROP raw column to prevent leakage
if "FARS__ALC_RES" in df.columns:
    alc_raw = pd.to_numeric(df["FARS__ALC_RES"], errors="coerce")
    INVALID_ALC_CODES = {995,996,997,998,999}
    alc = alc_raw.mask(alc_raw.isin(INVALID_ALC_CODES))
    alc_cat = pd.Series("Missing", index=df.index, dtype="object")
    alc_cat[alc.notna() & (alc <= 0)] = "No Alcohol"
    alc_cat[alc.notna() & (alc > 0)]  = "Alcohol Present"
    df["ALC_category"] = pd.Categorical(alc_cat, categories=["No Alcohol","Alcohol Present","Missing"], ordered=False)
    # Drop raw predictor to avoid target leakage later
    df.drop(columns=["FARS__ALC_RES"], inplace=True, errors="ignore")
    print("\nALC_category created and 'FARS__ALC_RES' dropped.")

print("\n✅ Cleaning & feature-engineering complete.")

# ===========================================
# Create Severity_2cat only
# ===========================================
if "Severity" not in df.columns:
    raise KeyError("Column 'Severity' not found; cannot build Severity categories")
_sev_num = pd.to_numeric(df["Severity"], errors="coerce").astype("Int64")
sev2 = pd.Series("Missing", index=df.index, dtype="object")
sev2[_sev_num.isin([1, 2])] = "Low (1–2)"
sev2[_sev_num.isin([3, 4])] = "High (3–4)"
df["Severity_2cat"] = pd.Categorical(sev2, categories=["Low (1–2)","High (3–4)","Missing"], ordered=False)
print("✅ Added 'Severity_2cat' categorical variable.")

# ===================================
# 4) EDA (plots + tables + heatmaps)
# ===================================
num_cols: list[str] = df.select_dtypes(include=[np.number]).columns.tolist()
bool_cols: list[str] = df.select_dtypes(include=["bool", "boolean"]).columns.tolist()
cat_cols: list[str] = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()

# ---- Spearman (numeric + boolean) ----
corr_spear = pd.DataFrame()
df_corr_input = pd.DataFrame(index=df.index)
if num_cols: df_corr_input = df[num_cols].copy()
if INCLUDE_BOOLS_IN_SPEARMAN and bool_cols:
    for bcol in bool_cols: df_corr_input[bcol] = df[bcol].astype("Int64")
if not df_corr_input.empty and df_corr_input.shape[1] >= 2:
    try:
        corr_spear = df_corr_input.corr(method="spearman")
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_spear, square=False, linewidths=0.5, cmap="coolwarm", center=0)
        plt.title("Spearman Correlation (Numeric + Boolean)")
        ensure_fig_saved_close(IMG_HEAT_SPEARMAN, dpi=150)
        print("✅ Spearman heatmap saved:", IMG_HEAT_SPEARMAN)
    except (ValueError, TypeError, RuntimeError) as e:
        print("⚠️ Spearman heatmap skipped:", e)

# ---- Numeric histograms & boxplots ----
for nc in num_cols:
    s_num = pd.to_numeric(df[nc], errors="coerce").dropna()
    p_hist = os.path.join(HISTS_DIR, f"{_safe_name(nc)}_hist.png")
    if s_num.empty:
        fig = plt.figure(figsize=(8, 6)); plt.title(f"{nc} — (no non-null values)"); plt.axis("off")
        fig.savefig(p_hist, dpi=140, bbox_inches="tight"); plt.close(fig)
    else:
        try:
            plt.figure(figsize=(8, 6)); sns.histplot(s_num, kde=True, bins=30)
            plt.title(f"Distribution of {nc}"); plt.xlabel(nc); plt.ylabel("Density")
            ensure_fig_saved_close(p_hist, dpi=150)
        except (ValueError, TypeError, RuntimeError) as e:
            print(f"⚠️ Histogram skipped for {nc}:", e)
    p_box = os.path.join(BOXES_DIR, f"{_safe_name(nc)}_box.png")
    if s_num.empty:
        fig = plt.figure(figsize=(8, 3)); plt.title(f"{nc} — (no non-null values)"); plt.axis("off")
        fig.savefig(p_box, dpi=140, bbox_inches="tight"); plt.close(fig)
    else:
        try:
            plt.figure(figsize=(10, 2.8)); sns.boxplot(x=s_num, orient="h", showfliers=True)
            plt.title(f"Boxplot of {nc}"); plt.xlabel(nc)
            ensure_fig_saved_close(p_box, dpi=150)
        except (ValueError, TypeError, RuntimeError) as e:
            print(f"⚠️ Boxplot skipped for {nc}:", e)

# ---- Grid of all numeric boxplots ----
if num_cols:
    df_num_for_grid = df[num_cols].dropna(how="all", axis=1)
    if not df_num_for_grid.empty:
        save_multi_boxplot_grid(df_num_for_grid, IMG_ALL_NUM_BOX, ncols=4)
        print("✅ Boxplot grid saved:", IMG_ALL_NUM_BOX)

# ---- Categorical/boolean bars + tables ----
cat_bool_cols = cat_cols + bool_cols
cat_dist_rows: list[dict[str, Any]] = []
cat_missing_rows: list[dict[str, Any]] = []
for var in cat_bool_cols:
    s = df[var]; total = int(len(s)); nonnull = int(s.notna().sum())
    missing = total - nonnull; miss_pct = (missing / total * 100.0) if total else 0.0
    cat_missing_rows.append({"variable": var, "missing": missing, "missing_pct": round(miss_pct, 3)})
    counts = s.value_counts(dropna=False); total_counts = int(counts.sum())
    p_bar = os.path.join(BARS_DIR, f"{_safe_name(var)}_bar.png")
    if total_counts > 0:
        for level, cnt in counts.items():
            level_label = "NaN" if pd.isna(level) else str(level)
            pct = (cnt / total_counts) * 100.0
            cat_dist_rows.append({"variable": var, "level": level_label, "count": int(cnt), "percent": round(float(pct), 3)})
        counts_plot = counts.copy(); counts_plot.index = counts_plot.index.map(lambda x: "NaN" if pd.isna(x) else str(x))
        perc = (counts_plot.astype(float) / float(total_counts) * 100.0).sort_values(ascending=False)
        if len(perc) > TOP_K:
            top = perc.iloc[:TOP_K - 1]; other_val = float(perc.iloc[TOP_K - 1:].sum())
            perc_plot = pd.concat([top, pd.Series({"OTHER": other_val})])
        else: perc_plot = perc
        try:
            plt.figure(figsize=(10, max(4.0, 0.4 * len(perc_plot))))
            ax_bar = sns.barplot(x=perc_plot.values.astype(float), y=perc_plot.index.astype(str), orient="h")
            ax_bar.set_xlabel("Percent"); ax_bar.set_ylabel(var); ax_bar.set_title(f"{var} — Distribution (%)")
            for patch in ax_bar.patches:
                if isinstance(patch, Rectangle):
                    w = float(patch.get_width()); y = float(patch.get_y() + patch.get_height() / 2)
                    if np.isfinite(w): ax_bar.text(w + 0.5, y, f"{w:.1f}%", va="center")
            ensure_fig_saved_close(p_bar, dpi=150)
        except (ValueError, TypeError, RuntimeError) as e:
            print(f"⚠️ Barplot skipped for {var}:", e)
    else:
        fig = plt.figure(figsize=(8, 3)); plt.title(f"{var} — (no data)"); plt.axis("off")
        fig.savefig(p_bar, dpi=140, bbox_inches="tight"); plt.close(fig)

cat_dist_df = pd.DataFrame(cat_dist_rows)
cat_missing_df = pd.DataFrame(cat_missing_rows).sort_values(["missing","variable"], ascending=[False, True])

# ---- Categorical heatmap (Cramér's V) ----
cramers_matrix: Optional[pd.DataFrame] = None
if SCIPY_OK and len(cat_bool_cols) >= 2:
    cols_for_heat = [c for c in cat_bool_cols if df[c].nunique(dropna=True) <= MAX_LEVELS_CRAMERS]
    if len(cols_for_heat) >= 2:
        m_len = len(cols_for_heat)
        cramers_matrix = pd.DataFrame(np.nan, index=cols_for_heat, columns=cols_for_heat, dtype=float)
        from scipy.stats import chi2_contingency

        def _cramers_v_corrected(x_series: pd.Series, y_series: pd.Series) -> float:
            crosstab = pd.crosstab(x_series, y_series)
            if crosstab.size == 0: return float("nan")
            chi2_stat, _, _, _ = chi2_contingency(crosstab, correction=False)
            n_observations = int(crosstab.values.sum())
            if n_observations == 0: return float("nan")
            phi2 = float(chi2_stat) / n_observations; r_dim, k_dim = crosstab.shape
            if n_observations > 1:
                phi2corr = max(0.0, phi2 - (k_dim - 1)*(r_dim - 1)/(n_observations - 1))
                rcorr = r_dim - (r_dim - 1)**2/(n_observations - 1)
                kcorr = k_dim - (k_dim - 1)**2/(n_observations - 1)
            else:
                phi2corr = 0.0; rcorr, kcorr = float(r_dim), float(k_dim)
            denom = min((kcorr - 1), (rcorr - 1))
            if denom <= 0: return float("nan")
            return float(np.sqrt(phi2corr / denom))
        for i, ci in enumerate(cols_for_heat):
            for j, cj in enumerate(cols_for_heat):
                if j < i: continue
                v = 1.0 if ci == cj else _cramers_v_corrected(df[ci], df[cj])
                cramers_matrix.loc[ci, cj] = v; cramers_matrix.loc[cj, ci] = v
        side = min(18.0, 1.0 + 0.6 * m_len)
        try:
            plt.figure(figsize=(side, side))
            sns.heatmap(cramers_matrix, vmin=0, vmax=1, cmap="viridis", linewidths=0.5, square=True)
            plt.title("Categorical Association (Cramér's V)")
            ensure_fig_saved_close(IMG_HEAT_CRAMERS, dpi=150)
            print("✅ Cramér's V heatmap saved:", IMG_HEAT_CRAMERS)
        except (ValueError, TypeError, RuntimeError) as e:
            print("⚠️ Cramér's V heatmap skipped:", e)

# ---- Numeric detailed table ----
numeric_detailed = summarize_numeric(df[num_cols]) if num_cols else pd.DataFrame({"note": ["No numeric columns"]})

# ---- Pairplot (numeric-only) ----
if num_cols:
    df_num_pp = df[num_cols].dropna(how="all", axis=1)
    if not df_num_pp.empty:
        stds_series = df_num_pp.std(numeric_only=True)
        good_cols = stds_series[stds_series > 0].index.tolist()
        df_num_pp = df_num_pp[good_cols]
        if not df_num_pp.empty:
            if df_num_pp.shape[1] > MAX_PAIRPLOT_VARS:
                top_cols = stds_series[good_cols].nlargest(MAX_PAIRPLOT_VARS).index.tolist()
                df_num_pp = df_num_pp[top_cols]
            try:
                pairgrid_obj = sns.pairplot(df_num_pp, diag_kind="hist", corner=False, plot_kws={"s":8,"alpha":0.6,"edgecolor":"none"})
                nvars_pair = df_num_pp.shape[1]
                pairgrid_obj.figure.set_size_inches(max(8, 2.3 * nvars_pair), max(8, 2.3 * nvars_pair))
                pairgrid_obj.figure.suptitle("Pairplot — Numeric Variables", y=1.02)
                pairgrid_obj.figure.tight_layout()
                pairgrid_obj.figure.savefig(IMG_PAIRPLOT_NUM, dpi=160, bbox_inches="tight")
                plt.close(pairgrid_obj.figure)
                print("✅ Pairplot saved:", IMG_PAIRPLOT_NUM)
            except (ValueError, TypeError, RuntimeError, MemoryError) as e:
                print("⚠️ Pairplot skipped:", e)

# ---- Write EDA tables to Excel ----
write_eda_excel_safely(
    xlsx_output_path=EDA_XLSX_PATH,
    numeric_detailed_tbl=numeric_detailed,
    cat_dist_tbl=cat_dist_df,
    cat_missing_tbl=cat_missing_df,
    corr_spear_tbl=corr_spear,
    cat_heat_matrix_tbl=cramers_matrix,
)

print(
    "\n📁 Outputs directory: {}\n"
    " - Excel (tables): {}\n"
    " - Spearman PNG : {}\n"
    " - Cramér's PNG : {}\n"
    " - Boxplot grid : {}\n"
    " - Pairplot PNG : {}\n"
    " - Hist dir : {}\n"
    " - Box dir : {}\n"
    " - Bar dir : {}\n".format(
        OUTPUT_DIR, EDA_XLSX_PATH, IMG_HEAT_SPEARMAN, IMG_HEAT_CRAMERS,
        IMG_ALL_NUM_BOX, IMG_PAIRPLOT_NUM, HISTS_DIR, BOXES_DIR, BARS_DIR
    )
)

# ==========================================
# 5) Statistical tests vs target (Severity_2cat or Severity_bin)
# ==========================================
TARGET_COL = "Severity_2cat" if "Severity_2cat" in df.columns else "Severity_bin"
if TARGET_COL not in df.columns:
    raise KeyError(f"Missing target column: {TARGET_COL}")

if not SCIPY_OK:
    print("⚠️ Skipping statistical tests - SciPy not available")
else:
    tests_rows: list[dict[str, Any]] = []
    y_target = df[TARGET_COL]

    def _is_numeric(series_in: pd.Series) -> bool:
        return is_numeric_dtype(series_in)
    def _is_bool(series_in: pd.Series) -> bool:
        return is_bool_dtype(series_in) or str(series_in.dtype) == "boolean"
    def _is_categorical_like(series_in: pd.Series) -> bool:
        return isinstance(series_in.dtype, CategoricalDtype) or is_object_dtype(series_in) or is_string_dtype(series_in)

    for col in [c for c in df.columns if c not in {TARGET_COL} and df[c].notna().any()]:
        s_pred = df[col]; row: dict[str, Any] = {"feature": col}
        try:
            if _is_numeric(s_pred):
                if y_target.nunique(dropna=True) == 2:
                    y_groups = y_target.dropna().unique().tolist()
                    g1 = s_pred[y_target == y_groups[0]].dropna(); g2 = s_pred[y_target == y_groups[1]].dropna()
                    if len(g1) > 0 and len(g2) > 0:
                        u_stat, p_val = mannwhitneyu(g1, g2, alternative="two-sided")
                        row.update({"test":"Mann–Whitney U","n":int(len(g1)+len(g2)),"stat":float(u_stat),"p_value":float(p_val)})
                    else:
                        row.update({"test":"Mann–Whitney U","note":"empty groups"})
                else:
                    row.update({"test":"Mann–Whitney U","note":"target not binary"})
            elif _is_categorical_like(s_pred) or _is_bool(s_pred):
                cont = pd.crosstab(s_pred, y_target); r, k = cont.shape
                if r == 2 and k == 2:
                    odds, p_val = fisher_exact(cont)
                    row.update({"test":"Fisher exact (2×2)","n":int(cont.values.sum()),"stat":float(odds),"p_value":float(p_val)})
                else:
                    chi2, p_val, dof, expected = chi2_contingency_stats(cont)
                    n_obs = int(cont.values.sum()); denom_local = max(1.0, min(r-1, k-1))
                    cramers_v = float(np.sqrt((chi2 / n_obs) / denom_local)) if n_obs > 0 else np.nan
                    row.update({"test":"Chi-square","n":n_obs,"stat":float(chi2),"p_value":float(p_val),"dof":int(dof),"cramers_v":cramers_v})
            else:
                row.update({"test":"Unsupported dtype"})
        except Exception as ex:
            row.update({"test_error": str(ex)})
        tests_rows.append(row)

    tests_df = pd.DataFrame(tests_rows)
    tests_df.sort_values("p_value", inplace=True, na_position="last")
    signif_df = tests_df[pd.to_numeric(tests_df["p_value"], errors="coerce") < 0.05].copy()
    try:
        with pd.ExcelWriter(TESTS_XLSX, engine="xlsxwriter") as xw:
            tests_df.to_excel(xw, sheet_name="All tests", index=False)
            signif_df.to_excel(xw, sheet_name="Significant (p<0.05)", index=False)
        tests_df.to_csv(TESTS_CSV_SUMMARY, index=False); signif_df.to_csv(TESTS_CSV_SIGNIF, index=False)
        print("\n✅ Saved updated statistical test results:")
        print(f" - Excel full: {TESTS_XLSX}")
        print(f" - CSV summary: {TESTS_CSV_SUMMARY}")
        print(f" - CSV significant: {TESTS_CSV_SIGNIF}")
    except Exception as e:
        print("⚠️ Failed to save updated statistical results:", e)

# =====================================
# 6) Export full post-cleaning dataframe
# =====================================
try:
    df.to_csv(POST_CLEAN_CSV, index=False, encoding="utf-8")
    print("✅ Full cleaned dataframe saved to:", POST_CLEAN_CSV)
except (IOError, OSError, PermissionError) as e:
    print("⚠️ Failed to save post-cleaning CSV:", e)

# =========================================
# 7) Create Severity_bin and stratified split
# =========================================
print("\n📊 Creating binary target and splitting data...")
if "Severity" not in df.columns:
    raise KeyError("Column 'Severity' not found; cannot create Severity_bin")
sev = pd.to_numeric(df["Severity"], errors="coerce")
severity_bin = np.where(sev.isin([1, 2]), 0, np.where(sev.isin([3, 4]), 1, np.nan))
df_split = df.copy(); df_split["Severity_bin"] = severity_bin
n_before = len(df_split)
df_split = df_split.dropna(subset=["Severity_bin"]).copy()
df_split["Severity_bin"] = df_split["Severity_bin"].astype("int8")
n_removed = n_before - len(df_split)
print("✅ Severity_bin created: 0=[1,2], 1=[3,4]")
if n_removed > 0: print(f"⚠️ Removed {n_removed:,} rows with invalid Severity")

print_bin_dist("FULL", df_split["Severity_bin"])
check_stratify_okay(df_split["Severity_bin"], min_per_class=2)

# Split
train_val, test = train_test_split(
    df_split, test_size=TEST_FRAC, random_state=RANDOM_STATE,
    shuffle=True, stratify=df_split["Severity_bin"]
)
val_rel = VAL_FRAC / (TRAIN_FRAC + VAL_FRAC)
check_stratify_okay(train_val["Severity_bin"], min_per_class=2)
train, val = train_test_split(
    train_val, test_size=val_rel, random_state=RANDOM_STATE,
    shuffle=True, stratify=train_val["Severity_bin"]
)

# Validate disjointness
if set(train.index) & set(val.index) or set(train.index) & set(test.index) or set(val.index) & set(test.index):
    raise RuntimeError("Overlap detected between splits!")

# Save splits
try:
    train.to_csv(TRAIN_OUT_PATH, index=False, encoding="utf-8")
    val.to_csv(VAL_OUT_PATH, index=False, encoding="utf-8")
    test.to_csv(TEST_OUT_PATH, index=False, encoding="utf-8")
    print("\n✅ Saved splits (stratified by Severity_bin):")
    print(" - Train:", TRAIN_OUT_PATH)
    print(" - Val :", VAL_OUT_PATH)
    print(" - Test :", TEST_OUT_PATH)
except (IOError, OSError, PermissionError) as e:
    print("❌ Failed to write splits:", e)

print(f"Shapes: train={train.shape}, val={val.shape}, test={test.shape}\n")
print_bin_dist("Train", train["Severity_bin"])
print_bin_dist("Val", val["Severity_bin"])
print_bin_dist("Test", test["Severity_bin"])

print("\n🎉 Done: cleaning, EDA, stats, and split complete.")
