from __future__ import annotations
import os
import math
from typing import List, Dict, Optional, cast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import seaborn as sns
from matplotlib.patches import Rectangle

# Optional SciPy (only what we actually use)
SCIPY_OK = True
try:
    from scipy.stats import chi2_contingency as scipy_chi2_contingency
except ImportError:
    SCIPY_OK = False
    scipy_chi2_contingency = None  # type: ignore

# Optional TextBlob
TEXTBLOB_OK = True
try:
    from textblob import TextBlob
except ImportError:
    TEXTBLOB_OK = False
    TextBlob = None  # type: ignore

# ==============================
# Config: IO paths
# ==============================
CSV_PATH = r"C:\Users\LEA GUR-ARIE\PycharmProjects\USA_Accidents_ML\src\data\raw\interim\output_data\df_prepared.csv"

EDA_DIR = r"C:\Users\LEA GUR-ARIE\PycharmProjects\USA_Accidents_ML\src\data\raw\interim\EDA_plot"
os.makedirs(EDA_DIR, exist_ok=True)

# EDA outputs
OUTPUT_FILE_MAIN = "analysis_output.xlsx"
OUTPUT_XLSX_PATH = os.path.join(EDA_DIR, OUTPUT_FILE_MAIN)

IMG_HEAT_SPEARMAN = os.path.join(EDA_DIR, "spearman_heatmap.png")
IMG_HEAT_CRAMERS = os.path.join(EDA_DIR, "categorical_cramersV_heatmap.png")
IMG_HEXBIN_PLAIN = os.path.join(EDA_DIR, "hexbin_StartLat_StartLng.png")
IMG_HEXBIN_LABELS = os.path.join(EDA_DIR, "hexbin_StartLat_StartLng_labeled.png")
IMG_ALL_NUM_BOX = os.path.join(EDA_DIR, "boxplot_all_numeric_grid.png")
IMG_PAIRPLOT_NUM = os.path.join(EDA_DIR, "pairplot_numeric_grid.png")

HISTS_DIR = os.path.join(EDA_DIR, "histograms_numeric")
BOXES_DIR = os.path.join(EDA_DIR, "boxplots_numeric")
BARS_DIR = os.path.join(EDA_DIR, "barplots_categorical")
for _p in (HISTS_DIR, BOXES_DIR, BARS_DIR):
    os.makedirs(_p, exist_ok=True)

# Plot settings
TOP_K = 15
MAX_LEVELS_CRAMERS = 30
INCLUDE_BOOLS_IN_SPEARMAN = True
MAX_PAIRPLOT_VARS = 12

# RNG
SEED = 42
RNG_GLOBAL = np.random.default_rng(SEED)

# ==============================
# Helpers
# ==============================
def ensure_fig_saved_close(path: str, dpi: int = 150) -> None:
    """Tight layout, save, close figure."""
    try:
        plt.tight_layout()
    except (RuntimeError, ValueError):
        pass
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()

def summarize_numeric(df_num: pd.DataFrame) -> pd.DataFrame:
    """Extended numeric summary with IQR fences and outlier counts."""
    rows_out: List[dict[str, object]] = []

    for col_name in map(str, df_num.columns):
        s_vals = pd.to_numeric(df_num[col_name], errors="coerce")

        n_total = int(s_vals.size)
        n_nonnull = int(s_vals.notna().sum())
        n_missing = n_total - n_nonnull
        pct_missing = (n_missing / n_total) * 100.0 if n_total else 0.0

        mean_val = float(s_vals.mean()) if n_nonnull else np.nan
        std_val = float(s_vals.std()) if n_nonnull else np.nan
        se_mean = float(std_val / np.sqrt(n_nonnull)) if (n_nonnull > 0 and np.isfinite(std_val)) else np.nan

        p01 = p05 = q1 = med = q3 = p95 = p99 = np.nan
        iqr_val = lower = upper = np.nan
        n_out = 0
        pct_out_iqr = np.nan

        if n_nonnull > 0:
            q = s_vals.quantile([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
            p01 = float(q.get(0.01, np.nan))
            p05 = float(q.get(0.05, np.nan))
            q1 = float(q.get(0.25, np.nan))
            med = float(q.get(0.50, np.nan))
            q3 = float(q.get(0.75, np.nan))
            p95 = float(q.get(0.95, np.nan))
            p99 = float(q.get(0.99, np.nan))

            if np.isfinite(q1) and np.isfinite(q3):
                iqr_val = q3 - q1
                lower = q1 - 1.5 * iqr_val
                upper = q3 + 1.5 * iqr_val
                if np.isfinite(lower) and np.isfinite(upper):
                    is_out = (s_vals < lower) | (s_vals > upper)
                    n_out = int(is_out.sum())
                    pct_out_iqr = (n_out / n_nonnull) * 100.0

        rows_out.append({
            "column": col_name,
            "dtype": str(df_num[col_name].dtype),
            "count": n_total,
            "non_null": n_nonnull,
            "missing": n_missing,
            "missing_pct": pct_missing,
            "mean": mean_val,
            "std": std_val,
            "se_mean": se_mean,
            "min": float(s_vals.min()) if n_nonnull else np.nan,
            "p01": p01, "p05": p05, "q1": q1, "median": med, "q3": q3, "p95": p95, "p99": p99,
            "max": float(s_vals.max()) if n_nonnull else np.nan,
            "iqr": iqr_val, "lower_fence": lower, "upper_fence": upper,
            "n_outliers_iqr": n_out,
            "pct_outliers_iqr": pct_out_iqr,
        })

    out_df = pd.DataFrame(rows_out).set_index("column").sort_index()

    round_3_cols = [
        "missing_pct", "mean", "std", "min", "p01", "p05", "q1", "median", "q3",
        "p95", "p99", "max", "iqr", "lower_fence", "upper_fence", "pct_outliers_iqr"
    ]
    for c_name in round_3_cols:
        if c_name in out_df.columns:
            out_df[c_name] = pd.to_numeric(out_df[c_name], errors="coerce").round(3)
    if "se_mean" in out_df.columns:
        out_df["se_mean"] = pd.to_numeric(out_df["se_mean"], errors="coerce").round(6)
    return out_df

def cramers_v_corrected(x_series: pd.Series, y_series: pd.Series) -> float:
    """Cramér's V with bias correction."""
    if not SCIPY_OK:
        return float("nan")
    ct = pd.crosstab(x_series, y_series)
    if ct.size == 0:
        return float("nan")
    chi2_stat, _, _, _ = scipy_chi2_contingency(ct, correction=False)
    n_obs = int(ct.values.sum())
    if n_obs == 0:
        return float("nan")
    phi2 = float(chi2_stat) / n_obs
    r_dim, k_dim = ct.shape
    if n_obs > 1:
        phi2corr = max(0.0, phi2 - (k_dim - 1) * (r_dim - 1) / (n_obs - 1))
        rcorr = r_dim - (r_dim - 1) ** 2 / (n_obs - 1)
        kcorr = k_dim - (k_dim - 1) ** 2 / (n_obs - 1)
    else:
        phi2corr = 0.0
        rcorr, kcorr = float(r_dim), float(k_dim)
    denom = min((kcorr - 1), (rcorr - 1))
    if denom <= 0:
        return float("nan")
    return float(np.sqrt(phi2corr / denom))

def timestamped_copy_path(base_dir: str, stem: str = "analysis_output") -> str:
    from datetime import datetime as _dt
    ts = _dt.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(base_dir, f"{stem}_{ts}.xlsx")

def safe_write_eda_excel(
        output_path: str,
        numeric_detailed_tbl: pd.DataFrame,
        cat_dist_tbl: pd.DataFrame,
        cat_missing_tbl: pd.DataFrame,
        corr_spear_tbl: pd.DataFrame,
        cat_heat_matrix_tbl: Optional[pd.DataFrame],
        borough_counts_tbl: Dict[str, int],
) -> None:
    """Write all EDA tables; if file is locked, write a timestamped copy."""
    try:
        import xlsxwriter  # noqa: F401 - needed by pandas ExcelWriter engine
        with pd.ExcelWriter(output_path, engine="xlsxwriter") as xlsx_writer:
            numeric_detailed_tbl.to_excel(xlsx_writer, sheet_name="Numeric Detailed")
            if not cat_dist_tbl.empty:
                cat_dist_tbl.sort_values(["variable", "percent"], ascending=[True, False]) \
                    .to_excel(xlsx_writer, sheet_name="Categorical Dist (%)", index=False)
            if not cat_missing_tbl.empty:
                cat_missing_tbl.to_excel(xlsx_writer, sheet_name="Categorical Missing", index=False)
            if not corr_spear_tbl.empty:
                corr_spear_tbl.to_excel(xlsx_writer, sheet_name="Spearman Corr")
            if cat_heat_matrix_tbl is not None:
                cat_heat_matrix_tbl.to_excel(xlsx_writer, sheet_name="CramersV (Cat)")
            if borough_counts_tbl:
                pd.DataFrame(
                    sorted(borough_counts_tbl.items(), key=lambda kv: kv[1], reverse=True),
                    columns=["Borough", "Count"]
                ).to_excel(xlsx_writer, sheet_name="Borough Counts", index=False)
        print(f"✅ File saved to: {output_path}")
    except PermissionError:
        alt_path = timestamped_copy_path(EDA_DIR, stem=os.path.splitext(os.path.basename(output_path))[0])
        with pd.ExcelWriter(alt_path, engine="xlsxwriter") as xlsx_writer_alt:
            numeric_detailed_tbl.to_excel(xlsx_writer_alt, sheet_name="Numeric Detailed")
            if not cat_dist_tbl.empty:
                cat_dist_tbl.sort_values(["variable", "percent"], ascending=[True, False]) \
                    .to_excel(xlsx_writer_alt, sheet_name="Categorical Dist (%)", index=False)
            if not cat_missing_tbl.empty:
                cat_missing_tbl.to_excel(xlsx_writer_alt, sheet_name="Categorical Missing", index=False)
            if not corr_spear_tbl.empty:
                corr_spear_tbl.to_excel(xlsx_writer_alt, sheet_name="Spearman Corr")
            if cat_heat_matrix_tbl is not None:
                cat_heat_matrix_tbl.to_excel(xlsx_writer_alt, sheet_name="CramersV (Cat)")
            if borough_counts_tbl:
                pd.DataFrame(
                    sorted(borough_counts_tbl.items(), key=lambda kv: kv[1], reverse=True),
                    columns=["Borough", "Count"]
                ).to_excel(xlsx_writer_alt, sheet_name="Borough Counts", index=False)
        print("⚠️ Original file was locked (likely open in Excel).")
        print(f"✅ Wrote timestamped copy: {alt_path}")
    except (OSError, ValueError) as err:
        print(f"❌ Failed to write Excel: {err}")

def save_multi_boxplot_grid(df_num_grid: pd.DataFrame, out_file_path: str, ncols: int = 4) -> None:
    """Create grid of boxplots for all numeric columns."""
    from matplotlib.axes import Axes

    cols_present = [
        c for c in df_num_grid.columns
        if pd.to_numeric(df_num_grid[c], errors="coerce").dropna().size > 0
    ]
    if not cols_present:
        print("⚠️ No valid numeric columns for boxplot grid")
        return

    n_vars = len(cols_present)
    nrows = max(1, math.ceil(n_vars / ncols))
    fig_grid, axes_arr = plt.subplots(nrows, ncols, figsize=(ncols * 5.0, nrows * 4.0), squeeze=False)
    axes_flat: List[Axes] = [cast(Axes, ax) for ax in axes_arr.flatten()]

    for idx, varname in enumerate(cols_present):
        ax_current = axes_flat[idx]
        try:
            data = pd.to_numeric(df_num_grid[varname], errors="coerce").dropna()
            sns.boxplot(y=data, ax=ax_current, orient="v", showfliers=True)
            ax_current.set_title(str(varname), fontsize=10)
            ax_current.set_xlabel("")
            ax_current.set_ylabel("")
            ax_current.grid(True, axis="y", alpha=0.3)
        except (ValueError, RuntimeError) as err:
            print(f"⚠️ Boxplot failed for {varname}: {err}")
            ax_current.axis("off")

    for idx in range(len(cols_present), len(axes_flat)):
        axes_flat[idx].axis("off")

    try:
        fig_grid.tight_layout()
    except (RuntimeError, ValueError):
        pass
    fig_grid.savefig(out_file_path, dpi=160, bbox_inches="tight")
    plt.close(fig_grid)
    print(f"✅ Saved boxplot grid: {out_file_path}")

# ==============================
# Load data
# ==============================
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"Input file not found: {CSV_PATH}")

df= pd.read_csv(CSV_PATH)
df.columns = df.columns.str.strip()
print("✅ Data loaded successfully.")

# ==============================
# Load data
# ==============================
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"Input file not found: {CSV_PATH}")

df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.strip()
print("✅ Data loaded successfully (no columns dropped).")

# Cast object to category (excluding raw time text columns)
time_cols = [c for c in ("Start_Time", "End_Time") if c in df.columns]
object_cols = df.select_dtypes(include=["object"]).columns.tolist()
object_cols_cast = [c for c in object_cols if c not in time_cols]
if object_cols_cast:
    try:
        df[object_cols_cast] = df[object_cols_cast].astype("category")
    except (TypeError, ValueError):
        pass


# Cast object to category (excluding raw time text columns)
time_cols = [c for c in ("Start_Time", "End_Time") if c in df.columns]
object_cols = df.select_dtypes(include=["object"]).columns.tolist()
object_cols_cast = [c for c in object_cols if c not in time_cols]
if object_cols_cast:
    try:
        df[object_cols_cast] = df[object_cols_cast].astype("category")
    except (TypeError, ValueError):
        pass



# ==============================
# Separate by dtype
# ==============================
num_cols: List[str] = df.select_dtypes(include=[np.number]).columns.tolist()
bool_cols: List[str] = df.select_dtypes(include=["bool", "boolean"]).columns.tolist()
cat_cols: List[str] = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()

print(f"\nColumn types: {len(num_cols)} numeric, {len(bool_cols)} boolean, {len(cat_cols)} categorical")

# ==============================
# Spearman correlation
# ==============================
df_corr_input = pd.DataFrame(index=df.index)
if num_cols:
    df_corr_input = df[num_cols].copy()
if INCLUDE_BOOLS_IN_SPEARMAN and bool_cols:
    for bname in bool_cols:
        df_corr_input[bname] = df[bname].astype("Int64")

if (not df_corr_input.empty) and (df_corr_input.shape[1] >= 2):
    try:
        corr_spear = df_corr_input.corr(method="spearman")
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_spear, square=False, linewidths=0.5, cmap="coolwarm", center=0)
        plt.title("Spearman Correlation (Numeric + Boolean)")
        ensure_fig_saved_close(IMG_HEAT_SPEARMAN, dpi=150)
        print(f"✅ Saved: {IMG_HEAT_SPEARMAN}")
    except (ValueError, RuntimeError) as e:
        print(f"⚠️ Spearman heatmap skipped: {e}")
        corr_spear = pd.DataFrame()
else:
    print("⚠️ Insufficient columns for Spearman heatmap")
    corr_spear = pd.DataFrame()

# ==============================
# Individual numeric plots
# ==============================
print("\n📊 Generating individual numeric plots...")
for nc in num_cols:
    s_num = pd.to_numeric(df[nc], errors="coerce").dropna()

    # Histogram
    hist_path = os.path.join(HISTS_DIR, f"{nc}_hist.png")
    if s_num.empty:
        fig = plt.figure(figsize=(8, 6))
        plt.title(f"{nc} — (no non-null values)")
        plt.axis("off")
        fig.savefig(hist_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
    else:
        try:
            plt.figure(figsize=(8, 6))
            sns.histplot(s_num, kde=True, bins=30)
            plt.title(f"Distribution of {nc}")
            plt.xlabel(nc)
            plt.ylabel("Density")
            ensure_fig_saved_close(hist_path, dpi=150)
        except (ValueError, RuntimeError) as hist_err:
            print(f"⚠️ Histogram skipped for {nc}: {hist_err}")

    # Boxplot
    box_path = os.path.join(BOXES_DIR, f"{nc}_box.png")
    if s_num.empty:
        fig = plt.figure(figsize=(8, 3))
        plt.title(f"{nc} — (no non-null values)")
        plt.axis("off")
        fig.savefig(box_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
    else:
        try:
            plt.figure(figsize=(10, 2.8))
            sns.boxplot(x=s_num, orient="h", showfliers=True)
            plt.title(f"Boxplot of {nc}")
            plt.xlabel(nc)
            ensure_fig_saved_close(box_path, dpi=150)
        except (ValueError, RuntimeError) as box_err:
            print(f"⚠️ Boxplot skipped for {nc}: {box_err}")

# ==============================
# Grid of all numeric boxplots
# ==============================
if num_cols:
    print("\n📊 Generating boxplot grid...")
    save_multi_boxplot_grid(df[num_cols], IMG_ALL_NUM_BOX, ncols=4)

# ==============================
# Categorical barplots + tables
# ==============================
print("\n📊 Generating categorical barplots...")
cat_bool_cols = cat_cols + bool_cols
cat_dist_rows: List[Dict[str, object]] = []
cat_missing_rows: List[Dict[str, object]] = []

for var in cat_bool_cols:
    s = df[var]
    total = len(s)
    nonnull = int(s.notna().sum())
    missing = total - nonnull
    miss_pct = (missing / total * 100) if total else 0.0
    cat_missing_rows.append({"variable": var, "missing": missing, "missing_pct": round(miss_pct, 3)})

    counts = s.value_counts(dropna=False)
    total_counts = int(counts.sum())

    img_path = os.path.join(BARS_DIR, f"{var}_bar.png")
    if total_counts == 0:
        fig = plt.figure(figsize=(8, 3))
        plt.title(f"{var} — (no data)")
        plt.axis("off")
        fig.savefig(img_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        continue

    for level, cnt in counts.items():
        label = "NaN" if pd.isna(level) else str(level)
        pct = (cnt / total_counts * 100)
        cat_dist_rows.append({"variable": var, "level": label, "count": int(cnt), "percent": round(float(pct), 3)})

    counts_plot = counts.copy()
    counts_plot.index = counts_plot.index.map(lambda x: "NaN" if pd.isna(x) else str(x))
    perc = (counts_plot / total_counts * 100).sort_values(ascending=False)

    if len(perc) > TOP_K:
        top = perc.iloc[:TOP_K - 1]
        other = perc.iloc[TOP_K - 1:].sum()
        perc_plot = pd.concat([top, pd.Series({"OTHER": other})])
    else:
        perc_plot = perc

    try:
        plt.figure(figsize=(10, max(4.0, 0.4 * len(perc_plot))))
        ax_bar = sns.barplot(x=perc_plot.values, y=perc_plot.index.astype(str), orient="h")
        ax_bar.set_xlabel("Percent")
        ax_bar.set_ylabel(var)
        ax_bar.set_title(f"{var} — Distribution (%)")
        for patch in ax_bar.patches:
            if isinstance(patch, Rectangle):
                width = patch.get_width()
                y = patch.get_y() + patch.get_height() / 2
                if np.isfinite(width):
                    ax_bar.text(width + 0.5, y, f"{width:.1f}%", va="center")
        ensure_fig_saved_close(img_path, dpi=150)
    except (ValueError, RuntimeError) as bar_err:
        print(f"⚠️ Barplot skipped for {var}: {bar_err}")

cat_dist_df = pd.DataFrame(cat_dist_rows)
cat_missing_df = pd.DataFrame(cat_missing_rows).sort_values(["missing", "variable"], ascending=[False, True])

# ==============================
# Cramér's V heatmap
# ==============================
cat_heat_matrix: Optional[pd.DataFrame] = None
if SCIPY_OK and len(cat_bool_cols) >= 2:
    cols_for_heat = [c for c in cat_bool_cols if df[c].nunique(dropna=True) <= MAX_LEVELS_CRAMERS]
    if len(cols_for_heat) >= 2:
        print("\n📊 Computing Cramér's V heatmap...")
        m_len = len(cols_for_heat)
        cat_heat_matrix = pd.DataFrame(np.nan, index=cols_for_heat, columns=cols_for_heat)

        for i, col_i in enumerate(cols_for_heat):
            for j, col_j in enumerate(cols_for_heat):
                if j < i:
                    continue
                if col_i == col_j:
                    cat_heat_matrix.loc[col_i, col_j] = 1.0
                else:
                    try:
                        v = cramers_v_corrected(df[col_i], df[col_j])
                    except (FloatingPointError, ValueError):
                        v = float("nan")
                    cat_heat_matrix.loc[col_i, col_j] = v
                    cat_heat_matrix.loc[col_j, col_i] = v

        side = min(18.0, 1.0 + 0.6 * m_len)
        try:
            plt.figure(figsize=(side, side))
            sns.heatmap(cat_heat_matrix, vmin=0, vmax=1, cmap="viridis", linewidths=0.5, square=True)
            plt.title("Categorical Association (Cramér's V)")
            ensure_fig_saved_close(IMG_HEAT_CRAMERS, dpi=150)
            print(f"✅ Saved: {IMG_HEAT_CRAMERS}")
        except (ValueError, RuntimeError) as e:
            print(f"⚠️ Cramér's V heatmap skipped: {e}")

# ==============================
# Hexbin maps
# ==============================
borough_counts_map: Dict[str, int] = {}
if {"Start_Lat", "Start_Lng"}.issubset(df.columns):
    print("\n📊 Generating hexbin maps...")
    lat = pd.to_numeric(df["Start_Lat"], errors="coerce")
    lng = pd.to_numeric(df["Start_Lng"], errors="coerce")
    valid = lat.notna() & lng.notna()

    if valid.any():
        # Plain hexbin
        try:
            plt.figure(figsize=(8, 7))
            plt.hexbin(x=lng[valid], y=lat[valid], gridsize=60, mincnt=1, cmap="YlOrRd")
            plt.xlabel("Start_Lng")
            plt.ylabel("Start_Lat")
            plt.title("Hexbin Density: Start_Lat vs Start_Lng")
            cbar = plt.colorbar()
            cbar.set_label("Count")
            ensure_fig_saved_close(IMG_HEXBIN_PLAIN, dpi=160)
            print(f"✅ Saved: {IMG_HEXBIN_PLAIN}")
        except (ValueError, RuntimeError) as e:
            print(f"⚠️ Hexbin (plain) skipped: {e}")

        # Borough boxes
        borough_boxes = {
            "Staten Island": ((40.48, 40.65), (-74.26, -74.05)),
            "Brooklyn": ((40.56, 40.74), (-74.05, -73.85)),
            "Queens": ((40.55, 40.80), (-73.96, -73.70)),
            "Manhattan": ((40.69, 40.88), (-74.03, -73.92)),
            "Bronx": ((40.80, 40.92), (-73.93, -73.76)),
        }

        # Count accidents per borough
        for bname, ((lat_min, lat_max), (lng_min, lng_max)) in borough_boxes.items():
            mask = lat.between(lat_min, lat_max) & lng.between(lng_min, lng_max)
            borough_counts_map[bname] = int(mask.sum())

        print("\n=== Borough accident counts (rough boxes) ===")
        for bname, cnt in sorted(borough_counts_map.items(), key=lambda x: x[1], reverse=True):
            print(f"{bname:14s}: {cnt:,}")

        # Labeled hexbin
        try:
            plt.figure(figsize=(8, 7))
            plt.hexbin(x=lng[valid], y=lat[valid], gridsize=60, mincnt=1, cmap="YlOrRd")
            plt.xlabel("Start_Lng")
            plt.ylabel("Start_Lat")
            plt.title("Hexbin Density with Borough Labels")
            cbar = plt.colorbar()
            cbar.set_label("Count")

            for bname, ((lat_min, lat_max), (lng_min, lng_max)) in borough_boxes.items():
                mask = valid & lat.between(lat_min, lat_max) & lng.between(lng_min, lng_max)
                if not mask.any():
                    continue
                x_med = float(lng[mask].median())
                y_med = float(lat[mask].median())
                n = borough_counts_map.get(bname, 0)
                txt = plt.text(
                    x_med, y_med, f"{bname}\n(n={n:,})",
                    ha="center", va="center", fontsize=10, weight="bold", color="white", zorder=10
                )
                txt.set_path_effects([pe.withStroke(linewidth=3, foreground="black")])

            ensure_fig_saved_close(IMG_HEXBIN_LABELS, dpi=160)
            print(f"✅ Saved: {IMG_HEXBIN_LABELS}")
        except (ValueError, RuntimeError) as e:
            print(f"⚠️ Labeled hexbin skipped: {e}")

# ==============================
# Numeric detailed table
# ==============================
numeric_detailed_df = summarize_numeric(df[num_cols]) if num_cols else pd.DataFrame({"note": ["No numeric columns"]})

# ==============================
# Pairplot
# ==============================
if num_cols:
    print("\n📊 Generating pairplot...")
    df_num_pp = df[num_cols].dropna(how="all", axis=1)
    if not df_num_pp.empty:
        stds = df_num_pp.std(numeric_only=True)
        cols_good = stds[stds > 0].index.tolist()
        df_num_pp = df_num_pp[cols_good]

        if (not df_num_pp.empty) and (df_num_pp.shape[1] > MAX_PAIRPLOT_VARS):
            top_cols = stds[cols_good].nlargest(MAX_PAIRPLOT_VARS).index.tolist()
            df_num_pp = df_num_pp[top_cols]

        if not df_num_pp.empty:
            try:
                pg = sns.pairplot(
                    df_num_pp, diag_kind="hist", corner=False,
                    plot_kws={"s": 8, "alpha": 0.6, "edgecolor": "none"}
                )
                nvars = df_num_pp.shape[1]
                pg.figure.set_size_inches(max(8, 2.3 * nvars), max(8, 2.3 * nvars))
                pg.figure.suptitle("Pairplot — Numeric Variables", y=1.02)
                pg.figure.tight_layout()
                pg.figure.savefig(IMG_PAIRPLOT_NUM, dpi=160, bbox_inches="tight")
                plt.close(pg.figure)
                print(f"✅ Saved: {IMG_PAIRPLOT_NUM}")
            except (ValueError, RuntimeError) as e:
                print(f"⚠️ Pairplot skipped: {e}")

# ==============================
# Write EDA Excel
# ==============================
print("\n📄 Writing EDA summary to Excel...")

# Build inputs
cat_dist_df = pd.DataFrame(cat_dist_rows)
cat_missing_df = pd.DataFrame(cat_missing_rows)
if not cat_missing_df.empty and {"missing", "variable"}.issubset(cat_missing_df.columns):
    cat_missing_df = cat_missing_df.sort_values(["missing", "variable"], ascending=[False, True])

# Sanity/fallbacks with different names to avoid shadowing function params
corr_spear_df = corr_spear if isinstance(corr_spear, pd.DataFrame) else pd.DataFrame()
cat_heat_matrix_to_write = cat_heat_matrix if isinstance(cat_heat_matrix, pd.DataFrame) else None
borough_counts_to_write = borough_counts_map if isinstance(borough_counts_map, dict) else {}

# Write
safe_write_eda_excel(
    output_path=OUTPUT_XLSX_PATH,
    numeric_detailed_tbl=numeric_detailed_df,
    cat_dist_tbl=cat_dist_df,
    cat_missing_tbl=cat_missing_df,
    corr_spear_tbl=corr_spear_df,
    cat_heat_matrix_tbl=cat_heat_matrix_to_write,
    borough_counts_tbl=borough_counts_to_write,
)



print(
    f"\n📁 EDA outputs in: {EDA_DIR}\n"
    f"   - Excel: {OUTPUT_XLSX_PATH}\n"
    f"   - Spearman heatmap: {IMG_HEAT_SPEARMAN}\n"
    f"   - Cramér's V heatmap: {IMG_HEAT_CRAMERS}\n"
    f"   - Hexbin (plain): {IMG_HEXBIN_PLAIN}\n"
    f"   - Hexbin (labeled): {IMG_HEXBIN_LABELS}\n"
    f"   - Boxplot grid: {IMG_ALL_NUM_BOX}\n"
    f"   - Pairplot: {IMG_PAIRPLOT_NUM}\n"
    f"   - Histograms: {HISTS_DIR}\n"
    f"   - Boxplots: {BOXES_DIR}\n"
    f"   - Barplots: {BARS_DIR}\n"
)

# ==============================
# Sentiment Analysis
# ==============================
if TEXTBLOB_OK and {"Description", "Severity"}.issubset(df.columns):
    print("\n📊 Performing sentiment analysis...")
    sentiment_out_dir = EDA_DIR

    desc = df["Description"].dropna().astype(str)
    polarity_all = pd.Series(index=df.index, dtype="float64")
    subjectivity_all = pd.Series(index=df.index, dtype="float64")

    try:
        polarity_all.loc[desc.index] = desc.apply(lambda t: TextBlob(t).sentiment.polarity)
        subjectivity_all.loc[desc.index] = desc.apply(lambda t: TextBlob(t).sentiment.subjectivity)
    except (ValueError, AttributeError) as e:
        print(f"⚠️ TextBlob sentiment skipped: {e}")

    df_sent = df[["Severity"]].copy()
    df_sent["polarity"] = polarity_all
    df_sent["subjectivity"] = subjectivity_all
    df_sent = df_sent.dropna(subset=["polarity", "subjectivity"])

    df_sent["sentiment"] = np.select(
        [df_sent["polarity"] > 0, df_sent["polarity"] < 0],
        [1, -1],
        default=0
    ).astype("int8")

    print(f"Rows with sentiment: {len(df_sent):,}")

    # Summary by Severity
    agg = df_sent.groupby("Severity", observed=False).agg(
        n_rows=("sentiment", "size"),
        mean_polarity=("polarity", "mean"),
        mean_subjectivity=("subjectivity", "mean"),
    )
    dist = (
        df_sent.groupby(["Severity", "sentiment"], observed=False)
        .size()
        .unstack("sentiment", fill_value=0)
        .rename(columns={-1: "neg_count", 0: "neu_count", 1: "pos_count"})
    )
    summary = agg.join(dist, how="left").fillna(0)
    for c in ["neg_count", "neu_count", "pos_count"]:
        summary[c] = summary[c].astype(int)
    summary["neg_pct"] = (summary["neg_count"] / summary["n_rows"] * 100).round(2)
    summary["neu_pct"] = (summary["neu_count"] / summary["n_rows"] * 100).round(2)
    summary["pos_pct"] = (summary["pos_count"] / summary["n_rows"] * 100).round(2)

    print("\n=== Sentiment summary by Severity ===")
    print(summary[[
        "n_rows", "mean_polarity", "mean_subjectivity",
        "neg_count", "neu_count", "pos_count", "neg_pct", "neu_pct", "pos_pct"
    ]].sort_index())

    # Save sentiment Excel
    from datetime import datetime as dt_mod

    sentiment_xlsx = os.path.join(sentiment_out_dir, "severity_sentiment_summary.xlsx")
    try:
        with pd.ExcelWriter(sentiment_xlsx, engine="xlsxwriter") as writer:
            summary.reset_index().to_excel(writer, sheet_name="summary_by_severity", index=False)
            overall = (
                df_sent["sentiment"].value_counts().rename_axis("sentiment")
                .reindex([-1, 0, 1]).fillna(0).astype(int).rename("count").reset_index()
            )
            overall.to_excel(writer, sheet_name="overall_counts", index=False)
        print(f"✅ Saved sentiment Excel: {sentiment_xlsx}")
    except PermissionError:
        alt = os.path.join(
            sentiment_out_dir,
            f"severity_sentiment_summary_{dt_mod.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        )
        with pd.ExcelWriter(alt, engine="xlsxwriter") as writer:
            summary.reset_index().to_excel(writer, sheet_name="summary_by_severity", index=False)
            overall = (
                df_sent["sentiment"].value_counts().rename_axis("sentiment")
                .reindex([-1, 0, 1]).fillna(0).astype(int).rename("count").reset_index()
            )
            overall.to_excel(writer, sheet_name="overall_counts", index=False)
        print(f"⚠️ Excel locked. Saved timestamped copy: {alt}")
    except (OSError, ValueError) as e:
        print(f"❌ Failed to write sentiment Excel: {e}")

    # Plot 1: Overall sentiment counts
    overall_counts = df_sent["sentiment"].value_counts().sort_index()
    labels = ["Negative", "Neutral", "Positive"]
    values = [int(overall_counts.get(-1, 0)), int(overall_counts.get(0, 0)), int(overall_counts.get(1, 0))]
    try:
        plt.figure(figsize=(8, 6))
        plt.bar(labels, values, color=["red", "gray", "green"])
        plt.xlabel("Sentiment")
        plt.ylabel("Number of Occurrences")
        plt.title("Sentiment Analysis Results (Overall)")
        bar_path = os.path.join(sentiment_out_dir, "sentiment_counts.png")
        ensure_fig_saved_close(bar_path, dpi=150)
        print(f"✅ Saved: {bar_path}")
    except (ValueError, RuntimeError) as e:
        print(f"⚠️ Sentiment bar plot skipped: {e}")

    # Plot 2: Polarity vs Subjectivity
    try:
        color_map = {-1: "red", 0: "gray", 1: "green"}
        plt.figure(figsize=(10, 6))
        for sval in [-1, 0, 1]:
            sub = df_sent[df_sent["sentiment"] == sval]
            if not sub.empty:
                plt.scatter(
                    sub["polarity"], sub["subjectivity"],
                    s=16, alpha=0.5, c=color_map[sval],
                    label={-1: "Negative", 0: "Neutral", 1: "Positive"}[sval]
                )
        plt.title("Polarity vs. Subjectivity by Sentiment")
        plt.xlabel("Polarity (−1..+1)")
        plt.ylabel("Subjectivity (0..1)")
        plt.axhline(0, color="black", linestyle="--", linewidth=0.7)
        plt.axvline(0, color="black", linestyle="--", linewidth=0.7)
        plt.legend()
        scatter_path = os.path.join(sentiment_out_dir, "polarity_vs_subjectivity.png")
        ensure_fig_saved_close(scatter_path, dpi=150)
        print(f"✅ Saved: {scatter_path}")
    except (ValueError, RuntimeError) as e:
        print(f"⚠️ Sentiment scatter plot skipped: {e}")
else:
    if not TEXTBLOB_OK:
        print("\n⚠️ Skipping sentiment analysis: TextBlob not available")
    else:
        print("\n⚠️ Skipping sentiment analysis: 'Description' or 'Severity' columns missing")

print("\n✅ EDA complete! (No train/val/test split performed.)")



# ==============================
# Export full EDA dataframe
# ==============================
POST_EDA_CSV_PATH = r"C:\Users\LEA GUR-ARIE\PycharmProjects\USA_Accidents_ML\src\data\raw\interim\output_data\post_eda.csv"

try:
    df.to_csv(POST_EDA_CSV_PATH, index=False, encoding="utf-8")
    print(f"✅ Full EDA dataframe saved successfully to:\n{POST_EDA_CSV_PATH}")
except Exception as e:
    print(f"❌ Failed to save CSV: {e}")
