import pandas as pd
import numpy as np

# ==========================================
# Feature sanity-check + pruning script
# - Loads a FE dataset
# - Drops target-leakage and redundant features
# - Reports missing %, zero-variance, correlations, cardinality
# - Casts binary-like floats (0/1) to nullable Int
# - Prints concise post-drop summary
# ==========================================

# -------------------------------
# 0) Config
# -------------------------------
CSV_PATH = r"C:\Users\LEA GUR-ARIE\PycharmProjects\USA_Accidents_ML\src\data\raw\interim\output_data\split\_fe_outputs\train_fe.csv"
TARGET = "Severity_bin_ord"  # <- current target to KEEP

# Columns we agreed to drop (keep Distance_group_ord)
DROP_COLS = [
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

if TARGET in DROP_COLS:
    DROP_COLS.remove(TARGET)

# -------------------------------
# 1) Load
# -------------------------------
df = pd.read_csv(CSV_PATH)
print("Shape:", df.shape)
print("Columns:", len(df.columns))
print(df.head())

# -------------------------------
# 2) Drop any target-leakage columns
#    Keep exactly TARGET, drop any other 'Severity*'
# -------------------------------
leak_cols = [c for c in df.columns if c.startswith("Severity") and c != TARGET]
if leak_cols:
    print("\n[INFO] Dropping leakage columns:", leak_cols)
    df = df.drop(columns=leak_cols)

# -------------------------------
# 3) Missingness report
# -------------------------------
missing = df.isna().mean().sort_values(ascending=False)
print("\n=== Missing ratio ===")
print(missing)

# Flag features with too many missing values (e.g., > 40%)
high_missing = missing[missing > 0.40].index.tolist()
print("\nHigh-missing (>40%) candidates to drop:", high_missing)

# -------------------------------
# 4) Zero-variance features
# -------------------------------
nzv = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
print("\nZero-variance features:", nzv)

# -------------------------------
# 5) Numeric correlation check
#    Note: we'll compute on float view (nullable ints coerced safely)
# -------------------------------
# Identify numeric columns (before any dtype changes)
num_cols_initial = df.select_dtypes(include=[np.number]).columns.tolist()

if len(num_cols_initial) > 1:
    # Coerce to float to include any Int64 later if needed
    corr = df[num_cols_initial].astype(float).corr().abs()
    high_corr_pairs = []
    for i in range(len(num_cols_initial)):
        for j in range(i + 1, len(num_cols_initial)):
            if corr.iloc[i, j] > 0.92:
                high_corr_pairs.append((num_cols_initial[i], num_cols_initial[j], corr.iloc[i, j]))

    print("\nHighly correlated numeric pairs (>0.92):")
    if high_corr_pairs:
        for p in high_corr_pairs:
            print("  ", p)
    else:
        print("  (none)")

# -------------------------------
# 6) Categorical cardinality
# -------------------------------
cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
print("\n=== Categorical cardinality ===")
for c in cat_cols:
    print(f"{c}: {df[c].nunique()} unique values")

high_cardinality = [c for c in cat_cols if df[c].nunique() > 40]
print("\nHigh-cardinality categoricals (>40 values):", high_cardinality)

# -------------------------------
# 7) Numeric distributions
# -------------------------------
num_cols_now = df.select_dtypes(include=[np.number]).columns.tolist()
print("\n=== Numeric distributions (min/max/mean/std) ===")
if num_cols_now:
    print(df[num_cols_now].describe().T)
else:
    print("(no numeric columns detected)")

# -------------------------------
# 8) Detect binary-like floats (values ⊆ {0,1}) that are not integers
# -------------------------------
binary_like = []
for c in num_cols_now:
    vals = set(pd.unique(df[c].dropna()))
    # normalize possible np.float64 0.0/1.0 to python ints for the subset check
    vals_norm = {int(v) if v in (0.0, 1.0) else v for v in vals}
    if vals_norm.issubset({0, 1}) and str(df[c].dtype) not in ("int64", "Int64"):
        binary_like.append(c)

print("\nBinary-like numeric columns requiring cast to int:", binary_like)

# -------------------------------
# 9) Duplicate columns (exact equality)
# -------------------------------
dupes = []
cols = df.columns.tolist()
for i in range(len(cols)):
    for j in range(i + 1, len(cols)):
        if df.iloc[:, i].equals(df.iloc[:, j]):
            dupes.append((cols[i], cols[j]))

print("\nDuplicate columns:", dupes)

# -------------------------------
# 10) Apply agreed DROPs and tidy types
# -------------------------------
existing_to_drop = [c for c in DROP_COLS if c in df.columns]
print("\n[PLAN] Columns planned to drop:", DROP_COLS)
print("[INFO] Columns found & will be dropped:", existing_to_drop)
if existing_to_drop:
    df = df.drop(columns=existing_to_drop)

# Keep Distance_group_ord by design; just confirm it's still present
print("[CHECK] Distance_group_ord in df:", "Distance_group_ord" in df.columns)

# Cast binary-like floats to nullable Int64 (keeps NA if present)
if binary_like:
    castable = [c for c in binary_like if c in df.columns]
    if castable:
        print("[INFO] Casting binary-like float → Int64 for:", castable)
        df[castable] = df[castable].astype("Int64")

# -------------------------------
# 11) Post-drop summary
# -------------------------------
print("\n=== Post-drop summary ===")
print("New shape:", df.shape)
print("Remaining columns:", len(df.columns))
print("\nHead after drop/casts:")
print(df.head())

# Rebuild helper lists (numeric will include nullable Int via explicit include)
num_cols_final = df.select_dtypes(include=[np.number, "Int64"]).columns.tolist()
cat_cols_final = df.select_dtypes(include=["object", "category"]).columns.tolist()
print("\n[CHECK] Numeric cols:", len(num_cols_final))
print("[CHECK] Categorical cols:", len(cat_cols_final))

# Optional: recompute correlations after casting/dropping (commented out by default)
# corr_final = df[num_cols_final].astype(float).corr().abs()
# print("\n[OPTIONAL] Correlation (post-drop):")
# print(corr_final)


