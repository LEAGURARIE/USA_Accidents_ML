# ==========================================================
# Apply Stage-2 Feature Selection to Train/Valid/Test
#
# - Rebuilds the SAME preprocessing as in FeatureSelection Stage 2
#   (ordinal encoding, one-hot, numeric imputation, bool handling, etc.)
# - Fits the ColumnTransformer ONLY on TRAIN_FE
# - Uses Stage-2 selected feature indices (Final_Index) to keep a subset of
#   the transformed feature matrix
# - Applies the fitted preprocessing + feature mask to:
#       * train_fe.csv
#       * valid_fe.csv
#       * test_fe.csv
# - Exports final processed matrices to:
#       src/data/raw/processed
#
# Notes:
#   * Train/Valid outputs: X_selected + target column in the same file
#   * Test output:
#       - test_stage2_processed.csv    → features only (no target)
#       - test_stage2_target.csv       → target column only (for later evaluation)
#   * No feature-selection models are refitted on valid/test
# ==========================================================

from __future__ import annotations

import os
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer

# -------------------------
# Configuration
# -------------------------

RANDOM_STATE = 42

# Paths: adjust if needed
PROJECT_ROOT = r"C:\Users\LEA GUR-ARIE\PycharmProjects\USA_Accidents_ML"

SPLIT_FE_DIR = os.path.join(
    PROJECT_ROOT,
    r"src\data\raw\interim\output_data\split\_fe_outputs"
)

TRAIN_FE_PATH = os.path.join(SPLIT_FE_DIR, "train_fe.csv")
VALID_FE_PATH = os.path.join(SPLIT_FE_DIR, "valid_fe.csv")
TEST_FE_PATH = os.path.join(SPLIT_FE_DIR, "test_fe.csv")

# Stage-2 selected indices (from FeatureSelection Stage 2)
STAGE2_IDX_CSV = os.path.join(SPLIT_FE_DIR, "stage2_selected_feature_indices.csv")

# Output directory for final processed datasets
PROCESSED_DIR = os.path.join(
    PROJECT_ROOT,
    r"src\data\raw\processed"
)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Target resolution preference: choose exactly one, prefer ordinal
TARGET_ORDINAL_CANDIDATES = ["Severity_bin_ord", "Severity_ord", "Severety_bin_ord"]
TARGET_FALLBACK_CANDIDATES = ["Severity", "Severity_bin", "Severety_bin"]

# Ordinal feature helpers (same as FeatureSelection)
ORDINAL_MAP: Dict[str, List[str]] = {
    # Example: "Visibility_2cat_ord": ["Low", "High"],
}
ORDINAL_SUFFIXES: Tuple[str, ...] = ("_ord", "_ordinal")

# Columns that are logically categorical even if stored as numbers
FORCED_CATEGORICAL_COLS: List[str] = [
    "FARS__SEX_3cat",
]

# Missing handling (no indicators). Textual missing aliases normalized to MISSING_TOKEN.
MISSING_TOKEN = "__MISSING__"
MISSING_ALIASES = {
    "__MISSING__", "missing", "Missing", "MISSING",
    "na", "NA", "Na", "none", "None", "NONE",
    "nan", "NaN", "NAN", "null", "Null", "NULL",
    ""
}

# Target-like column stems to drop from X (leak guard)
TARGET_STEMS = {"severity", "severety"}  # includes common typo

# -------------------------
# Utilities
# -------------------------


def coerce_boolish(series: pd.Series) -> pd.Series:
    """
    Attempts to coerce object-like columns that are mostly boolean tokens
    into numeric 0/1. Leaves numeric/bool dtypes unchanged.
    """
    if series.dtype == bool or series.dtype.kind in {"b", "i", "u", "f"}:
        return series
    s = series.astype(str).str.strip().str.lower()
    mask_boolish = s.isin({"true", "false", "1", "0", "nan", "none"})
    if float(mask_boolish.mean()) >= 0.9:
        mapping_bool = {"true": 1, "1": 1, "false": 0, "0": 0}
        out = s.map(mapping_bool)
        return pd.to_numeric(out, errors="coerce")
    return series


def normalize_missing_tokens(series: pd.Series) -> pd.Series:
    """
    Normalizes textual missing-like tokens to a single placeholder (MISSING_TOKEN).
    Leaves actual NaN values as NaN for imputers to handle.
    """
    if series.dtype.name not in ("object", "category", "string"):
        return series
    s = series.astype(str).str.strip()
    s = s.where(~s.isin(MISSING_ALIASES), MISSING_TOKEN)
    return s


def get_base_categorical_col(feature_name: str, cat_cols: List[str]) -> Optional[str]:
    """
    Maps a one-hot feature name to its original categorical column name.

    Example:
        feature_name = "cat__Borough_Manhattan"  ->  "Borough"

    Returns:
        Base categorical column name, or None if not an OHE feature.
    """
    if not feature_name.startswith("cat__"):
        return None
    for base_col_name in cat_cols:
        cand_prefix = f"cat__{base_col_name}_"
        if feature_name.startswith(cand_prefix):
            return base_col_name
    return None


def resolve_target_name(df: pd.DataFrame) -> str:
    """
    Resolves the target column name from the dataframe, preferring ordinal forms.
    """
    resolved: Optional[str] = None
    for cand in TARGET_ORDINAL_CANDIDATES + TARGET_FALLBACK_CANDIDATES:
        if cand in df.columns:
            resolved = cand
            break
    if resolved is None:
        raise ValueError(
            f"Could not find any target among: {TARGET_ORDINAL_CANDIDATES + TARGET_FALLBACK_CANDIDATES}"
        )
    print(f"[INFO] Using target column: {resolved}")
    return resolved


def split_x_y_with_leak_guard(df: pd.DataFrame, target_name: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Splits dataframe into (X, y) and removes any target-like columns from X.
    """
    y_series = df[target_name].copy()
    x_df = df.drop(columns=[target_name])

    leak_cols: Set[str] = set()
    for col_name in x_df.columns:
        low = col_name.lower()
        if any(stem in low for stem in TARGET_STEMS):
            leak_cols.add(col_name)

    if leak_cols:
        print(f"[LEAK-GUARD] Dropping target-like columns from X: {sorted(leak_cols)}")
        x_df = x_df.drop(columns=sorted(leak_cols))

    return x_df, y_series


def detect_ordinal_columns(all_feature_cols: List[str]) -> List[str]:
    """
    Detects ordinal feature columns by suffix or explicit ORDINAL_MAP membership.
    """
    name_ord_cols = [c for c in all_feature_cols if c.endswith(ORDINAL_SUFFIXES)]
    explicit_ord_cols = [c for c in all_feature_cols if c in ORDINAL_MAP]
    ordinal_cols_local = sorted(set(name_ord_cols + explicit_ord_cols))
    return ordinal_cols_local


def apply_basic_dtype_fixups(
    x_df: pd.DataFrame,
    ordinal_cols: List[str],
    forced_cats: List[str]
) -> pd.DataFrame:
    """
    Applies the same basic dtype and text normalization steps used in FeatureSelection:
    - Coerce boolean-like objects to numeric 0/1 where appropriate
    - Ensure ordinal columns are string-typed
    - Force some columns to string (categorical)
    - Normalize missing tokens
    - Cast pure bool dtype to int8
    """
    x_fixed = x_df.copy()

    # Coerce boolean-like object columns
    for obj_col_name in x_fixed.select_dtypes(include=["object"]).columns:
        x_fixed[obj_col_name] = coerce_boolish(x_fixed[obj_col_name])

    # Ensure ordinal features use textual dtype
    for ord_name in ordinal_cols:
        if ord_name in x_fixed.columns:
            x_fixed[ord_name] = x_fixed[ord_name].astype("string")

    # Force specific columns to be treated as categorical/strings
    for forced_col in forced_cats:
        if forced_col in x_fixed.columns and forced_col not in ordinal_cols:
            x_fixed[forced_col] = x_fixed[forced_col].astype("string")

    # Normalize textual missing tokens across all non-numeric columns
    for any_col_name in x_fixed.columns:
        x_fixed[any_col_name] = normalize_missing_tokens(x_fixed[any_col_name])

    # Cast pure boolean dtype to int8 (for numeric imputers)
    pure_bool_cols = x_fixed.select_dtypes(include=["bool"]).columns.tolist()
    if pure_bool_cols:
        x_fixed[pure_bool_cols] = x_fixed[pure_bool_cols].astype(np.int8)

    return x_fixed


def build_preprocess_from_train(
    x_train: pd.DataFrame,
    ordinal_cols: List[str]
) -> Tuple[ColumnTransformer, List[str], List[str], List[str], List[str]]:
    """
    Builds and fits the ColumnTransformer on TRAIN only, using the same logic
    as in FeatureSelection Stage 2.

    Returns:
        preprocess       : fitted ColumnTransformer
        feature_names    : list of transformed feature names
        numeric_cols     : numeric columns excluding ordinal ones
        categorical_cols : categorical-like columns excluding ordinal ones
        ordinal_cols     : ordinal columns (unchanged)
    """
    # Recompute groups after dtype fixes
    numeric_cols_all = x_train.select_dtypes(include=["number"]).columns.tolist()
    catlike_cols_all = x_train.select_dtypes(include=["object", "category", "string"]).columns.tolist()

    numeric_cols = [c for c in numeric_cols_all if c not in ordinal_cols]
    categorical_cols = [c for c in catlike_cols_all if c not in ordinal_cols]

    numeric_with_missing = [c for c in numeric_cols if x_train[c].isna().any()]
    numeric_without_missing = [c for c in numeric_cols if c not in numeric_with_missing]

    num_impute = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    def build_ordinal_pipeline(ordinal_columns: List[str]) -> Pipeline:
        """
        Builds an ordinal pipeline with explicit category order per feature.
        The MISSING_TOKEN is prepended and encoded as 0, real categories follow in sorted order.
        unknown_value is set to -1 to avoid clashing with the 0 assigned to MISSING_TOKEN.
        """
        if not ordinal_columns:
            return Pipeline(steps=[("dropper", "drop")])

        categories: List[List[str]] = []
        for ord_feat_name in ordinal_columns:
            if ord_feat_name in ORDINAL_MAP:
                cats_wo_missing = [v for v in ORDINAL_MAP[ord_feat_name] if v != MISSING_TOKEN]
                cats = [MISSING_TOKEN] + cats_wo_missing
            else:
                vals_series: pd.Series = x_train[ord_feat_name].dropna().astype(str)
                vals_sorted = sorted([v for v in vals_series.unique().tolist() if v != MISSING_TOKEN])
                cats = [MISSING_TOKEN] + vals_sorted
            categories.append(cats)

        return Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=MISSING_TOKEN)),
            ("encoder", OrdinalEncoder(
                categories=categories,
                handle_unknown="use_encoded_value",
                unknown_value=-1
            )),
        ])

    ordinal_pipe: Pipeline | str = build_ordinal_pipeline(ordinal_cols) if ordinal_cols else "drop"

    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value=MISSING_TOKEN)),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=int)),
    ])

    transformers: List[tuple] = []
    if numeric_with_missing:
        transformers.append(("num_imp", num_impute, numeric_with_missing))
    if numeric_without_missing:
        transformers.append(("num_passthrough", "passthrough", numeric_without_missing))
    if ordinal_cols:
        transformers.append(("ord", ordinal_pipe, ordinal_cols))
    if categorical_cols:
        transformers.append(("cat", categorical_pipe, categorical_cols))

    preprocess = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=True,
    )

    # Fit on TRAIN only
    x_train_t = preprocess.fit_transform(x_train)
    feature_names = list(preprocess.get_feature_names_out())
    print(f"[INFO] Fitted ColumnTransformer on TRAIN: {x_train_t.shape[1]} transformed features")

    return preprocess, feature_names, numeric_cols, categorical_cols, ordinal_cols


def export_processed_split(
    split_name: str,
    x_selected: np.ndarray,
    y_series: Optional[pd.Series],
    out_dir: str,
    feature_names_selected: List[str],
) -> None:
    """
    Exports a processed split.
    If y_series is provided → X_selected + target in the same file.
    If y_series is None       → X_selected only (for example, TEST features).
    """
    df_out = pd.DataFrame(x_selected, columns=feature_names_selected)

    if y_series is not None:
        df_out[y_series.name] = y_series.values

    csv_path = os.path.join(out_dir, f"{split_name}_stage2_processed.csv")
    parquet_path = os.path.join(out_dir, f"{split_name}_stage2_processed.parquet")

    df_out.to_csv(csv_path, index=False)
    try:
        df_out.to_parquet(parquet_path, index=False)
    except Exception as ex:
        print(f"[WARN] Could not write Parquet for {split_name}: {ex}")

    print(f"[OK] Saved {split_name} to:")
    print(f"     {csv_path}")
    if os.path.exists(parquet_path):
        print(f"     {parquet_path}")


def export_target_vector(
    split_name: str,
    y_series: pd.Series,
    out_dir: str
) -> None:
    """
    Exports only the target column for a given split.
    Used for TEST to keep labels separate from the feature matrix.
    """
    df_target = pd.DataFrame({y_series.name: y_series.values})
    csv_path = os.path.join(out_dir, f"{split_name}_stage2_target.csv")
    df_target.to_csv(csv_path, index=False)
    print(f"[OK] Saved {split_name} target to:")
    print(f"     {csv_path}")


# -------------------------
# MAIN
# -------------------------


def main() -> None:
    # --- Load FE splits ---
    if not os.path.exists(TRAIN_FE_PATH):
        raise FileNotFoundError(f"TRAIN_FE not found: {TRAIN_FE_PATH}")
    if not os.path.exists(VALID_FE_PATH):
        raise FileNotFoundError(f"VALID_FE not found: {VALID_FE_PATH}")
    if not os.path.exists(TEST_FE_PATH):
        raise FileNotFoundError(f"TEST_FE not found: {TEST_FE_PATH}")

    train_df = pd.read_csv(TRAIN_FE_PATH)
    valid_df = pd.read_csv(VALID_FE_PATH)
    test_df = pd.read_csv(TEST_FE_PATH)

    print("[INFO] Loaded FE splits:")
    print("  train_fe:", train_df.shape)
    print("  valid_fe:", valid_df.shape)
    print("  test_fe :", test_df.shape)

    # --- Resolve target column name ONCE ---
    resolved_target_col = resolve_target_name(train_df)

    # --- Split X/y for each split and apply leak guard ---
    x_train_raw, y_train = split_x_y_with_leak_guard(train_df, resolved_target_col)
    x_valid_raw, y_valid = split_x_y_with_leak_guard(valid_df, resolved_target_col)
    x_test_raw,  y_test  = split_x_y_with_leak_guard(test_df,  resolved_target_col)

    # --- Detect ordinal columns based on TRAIN schema ---
    all_feature_cols_train = x_train_raw.columns.tolist()
    ordinal_cols = detect_ordinal_columns(all_feature_cols_train)

    print(f"[INFO] Ordinal columns (by name/map): {ordinal_cols}")

    # --- Apply basic dtype + text fixups to each split (same logic) ---
    x_train_fixed = apply_basic_dtype_fixups(x_train_raw, ordinal_cols, FORCED_CATEGORICAL_COLS)
    x_valid_fixed = apply_basic_dtype_fixups(x_valid_raw, ordinal_cols, FORCED_CATEGORICAL_COLS)
    x_test_fixed  = apply_basic_dtype_fixups(x_test_raw,  ordinal_cols, FORCED_CATEGORICAL_COLS)

    # --- Build and fit ColumnTransformer on TRAIN only ---
    preprocess, feature_names, numeric_cols, categorical_cols, ordinal_cols_final = \
        build_preprocess_from_train(x_train_fixed, ordinal_cols)

    _name_to_idx = {fname: i for i, fname in enumerate(feature_names)}

    print(f"[INFO] Numeric cols: {len(numeric_cols)}")
    print(f"[INFO] Ordinal cols: {len(ordinal_cols_final)} | Categorical cols: {len(categorical_cols)}")

    # --- Transform all splits with the fitted preprocessing ---
    x_train_t = np.asarray(preprocess.transform(x_train_fixed))
    x_valid_t = np.asarray(preprocess.transform(x_valid_fixed))
    x_test_t  = np.asarray(preprocess.transform(x_test_fixed))

    print(f"[INFO] Transformed shapes:")
    print(f"  TRAIN: {x_train_t.shape}")
    print(f"  VALID: {x_valid_t.shape}")
    print(f"  TEST : {x_test_t.shape}")

    # --- Load Stage-2 selected feature indices ---
    if not os.path.exists(STAGE2_IDX_CSV):
        raise FileNotFoundError(f"Stage-2 indices CSV not found: {STAGE2_IDX_CSV}")

    stage2_idx_df = pd.read_csv(STAGE2_IDX_CSV)
    if "Final_Index" not in stage2_idx_df.columns:
        raise ValueError("Stage-2 indices file must contain a 'Final_Index' column.")

    stage2_indices = stage2_idx_df["Final_Index"].to_numpy(dtype=int)
    feature_names_selected = stage2_idx_df["Feature"].tolist()
    print(f"[INFO] Stage-2 selected feature count: {stage2_indices.size}")

    # Sanity check: ensure indices are within range
    max_idx = x_train_t.shape[1] - 1
    if (stage2_indices < 0).any() or (stage2_indices > max_idx).any():
        raise ValueError(
            f"Stage-2 indices out of bounds. Max index in transformed X is {max_idx}, "
            f"but got indices in [{stage2_indices.min()}, {stage2_indices.max()}]."
        )

    # --- Apply Stage-2 mask to all splits ---
    x_train_sel = x_train_t[:, stage2_indices]
    x_valid_sel = x_valid_t[:, stage2_indices]
    x_test_sel  = x_test_t[:, stage2_indices]

    print(f"[INFO] After Stage-2 selection (columns): {x_train_sel.shape[1]}")

    # --- Export processed splits ---
    # Train & Valid: features + target in same file
    export_processed_split("train", x_train_sel, y_train, PROCESSED_DIR, feature_names_selected)
    export_processed_split("valid", x_valid_sel, y_valid, PROCESSED_DIR, feature_names_selected)

    # Test: features only, and target in a separate file
    export_processed_split("test", x_test_sel, None, PROCESSED_DIR, feature_names_selected)
    export_target_vector("test", y_test, PROCESSED_DIR)

    print("\n✅ Done. Stage-2 processed datasets are ready.")
    print("   Output directory:", PROCESSED_DIR)


if __name__ == "__main__":
    main()
