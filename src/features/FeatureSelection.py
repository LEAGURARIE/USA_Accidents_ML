# ==========================================================
# Feature Selection Vote (LogisticL1, LogisticL2, GBC, RFC, LinearSVC)
#
# Classification-based feature selection for binary/ordinal target (Severity)
# ==========================================================
from src.utils.helpers import *

# File-specific configuration
ALPHA_LASSO = 1.0
ALPHA_RIDGE = 5.0

CSV_PATH = os.path.join(FE_OUTPUT_DIR, "train_fe.csv")

TARGET_FALLBACK_CANDIDATES = ["Severity", "Severity_bin", "Severety_bin"]
TARGET_IS_ORDINAL = True
TARGET_ORDER: List[str] = ["0", "1"]

ORDINAL_MAP: Dict[str, List[str]] = {}
ORDINAL_SUFFIXES = ("_ord", "_ordinal")

FORCED_CATEGORICAL_COLS: List[str] = ["FARS__SEX_3cat"]

MISSING_TOKEN = "__MISSING__"
MISSING_ALIASES = {
    "__MISSING__", "missing", "Missing", "MISSING",
    "na", "NA", "Na", "none", "None", "NONE",
    "nan", "NaN", "NAN", "null", "Null", "NULL", ""
}

SELECT_SUM_THRESHOLD = 3
TOP_K_PRINT = 40

RFE_N_FEATURES_TO_SELECT = 20
PERM_N_REPEATS = 5

DOMAIN_BASE_COLS_STAGE2: List[str] = ["ALC_category", "FARS__SEX_3cat"]


# Utility functions
def coerce_boolish(series: pd.Series) -> pd.Series:
    """Coerce object-like columns that are mostly boolean tokens into numeric 0/1."""
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
    """Normalizes textual missing-like tokens to MISSING_TOKEN."""
    if series.dtype.name not in ("object", "category", "string"):
        return series
    s = series.astype(str).str.strip()
    s = s.where(~s.isin(MISSING_ALIASES), MISSING_TOKEN)
    return s


def get_base_categorical_col(feature_name: str, cat_cols: List[str]) -> Optional[str]:
    """Maps a one-hot feature name to its original categorical column name."""
    if not feature_name.startswith("cat__"):
        return None
    for cat_base in cat_cols:
        cand_prefix = f"cat__{cat_base}_"
        if feature_name.startswith(cand_prefix):
            return cat_base
    return None


# Load data
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

df: pd.DataFrame = load_csv_safe(CSV_PATH)

# Resolve and prepare target
target_col: Optional[str] = None
for cand in TARGET_ORDINAL_CANDIDATES + TARGET_FALLBACK_CANDIDATES:
    if cand in df.columns:
        target_col = cand
        break
if target_col is None:
    raise ValueError(
        f"Could not find any target among: {TARGET_ORDINAL_CANDIDATES + TARGET_FALLBACK_CANDIDATES}"
    )

print_info(f"Using target column: {target_col}")

y_raw: pd.Series = df[target_col]
target_mapping: Optional[Dict[str, int]] = None

if TARGET_IS_ORDINAL:
    if pd.api.types.is_numeric_dtype(y_raw):
        y: pd.Series = pd.to_numeric(y_raw, errors="coerce")
    else:
        y_str: pd.Series = y_raw.astype(str)
        if TARGET_ORDER:
            ordered_classes = [str(v) for v in TARGET_ORDER]
        else:
            ordered_classes = sorted(pd.Series(y_str.dropna().unique()).tolist())
        map_local: Dict[str, int] = {cls: i for i, cls in enumerate(ordered_classes)}
        y = y_str.map(map_local)
        target_mapping = map_local
else:
    y = y_raw

if y.isna().any():
    keep_mask = ~y.isna()
    df = df.loc[keep_mask].reset_index(drop=True)
    y = y.loc[keep_mask].reset_index(drop=True)

# Build X and apply leak guard
X: pd.DataFrame = df.drop(columns=[target_col])

TARGET_STEMS = {"severity", "severety"}
leak_cols: Set[str] = set()
for col_name in X.columns:
    low = col_name.lower()
    if any(stem in low for stem in TARGET_STEMS):
        leak_cols.add(col_name)

if leak_cols:
    print("[LEAK-GUARD] Dropping target-like columns from X:", sorted(leak_cols))
    X = X.drop(columns=sorted(leak_cols))

# Coerce boolean-like object columns
for obj_col_name in X.select_dtypes(include=["object"]).columns:
    X[obj_col_name] = coerce_boolish(X[obj_col_name])

# Detect ordinal feature columns
all_cols = X.columns.tolist()
name_ord_cols = [c for c in all_cols if c.endswith(ORDINAL_SUFFIXES)]
explicit_ord_cols = [c for c in all_cols if c in ORDINAL_MAP]
ordinal_cols = sorted(set(name_ord_cols + explicit_ord_cols))

for ord_name in ordinal_cols:
    if ord_name in X.columns:
        X[ord_name] = X[ord_name].astype("string")

for forced_col in FORCED_CATEGORICAL_COLS:
    if forced_col in X.columns and forced_col not in ordinal_cols:
        X[forced_col] = X[forced_col].astype("string")

for any_col_name in X.columns:
    X[any_col_name] = normalize_missing_tokens(X[any_col_name])

pure_bool_cols = X.select_dtypes(include=["bool"]).columns.tolist()
if pure_bool_cols:
    X[pure_bool_cols] = X[pure_bool_cols].astype(np.int8)

# Recompute column groups
numeric_cols_all = X.select_dtypes(include=["number"]).columns.tolist()
catlike_cols_all = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()

numeric_cols = [c for c in numeric_cols_all if c not in ordinal_cols]
categorical_cols = [c for c in catlike_cols_all if c not in ordinal_cols]

numeric_with_missing = [c for c in numeric_cols if X[c].isna().any()]
numeric_without_missing = [c for c in numeric_cols if c not in numeric_with_missing]

# Preprocessing pipelines
num_impute = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])


def build_ordinal(ordinal_columns: List[str]) -> Pipeline:
    """Builds an ordinal pipeline with explicit category order per feature."""
    if not ordinal_columns:
        return Pipeline(steps=[("dropper", "drop")])

    categories: List[List[str]] = []
    for ord_feat_name in ordinal_columns:
        if ord_feat_name in ORDINAL_MAP:
            cats_wo_missing = [v for v in ORDINAL_MAP[ord_feat_name] if v != MISSING_TOKEN]
            cats = [MISSING_TOKEN] + cats_wo_missing
        else:
            vals_series: pd.Series = X[ord_feat_name].dropna().astype(str)
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


ordinal_pipe: Pipeline | str = build_ordinal(ordinal_cols) if ordinal_cols else "drop"

categorical_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value=MISSING_TOKEN)),
    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=int)),
])

# ColumnTransformer assembly
transformers = []
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

# Fit preprocessing and build feature matrix
X_t_np: np.ndarray = np.asarray(preprocess.fit_transform(X))
feature_names: List[str] = list(preprocess.get_feature_names_out())
name_to_idx = {fname: i for i, fname in enumerate(feature_names)}

# Stage 1: Model-based selection votes
scaler_linear = StandardScaler(with_mean=True, with_std=True)
X_scaled = scaler_linear.fit_transform(X_t_np)

# L1 Logistic Regression
logistic_l1 = LogisticRegression(
    penalty="l1", solver="saga", C=1.0/ALPHA_LASSO,
    random_state=RANDOM_STATE, max_iter=10000
)
logistic_l1.fit(X_scaled, y)
logistic_l1_coef = logistic_l1.coef_.ravel() if logistic_l1.coef_.ndim > 1 else logistic_l1.coef_
logistic_l1_selected = np.asarray(np.abs(logistic_l1_coef) > 0, dtype=int)

# L2 Logistic Regression
logistic_l2_base = LogisticRegression(
    penalty="l2", C=1.0/ALPHA_RIDGE, random_state=RANDOM_STATE, max_iter=10000
)
logistic_l2_sel = SelectFromModel(estimator=logistic_l2_base, threshold="median")
logistic_l2_sel.fit(X_scaled, y)
logistic_l2_selected = np.asarray(logistic_l2_sel.get_support(), dtype=int)

# Gradient Boosting Classifier
gb_sel = SelectFromModel(
    estimator=GradientBoostingClassifier(random_state=RANDOM_STATE),
    threshold="1.25*median"
)
gb_sel.fit(X_t_np, y)
gb_selected = np.asarray(gb_sel.get_support(), dtype=int)

# Random Forest Classifier
rf_sel = SelectFromModel(
    estimator=RandomForestClassifier(n_estimators=400, random_state=RANDOM_STATE, n_jobs=-1),
    threshold="1.5*median"
)
rf_sel.fit(X_t_np, y)
rf_selected = np.asarray(rf_sel.get_support(), dtype=int)

# Linear SVC
lin_svc = LinearSVC(random_state=RANDOM_STATE, C=1.0, dual="auto", max_iter=10000)
svc_sel = SelectFromModel(estimator=lin_svc, threshold="median")
svc_sel.fit(X_scaled, y)
svc_selected = np.asarray(svc_sel.get_support(), dtype=int)

# Voting table (Stage 1)
selection_df = pd.DataFrame({
    "Feature": feature_names,
    "LogisticL1": logistic_l1_selected,
    "LogisticL2": logistic_l2_selected,
    "GradientBoost": gb_selected,
    "RandomForest": rf_selected,
    "LinearSVC": svc_selected,
})
selection_df["Sum"] = selection_df[["LogisticL1", "LogisticL2", "GradientBoost", "RandomForest", "LinearSVC"]].sum(axis=1)
selection_df.sort_values(["Sum", "Feature"], ascending=[False, True], inplace=True)

# Group Feature Selection for categorical OHE variables (Stage 1)
base_selected_df = selection_df[selection_df["Sum"] >= SELECT_SUM_THRESHOLD].copy()

selected_base_cols: Set[str] = set()
for selected_feature in base_selected_df["Feature"]:
    base_col = get_base_categorical_col(selected_feature, categorical_cols)
    if base_col is not None:
        selected_base_cols.add(base_col)

group_mask_stage1 = np.zeros(len(selection_df), dtype=bool)
if selected_base_cols:
    for base_col in selected_base_cols:
        group_prefix = f"cat__{base_col}_"
        group_mask_stage1 |= selection_df["Feature"].str.startswith(group_prefix).to_numpy()

final_selected_df = selection_df[
    (selection_df["Sum"] >= SELECT_SUM_THRESHOLD) | group_mask_stage1
].copy()
final_selected_df["Index"] = final_selected_df["Feature"].map(name_to_idx)

final_selected_df = cast(pd.DataFrame, final_selected_df).sort_values(by="Index").reset_index(drop=True)

# Stage 1 exports
out_dir = os.path.dirname(CSV_PATH) or "."
votes_csv = os.path.join(out_dir, "feature_votes.csv")
selection_df.to_csv(votes_csv, index=False)

selected_csv = os.path.join(out_dir, f"selected_features_sum_ge_{SELECT_SUM_THRESHOLD}.csv")
final_selected_df.to_csv(selected_csv, index=False)

selected_names_txt = os.path.join(out_dir, f"selected_feature_names_sum_ge_{SELECT_SUM_THRESHOLD}.txt")
with open(selected_names_txt, "w", encoding="utf-8") as fh:
    for fname in final_selected_df["Feature"].tolist():
        fh.write(f"{fname}\n")

idx_csv = os.path.join(out_dir, f"selected_feature_indices_sum_ge_{SELECT_SUM_THRESHOLD}.csv")
final_selected_df[["Index", "Feature", "LogisticL1", "LogisticL2", "GradientBoost", "RandomForest", "LinearSVC", "Sum"]].to_csv(
    idx_csv, index=False
)

# Stage 2: XGBoost + RFE + Permutation Importance
domain_mask = np.zeros(len(selection_df), dtype=bool)
for domain_base in DOMAIN_BASE_COLS_STAGE2:
    if domain_base in categorical_cols:
        dom_prefix = f"cat__{domain_base}_"
        domain_mask |= selection_df["Feature"].str.startswith(dom_prefix).to_numpy()

domain_stage2_df = selection_df[domain_mask].copy()

stage1_block = final_selected_df.loc[:, ["Feature", "Index"]].copy()
stage1_block.columns = ["Feature", "Stage1_Index"]

stage2_union_df = pd.concat(
    [
        stage1_block,
        pd.DataFrame({
            "Feature": domain_stage2_df["Feature"],
            "Stage1_Index": domain_stage2_df["Feature"].map(name_to_idx),
        }),
    ],
    ignore_index=True,
)

stage2_union_df.drop_duplicates(subset="Feature", inplace=True)
stage2_union_df["Stage1_Index"] = stage2_union_df["Stage1_Index"].astype(int)

stage2_feature_names: List[str] = stage2_union_df["Feature"].tolist()
stage2_indices: np.ndarray = stage2_union_df["Stage1_Index"].to_numpy()
X_stage2: np.ndarray = X_t_np[:, stage2_indices]

# XGBoost classifier
xgb_est = XGBClassifier(
    random_state=RANDOM_STATE,
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    n_jobs=-1,
)
xgb_est.fit(X_stage2, y)

xgb_importances: np.ndarray = xgb_est.feature_importances_

# RFE over XGBoost estimator
n_features_rfe = min(RFE_N_FEATURES_TO_SELECT, X_stage2.shape[1])
rfe = RFE(estimator=xgb_est, n_features_to_select=n_features_rfe, step=1)
rfe.fit(X_stage2, y)
rfe_support = rfe.support_.astype(int)
rfe_ranking = rfe.ranking_

# Permutation importance on Stage-2 features
perm_result = permutation_importance(
    xgb_est,
    X_stage2,
    y,
    n_repeats=PERM_N_REPEATS,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
perm_mean = perm_result.importances_mean
perm_std = perm_result.importances_std

# Assemble Stage-2 metrics table
stage2_df = pd.DataFrame({
    "Feature": stage2_feature_names,
    "Stage1_Index": stage2_indices,
    "XGB_importance": xgb_importances,
    "RFE_selected": rfe_support,
    "RFE_rank": rfe_ranking,
    "PermImp_mean": perm_mean,
    "PermImp_std": perm_std,
})

# Combined Stage-2 selection rule
imp_median = float(np.median(xgb_importances)) if xgb_importances.size > 0 else 0.0
perm_median = float(np.median(perm_mean)) if perm_mean.size > 0 else 0.0

stage2_keep_mask = (
    (stage2_df["RFE_selected"] == 1)
    | (stage2_df["XGB_importance"] >= imp_median)
    | (stage2_df["PermImp_mean"] >= perm_median)
)

stage2_selected_base_df = stage2_df[stage2_keep_mask].copy()

# Group Feature Selection in Stage 2
stage2_base_cols: Set[str] = set()
for feat_name in stage2_selected_base_df["Feature"]:
    base_col = get_base_categorical_col(feat_name, categorical_cols)
    if base_col is not None:
        stage2_base_cols.add(base_col)

group_mask_stage2 = np.zeros(len(stage2_df), dtype=bool)
if stage2_base_cols:
    for base_col in stage2_base_cols:
        group_prefix = f"cat__{base_col}_"
        group_mask_stage2 |= stage2_df["Feature"].str.startswith(group_prefix).to_numpy()

final_stage2_grouped_df = stage2_df[
    stage2_keep_mask | group_mask_stage2
].copy()
final_stage2_grouped_df["Final_Index"] = final_stage2_grouped_df["Stage1_Index"]

final_stage2_grouped_df = cast(pd.DataFrame, final_stage2_grouped_df).sort_values(by="Final_Index").reset_index(drop=True)

# Stage 2 exports
stage2_full_csv = os.path.join(out_dir, "stage2_xgb_rfe_perm_metrics.csv")
stage2_df.to_csv(stage2_full_csv, index=False)

stage2_selected_csv = os.path.join(out_dir, "stage2_selected_features.csv")
final_stage2_grouped_df.to_csv(stage2_selected_csv, index=False)

stage2_names_txt = os.path.join(out_dir, "stage2_selected_feature_names.txt")
with open(stage2_names_txt, "w", encoding="utf-8") as fh:
    for fname in final_stage2_grouped_df["Feature"].tolist():
        fh.write(f"{fname}\n")

stage2_idx_csv = os.path.join(out_dir, "stage2_selected_feature_indices.csv")
final_stage2_grouped_df[[
    "Final_Index", "Feature", "XGB_importance", "RFE_selected", "RFE_rank", "PermImp_mean", "PermImp_std",
]].to_csv(stage2_idx_csv, index=False)

# Console summary
print(f"Rows: {X.shape[0]}  |  Original cols: {X.shape[1]}  |  Expanded features: {X_t_np.shape[1]}")
print(f"Numeric cols: {len(numeric_cols)} (with missing: {len(numeric_with_missing)})")
print(f"Ordinal cols: {len(ordinal_cols)}  |  Categorical cols: {len(categorical_cols)}")
if TARGET_IS_ORDINAL and target_mapping is not None:
    print("\nTarget ordinal mapping:")
    for k, v in target_mapping.items():
        print(f"  {k} -> {v}")

print(f"\n=== Top {TOP_K_PRINT} features by votes (Stage 1) ===")
print(selection_df.head(TOP_K_PRINT).to_string(index=False))

counts = selection_df[["LogisticL1", "LogisticL2", "GradientBoost", "RandomForest", "LinearSVC"]].sum()
print("\nSelected counts per model (Stage 1):")
print(counts.to_string())

print(f"\nSelected (Sum >= {SELECT_SUM_THRESHOLD}) before grouping: {base_selected_df.shape[0]} features")
print(f"Selected after grouping (Stage 1): {final_selected_df.shape[0]} features")
if selected_base_cols:
    print("\nGrouped categorical bases in Stage 1:")
    for base_col in sorted(selected_base_cols):
        print(" -", base_col)

present_domain_bases = []
for domain_base in DOMAIN_BASE_COLS_STAGE2:
    prefix = f"cat__{domain_base}_"
    if any(stage2_union_df["Feature"].str.startswith(prefix)):
        present_domain_bases.append(domain_base)

print(f"\nStage 2: XGBoost + RFE + Permutation importance on {X_stage2.shape[1]} features")
if present_domain_bases:
    print("Domain-important bases included as Stage-2 candidates:")
    for db_name in present_domain_bases:
        print(" -", db_name)

if stage2_base_cols:
    print("\nGrouped categorical bases in Stage 2:")
    for base_col in sorted(stage2_base_cols):
        print(" -", base_col)

print(f"Stage 2 final grouped selected features: {final_stage2_grouped_df.shape[0]}")

print("\nSaved:")
print(" -", votes_csv)
print(" -", selected_csv)
print(" -", selected_names_txt)
print(" -", idx_csv)
print(" -", stage2_full_csv)
print(" -", stage2_selected_csv)
print(" -", stage2_names_txt)
print(" -", stage2_idx_csv)
