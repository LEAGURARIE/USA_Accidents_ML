# ==========================================================
# Model Selection Stage 2 - Two-Stage Hyperparameter Tuning
#
# Stage-2 processed inputs:
#   - train_stage2_processed.csv
#   - valid_stage2_processed.csv
#
# What this script does:
#   1) Loads train/valid Stage-2 processed datasets.
#   2) For each model (RandomForestClassifier, XGBClassifier):
#        a) Runs a WIDE RandomizedSearchCV
#        b) Builds a NARROW GridSearchCV around the best random params
#        c) Evaluates the best GridSearch model on VALID
#   3) Saves a summary table
# ==========================================================
from src.utils.helpers import *

# File-specific configuration
RESULTS_XLSX_PATH = os.path.join(PROCESSED_DIR, "model_selection_stage2.xlsx")
TARGET_FALLBACK_CANDIDATES: List[str] = ["Severity", "Severity_bin", "Severety_bin"]
TARGET_STEMS = {"severity", "severety"}


class ModelConfig(TypedDict):
    model: Any
    random_grid: Dict[str, List[Any]]


def resolve_target_name(df: pd.DataFrame) -> str:
    """Resolves the target column name from the dataframe."""
    target_name: Optional[str] = None
    for cand in TARGET_ORDINAL_CANDIDATES + TARGET_FALLBACK_CANDIDATES:
        if cand in df.columns:
            target_name = cand
            break
    if target_name is None:
        raise ValueError(
            f"Could not find any target among: "
            f"{TARGET_ORDINAL_CANDIDATES + TARGET_FALLBACK_CANDIDATES}"
        )
    print_info(f"Using target column: {target_name}")
    return target_name


def split_x_y_with_leak_guard(df: pd.DataFrame, target_name: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Splits dataframe into (X, y) and removes any target-like columns from X."""
    if target_name not in df.columns:
        raise ValueError(f"Target column '{target_name}' not found in dataframe.")

    y_series = df[target_name].copy()
    x_df = df.drop(columns=[target_name])

    leak_cols: List[str] = []
    for col_name in x_df.columns:
        low = col_name.lower()
        if any(stem in low for stem in TARGET_STEMS):
            leak_cols.append(col_name)

    if leak_cols:
        print(f"[LEAK-GUARD] Dropping target-like columns from X: {sorted(leak_cols)}")
        x_df = x_df.drop(columns=sorted(leak_cols))

    return x_df, y_series


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """Computes standard classification metrics."""
    accuracy_val = float(accuracy_score(y_true, y_pred))
    precision_val = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    recall_val = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    f1_val = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    roc_auc_val: float = 0.0
    if y_prob is not None:
        try:
            if y_prob.ndim == 1 or y_prob.shape[1] == 1:
                roc_auc_val = float(roc_auc_score(y_true, y_prob.ravel()))
            else:
                roc_auc_val = float(roc_auc_score(y_true, y_prob[:, 1]))
        except (ValueError, IndexError):
            roc_auc_val = 0.0

    return {
        "Accuracy": accuracy_val,
        "Precision": precision_val,
        "Recall": recall_val,
        "F1": f1_val,
        "ROC_AUC": roc_auc_val,
    }


def _neighbor_int(value: int, step: int, low: int, high: int) -> List[int]:
    """Helper to build an integer neighborhood."""
    v = int(value)
    candidates = {v}
    candidates.add(max(low, v - step))
    candidates.add(min(high, v + step))
    return sorted(candidates)


def _neighbor_float(value: float, factor: float, low: float, high: float) -> List[float]:
    """Helper to build a float neighborhood."""
    v = float(value)
    candidates = {v}
    candidates.add(max(low, v / factor))
    candidates.add(min(high, v * factor))
    return sorted({round(c, 4) for c in candidates})


def build_rf_refined_grid(best_params: Dict[str, Any]) -> Dict[str, List[Any]]:
    """Builds a narrow GridSearch param grid for RandomForestClassifier."""
    grid: Dict[str, List[Any]] = {}

    n_est = int(best_params.get("n_estimators", 400))
    grid["n_estimators"] = _neighbor_int(n_est, step=100, low=100, high=1000)

    if best_params.get("max_depth") is None:
        grid["max_depth"] = [None, 8, 12, 16]
    else:
        md = int(best_params["max_depth"])
        grid["max_depth"] = _neighbor_int(md, step=2, low=4, high=24)

    mss = int(best_params.get("min_samples_split", 2))
    grid["min_samples_split"] = _neighbor_int(mss, step=2, low=2, high=20)

    msl = int(best_params.get("min_samples_leaf", 1))
    grid["min_samples_leaf"] = _neighbor_int(msl, step=1, low=1, high=10)

    best_max_features = best_params.get("max_features", "sqrt")
    if best_max_features == "sqrt":
        grid["max_features"] = ["sqrt", "log2"]
    elif best_max_features == "log2":
        grid["max_features"] = ["log2", "sqrt"]
    else:
        grid["max_features"] = [best_max_features]

    return grid


def build_xgb_refined_grid(best_params: Dict[str, Any]) -> Dict[str, List[Any]]:
    """Builds a narrow GridSearch param grid for XGBClassifier."""
    grid: Dict[str, List[Any]] = {}

    n_est = int(best_params.get("n_estimators", 400))
    grid["n_estimators"] = _neighbor_int(n_est, step=100, low=200, high=800)

    md = int(best_params.get("max_depth", 6))
    grid["max_depth"] = _neighbor_int(md, step=1, low=3, high=12)

    lr = float(best_params.get("learning_rate", 0.05))
    grid["learning_rate"] = _neighbor_float(lr, factor=2.0, low=0.01, high=0.3)

    subs = float(best_params.get("subsample", 0.8))
    grid["subsample"] = _neighbor_float(subs, factor=1.25, low=0.5, high=1.0)

    colsample = float(best_params.get("colsample_bytree", 0.8))
    grid["colsample_bytree"] = _neighbor_float(colsample, factor=1.25, low=0.5, high=1.0)

    return grid


def main() -> None:
    # Load processed train/valid (Stage 2)
    if not os.path.exists(TRAIN_PROCESSED_PATH):
        raise FileNotFoundError(f"TRAIN processed file not found: {TRAIN_PROCESSED_PATH}")
    if not os.path.exists(VALID_PROCESSED_PATH):
        raise FileNotFoundError(f"VALID processed file not found: {VALID_PROCESSED_PATH}")

    train_df = load_csv_safe(TRAIN_PROCESSED_PATH)
    valid_df = load_csv_safe(VALID_PROCESSED_PATH)

    print_info("Loaded processed splits:")
    print("  train_stage2_processed:", train_df.shape)
    print("  valid_stage2_processed:", valid_df.shape)

    target_name = resolve_target_name(train_df)

    x_train_df, y_train = split_x_y_with_leak_guard(train_df, target_name)
    x_valid_df, y_valid = split_x_y_with_leak_guard(valid_df, target_name)

    x_train_np = x_train_df.to_numpy(dtype=float)
    x_valid_np = x_valid_df.to_numpy(dtype=float)
    y_train_np = y_train.to_numpy(dtype=int)
    y_valid_np = y_valid.to_numpy(dtype=int)

    print_info(f"X_train shape: {x_train_np.shape}")
    print_info(f"X_valid shape: {x_valid_np.shape}")
    print_info(f"Class distribution in train: {np.bincount(y_train_np)}")
    print_info(f"Class distribution in valid: {np.bincount(y_valid_np)}")

    # Define models + RANDOM grids
    models_and_random_grids: Dict[str, ModelConfig] = {
        "RandomForestClassifier": {
            "model": RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=N_JOBS),
            "random_grid": {
                "n_estimators": [100, 200, 300, 400, 600, 800],
                "max_depth": [None, 4, 6, 8, 10, 12, 16, 20],
                "min_samples_split": [2, 5, 10, 20],
                "min_samples_leaf": [1, 2, 4, 8],
                "max_features": ["sqrt", "log2"],
            },
        }
    }

    if XGB_OK and XGBClassifier is not None:
        models_and_random_grids["XGBClassifier"] = {
            "model": XGBClassifier(
                objective="binary:logistic",
                random_state=RANDOM_STATE,
                n_jobs=N_JOBS,
                tree_method="hist",
                eval_metric="logloss",
            ),
            "random_grid": {
                "n_estimators": [200, 300, 400, 500, 600, 800],
                "max_depth": [3, 4, 5, 6, 8, 10],
                "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
                "subsample": [0.6, 0.8, 1.0],
                "colsample_bytree": [0.6, 0.8, 1.0],
            },
        }
    else:
        print_warn("xgboost not available - XGBClassifier will be skipped.")

    # CV setup (Stratified for classification)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    results_rows: List[Dict[str, Any]] = []

    # Two-stage tuning per model
    for model_name, cfg in models_and_random_grids.items():
        base_model: Any = cfg["model"]
        random_grid: Dict[str, List[Any]] = cfg["random_grid"]

        print_info(f"Stage 1 - RandomizedSearchCV for {model_name} ...")
        rand_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=random_grid,
            n_iter=20,
            scoring="f1_weighted",
            cv=cv,
            n_jobs=N_JOBS,
            random_state=RANDOM_STATE,
            verbose=1,
        )
        rand_search.fit(x_train_np, y_train_np)

        rand_best_score: float = float(rand_search.best_score_)
        rand_best_params: Dict[str, Any] = dict(rand_search.best_params_)

        print_info(f"{model_name} RandomizedSearch best CV F1: {rand_best_score:.6f}")
        print_info(f"{model_name} RandomizedSearch best params: {rand_best_params}")

        if model_name == "RandomForestClassifier":
            refined_grid = build_rf_refined_grid(rand_best_params)
        elif model_name == "XGBClassifier":
            refined_grid = build_xgb_refined_grid(rand_best_params)
        else:
            raise ValueError(f"Unsupported model in two-stage tuning: {model_name}")

        print_info(f"Stage 2 - GridSearchCV for {model_name} (refined around random best)...")
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=refined_grid,
            scoring="f1_weighted",
            cv=cv,
            n_jobs=N_JOBS,
            verbose=1,
        )
        grid_search.fit(x_train_np, y_train_np)

        grid_best_model: Any = grid_search.best_estimator_
        grid_best_score: float = float(grid_search.best_score_)
        grid_best_params: Dict[str, Any] = dict(grid_search.best_params_)

        print_info(f"{model_name} GridSearch best CV F1: {grid_best_score:.6f}")
        print_info(f"{model_name} GridSearch best params: {grid_best_params}")

        y_valid_pred = grid_best_model.predict(x_valid_np)

        y_valid_prob: Optional[np.ndarray] = None
        if hasattr(grid_best_model, "predict_proba"):
            y_valid_prob = grid_best_model.predict_proba(x_valid_np)

        metrics_dict = classification_metrics(y_valid_np, y_valid_pred, y_valid_prob)

        result_row: Dict[str, Any] = {
            "Model": model_name,
            "Rand_best_F1": rand_best_score,
            "Grid_best_F1": grid_best_score,
            "VALID_Accuracy": metrics_dict["Accuracy"],
            "VALID_Precision": metrics_dict["Precision"],
            "VALID_Recall": metrics_dict["Recall"],
            "VALID_F1": metrics_dict["F1"],
            "VALID_ROC_AUC": metrics_dict["ROC_AUC"],
            "Best_Params_Random": str(rand_best_params),
            "Best_Params_Grid": str(grid_best_params),
        }
        results_rows.append(result_row)

    results_df = pd.DataFrame(results_rows)
    results_df.sort_values(by="VALID_F1", ascending=False, inplace=True)
    results_df.reset_index(drop=True, inplace=True)

    print("\n=== VALID Results after Two-Stage Hyperparameter Tuning (sorted by VALID_F1) ===")
    print(results_df.to_string(index=False))

    # Save to Excel (or CSV if openpyxl not available)
    try:
        results_df.to_excel(RESULTS_XLSX_PATH, index=False)
        print_ok(f"Saved model selection results to Excel:\n     {RESULTS_XLSX_PATH}")
    except (OSError, PermissionError, ValueError, ModuleNotFoundError, ImportError) as exc:
        print_warn(f"Could not save Excel ({exc}), saving as CSV instead.")
        csv_path = RESULTS_XLSX_PATH.replace(".xlsx", ".csv")
        results_df.to_csv(csv_path, index=False)
        print_ok(f"Saved model selection results to CSV:\n     {csv_path}")


if __name__ == "__main__":
    main()
