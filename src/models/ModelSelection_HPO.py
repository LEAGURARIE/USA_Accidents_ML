# ==========================================================
# Model Selection Stage 2 – Two-Stage Hyperparameter Tuning
#
# Stage-2 processed inputs:
#   - train_stage2_processed.csv
#   - valid_stage2_processed.csv
#
# What this script does:
#   1) Loads train/valid Stage-2 processed datasets.
#   2) Resolves the target column and applies a leak guard.
#   3) For each model (RandomForest, XGB if available):
#        a) Runs a WIDE RandomizedSearchCV to explore a large hyperparameter space.
#        b) Builds a NARROW GridSearchCV around the best random params.
#        c) Evaluates the best GridSearch model on VALID and logs metrics.
#   4) Saves a summary table to Excel: model_selection_stage2.xlsx
#
# IMPORTANT:
#   - The test set is NOT touched here.
#   - This script is only for model + hyperparameter selection.
# ==========================================================
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple, TypedDict

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV

# Optional XGBoost
XGB_OK = True
try:
    import xgboost  # noqa: F401
    XGBRegressor = xgboost.XGBRegressor
except ImportError:
    # Ensure these names exist even if xgboost is not installed (for linters/type checkers)
    XGB_OK = False
    xgboost = None          # type: ignore[assignment]
    XGBRegressor = None     # type: ignore[assignment]

# -------------------------
# Configuration
# -------------------------
RANDOM_STATE = 42
N_JOBS = -1
CV_FOLDS = 3

PROJECT_ROOT = r"C:\Users\LEA GUR-ARIE\PycharmProjects\USA_Accidents_ML"
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "src", "data", "raw", "processed")

TRAIN_PROCESSED_PATH = os.path.join(PROCESSED_DIR, "train_stage2_processed.csv")
VALID_PROCESSED_PATH = os.path.join(PROCESSED_DIR, "valid_stage2_processed.csv")

RESULTS_XLSX_PATH = os.path.join(PROCESSED_DIR, "model_selection_stage2.xlsx")

# Target resolution preference
TARGET_ORDINAL_CANDIDATES: List[str] = ["Severity_bin_ord", "Severity_ord", "Severety_bin_ord"]
TARGET_FALLBACK_CANDIDATES: List[str] = ["Severity", "Severity_bin", "Severety_bin"]

# Defensive leak guard: drop any feature whose name looks like a target
TARGET_STEMS = {"severity", "severety"}


# -------------------------
# Typed model config
# -------------------------
class ModelConfig(TypedDict):
    model: Any                          # estimator with fit/predict
    random_grid: Dict[str, List[Any]]   # param_distributions for RandomizedSearchCV


# -------------------------
# Utilities
# -------------------------
def resolve_target_name(df: pd.DataFrame) -> str:
    """
    Resolves the target column name from the dataframe, preferring ordinal variants.
    """
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
    print(f"[INFO] Using target column: {target_name}")
    return target_name


def split_x_y_with_leak_guard(df: pd.DataFrame, target_name: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Splits dataframe into (X, y) and removes any target-like columns from X.
    """
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


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes standard regression metrics: MSE, RMSE, MAE, RMSLE, R2.
    """
    mse_val = metrics.mean_squared_error(y_true, y_pred)
    rmse_val = float(np.sqrt(mse_val))
    mae_val = metrics.mean_absolute_error(y_true, y_pred)
    r2_val = metrics.r2_score(y_true, y_pred)

    def safe_rmsle(y_t: np.ndarray, y_p: np.ndarray, clip_eps: float = 1e-9) -> float:
        """
        RMSLE with basic protection for non-positive values.
        If there are zeros/negatives, we shift everything so it becomes positive.
        """
        y_t = y_t.astype(float).copy()
        y_p = y_p.astype(float).copy()
        min_true = float(y_t.min())
        min_pred = float(y_p.min())
        if min_true <= 0 or min_pred <= 0:
            shift = max(0.0, 1.0 - min(min_true, min_pred))
            y_t = y_t + shift
            y_p = y_p + shift
        y_t = np.clip(y_t, clip_eps, None)
        y_p = np.clip(y_p, clip_eps, None)
        return float(np.sqrt(metrics.mean_squared_log_error(y_t, y_p)))

    rmsle_val = safe_rmsle(y_true, y_pred)

    return {
        "MSE": mse_val,
        "RMSE": rmse_val,
        "MAE": mae_val,
        "RMSLE": rmsle_val,
        "R2": r2_val,
    }


# -------------------------
# Helper: build refined grids
# -------------------------
def _neighbor_int(value: int, step: int, low: int, high: int) -> List[int]:
    """
    Helper to build an integer neighborhood, clamped between [low, high].
    """
    v = int(value)
    candidates = {v}
    candidates.add(max(low, v - step))
    candidates.add(min(high, v + step))
    return sorted(candidates)


def _neighbor_float(value: float, factor: float, low: float, high: float) -> List[float]:
    """
    Helper to build a float neighborhood by multiplying/dividing by factor.
    """
    v = float(value)
    candidates = {v}
    candidates.add(max(low, v / factor))
    candidates.add(min(high, v * factor))
    return sorted({round(c, 4) for c in candidates})


def build_rf_refined_grid(best_params: Dict[str, Any]) -> Dict[str, List[Any]]:
    """
    Builds a narrow GridSearch param grid around the best RandomForest params
    found by RandomizedSearchCV.
    """
    grid: Dict[str, List[Any]] = {}

    # n_estimators neighborhood
    n_est = int(best_params.get("n_estimators", 400))
    grid["n_estimators"] = _neighbor_int(n_est, step=100, low=100, high=1000)

    # max_depth neighborhood
    if best_params.get("max_depth") is None:
        grid["max_depth"] = [None, 8, 12, 16]
    else:
        md = int(best_params["max_depth"])
        grid["max_depth"] = _neighbor_int(md, step=2, low=4, high=24)

    # min_samples_split neighborhood
    mss = int(best_params.get("min_samples_split", 2))
    grid["min_samples_split"] = _neighbor_int(mss, step=2, low=2, high=20)

    # min_samples_leaf neighborhood
    msl = int(best_params.get("min_samples_leaf", 1))
    grid["min_samples_leaf"] = _neighbor_int(msl, step=1, low=1, high=10)

    # max_features neighborhood
    best_max_features = best_params.get("max_features", "sqrt")
    if best_max_features == "sqrt":
        grid["max_features"] = ["sqrt", "log2"]
    elif best_max_features == "log2":
        grid["max_features"] = ["log2", "sqrt"]
    else:
        grid["max_features"] = [best_max_features]

    return grid


def build_xgb_refined_grid(best_params: Dict[str, Any]) -> Dict[str, List[Any]]:
    """
    Builds a narrow GridSearch param grid around the best XGBRegressor params
    found by RandomizedSearchCV.
    """
    grid: Dict[str, List[Any]] = {}

    # n_estimators neighborhood
    n_est = int(best_params.get("n_estimators", 400))
    grid["n_estimators"] = _neighbor_int(n_est, step=100, low=200, high=800)

    # max_depth neighborhood
    md = int(best_params.get("max_depth", 6))
    grid["max_depth"] = _neighbor_int(md, step=1, low=3, high=12)

    # learning_rate neighborhood
    lr = float(best_params.get("learning_rate", 0.05))
    grid["learning_rate"] = _neighbor_float(lr, factor=2.0, low=0.01, high=0.3)

    # subsample neighborhood
    subs = float(best_params.get("subsample", 0.8))
    grid["subsample"] = _neighbor_float(subs, factor=1.25, low=0.5, high=1.0)

    # colsample_bytree neighborhood
    colsample = float(best_params.get("colsample_bytree", 0.8))
    grid["colsample_bytree"] = _neighbor_float(colsample, factor=1.25, low=0.5, high=1.0)

    return grid


# -------------------------
# Main
# -------------------------
def main() -> None:
    # --- Load processed train/valid (Stage 2) ---
    if not os.path.exists(TRAIN_PROCESSED_PATH):
        raise FileNotFoundError(f"TRAIN processed file not found: {TRAIN_PROCESSED_PATH}")
    if not os.path.exists(VALID_PROCESSED_PATH):
        raise FileNotFoundError(f"VALID processed file not found: {VALID_PROCESSED_PATH}")

    train_df = pd.read_csv(TRAIN_PROCESSED_PATH)
    valid_df = pd.read_csv(VALID_PROCESSED_PATH)

    print("[INFO] Loaded processed splits:")
    print("  train_stage2_processed:", train_df.shape)
    print("  valid_stage2_processed:", valid_df.shape)

    # --- Resolve target column name (once, from train) ---
    target_name = resolve_target_name(train_df)

    # --- Split X/y + leak guard ---
    x_train_df, y_train = split_x_y_with_leak_guard(train_df, target_name)
    x_valid_df, y_valid = split_x_y_with_leak_guard(valid_df, target_name)

    # Ensure numeric arrays (Stage-2 features are numeric already)
    x_train_np = x_train_df.to_numpy(dtype=float)
    x_valid_np = x_valid_df.to_numpy(dtype=float)
    y_train_np = y_train.to_numpy(dtype=float)
    y_valid_np = y_valid.to_numpy(dtype=float)

    print(f"[INFO] X_train shape: {x_train_np.shape}")
    print(f"[INFO] X_valid shape: {x_valid_np.shape}")

    # -------------------------
    # Define models + RANDOM grids
    # -------------------------
    models_and_random_grids: Dict[str, ModelConfig] = {
        "RandomForestRegressor": {
            "model": RandomForestRegressor(
                random_state=RANDOM_STATE,
                n_jobs=N_JOBS,
            ),
            # Wide random grid for good diversity
            "random_grid": {
                "n_estimators": [100, 200, 300, 400, 600, 800],
                "max_depth": [None, 4, 6, 8, 10, 12, 16, 20],
                "min_samples_split": [2, 5, 10, 20],
                "min_samples_leaf": [1, 2, 4, 8],
                "max_features": ["sqrt", "log2"],
            },
        }
    }

    # XGBoost (optional)
    if XGB_OK and XGBRegressor is not None:
        models_and_random_grids["XGBRegressor"] = {
            "model": XGBRegressor(
                objective="reg:squarederror",
                random_state=RANDOM_STATE,
                n_jobs=N_JOBS,
                tree_method="hist",
            ),
            # Wide random grid for good diversity
            "random_grid": {
                "n_estimators": [200, 300, 400, 500, 600, 800],
                "max_depth": [3, 4, 5, 6, 8, 10],
                "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
                "subsample": [0.6, 0.8, 1.0],
                "colsample_bytree": [0.6, 0.8, 1.0],
            },
        }
    else:
        print("[WARN] xgboost not available – XGBRegressor will be skipped.")

    # -------------------------
    # CV setup
    # -------------------------
    cv = KFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    results_rows: List[Dict[str, Any]] = []

    # -------------------------
    # Two-stage tuning per model
    # -------------------------
    for model_name, cfg in models_and_random_grids.items():
        base_model: Any = cfg["model"]
        random_grid: Dict[str, List[Any]] = cfg["random_grid"]

        print(f"\n[INFO] Stage 1 – RandomizedSearchCV for {model_name} ...")
        rand_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=random_grid,
            n_iter=40,  # reasonably large for good diversity
            scoring="neg_root_mean_squared_error",
            cv=cv,
            n_jobs=N_JOBS,
            random_state=RANDOM_STATE,
            verbose=1,
        )
        rand_search.fit(x_train_np, y_train_np)

        rand_best_score: float = float(rand_search.best_score_)  # negative RMSE
        rand_best_params: Dict[str, Any] = dict(rand_search.best_params_)

        print(f"[INFO] {model_name} RandomizedSearch best CV RMSE: {-rand_best_score:.6f}")
        print(f"[INFO] {model_name} RandomizedSearch best params: {rand_best_params}")

        # Build a narrow GridSearch grid around best random params
        if model_name == "RandomForestRegressor":
            refined_grid = build_rf_refined_grid(rand_best_params)
        elif model_name == "XGBRegressor":
            refined_grid = build_xgb_refined_grid(rand_best_params)
        else:
            raise ValueError(f"Unsupported model in two-stage tuning: {model_name}")

        print(f"[INFO] Stage 2 – GridSearchCV for {model_name} (refined around random best)...")
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=refined_grid,
            scoring="neg_root_mean_squared_error",
            cv=cv,
            n_jobs=N_JOBS,
            verbose=1,
        )
        grid_search.fit(x_train_np, y_train_np)

        grid_best_model: Any = grid_search.best_estimator_
        grid_best_score: float = float(grid_search.best_score_)  # negative RMSE
        grid_best_params: Dict[str, Any] = dict(grid_search.best_params_)

        print(f"[INFO] {model_name} GridSearch best CV RMSE: {-grid_best_score:.6f}")
        print(f"[INFO] {model_name} GridSearch best params: {grid_best_params}")

        # Evaluate the best GridSearch model on VALID
        y_valid_pred = grid_best_model.predict(x_valid_np)
        metrics_dict = regression_metrics(y_valid_np, y_valid_pred)

        result_row: Dict[str, Any] = {
            "Model": model_name,
            "Rand_best_neg_RMSE": rand_best_score,
            "Rand_best_RMSE": -rand_best_score,
            "Grid_best_neg_RMSE": grid_best_score,
            "Grid_best_RMSE": -grid_best_score,
            "VALID_MSE": metrics_dict["MSE"],
            "VALID_RMSE": metrics_dict["RMSE"],
            "VALID_MAE": metrics_dict["MAE"],
            "VALID_RMSLE": metrics_dict["RMSLE"],
            "VALID_R2": metrics_dict["R2"],
            "Best_Params_Random": str(rand_best_params),
            "Best_Params_Grid": str(grid_best_params),
        }
        results_rows.append(result_row)

    results_df = pd.DataFrame(results_rows)
    results_df.sort_values(by="VALID_RMSE", ascending=True, inplace=True)
    results_df.reset_index(drop=True, inplace=True)

    print("\n=== VALID Results after Two-Stage Hyperparameter Tuning (sorted by VALID_RMSE) ===")
    print(results_df.to_string(index=False))

    # Save to Excel
    try:
        results_df.to_excel(RESULTS_XLSX_PATH, index=False)
        print(f"\n[OK] Saved model selection results to Excel:\n     {RESULTS_XLSX_PATH}")
    except (OSError, PermissionError, ValueError) as exc:
        print(f"[WARN] Could not save Excel: {exc}")


if __name__ == "__main__":
    main()
