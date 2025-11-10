# ==========================================================
# Model Selection after Stage-2 Feature Selection
#
# - Uses the Stage-2 processed CSVs (already:
#     * cleaned
#     * feature-engineered
#     * imputed
#     * one-hot/ordinal encoded
#     * reduced to Stage-2 selected features only)
# - In this stage:
#       * Train on TRAIN
#       * Evaluate performance on VALID
#       * Do NOT touch TEST at all
# - Models: Linear Regression, Decision Tree, Random Forest,
#           AdaBoost, Gradient Boosting, SVR (+ optional XGBRegressor)
# ==========================================================
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor,
)
from sklearn.svm import SVR

# Optional XGBoost (import guarded only for import-related errors)
XGB_OK = True
try:
    import xgboost
    XGBRegressor = xgboost.XGBRegressor
except (ImportError, ModuleNotFoundError):
    XGB_OK = False
    XGBRegressor = None  # type: ignore[assignment]

# -------------------------
# Configuration
# -------------------------
RANDOM_STATE = 42

PROJECT_ROOT = r"C:\Users\LEA GUR-ARIE\PycharmProjects\USA_Accidents_ML"
PROCESSED_DIR = os.path.join(PROJECT_ROOT, r"src\data\raw\processed")

TRAIN_PROCESSED_PATH = os.path.join(PROCESSED_DIR, "train_stage2_processed.csv")
VALID_PROCESSED_PATH = os.path.join(PROCESSED_DIR, "valid_stage2_processed.csv")

# Target resolution preference: choose exactly one, prefer the ordinal variant
TARGET_ORDINAL_CANDIDATES: List[str] = ["Severity_bin_ord", "Severity_ord", "Severety_bin_ord"]
TARGET_FALLBACK_CANDIDATES: List[str] = ["Severity", "Severity_bin", "Severety_bin"]


def resolve_target_column(df: pd.DataFrame) -> str:
    """Pick the target column name (same logic as earlier stages)."""
    target_col: Optional[str] = None
    for cand in TARGET_ORDINAL_CANDIDATES + TARGET_FALLBACK_CANDIDATES:
        if cand in df.columns:
            target_col = cand
            break
    if target_col is None:
        raise ValueError(
            f"Could not find any target among: "
            f"{TARGET_ORDINAL_CANDIDATES + TARGET_FALLBACK_CANDIDATES}"
        )
    print(f"[INFO] Using target column: {target_col}")
    return target_col


# -------------------------
# Metrics helpers
# -------------------------
def safe_rmsle(y_true: np.ndarray, y_pred: np.ndarray, clip_eps: float = 1e-9) -> float:
    """
    Computes RMSLE robustly, clipping or shifting non-positive values as needed.
    Works also when target is 0/1.
    """
    y_true = y_true.astype(float).copy()
    y_pred = y_pred.astype(float).copy()

    min_true = float(y_true.min())
    min_pred = float(y_pred.min())

    if min_true <= 0 or min_pred <= 0:
        shift = max(0.0, 1.0 - min(min_true, min_pred))
        y_true = y_true + shift
        y_pred = y_pred + shift

    y_true = np.clip(y_true, clip_eps, None)
    y_pred = np.clip(y_pred, clip_eps, None)

    return float(np.sqrt(metrics.mean_squared_log_error(y_true, y_pred)))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Return a dictionary of standard regression metrics as plain floats."""
    mse_val: float = float(metrics.mean_squared_error(y_true, y_pred))
    rmse_val: float = float(np.sqrt(mse_val))
    mae_val: float = float(metrics.mean_absolute_error(y_true, y_pred))
    rmsle_val: float = safe_rmsle(y_true, y_pred)
    r2_val: float = float(metrics.r2_score(y_true, y_pred))

    return {
        "MSE": mse_val,
        "RMSE": rmse_val,
        "MAE": mae_val,
        "RMSLE": rmsle_val,
        "R2": r2_val,
    }


def main() -> None:
    # -------------------------
    # Load processed datasets (Stage 2 outputs)
    # -------------------------
    if not os.path.exists(TRAIN_PROCESSED_PATH):
        raise FileNotFoundError(f"TRAIN processed file not found: {TRAIN_PROCESSED_PATH}")
    if not os.path.exists(VALID_PROCESSED_PATH):
        raise FileNotFoundError(f"VALID processed file not found: {VALID_PROCESSED_PATH}")

    train_df = pd.read_csv(TRAIN_PROCESSED_PATH)
    valid_df = pd.read_csv(VALID_PROCESSED_PATH)

    print("[INFO] Loaded Stage-2 processed splits:")
    print("  train:", train_df.shape)
    print("  valid:", valid_df.shape)

    # -------------------------
    # Resolve target column (same in both splits)
    # -------------------------
    target_col = resolve_target_column(train_df)

    if target_col not in valid_df.columns:
        raise ValueError(f"Target column '{target_col}' not found in VALID dataframe.")

    # -------------------------
    # Build X / y for train & valid
    # -------------------------
    x_train_df = train_df.drop(columns=[target_col])
    y_train_series = train_df[target_col]

    x_valid_df = valid_df.drop(columns=[target_col])
    y_valid_series = valid_df[target_col]

    x_train = x_train_df.to_numpy(dtype=float)
    x_valid = x_valid_df.to_numpy(dtype=float)

    y_train = y_train_series.to_numpy(dtype=float)
    y_valid = y_valid_series.to_numpy(dtype=float)

    print(f"[INFO] X_train shape: {x_train.shape}, X_valid shape: {x_valid.shape}")
    print(f"[INFO] y_train size: {y_train.shape[0]}, y_valid size: {y_valid.shape[0]}")

    # -------------------------
    # Define models
    # -------------------------
    models: Dict[str, Any] = {
        "LinearRegression": make_pipeline(
            StandardScaler(with_mean=True, with_std=True),
            LinearRegression(),
        ),
        "DecisionTreeRegressor": DecisionTreeRegressor(random_state=RANDOM_STATE),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=400,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            max_features="sqrt",
        ),
        "AdaBoostRegressor": AdaBoostRegressor(random_state=RANDOM_STATE),
        "GradientBoostingRegressor": GradientBoostingRegressor(random_state=RANDOM_STATE),
        "SVR": make_pipeline(
            StandardScaler(with_mean=True, with_std=True),
            SVR(kernel="rbf", C=1.0, epsilon=0.1),
        ),
    }

    if XGB_OK and XGBRegressor is not None:
        models["XGBRegressor"] = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            tree_method="hist",
        )

    # -------------------------
    # Train → predict → evaluate on VALID
    # -------------------------
    results: List[Dict[str, Any]] = []

    for name, model in models.items():
        print(f"\n[INFO] Training model: {name}")
        model.fit(x_train, y_train)
        y_valid_pred = model.predict(x_valid)
        metrics_dict = regression_metrics(y_valid, y_valid_pred)
        row: Dict[str, Any] = {"Model": name}
        row.update(metrics_dict)
        results.append(row)

    results_df = (
        pd.DataFrame(results)
        .sort_values(by=["RMSE", "MAE"], ascending=[True, True])
        .reset_index(drop=True)
    )

    print("\n=== VALID Results (sorted by RMSE) ===")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
