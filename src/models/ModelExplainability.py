from __future__ import annotations

"""
ModelExplainabilitySHAP.py

- Loads Stage-2 processed train/valid/test:
    * train_stage2_processed.csv
    * valid_stage2_processed.csv
    * test_stage2_processed.csv

- Loads best model choice + hyperparameters from:
    * model_selection_stage2.xlsx

- Rebuilds the best model (RandomForestRegressor or XGBRegressor),
  fits it on TRAIN + VALID (full data, no test leakage),
  and computes SHAP values on ALL rows of TEST.

- For RandomForestRegressor:
    * Uses SHAP TreeExplainer (fast & native tree support).

- For XGBRegressor:
    * Uses SHAP permutation explainer as a workaround for a base_score
      parsing bug between recent xgboost and shap's TreeExplainer.

- Saves:
    * SHAP summary plot (beeswarm)
    * SHAP summary bar plot
    * Raw SHAP values and expected_value as .npy files
"""

import os
import ast
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# SHAP (must be installed via poetry: `poetry add shap`)
import shap

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score  # >>> ADDED

# Optional XGBoost
XGB_OK = True
try:
    import xgboost as xgb
    XGBRegressor = xgb.XGBRegressor
except ImportError:
    XGB_OK = False
    xgb = None  # ensure 'xgb' is defined even if import fails
    XGBRegressor = None  # type: ignore[assignment]

# -------------------------
# Configuration
# -------------------------

RANDOM_STATE = 42

PROJECT_ROOT = r"C:\Users\LEA GUR-ARIE\PycharmProjects\USA_Accidents_ML"
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "src", "data", "raw", "processed")

TRAIN_PROCESSED_PATH = os.path.join(PROCESSED_DIR, "train_stage2_processed.csv")
VALID_PROCESSED_PATH = os.path.join(PROCESSED_DIR, "valid_stage2_processed.csv")
TEST_PROCESSED_PATH = os.path.join(PROCESSED_DIR, "test_stage2_processed.csv")
TEST_TARGET_PATH = os.path.join(PROCESSED_DIR, "test_stage2_target.csv")  # >>> ADDED

MODEL_SELECTION_XLSX = os.path.join(PROCESSED_DIR, "model_selection_stage2.xlsx")

SHAP_OUTPUT_DIR = os.path.join(PROCESSED_DIR, "shap_outputs")
os.makedirs(SHAP_OUTPUT_DIR, exist_ok=True)

# Target resolution preference
TARGET_ORDINAL_CANDIDATES: List[str] = ["Severity_bin_ord", "Severity_ord", "Severety_bin_ord"]
TARGET_FALLBACK_CANDIDATES: List[str] = ["Severity", "Severity_bin", "Severety_bin"]

# To be safe, drop any columns whose names look like target from X
TARGET_STEMS = {"severity", "severety"}


# -------------------------
# Utilities
# -------------------------

def resolve_target_name(df: pd.DataFrame) -> str:
    """
    Resolves the target column name from the dataframe, preferring ordinal forms.
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
    Splits dataframe into (X, y) and removes any target-like columns from X
    (e.g., columns whose names contain 'severity' / 'severety').
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


def drop_target_like_if_present(df: pd.DataFrame, target_name: str) -> pd.DataFrame:
    """
    Ensures the test dataframe does NOT contain the target or target-like columns.
    This avoids any accidental leakage at explainability time.
    """
    x_df = df.copy()
    drop_cols: List[str] = []

    if target_name in x_df.columns:
        drop_cols.append(target_name)

    for col_name in x_df.columns:
        low = col_name.lower()
        if any(stem in low for stem in TARGET_STEMS):
            drop_cols.append(col_name)

    if drop_cols:
        print(f"[INFO] Dropping target-like columns from TEST: {sorted(set(drop_cols))}")
        x_df = x_df.drop(columns=sorted(set(drop_cols)))

    return x_df


def build_best_model_from_excel(results_xlsx_path: str):
    """
    Loads model_selection_stage2.xlsx, finds the best row (min VALID_RMSE),
    and instantiates the corresponding model with its best hyperparameters.

    Assumes a two-stage HPO format with columns:
        - 'Model'
        - 'VALID_RMSE'
        - 'Best_Params_Grid'  (final tuned hyperparameters, as a Python dict string)
    """
    if not os.path.exists(results_xlsx_path):
        raise FileNotFoundError(f"Model selection Excel not found: {results_xlsx_path}")

    df_res = pd.read_excel(results_xlsx_path)

    required_cols = {"Model", "VALID_RMSE", "Best_Params_Grid"}
    if not required_cols.issubset(df_res.columns):
        raise ValueError(
            f"Excel file must contain columns: {required_cols}. "
            f"Found: {set(df_res.columns)}"
        )

    # Pick the row with minimal VALID_RMSE
    best_idx = df_res["VALID_RMSE"].idxmin()
    best_row = df_res.loc[best_idx]

    model_name = str(best_row["Model"])
    best_rmse = float(best_row["VALID_RMSE"])
    best_params_str = str(best_row["Best_Params_Grid"])

    print(f"[INFO] Best model from Excel: {model_name} (VALID_RMSE={best_rmse:.6f})")
    print(f"[INFO] Using hyperparameters from 'Best_Params_Grid'")
    print(f"[INFO] Best params (raw string): {best_params_str}")

    try:
        best_params: Dict[str, object] = ast.literal_eval(best_params_str)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"Could not parse Best_Params_Grid as dict: {exc}") from exc

    # Defensive: remove keys that might conflict with fixed args
    for key in ["random_state", "n_jobs", "tree_method", "objective"]:
        if key in best_params:
            print(f"[WARN] Removing '{key}' from Best_Params_Grid to avoid conflicts.")
            best_params.pop(key, None)

    if model_name == "RandomForestRegressor":
        model = RandomForestRegressor(
            random_state=RANDOM_STATE,
            n_jobs=-1,
            **best_params,
        )
        return model, model_name

    if model_name == "XGBRegressor":
        if not XGB_OK or XGBRegressor is None:
            raise RuntimeError("Best model is XGBRegressor, but xgboost is not available.")
        model = XGBRegressor(
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            tree_method="hist",
            **best_params,
        )
        return model, model_name

    raise ValueError(f"Unsupported best model type: {model_name}")


def compute_shap_values_for_tree_model(
    model: Any,
    x_test_np: np.ndarray,
    x_background_np: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float]:
    """
    Computes SHAP values for a tree-based model.

    * For RandomForestRegressor:
        - Uses SHAP TreeExplainer (fast and native for tree ensembles).
    * For XGBRegressor:
        - Uses SHAP permutation explainer as a workaround for
          the base_score parsing issue in shap + xgboost.

    Parameters
    ----------
    model:
        Fitted tree-based model (RandomForestRegressor or XGBRegressor).
    x_test_np:
        Test feature matrix (n_samples, n_features).
    x_background_np:
        Background data used by the explainer. If None, uses x_test_np.

    Returns
    -------
    shap_values:
        Array of shape (n_samples, n_features).
    expected_value:
        Scalar expected value used by SHAP.
    """
    # Background data for permutation explainer (for XGB) if not provided
    if x_background_np is None:
        x_background_np = x_test_np

    # --- XGBoost: use permutation explainer (workaround for base_score bug) ---
    if XGB_OK and XGBRegressor is not None and isinstance(model, XGBRegressor):
        print(
            "[INFO] Using SHAP permutation explainer for XGBRegressor "
            "(workaround for base_score bug)."
        )

        model_any: Any = model
        explainer: Any = shap.Explainer(
            model_any.predict,
            x_background_np,
            algorithm="permutation",
        )
        shap_values_obj: Any = explainer(x_test_np)

        # Values: shape (n_samples, n_features)
        shap_values = np.asarray(shap_values_obj.values)

        # For permutation explainer, the per-sample base_values can be used
        # as a proxy for expected_value. We will later reduce it to a scalar.
        expected_value_raw: Any = np.asarray(shap_values_obj.base_values)

    # --- RandomForestRegressor (and other tree models) → TreeExplainer ---
    else:
        print("[INFO] Building SHAP TreeExplainer...")
        explainer: Any = shap.TreeExplainer(model)

        print("[INFO] Computing SHAP values on ALL test rows...")
        shap_values_raw: Any = explainer.shap_values(x_test_np)

        if isinstance(shap_values_raw, list):
            if len(shap_values_raw) == 0:
                raise ValueError("Received empty list of SHAP values.")
            shap_values = np.asarray(shap_values_raw[0])
        else:
            shap_values = np.asarray(shap_values_raw)

        expected_value_raw = explainer.expected_value

    # Sanity check on shape
    if shap_values.ndim != 2 or shap_values.shape[0] != x_test_np.shape[0]:
        raise ValueError(
            f"Unexpected SHAP values shape {shap_values.shape}, "
            f"expected (n_samples, n_features) = {x_test_np.shape[0], x_test_np.shape[1]}"
        )

    # Normalize expected_value to a scalar
    if isinstance(expected_value_raw, (list, np.ndarray)):
        expected_value = float(np.asarray(expected_value_raw).mean())
    else:
        expected_value = float(expected_value_raw)

    print(f"[INFO] SHAP values shape: {shap_values.shape}")
    print(f"[INFO] SHAP expected_value: {expected_value:.6f}")

    return shap_values, expected_value


def save_shap_plots(
    shap_values: np.ndarray,
    x_test_df: pd.DataFrame,
    model_name: str,
    output_dir: str,
) -> None:
    """
    Saves SHAP summary (beeswarm) and SHAP summary bar plots.
    """
    feature_names = list(x_test_df.columns)

    # Summary beeswarm plot
    plt.figure()
    shap.summary_plot(
        shap_values,
        x_test_df.values,
        feature_names=feature_names,
        show=False,
        plot_type="dot",
        max_display=30,
    )
    beeswarm_path = os.path.join(output_dir, f"shap_summary_beeswarm_{model_name}.png")
    plt.tight_layout()
    plt.savefig(beeswarm_path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved SHAP beeswarm summary plot: {beeswarm_path}")

    # Summary bar plot (global feature importance)
    plt.figure()
    shap.summary_plot(
        shap_values,
        x_test_df.values,
        feature_names=feature_names,
        show=False,
        plot_type="bar",
        max_display=30,
    )
    bar_path = os.path.join(output_dir, f"shap_summary_bar_{model_name}.png")
    plt.tight_layout()
    plt.savefig(bar_path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved SHAP bar summary plot: {bar_path}")


def save_shap_arrays(
    shap_values: np.ndarray,
    expected_value: float,
    output_dir: str,
    model_name: str,
) -> None:
    """
    Saves raw SHAP values and expected_value as .npy files for further analysis.
    """
    shap_path = os.path.join(output_dir, f"shap_values_{model_name}.npy")
    exp_path = os.path.join(output_dir, f"shap_expected_value_{model_name}.npy")

    np.save(shap_path, shap_values)
    np.save(exp_path, np.array([expected_value], dtype="float64"))

    print(f"[OK] Saved raw SHAP values to: {shap_path}")
    print(f"[OK] Saved expected_value to: {exp_path}")


# -------------------------
# Main
# -------------------------

def main() -> None:
    # --- Load processed train/valid/test (Stage 2) ---
    if not os.path.exists(TRAIN_PROCESSED_PATH):
        raise FileNotFoundError(f"TRAIN processed file not found: {TRAIN_PROCESSED_PATH}")
    if not os.path.exists(VALID_PROCESSED_PATH):
        raise FileNotFoundError(f"VALID processed file not found: {VALID_PROCESSED_PATH}")
    if not os.path.exists(TEST_PROCESSED_PATH):
        raise FileNotFoundError(f"TEST processed file not found: {TEST_PROCESSED_PATH}")
    if not os.path.exists(TEST_TARGET_PATH):  # >>> ADDED
        raise FileNotFoundError(f"TEST target file not found: {TEST_TARGET_PATH}")  # >>> ADDED

    train_df = pd.read_csv(TRAIN_PROCESSED_PATH)
    valid_df = pd.read_csv(VALID_PROCESSED_PATH)
    test_df = pd.read_csv(TEST_PROCESSED_PATH)
    test_target_df = pd.read_csv(TEST_TARGET_PATH)  # >>> ADDED

    print("[INFO] Loaded Stage-2 processed splits:")
    print("  train_stage2_processed:", train_df.shape)
    print("  valid_stage2_processed:", valid_df.shape)
    print("  test_stage2_processed :", test_df.shape)

    # --- Resolve target name from train ---
    target_name = resolve_target_name(train_df)

    # --- Extract y_test from separate target file ---  # >>> ADDED
    if target_name not in test_target_df.columns:  # >>> ADDED
        raise ValueError(  # >>> ADDED
            f"Target column '{target_name}' not found in TEST target file: {TEST_TARGET_PATH}"  # >>> ADDED
        )  # >>> ADDED
    y_test = test_target_df[target_name].to_numpy(dtype=float)  # >>> ADDED

    # --- Split X/y for train and valid + leak guard ---
    x_train_df, y_train = split_x_y_with_leak_guard(train_df, target_name)
    x_valid_df, y_valid = split_x_y_with_leak_guard(valid_df, target_name)

    # --- Ensure TEST does NOT contain target or target-like columns ---
    x_test_df = drop_target_like_if_present(test_df, target_name)

    # --- Combine TRAIN + VALID for final training ---
    x_train_full_df = pd.concat([x_train_df, x_valid_df], axis=0).reset_index(drop=True)
    y_train_full = pd.concat([y_train, y_valid], axis=0).reset_index(drop=True)

    # --- Convert to numpy arrays (float) ---
    x_train_full_np = x_train_full_df.to_numpy(dtype=float)
    x_test_np = x_test_df.to_numpy(dtype=float)
    y_train_full_np = y_train_full.to_numpy(dtype=float)

    print(f"[INFO] X_train_full shape: {x_train_full_np.shape}")
    print(f"[INFO] X_test shape      : {x_test_np.shape}")

    # --- Build best model from Excel (model_selection_stage2.xlsx) ---
    model, best_model_name = build_best_model_from_excel(MODEL_SELECTION_XLSX)

    # Optional strict check: ensure the selected model is XGBRegressor
    if best_model_name != "XGBRegressor":
        raise RuntimeError(
            f"Expected best model to be XGBRegressor, but got '{best_model_name}'. "
            f"Check 'model_selection_stage2.xlsx' if this is unintended."
        )

    # --- Fit model on TRAIN + VALID (full data, no TEST) ---
    print(f"[INFO] Fitting best model on TRAIN+VALID: {best_model_name}")
    model.fit(x_train_full_np, y_train_full_np)

    # >>> ADDED: Evaluate model on TRAIN+VALID and TEST
    y_pred_train = model.predict(x_train_full_np)
    y_pred_test = model.predict(x_test_np)

    train_rmse = float(np.sqrt(mean_squared_error(y_train_full_np, y_pred_train)))
    test_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))
    train_r2 = float(r2_score(y_train_full_np, y_pred_train))
    test_r2 = float(r2_score(y_test, y_pred_test))

    print("\n[RESULT] Final model performance (TRAIN+VALID vs TEST):")
    print(f"  TRAIN+VALID RMSE = {train_rmse:.4f} | R² = {train_r2:.4f}")
    print(f"  TEST        RMSE = {test_rmse:.4f} | R² = {test_r2:.4f}\n")
    # <<< END ADDED BLOCK

    # --- Compute SHAP values on ALL TEST rows ---
    # Pass x_train_full_np as background for the permutation explainer (XGB case).
    shap_values, expected_value = compute_shap_values_for_tree_model(
        model,
        x_test_np,
        x_background_np=x_train_full_np,
    )

    # --- Save plots ---
    save_shap_plots(shap_values, x_test_df, best_model_name, SHAP_OUTPUT_DIR)

    # --- Save raw SHAP arrays ---
    save_shap_arrays(shap_values, expected_value, SHAP_OUTPUT_DIR, best_model_name)

    print("\n✅ SHAP explainability completed.")
    print(f"   Outputs in: {SHAP_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
