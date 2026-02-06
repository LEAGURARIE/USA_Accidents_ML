# ==========================================================
# Model Explainability - SHAP Analysis
#
# - Loads Stage-2 processed train/valid/test
# - Loads best model from model_selection_stage2.csv
# - Fits best model on TRAIN + VALID, computes SHAP on TEST
# - Saves SHAP summary plots and raw values
# ==========================================================
from src.utils.helpers import *

# File-specific configuration
TEST_TARGET_PATH = os.path.join(PROCESSED_DIR, "test_stage2_target.csv")
MODEL_SELECTION_XLSX = os.path.join(PROCESSED_DIR, "model_selection_stage2.xlsx")
MODEL_SELECTION_CSV = os.path.join(PROCESSED_DIR, "model_selection_stage2.csv")
TARGET_FALLBACK_CANDIDATES: List[str] = ["Severity", "Severity_bin", "Severety_bin"]
TARGET_STEMS = {"severity", "severety"}

ensure_dirs(SHAP_OUTPUT_DIR, MODEL_OUTPUT_DIR, REPORTS_DIR)


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
    """Splits dataframe into (X, y) and removes target-like columns from X."""
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
    """Ensures the test dataframe does NOT contain target or target-like columns."""
    x_df = df.copy()
    drop_cols: List[str] = []

    if target_name in x_df.columns:
        drop_cols.append(target_name)

    for col_name in x_df.columns:
        low = col_name.lower()
        if any(stem in low for stem in TARGET_STEMS):
            drop_cols.append(col_name)

    if drop_cols:
        print_info(f"Dropping target-like columns from TEST: {sorted(set(drop_cols))}")
        x_df = x_df.drop(columns=sorted(set(drop_cols)))

    return x_df


def build_best_model_from_csv(results_path: str):
    """Loads model selection results (xlsx or csv) and instantiates the best model."""
    # Try xlsx first, then csv
    xlsx_path = results_path.replace(".csv", ".xlsx")
    csv_path = results_path.replace(".xlsx", ".csv")

    if os.path.exists(xlsx_path):
        df_res = pd.read_excel(xlsx_path)
        print_info(f"Loaded model selection results from: {xlsx_path}")
    elif os.path.exists(csv_path):
        df_res = load_csv_safe(csv_path)
        print_info(f"Loaded model selection results from: {csv_path}")
    else:
        raise FileNotFoundError(
            f"Model selection file not found. Tried:\n  {xlsx_path}\n  {csv_path}"
        )

    required_cols = {"Model", "VALID_F1", "Best_Params_Grid"}
    if not required_cols.issubset(df_res.columns):
        raise ValueError(
            f"CSV file must contain columns: {required_cols}. "
            f"Found: {set(df_res.columns)}"
        )

    best_idx = df_res["VALID_F1"].idxmax()
    best_row = df_res.loc[best_idx]

    model_name = str(best_row["Model"])
    best_f1 = float(best_row["VALID_F1"])
    best_params_str = str(best_row["Best_Params_Grid"])

    print_info(f"Best model from CSV: {model_name} (VALID_F1={best_f1:.6f})")
    print_info("Using hyperparameters from 'Best_Params_Grid'")
    print_info(f"Best params (raw string): {best_params_str}")

    try:
        best_params: Dict[str, object] = ast.literal_eval(best_params_str)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"Could not parse Best_Params_Grid as dict: {exc}") from exc

    # Store original params for report
    original_params = dict(best_params)

    for key in ["random_state", "n_jobs", "tree_method", "objective", "eval_metric"]:
        if key in best_params:
            print_warn(f"Removing '{key}' from Best_Params_Grid to avoid conflicts.")
            best_params.pop(key, None)

    if model_name == "RandomForestClassifier":
        model = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, **best_params)
        return model, model_name, original_params

    if model_name == "XGBClassifier":
        if not XGB_OK or XGBClassifier is None:
            raise RuntimeError("Best model is XGBClassifier, but xgboost is not available.")
        model = XGBClassifier(
            objective="binary:logistic",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            tree_method="hist",
            eval_metric="logloss",
            **best_params,
        )
        return model, model_name, original_params

    raise ValueError(f"Unsupported best model type: {model_name}")


def compute_shap_values_for_tree_model(
    model: Any,
    x_test_np: np.ndarray,
    x_background_np: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float]:
    """Computes SHAP values for a tree-based classifier."""
    if not SHAP_OK:
        raise RuntimeError("SHAP is not available. Install with: pip install shap")

    if x_background_np is None:
        x_background_np = x_test_np

    # XGBoost Classifier: use permutation explainer (workaround for base_score bug)
    if XGB_OK and XGBClassifier is not None and isinstance(model, XGBClassifier):
        print_info("Using SHAP permutation explainer for XGBClassifier (workaround for base_score bug).")

        def predict_proba_positive(X: np.ndarray) -> np.ndarray:
            return model.predict_proba(X)[:, 1]

        explainer: Any = shap.Explainer(
            predict_proba_positive,
            x_background_np,
            algorithm="permutation",
        )
        shap_values_obj: Any = explainer(x_test_np)

        shap_values = np.asarray(shap_values_obj.values)
        expected_value_raw: Any = np.asarray(shap_values_obj.base_values)

    # RandomForestClassifier (and other tree models) -> TreeExplainer
    else:
        print_info("Building SHAP TreeExplainer...")
        explainer: Any = shap.TreeExplainer(model)

        print_info("Computing SHAP values on ALL test rows...")
        shap_values_raw: Any = explainer.shap_values(x_test_np)

        if isinstance(shap_values_raw, list):
            if len(shap_values_raw) == 0:
                raise ValueError("Received empty list of SHAP values.")
            if len(shap_values_raw) == 2:
                shap_values = np.asarray(shap_values_raw[1])
            else:
                shap_values = np.asarray(shap_values_raw[0])
        else:
            shap_values = np.asarray(shap_values_raw)

        expected_value_raw = explainer.expected_value
        if isinstance(expected_value_raw, (list, np.ndarray)) and len(expected_value_raw) == 2:
            expected_value_raw = expected_value_raw[1]

    if shap_values.ndim != 2 or shap_values.shape[0] != x_test_np.shape[0]:
        raise ValueError(
            f"Unexpected SHAP values shape {shap_values.shape}, "
            f"expected (n_samples, n_features) = {x_test_np.shape[0], x_test_np.shape[1]}"
        )

    if isinstance(expected_value_raw, (list, np.ndarray)):
        expected_value = float(np.asarray(expected_value_raw).mean())
    else:
        expected_value = float(expected_value_raw)

    print_info(f"SHAP values shape: {shap_values.shape}")
    print_info(f"SHAP expected_value: {expected_value:.6f}")

    return shap_values, expected_value


def save_shap_plots(
    shap_values: np.ndarray,
    x_test_df: pd.DataFrame,
    model_name: str,
    output_dir: str,
) -> None:
    """Saves SHAP summary (beeswarm) and SHAP summary bar plots."""
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
    print_ok(f"Saved SHAP beeswarm summary plot: {beeswarm_path}")

    # Summary bar plot
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
    print_ok(f"Saved SHAP bar summary plot: {bar_path}")


def save_shap_arrays(
    shap_values: np.ndarray,
    expected_value: float,
    output_dir: str,
    model_name: str,
) -> None:
    """Saves raw SHAP values and expected_value as .npy files."""
    shap_path = os.path.join(output_dir, f"shap_values_{model_name}.npy")
    exp_path = os.path.join(output_dir, f"shap_expected_value_{model_name}.npy")

    np.save(shap_path, shap_values)
    np.save(exp_path, np.array([expected_value], dtype="float64"))

    print_ok(f"Saved raw SHAP values to: {shap_path}")
    print_ok(f"Saved expected_value to: {exp_path}")


def generate_model_report(
    model_name: str,
    best_params: Dict[str, Any],
    train_shape: Tuple[int, int],
    test_shape: Tuple[int, int],
    y_train_full_np: np.ndarray,
    y_test: np.ndarray,
    y_pred_test: np.ndarray,
    shap_values: np.ndarray,
    feature_names: List[str],
    output_dir: str,
) -> str:
    """Generates and saves a comprehensive model report."""
    report_lines: List[str] = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Header
    report_lines.append("=" * 70)
    report_lines.append("FINAL MODEL REPORT - NYC Accidents Severity Prediction")
    report_lines.append("=" * 70)
    report_lines.append(f"Generated: {timestamp}")
    report_lines.append("")

    # Model Information
    report_lines.append("-" * 70)
    report_lines.append("1. MODEL INFORMATION")
    report_lines.append("-" * 70)
    report_lines.append(f"Model Type: {model_name}")
    report_lines.append(f"Hyperparameters:")
    for param, value in best_params.items():
        report_lines.append(f"  - {param}: {value}")
    report_lines.append("")

    # Dataset Information
    report_lines.append("-" * 70)
    report_lines.append("2. DATASET INFORMATION")
    report_lines.append("-" * 70)
    report_lines.append(f"Training Set (TRAIN + VALID): {train_shape[0]:,} samples, {train_shape[1]} features")
    report_lines.append(f"Test Set: {test_shape[0]:,} samples, {test_shape[1]} features")
    report_lines.append("")
    report_lines.append("Class Distribution:")
    train_counts = np.bincount(y_train_full_np)
    test_counts = np.bincount(y_test)
    report_lines.append(f"  Training: Class 0 = {train_counts[0]:,}, Class 1 = {train_counts[1]:,}")
    report_lines.append(f"  Test:     Class 0 = {test_counts[0]:,}, Class 1 = {test_counts[1]:,}")
    report_lines.append("")

    # Performance Metrics
    report_lines.append("-" * 70)
    report_lines.append("3. MODEL PERFORMANCE METRICS (TEST SET)")
    report_lines.append("-" * 70)

    test_accuracy = float(accuracy_score(y_test, y_pred_test))
    test_precision = float(precision_score(y_test, y_pred_test, average="weighted", zero_division=0))
    test_recall = float(recall_score(y_test, y_pred_test, average="weighted", zero_division=0))
    test_f1 = float(f1_score(y_test, y_pred_test, average="weighted", zero_division=0))

    report_lines.append(f"Accuracy:  {test_accuracy:.4f}")
    report_lines.append(f"Precision: {test_precision:.4f} (weighted)")
    report_lines.append(f"Recall:    {test_recall:.4f} (weighted)")
    report_lines.append(f"F1 Score:  {test_f1:.4f} (weighted)")
    report_lines.append("")

    # Confusion Matrix
    report_lines.append("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred_test)
    report_lines.append(f"                Predicted")
    report_lines.append(f"              Class 0  Class 1")
    report_lines.append(f"  Actual 0    {cm[0, 0]:>7}  {cm[0, 1]:>7}")
    report_lines.append(f"  Actual 1    {cm[1, 0]:>7}  {cm[1, 1]:>7}")
    report_lines.append("")

    # Classification Report
    report_lines.append("Classification Report:")
    report_lines.append(classification_report(y_test, y_pred_test))

    # SHAP Feature Importance
    report_lines.append("-" * 70)
    report_lines.append("4. SHAP FEATURE IMPORTANCE (Top 30)")
    report_lines.append("-" * 70)
    report_lines.append("Features ranked by mean absolute SHAP value:")
    report_lines.append("")

    # Calculate mean absolute SHAP values
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    feature_importance = list(zip(feature_names, mean_abs_shap))
    feature_importance.sort(key=lambda x: x[1], reverse=True)

    report_lines.append(f"{'Rank':<6}{'Feature':<50}{'Mean |SHAP|':<15}")
    report_lines.append("-" * 70)
    for rank, (feat, importance) in enumerate(feature_importance[:30], 1):
        report_lines.append(f"{rank:<6}{feat:<50}{importance:.6f}")
    report_lines.append("")

    # Output Files
    report_lines.append("-" * 70)
    report_lines.append("5. OUTPUT FILES")
    report_lines.append("-" * 70)
    report_lines.append(f"SHAP Beeswarm Plot: {os.path.join(SHAP_OUTPUT_DIR, f'shap_summary_beeswarm_{model_name}.png')}")
    report_lines.append(f"SHAP Bar Plot:      {os.path.join(SHAP_OUTPUT_DIR, f'shap_summary_bar_{model_name}.png')}")
    report_lines.append(f"SHAP Values (npy):  {os.path.join(SHAP_OUTPUT_DIR, f'shap_values_{model_name}.npy')}")
    report_lines.append(f"Report File:        {os.path.join(output_dir, f'final_model_report_{model_name}.txt')}")
    report_lines.append("")
    report_lines.append("=" * 70)
    report_lines.append("END OF REPORT")
    report_lines.append("=" * 70)

    # Join and save
    report_text = "\n".join(report_lines)

    # Save text report
    report_path = os.path.join(output_dir, f"final_model_report_{model_name}.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    # Save metrics as CSV for easy analysis
    metrics_df = pd.DataFrame({
        "Metric": ["Accuracy", "Precision (weighted)", "Recall (weighted)", "F1 Score (weighted)"],
        "Value": [test_accuracy, test_precision, test_recall, test_f1]
    })
    metrics_csv_path = os.path.join(output_dir, f"final_model_metrics_{model_name}.csv")
    metrics_df.to_csv(metrics_csv_path, index=False)

    # Save feature importance as CSV
    importance_df = pd.DataFrame({
        "Rank": list(range(1, len(feature_importance) + 1)),
        "Feature": [f[0] for f in feature_importance],
        "Mean_Abs_SHAP": [f[1] for f in feature_importance]
    })
    importance_csv_path = os.path.join(output_dir, f"shap_feature_importance_{model_name}.csv")
    importance_df.to_csv(importance_csv_path, index=False)

    print_ok(f"Saved model report to: {report_path}")
    print_ok(f"Saved metrics CSV to: {metrics_csv_path}")
    print_ok(f"Saved feature importance CSV to: {importance_csv_path}")

    return report_text


def main() -> None:
    # Load processed train/valid/test (Stage 2)
    if not os.path.exists(TRAIN_PROCESSED_PATH):
        raise FileNotFoundError(f"TRAIN processed file not found: {TRAIN_PROCESSED_PATH}")
    if not os.path.exists(VALID_PROCESSED_PATH):
        raise FileNotFoundError(f"VALID processed file not found: {VALID_PROCESSED_PATH}")
    if not os.path.exists(TEST_PROCESSED_PATH):
        raise FileNotFoundError(f"TEST processed file not found: {TEST_PROCESSED_PATH}")
    if not os.path.exists(TEST_TARGET_PATH):
        raise FileNotFoundError(f"TEST target file not found: {TEST_TARGET_PATH}")

    train_df = load_csv_safe(TRAIN_PROCESSED_PATH)
    valid_df = load_csv_safe(VALID_PROCESSED_PATH)
    test_df = load_csv_safe(TEST_PROCESSED_PATH)
    test_target_df = load_csv_safe(TEST_TARGET_PATH)

    print_info("Loaded Stage-2 processed splits:")
    print("  train_stage2_processed:", train_df.shape)
    print("  valid_stage2_processed:", valid_df.shape)
    print("  test_stage2_processed :", test_df.shape)

    target_name = resolve_target_name(train_df)

    if target_name not in test_target_df.columns:
        raise ValueError(
            f"Target column '{target_name}' not found in TEST target file: {TEST_TARGET_PATH}"
        )
    y_test = test_target_df[target_name].to_numpy(dtype=float)

    x_train_df, y_train = split_x_y_with_leak_guard(train_df, target_name)
    x_valid_df, y_valid = split_x_y_with_leak_guard(valid_df, target_name)

    x_test_df = drop_target_like_if_present(test_df, target_name)

    x_train_full_df = pd.concat([x_train_df, x_valid_df], axis=0).reset_index(drop=True)
    y_train_full = pd.concat([y_train, y_valid], axis=0).reset_index(drop=True)

    x_train_full_np = x_train_full_df.to_numpy(dtype=float)
    x_test_np = x_test_df.to_numpy(dtype=float)
    y_train_full_np = y_train_full.to_numpy(dtype=int)
    y_test = y_test.astype(int)

    print_info(f"X_train_full shape: {x_train_full_np.shape}")
    print_info(f"X_test shape      : {x_test_np.shape}")
    print_info(f"Class distribution in train+valid: {np.bincount(y_train_full_np)}")
    print_info(f"Class distribution in test: {np.bincount(y_test)}")

    model, best_model_name, best_params = build_best_model_from_csv(MODEL_SELECTION_CSV)

    if best_model_name != "XGBClassifier":
        print_warn(
            f"Expected best model to be XGBClassifier, but got '{best_model_name}'. "
            f"Proceeding with {best_model_name}."
        )

    print_info(f"Fitting best model on TRAIN+VALID: {best_model_name}")
    model.fit(x_train_full_np, y_train_full_np)

    y_pred_train = model.predict(x_train_full_np)
    y_pred_test = model.predict(x_test_np)

    train_accuracy = float(accuracy_score(y_train_full_np, y_pred_train))
    test_accuracy = float(accuracy_score(y_test, y_pred_test))
    train_f1 = float(f1_score(y_train_full_np, y_pred_train, average="weighted"))
    test_f1 = float(f1_score(y_test, y_pred_test, average="weighted"))

    print("\n[RESULT] Final model performance (TRAIN+VALID vs TEST):")
    print(f"  TRAIN+VALID Accuracy = {train_accuracy:.4f} | F1 = {train_f1:.4f}")
    print(f"  TEST        Accuracy = {test_accuracy:.4f} | F1 = {test_f1:.4f}")
    print("\n[INFO] Classification Report on TEST:")
    print(classification_report(y_test, y_pred_test))

    shap_values, expected_value = compute_shap_values_for_tree_model(
        model,
        x_test_np,
        x_background_np=x_train_full_np,
    )

    save_shap_plots(shap_values, x_test_df, best_model_name, SHAP_OUTPUT_DIR)
    save_shap_arrays(shap_values, expected_value, SHAP_OUTPUT_DIR, best_model_name)

    # Generate and save comprehensive report
    feature_names = list(x_test_df.columns)
    report = generate_model_report(
        model_name=best_model_name,
        best_params=best_params,
        train_shape=x_train_full_np.shape,
        test_shape=x_test_np.shape,
        y_train_full_np=y_train_full_np,
        y_test=y_test,
        y_pred_test=y_pred_test,
        shap_values=shap_values,
        feature_names=feature_names,
        output_dir=REPORTS_DIR,
    )

    print_ok("SHAP explainability completed.")
    print(f"   SHAP outputs in: {SHAP_OUTPUT_DIR}")
    print(f"   Reports in: {REPORTS_DIR}")


if __name__ == "__main__":
    main()
