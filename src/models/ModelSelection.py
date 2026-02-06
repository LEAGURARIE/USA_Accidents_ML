# ==========================================================
# Model Selection after Stage-2 Feature Selection
#
# - Uses the Stage-2 processed CSVs
# - Train on TRAIN, Evaluate on VALID
# - Models: Logistic Regression, Decision Tree, Random Forest,
#           AdaBoost, Gradient Boosting, SVC (+ optional XGBClassifier)
# - Target: Ordinal/Binary classification (Severity)
# ==========================================================
from src.utils.helpers import *

# File-specific configuration
TARGET_FALLBACK_CANDIDATES: List[str] = ["Severity", "Severity_bin", "Severety_bin"]


def resolve_target_col(df: pd.DataFrame) -> str:
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
    print_info(f"Using target column: {target_col}")
    return target_col


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """Return a dictionary of standard classification metrics as plain floats."""
    accuracy_val: float = float(accuracy_score(y_true, y_pred))
    precision_val: float = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    recall_val: float = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    f1_val: float = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    # ROC-AUC (only for binary classification with probability estimates)
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


def main() -> None:
    # Load processed datasets (Stage 2 outputs)
    if not os.path.exists(TRAIN_PROCESSED_PATH):
        raise FileNotFoundError(f"TRAIN processed file not found: {TRAIN_PROCESSED_PATH}")
    if not os.path.exists(VALID_PROCESSED_PATH):
        raise FileNotFoundError(f"VALID processed file not found: {VALID_PROCESSED_PATH}")

    train_df = load_csv_safe(TRAIN_PROCESSED_PATH)
    valid_df = load_csv_safe(VALID_PROCESSED_PATH)

    print_info("Loaded Stage-2 processed splits:")
    print("  train:", train_df.shape)
    print("  valid:", valid_df.shape)

    # Resolve target column
    target_col = resolve_target_col(train_df)

    if target_col not in valid_df.columns:
        raise ValueError(f"Target column '{target_col}' not found in VALID dataframe.")

    # Build X / y for train & valid
    x_train_df = train_df.drop(columns=[target_col])
    y_train_series = train_df[target_col]

    x_valid_df = valid_df.drop(columns=[target_col])
    y_valid_series = valid_df[target_col]

    x_train = x_train_df.to_numpy(dtype=float)
    x_valid = x_valid_df.to_numpy(dtype=float)

    y_train = y_train_series.to_numpy(dtype=int)
    y_valid = y_valid_series.to_numpy(dtype=int)

    print_info(f"X_train shape: {x_train.shape}, X_valid shape: {x_valid.shape}")
    print_info(f"y_train size: {y_train.shape[0]}, y_valid size: {y_valid.shape[0]}")
    print_info(f"Class distribution in train: {np.bincount(y_train)}")
    print_info(f"Class distribution in valid: {np.bincount(y_valid)}")

    # Define models (Classifiers for ordinal/binary target)
    models: Dict[str, Any] = {
        "LogisticRegression": make_pipeline(
            StandardScaler(with_mean=True, with_std=True),
            LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
        ),
        "DecisionTreeClassifier": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=400,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            max_features="sqrt",
        ),
        "AdaBoostClassifier": AdaBoostClassifier(random_state=RANDOM_STATE),
        "GradientBoostingClassifier": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "SVC": make_pipeline(
            StandardScaler(with_mean=True, with_std=True),
            SVC(kernel="rbf", C=1.0, probability=True, random_state=RANDOM_STATE),
        ),
    }

    if XGB_OK and XGBClassifier is not None:
        models["XGBClassifier"] = XGBClassifier(
            objective="binary:logistic",
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            tree_method="hist",
            eval_metric="logloss",
        )

    # Train -> predict -> evaluate on VALID
    results: List[Dict[str, Any]] = []

    for name, model in models.items():
        print_info(f"Training model: {name}")
        model.fit(x_train, y_train)
        y_valid_pred = model.predict(x_valid)

        y_valid_prob: Optional[np.ndarray] = None
        if hasattr(model, "predict_proba"):
            y_valid_prob = model.predict_proba(x_valid)
        elif hasattr(model, "named_steps") and hasattr(model.named_steps.get("svc", model), "predict_proba"):
            y_valid_prob = model.predict_proba(x_valid)

        metrics_dict = classification_metrics(y_valid, y_valid_pred, y_valid_prob)
        row: Dict[str, Any] = {"Model": name}
        row.update(metrics_dict)
        results.append(row)

    results_df = (
        pd.DataFrame(results)
        .sort_values(by=["F1", "Accuracy"], ascending=[False, False])
        .reset_index(drop=True)
    )

    print("\n=== VALID Results (sorted by F1 Score) ===")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
