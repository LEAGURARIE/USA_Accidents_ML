# ==========================================================
# Centralized Imports and Configuration
# ==========================================================
from __future__ import annotations

# --- Standard Library ---
import os
import re
import ast
import json
import math
import glob as pyglob
import platform
import time
import warnings
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Sequence, TypedDict, cast

# --- Third-party: Core ---
from collections import Counter
import numpy as np
import pandas as pd
from pandas.api.types import (
    CategoricalDtype,
    pandas_dtype,
    is_integer_dtype,
    is_float_dtype,
    is_bool_dtype,
    is_numeric_dtype,
    is_object_dtype,
    is_string_dtype,
    is_datetime64_any_dtype,
)

# --- Third-party: Joblib ---
import joblib

# --- Third-party: Requests/Network (optional) ---
REQUESTS_OK = True
try:
    import requests
    from requests import RequestException
except (ImportError, ModuleNotFoundError):
    REQUESTS_OK = False
    requests = None  # type: ignore[assignment]
    RequestException = Exception  # type: ignore[assignment,misc]

# --- Third-party: tqdm (optional) ---
TQDM_OK = True
try:
    from tqdm import tqdm
except (ImportError, ModuleNotFoundError):
    TQDM_OK = False
    tqdm = None  # type: ignore[assignment]

# --- Third-party: geopandas (optional) ---
GEOPANDAS_OK = True
try:
    import geopandas as gpd
except (ImportError, ModuleNotFoundError):
    GEOPANDAS_OK = False
    gpd = None  # type: ignore[assignment]

# --- Third-party: Visualization ---
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle

# --- Third-party: Seaborn (optional) ---
SEABORN_OK = True
try:
    import seaborn as sns
except (ImportError, ModuleNotFoundError):
    SEABORN_OK = False
    sns = None  # type: ignore[assignment]

# --- Third-party: SciPy (optional) ---
SCIPY_OK = True
try:
    from scipy.stats import (
        chi2_contingency,
        fisher_exact,
        spearmanr,
        kendalltau,
        mannwhitneyu,
        kruskal,
    )
except (ImportError, ModuleNotFoundError):
    SCIPY_OK = False
    chi2_contingency = None  # type: ignore[assignment]
    fisher_exact = None  # type: ignore[assignment]
    spearmanr = None  # type: ignore[assignment]
    kendalltau = None  # type: ignore[assignment]
    mannwhitneyu = None  # type: ignore[assignment]
    kruskal = None  # type: ignore[assignment]

# --- Third-party: TextBlob (optional) ---
TEXTBLOB_OK = True
try:
    from textblob import TextBlob
except (ImportError, ModuleNotFoundError):
    TEXTBLOB_OK = False
    TextBlob = None  # type: ignore[assignment]

# --- Third-party: Scikit-learn Core ---
from sklearn import metrics
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

# --- Third-party: Scikit-learn Imputers ---
import sklearn.experimental.enable_iterative_imputer  # noqa: F401
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.impute import IterativeImputer as SkIterativeImputer  # type: ignore[attr-defined]

# --- Third-party: Scikit-learn Feature Selection ---
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.inspection import permutation_importance

# --- Third-party: Scikit-learn Models ---
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
)
from sklearn.svm import SVC, LinearSVC

# --- Third-party: Scikit-learn Model Selection ---
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    KFold,
    train_test_split,
)

# --- Third-party: Scikit-learn Metrics ---
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

# --- Third-party: XGBoost (optional) ---
XGB_OK = True
try:
    import xgboost as xgb
    XGBClassifier = xgb.XGBClassifier
except (ImportError, ModuleNotFoundError):
    XGB_OK = False
    xgb = None  # type: ignore[assignment]
    XGBClassifier = None  # type: ignore[assignment]

# --- Third-party: SHAP (optional) ---
SHAP_OK = True
try:
    import shap
except (ImportError, ModuleNotFoundError):
    SHAP_OK = False
    shap = None  # type: ignore[assignment]


# ==========================================================
# Configuration Constants
# ==========================================================
RANDOM_STATE = 42

# Project paths
PROJECT_ROOT = r"C:\Users\LEA GUR-ARIE\PycharmProjects\USA_Accidents_ML"
DATA_DIR = os.path.join(PROJECT_ROOT, "src", "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
INTERIM_DIR = os.path.join(RAW_DIR, "interim", "output_data")
SPLIT_DIR = os.path.join(INTERIM_DIR, "split")
PROCESSED_DIR = os.path.join(RAW_DIR, "processed")
ARTIFACT_DIR = os.path.join(SPLIT_DIR, "_fe_artifacts")
FE_OUTPUT_DIR = os.path.join(SPLIT_DIR, "_fe_outputs")

# Output directories (reports, plots, figures)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "src", "output")
EDA_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "eda")
MODEL_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "models")
SHAP_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "shap")
REPORTS_DIR = os.path.join(OUTPUT_DIR, "reports")

# Data files
TRAIN_PATH = os.path.join(SPLIT_DIR, "train.csv")
VALID_PATH = os.path.join(SPLIT_DIR, "val.csv")
TEST_PATH = os.path.join(SPLIT_DIR, "test.csv")

TRAIN_PROCESSED_PATH = os.path.join(PROCESSED_DIR, "train_stage2_processed.csv")
VALID_PROCESSED_PATH = os.path.join(PROCESSED_DIR, "valid_stage2_processed.csv")
TEST_PROCESSED_PATH = os.path.join(PROCESSED_DIR, "test_stage2_processed.csv")

# Target column candidates
TARGET_ORDINAL_CANDIDATES: List[str] = ["Severity_bin_ord", "Severity_ord", "Severety_bin_ord"]

# Model selection parameters
N_JOBS = 1  # Sequential processing to avoid memory issues
CV_FOLDS = 3


# ==========================================================
# Helper Functions
# ==========================================================
def ensure_dirs(*paths: str) -> None:
    """Create directories if they don't exist."""
    for path in paths:
        os.makedirs(path, exist_ok=True)


def resolve_target_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Find the first available target column from candidates."""
    for col in candidates:
        if col in df.columns:
            return col
    return None


def load_csv_safe(path: str, **kwargs) -> pd.DataFrame:
    """Load CSV with low_memory=False by default."""
    return pd.read_csv(path, low_memory=False, **kwargs)


def print_info(msg: str) -> None:
    """Print info message."""
    print(f"[INFO] {msg}")


def print_ok(msg: str) -> None:
    """Print success message."""
    print(f"[OK] {msg}")


def print_warn(msg: str) -> None:
    """Print warning message."""
    print(f"[WARN] {msg}")


def print_error(msg: str) -> None:
    """Print error message."""
    print(f"[ERROR] {msg}")


def ensure_fig_saved_close(path: str, dpi: int = 150) -> None:
    """Tight layout, save, close figure."""
    try:
        plt.tight_layout()
    except (RuntimeError, ValueError):
        pass
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def safe_name(name: str) -> str:
    """Create a safe filename from a string."""
    cleaned = re.sub(r"[^\w\-]+", "_", str(name))
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned[:120]
