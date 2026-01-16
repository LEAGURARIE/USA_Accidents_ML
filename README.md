# NYC Accident Severity Prediction

Machine learning pipeline for predicting road accident severity in NYC, combining driver behavior, vehicle characteristics, and environmental conditions.

---

## Overview

**Target:** Severity (1-4 scale) → Binary classification (1-2: less severe, 3-4: more severe)

**Data Sources:**
- **NYC Accidents** — Timestamps, GPS, weather, road conditions
- **FARS** — Driver demographics, alcohol involvement, vehicle info

---

## Project Structure

```
PROJECT_ROOT/
├── src/
│   ├── data/
│   │   └── raw/
│   │       ├── interim/output_data/
│   │       │   ├── NYC_Accidents_with_FARS_raw.csv
│   │       │   ├── df_prepared.csv
│   │       │   ├── split/ (train/val/test + _fe_outputs/, _fe_artifacts/)
│   │       │   └── samples_for_git/
│   │       └── processed/
│   │           ├── train/valid/test_stage2_processed.csv
│   │           ├── model_selection_stage2.xlsx
│   │           └── shap_outputs/
│   └── models/
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/yourusername/nyc-accident-severity.git
cd nyc-accident-severity
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Requirements:** Python 3.11+, pandas, numpy, scikit-learn, xgboost, shap

---

## Quick Start

```python
# Option 1: Full pipeline
python src/pipeline/run_full_pipeline.py

# Option 2: From sample data
df = pd.read_csv('src/data/raw/interim/output_data/samples_for_git/df_prepared_sample_for_git.csv')
```

---

## Pipeline Stages

| Stage | Input | Output | Description |
|-------|-------|--------|-------------|
| 1. Data Prep | Raw CSV | df_prepared.csv | Highway detection, weather grouping, binning |
| 2. EDA | df_prepared.csv | post_eda.csv, plots | Correlations, spatial analysis, borough mapping |
| 3. Feature Engineering | train/val/test.csv | *_fe.csv + artifacts | Train-fitted transformations, imputation |
| 4. Model Training | processed CSVs | model_selection.xlsx | Two-stage tuning (Random → Grid search) |
| 5. Explainability | Trained model | SHAP plots | Feature importance analysis |

---

## Model Performance

**Best Model:** XGBoost Regressor (Validation RMSE: 0.25)

**Optimal Parameters:**
```python
{'n_estimators': 400, 'max_depth': 11, 'learning_rate': 0.015, 
 'subsample': 0.75, 'colsample_bytree': 0.8}
```

---

## Key Features

- Train-fitted transformations (no data leakage)
- Two-stage hyperparameter tuning
- SHAP explainability
- Artifact persistence for reproducibility
- Stratified splits (70/15/15)

**Top Predictive Factors:** Weather conditions, visibility, precipitation, alcohol involvement, driver demographics, vehicle type

---

## License

MIT License
