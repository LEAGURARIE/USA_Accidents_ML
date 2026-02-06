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
│   ├── data/           # Data loading, cleansing, preparation, EDA
│   │   └── raw/
│   │       ├── interim/output_data/
│   │       │   ├── NYC_Accidents_with_FARS_raw.csv
│   │       │   ├── df_prepared.csv
│   │       │   └── split/ (train/val/test + _fe_outputs/, _fe_artifacts/)
│   │       └── processed/
│   │           ├── train/valid/test_stage2_processed.csv
│   │           └── model_selection_stage2.xlsx
│   ├── features/       # Feature engineering and selection
│   ├── models/         # Model selection, HPO, explainability
│   ├── output/         # Generated outputs (eda, models, reports, shap)
│   └── utils/          # Centralized helpers and imports
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/LEAGURARIE/USA_Accidents_ML.git
cd USA_Accidents_ML
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Requirements:** Python 3.11+, pandas, numpy, scikit-learn, xgboost, shap, matplotlib, seaborn

---

## Quick Start

```python
# Run model explainability with SHAP analysis
python -m src.models.ModelExplainability

# Run model selection with hyperparameter optimization
python -m src.models.ModelSelection_HPO
```

---

## Pipeline Stages

| Stage | Input | Output | Description |
|-------|-------|--------|-------------|
| 1. Data Prep | Raw CSV | df_prepared.csv | Highway detection, weather grouping, binning |
| 2. EDA | df_prepared.csv | post_eda.csv, plots | Correlations, spatial analysis, borough mapping |
| 3. Feature Engineering | train/val/test.csv | *_fe.csv + artifacts | Train-fitted transformations, imputation |
| 4. Model Training | processed CSVs | model_selection.xlsx | Two-stage tuning (Random → Grid search) |
| 5. Explainability | Trained model | SHAP plots, reports | Feature importance analysis |

---

## Model Performance

**Best Model:** XGBoost Classifier

| Metric | Test Score |
|--------|------------|
| Accuracy | 92.55% |
| Precision (weighted) | 92.47% |
| Recall (weighted) | 92.55% |
| F1 Score (weighted) | 92.49% |

**Per-Class Performance:**
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0 (Less Severe) | 0.94 | 0.96 | 0.95 | 12,863 |
| 1 (More Severe) | 0.88 | 0.84 | 0.86 | 4,784 |

**Optimal Hyperparameters:**
```python
{'colsample_bytree': 0.75, 'learning_rate': 0.1, 'max_depth': 11,
 'n_estimators': 300, 'subsample': 1.0}
```

---

## SHAP Feature Importance

Top predictive features ranked by mean |SHAP value|:

1. **Event_Year** — Temporal trends in accident severity
2. **Distance(mi)_log** — Accident impact distance
3. **Start_Lat / Start_Lng** — Geographic location
4. **Event_DOW** — Day of week patterns
5. **Event_Month** — Seasonal effects
6. **Borough_Manhattan** — Manhattan-specific risk
7. **Event_Hour** — Time of day
8. **FARS_DRIVER_AGE** — Driver demographics
9. **Street_is_highway** — Highway vs local road
10. **Temperature(F)** — Weather conditions

![SHAP Summary](src/output/shap/shap_summary_bar_XGBClassifier.png)

---

## Key Features

- Train-fitted transformations (no data leakage)
- Two-stage hyperparameter tuning (RandomizedSearch → GridSearch)
- SHAP explainability with comprehensive reports
- Centralized utilities module for code reuse
- Artifact persistence for reproducibility
- Stratified splits (70/15/15)

---

## License

MIT License
