# NYC Accident Severity – Integrated Person / Vehicle / Environment Modeling

## 1. Objective

This project builds a data pipeline and machine learning model to predict the severity of road accidents in New York City, using a combined view of driver (person), vehicle, and environment:

- Person-level features from FARS (e.g., driver age, sex, alcohol involvement).
- Vehicle-level features (body type, make).
- Environment-level features from the NYC crash data & weather (time, location, distance, visibility, precipitation, road type, weather conditions, etc.).

The end product is:

1. A clean, feature-engineered dataset (train_fe, valid_fe, test_fe).
2. A tuned gradient-boosting model (XGBRegressor) predicting accident severity.
3. Model explainability via SHAP, highlighting which factors (person, vehicle, environment) drive severity.


## 2. Data Sources and Integration

### 2.1 Core data

- NYC Accidents – city crash records with:
  - Timestamps (Start_Time, End_Time),
  - GPS coordinates (Start_Lat, Start_Lng, End_Lat, End_Lng),
  - Road and infrastructure flags (e.g., Bump, Junction, Traffic_Signal),
  - Basic weather measurements (visibility, temperature, humidity, wind speed, precipitation),
  - Raw severity score (Severity from 1 to 4).

- FARS (Fatality Analysis Reporting System) – records at the person/vehicle level:
  - FARS__DRIVER_AGE, FARS__SEX, FARS__ALC_RES,
  - Vehicle body type (FARS__BODY_TYP), make (FARS__MAKE),
  - Other vehicle-level and coding variables.

These datasets are joined into a single, enriched table:
NYC_Accidents_with_FARS_raw.csv.

### 2.2 Person / Vehicle / Environment view

The pipeline is explicitly designed to capture three dimensions:

- Person: driver age, sex, and alcohol involvement (cleaned/encoded FARS fields).
- Vehicle: grouped vehicle body types and basic make categories.
- Environment: road type (highway vs street), weather conditions, visibility, precipitation, wind, day/night, weekend, spatial location (boroughs), distance of the accident, and time-of-day.

This integrated feature space is then used for modeling and SHAP-based explainability.


## 3. Project Structure (high level)

Paths below are relative to the project root (PROJECT_ROOT):

- src/data/raw/interim/output_data/
  - NYC_Accidents_with_FARS_raw.csv – merged accident + FARS input.
  - df_prepared.csv – after first preparation step.
  - post_eda.csv – after initial EDA & light transformations.
  - split/train.csv, split/val.csv, split/test.csv – stratified splits.
  - split/_fe_outputs/train_fe.csv, valid_fe.csv, test_fe.csv – train-fitted FE outputs.
  - split/_fe_artifacts/ – feature engineering artifacts (bins, imputers, schema).
- src/data/raw/interim/EDA_plot/ – first EDA outputs (plots + Excel).
- src/data/raw/interim/EDA1_plot/ – extended EDA + statistical tests.
- src/data/raw/processed/
  - train_stage2_processed.csv, valid_stage2_processed.csv, test_stage2_processed.csv – fully numeric “Stage 2” datasets.
  - model_selection_stage2.xlsx – model comparison and hyperparameters.
  - shap_outputs/ – SHAP plots and .npy files.
- src/models/
  - Scripts for model selection and explainability (Stage 2).


## 4. Data Preparation & EDA Pipeline

### 4.1 Initial preparation – df_prepared.csv

Script input:
NYC_Accidents_with_FARS_raw.csv  
Script output:
df_prepared.csv

Key steps:

1. Basic profiling
   - Source × Severity crosstabs (counts and row percentages).
   - Allows quick comparison of severity distribution across different data sources.

2. Street-level highway flag
   - Creates Street_is_highway based on:
     - Explicit highway tokens in the street name (e.g., EXPRESSWAY, PKWY, HWY, I-95, US-xx, NY-xx, etc.).
     - A curated list of major NYC expressways and parkways (BQE, LIE, FDR, Cross Bronx, Van Wyck, Belt Pkwy, Henry Hudson, etc.).
   - This produces a boolean indicator: accident on highway vs local street.

3. Airport proximity
   - Airport_Code frequency table: counts and percentages for each code, with "Missing" grouped separately.

4. Wind direction grouping
   - Raw Wind_Direction is normalized and grouped into:
     - NORTH, EAST, SOUTH, WEST, OTHER, and MISSING.
   - Stored as a categorical Wind_Direction_grp.

5. Weather condition grouping
   - Weather_Condition is cleaned and mapped to:
     - CLEAR, CLOUDY, RAIN, SNOW, FOG, STORM, OTHER, MISSING.
   - Stored as Weather_ConditionGroup, used later as an ordered categorical.

6. Distance grouping
   - Distance(mi) is converted to numeric and grouped into three bands:
     - ≤0.13 mi (Short), 0.13–0.76 mi (Medium), >0.76 mi (Long).
   - If those fixed cutpoints are not appropriate for a given dataset, the script dynamically falls back to median / 75th percentile to keep three meaningful groups.
   - Summary distribution of Distance_group is printed.

The result is a cleaned, enriched dataset (df_prepared.csv) with early engineered features used in downstream EDA.


### 4.2 EDA & Sentiment – analysis_output.xlsx + plots

Script input: df_prepared.csv  
Script outputs:

- analysis_output.xlsx with:
  - Numeric summary (“Numeric Detailed”),
  - Categorical distributions,
  - Missing-value tables,
  - Spearman correlation matrix,
  - Cramér’s V (categorical associations),
  - Borough-level counts (from geospatial boxes).
- Plots:
  - spearman_heatmap.png – numeric + boolean correlations,
  - categorical_cramersV_heatmap.png – association across categorical variables,
  - hexbin_StartLat_StartLng.png / _labeled.png – spatial density of accidents with borough labels,
  - boxplot_all_numeric_grid.png – grid of numeric distributions,
  - pairplot_numeric_grid.png,
  - Per-variable histograms/boxplots (numeric) and barplots (categorical),
  - sentiment analysis outputs (TextBlob).

Main responsibilities:

1. Numeric profiling
   - Extended summary for each numeric feature:
     - Mean, std, SE of mean,
     - Percentiles (1%, 5%, 25%, 50%, 75%, 95%, 99%),
     - IQR, Tukey fences, and percentage of IQR outliers.
   - Written to the “Numeric Detailed” sheet.

2. Correlation analysis
   - Spearman correlation across numeric and boolean features.
   - Heatmap saved as spearman_heatmap.png.

3. Categorical analysis
   - Value counts, percentages, and missingness for every categorical/boolean variable.
   - Capped barplots (top-K + “OTHER”) saved per variable.
   - Cramér’s V matrix (bias-corrected) for categorical variables with ≤ MAX_LEVELS_CRAMERS levels, plotted as categorical_cramersV_heatmap.png.

4. Spatial patterns
   - Hexbin maps of Start_Lat vs Start_Lng to visualize spatial density.
   - Rough geographic bounding boxes for the five NYC boroughs:
     - Staten Island, Brooklyn, Queens, Manhattan, Bronx.
   - Accident counts per borough are computed and reported; labeled hexbin map shows both density and borough labels with counts.

5. Sentiment analysis
   - If Description and Severity are present and TextBlob is installed:
     - Compute polarity and subjectivity for each description.
     - Derive categorical sentiment: negative (−1), neutral (0), positive (+1).
     - Summary by Severity:
       - Mean polarity, mean subjectivity,
       - Counts and percentages of negative / neutral / positive descriptions.
   - Save results to severity_sentiment_summary.xlsx and plots:
     - Overall sentiment bar chart,
     - Polarity vs subjectivity scatter plot colored by sentiment.

6. Export
   - The script writes a combined EDA Excel (analysis_output.xlsx) and saves all plots into EDA_plot.
   - It also exports an updated dataset as post_eda.csv for the next stage.


### 4.3 Cleaning, extended EDA & split – post_cleaning.csv + train/val/test

Script input: post_eda.csv  
Script outputs:

- post_cleaning.csv – fully cleaned dataset.
- split/train.csv, split/val.csv, split/test.csv – stratified splits.
- Extended EDA outputs in EDA1_plot/:
  - analysis_output.xlsx (updated),
  - spearman_heatmap.png, categorical_cramersV_heatmap.png, boxplot_all_numeric_grid.png, pairplot_numeric_grid.png,
  - Histograms, boxplots, barplots.
- Statistical tests vs target:
  - test_results.xlsx, test_results_summary.csv, test_results_significant.csv.

Key operations:

1. Column dropping
   - Removes high-cardinality IDs, location text fields, raw time strings, FARS technical codes, and text description fields not intended for modeling (e.g. ID, Street, Zipcode, City, County, Description, FARS lat/long codes, etc.).

2. Type casts and feature engineering
   - Cast object columns to category (excluding raw time strings).
   - Driver age (FARS__DRIVER_AGE):
     - Cleaned to numeric,
     - Invalid FARS codes and unrealistic ages (<14 or >100) set to NaN.
   - Visibility:
     - Visibility_is_low = visibility ≤ 3 miles (boolean),
     - Visibility_2cat = {“Low”, “Normal”, “Missing”}.
   - Boolean infrastructure flags cast to true booleans (if present):
     - Amenity, Bump, Crossing, Give_Way, Junction, No_Exit, Railway, Station, Stop, Traffic_Calming, Traffic_Signal, Turning_Loop, Street_is_highway, etc.
   - Twilight variables (Sunrise_Sunset, Civil_Twilight, Nautical_Twilight, Astronomical_Twilight):
     - Mapped to True (Day), False (Night).
   - Airport_Code:
     - Treated as categorical,
     - Missing values imputed with a “median” category that covers ≥50% of cases.
   - Vehicle body type (FARS__BODY_TYP):
     - Grouped into:
       - Passenger Car, SUV/CUV, Van, Pickup/Light Truck, Bus,
         Medium/Heavy Truck, Motorcycle/3-Wheel, Other Special Vehicle, Missing.
     - Encoded as FARS__BODY_TYP_Group.
   - Driver sex (FARS__SEX):
     - Cleaned to valid codes {1, 2}, others set to NaN.
     - Encoded as:
       - FARS__SEX_3cat ∈ {1, 2, 3} for {Male, Female, Missing}.
       - FARS__SEX_label ∈ {“Male”, “Female”, “Missing”}.
   - Alcohol test result (FARS__ALC_RES):
     - Invalid codes removed,
     - Encoded to ALC_category ∈ {“No Alcohol”, “Alcohol Present”, “Missing”}.
     - Raw FARS__ALC_RES subsequently dropped to avoid leakage.

3. Severity targets
   - Base numeric Severity (1–4) is required.
   - Severity_2cat (3-level label):
     - Low (1–2), High (3–4), Missing.
   - Later (Section 7), a binary target is created:
     - Severity_bin = 0 for Severity ∈ {1,2}, 1 for Severity ∈ {3,4}.

4. Extended EDA & correlation  
   - Same structure as Section 4.2 but now applied to the cleaned dataset, including:
     - Updated Spearman correlations,
     - Cramér’s V matrix,
     - Numeric and categorical summaries,
     - Pairplots and boxplot grids.

5. Statistical tests vs target
   - Target: Severity_2cat (if available), otherwise Severity_bin.
   - For each feature:
     - Numeric vs binary target:
       - Mann–Whitney U test.
     - Categorical vs target:
       - For 2×2 tables: Fisher’s exact test,
       - Otherwise: Chi-square test with Cramér’s V.
   - Results saved as:
     - test_results.xlsx (“All tests” and “Significant (p<0.05)” sheets),
     - test_results_summary.csv,
     - test_results_significant.csv.

6. Stratified train/validation/test split
   - Creates Severity_bin (0 = less severe [1–2], 1 = more severe [3–4]) and drops rows with invalid/missing severity.
   - Validates that each class has sufficient samples to support stratification.
   - Split ratios:
     - Train: 70%
     - Validation: 15%
     - Test: 15%
   - Stratified by Severity_bin using train_test_split in two stages (train+val vs test, then train vs val).
   - Sanity checks ensure no index overlap across splits.
   - Output saved to:
     - split/train.csv, split/val.csv, split/test.csv.

7. Final clean dataset
   - The full cleaned (pre-split) dataset is saved as post_cleaning.csv for reference.


## 5. Train-Fitted Feature Engineering & Artifacts

Script inputs:
- split/train.csv, split/val.csv, split/test.csv  

Script outputs:
- split/_fe_outputs/train_fe.csv, valid_fe.csv, test_fe.csv (+ Parquet)
- Artifacts in split/_fe_artifacts/:
  - bins_levels.json – feature bins and category orders,
  - group_stats.json – per-group medians,
  - rw_pools.joblib – random-weighted sampling pools,
  - knn_imputer.joblib – KNNImputer and its columns,
  - mice_imputer.joblib – IterativeImputer (MICE) and its columns,
  - schema.json – final TRAIN schema (dtypes + categories),
  - drops.json – columns to drop after FE,
  - manifest.json – metadata (seed, row counts, versions).

### 5.1 What is fitted on TRAIN

1. Bin definitions & category orders
   - Derived only from TRAIN:
     - Distance_group bins and labels,
     - Precipitation_cat bins and labels,
     - Category orders for Weather_ConditionGroup and Visibility_2cat.

2. Core feature engineering
   - Re-applies time/categorical/distance/precipitation/wind logic (Section 4.3) in a single function:
     - Start_Time_parsed, End_Time_parsed, Event_TS, Event_Year, Event_Month, Event_Day, Event_Hour, Event_DOW, Is_Weekend, DayNight_3cat.
     - Distance(mi), Distance(mi)_log, Distance_group, Distance_group_ord.
     - Precipitation(in), Precipitation_log, Precipitation_cat.
     - Wind_Speed(mph), Wind_Chill(F).
     - Ordered encodings and integer codes:
       - Weather_ConditionGroup + _ord,
       - Visibility_2cat + _ord,
       - Severity_bin + Severity_bin_ord.

3. Driver age normalization
   - Ensures consistent cleaning of FARS__DRIVER_AGE on TRAIN:
     - Invalid codes and unrealistic ages are set to NaN,
     - Stored as float32.

4. Group-level statistics
   - For each of:
     - Precipitation(in), Wind_Speed(mph), Wind_Chill(F)
   - Computes:
     - Per (Event_Year, Event_Month, Borough) median,
     - Global median.

5. Random-weighted (stochastic) imputation pools
   - Builds arrays of observed TRAIN values per (Event_Year, Event_Month, Borough) and globally:
     - rw_pools[feature][(year, month, borough)],
     - rw_pools[feature]["__GLOBAL__"].
   - These are used to fill missing values probabilistically, preserving empirical distributions and seasonal/borough patterns.

6. KNN & MICE imputers
   - KNNImputer (if available in TRAIN):
     - Fit on numeric subset: ["Pressure(in)", "Visibility(mi)", "Temperature(F)", "Humidity(%)"].
   - IterativeImputer (MICE) (if ≥2 of these exist):
     - ["FARS__DRIVER_AGE", "FARS__MAKE", "FARS__ALC_RES", "FARS__BODY_TYP", "FARS__SEX"].
   - Both imputers are saved and later only used in .transform() mode for VALID and TEST.

7. Column drops and schema extraction
   - A predefined list (DROP_AFTER_IMPUTE) is dropped after all derivations to avoid target leakage and keep only modeling features.
   - All remaining object columns are converted to categorical.
   - A column-level schema (dtype + categories + ordering) is stored in schema.json.
   - A manifest is saved with environment metadata.

### 5.2 Applying TRAIN artifacts to TRAIN/VALID/TEST (_fe.csv)

Using the saved artifacts, the script:

1. Re-loads artifacts (bins_levels, group_stats, rw_pools, imputers, drop list, schema).
2. For each of TRAIN, VALID, TEST:
   - Re-runs core FE (no refitting).
   - Normalizes driver age.
   - Applies group-based random-weighted imputation for the wind/precipitation variables.
   - Recomputes Precipitation_log and Precipitation_cat using TRAIN bins.
   - Applies KNNImputer and IterativeImputer in transform mode on their column subsets.
   - Drops the same columns as TRAIN.
   - Converts any remaining object columns to categorical.
   - Aligns the dataframe exactly to the TRAIN schema (columns and dtypes).

The result is three aligned, fully processed feature tables:

- train_fe.csv – final training set for modeling,
- valid_fe.csv – validation set,
- test_fe.csv – hold-out test set.


## 6. Stage 2 – Model Selection

Inputs:
- train_stage2_processed.csv
- valid_stage2_processed.csv  
(these are Stage 2, fully numeric versions of train_fe/valid_fe.)

Script:
Two-stage hyperparameter tuning (RandomizedSearchCV → GridSearchCV) for tree models.

### 6.1 Target resolution and leak guard

- The script auto-resolves the target column from:
  - Preferred ordinals: ["Severity_bin_ord", "Severity_ord", "Severety_bin_ord"]
  - Fallbacks: ["Severity", "Severity_bin", "Severety_bin"]
- A leak guard removes from X any feature whose name contains:
  - "severity" or "severety" (case-insensitive),
  - to ensure no target or target-like columns leak into the predictors.

### 6.2 Models and tuning strategy

Two candidate regressors:

1. RandomForestRegressor
2. XGBRegressor (if xgboost is installed)

For each model:

1. Stage 1 – RandomizedSearchCV
   - Wide hyperparameter search over:
     - For RF: n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features.
     - For XGB: n_estimators, max_depth, learning_rate, subsample, colsample_bytree.
   - CV = 3-fold KFold, scoring = negative RMSE.

2. Stage 2 – GridSearchCV (refinement)
   - Builds a narrow grid around the best RandomizedSearch parameters using small neighborhoods:
     - e.g., ±100 trees, ±1–2 around max_depth, neighbours around learning_rate, subsample, colsample.
   - Runs a focused GridSearchCV (same CV & scoring).

3. Validation evaluation
   - The best GridSearch model is applied to X_valid, and metrics are computed:
     - VALID_MSE, VALID_RMSE, VALID_MAE, VALID_RMSLE, VALID_R2.

4. Results export
   - A summary table with one row per model:
     - RandomizedSearch best RMSE & params,
     - GridSearch best RMSE & params,
     - Validation metrics and final hyperparameters (Best_Params_Grid).
   - Saved as model_selection_stage2.xlsx, sorted by VALID_RMSE.

In the current run, the best model is:

- XGBRegressor with:
  - colsample_bytree = 0.8  
  - learning_rate = 0.015  
  - max_depth = 11  
  - n_estimators = 400  
  - subsample = 0.75  
  - VALID_RMSE ≈ 0.25.


## 7. Stage 2 – Model Explainability with SHAP

Inputs:

- train_stage2_processed.csv
- valid_stage2_processed.csv
- test_stage2_processed.csv
- model_selection_stage2.xlsx

Outputs (under shap_outputs/):

- shap_summary_beeswarm_XGBRegressor.png
- shap_summary_bar_XGBRegressor.png
- shap_values_XGBRegressor.npy
- shap_expected_value_XGBRegressor.npy

### 7.1 Workflow

1. Load Stage 2 splits
   - Train, Validation, Test from src/data/raw/processed/.

2. Resolve target and apply leak guard
   - Same target resolution as in model selection.
   - Drops any target-like columns from the feature matrices.

3. Rebuild best model from Excel
   - Reads model_selection_stage2.xlsx.
   - Selects the row with minimal VALID_RMSE.
   - Parses Best_Params_Grid into a Python dict.
   - Removes conflicting keys (random_state, n_jobs, tree_method, objective) to preserve script-level configuration.
   - Instantiates the corresponding model:
     - In the current setup, forced to XGBRegressor (runtime check).

4. Fit on TRAIN + VALID
   - Concatenates train + valid to X_train_full, y_train_full.
   - Fits XGBRegressor on the full training data (no test leakage).

5. Compute SHAP values on TEST

   Because of a known compatibility issue between recent shap and xgboost (base_score parsing in TreeExplainer), the script uses:

   - SHAP permutation explainer for XGBRegressor:
     - explainer = shap.Explainer(model.predict, X_background, algorithm="permutation").
     - Background = full TRAIN+VALID feature matrix.
     - Computes SHAP values for all TEST rows:
       - shap_values.shape = (n_test_samples, n_features).
     - Expected value is derived as the mean of the explainer base values.

6. Save plots and arrays
   - Beeswarm plot:
     - Global view of feature contributions across all test observations.
   - Bar plot:
     - Mean absolute SHAP value per feature → overall feature importance ranking.
   - Raw arrays:
     - shap_values_XGBRegressor.npy – 2D SHAP matrix.
     - shap_expected_value_XGBRegressor.npy – scalar expected value.

### 7.2 Interpretability perspective

From the SHAP plots, the importance ranking typically highlights:

- Environment-related features:
  - Weather condition group and visibility,
  - Precipitation intensity and distance traveled,
  - Time-of-day, day/night, and weekend vs weekday,
  - Highway vs non-highway roads, borough/contextual features.

These align with the project goal: understanding how person, vehicle, and environment jointly influence accident severity,
with the model suggesting that environmental & contextual conditions play a major role in predicting severe crashes.


## 8. Environment & Dependencies

Main Python stack:

- Python 3.11
- pandas, numpy
- matplotlib, seaborn
- scikit-learn
- xgboost
- scipy (for statistical tests and Cramér’s V)
- shap
- textblob 
- xlsxwriter (Excel output)

All scripts are written to be reproducible, with:

- Fixed random seed (42) for splitting, imputations, and random-weighted sampling.
- Train-only fitting for all imputers, bins, and encoders.
- Explicit leak-guarding of target-like columns before model training and SHAP computation.


## 9. High-Level Run Order

1. Initial prep & feature engineering of raw merge  
   → df_prepared.csv  
2. EDA and sentiment  
   → analysis_output.xlsx, plots, post_eda.csv  
3. Cleaning, extended EDA, statistical tests, and stratified split  
   → post_cleaning.csv, split/train.csv, split/val.csv, split/test.csv  
4. Train-fitted feature engineering (FE) and artifact saving  
   → train_fe.csv, valid_fe.csv, test_fe.csv, FE artifacts  
5. Stage 2 preprocessing (not fully detailed here) to numeric  
   → train_stage2_processed.csv, valid_stage2_processed.csv, test_stage2_processed.csv  
6. Stage 2 model selection (RF/XGB with two-stage hyperparameter tuning)  
   → model_selection_stage2.xlsx  
7. Final model training (TRAIN+VALID) and SHAP explainability on TEST  
   → SHAP plots + .npy arrays

This README summarizes the full path from raw NYC + FARS accident records, through cleaning, feature engineering, EDA, statistical testing, modeling, and explainability, with a strong emphasis on combining person, vehicle, and environment into a single, interpretable predictive model for accident severity.


