# Insurance Management System – Telematics Claim Prediction

This project is the data analytics and ML backbone of an Insurance Management System (IMS) for auto insurance. It processes synthetic driver telematics data and builds models to predict whether a policy will generate at least one claim, exporting UI-friendly outputs for integration with a front-end web app.

---

## Problem & Dataset

- **Goal**: Predict a **binary claim flag** (`NB_Claim > 0`) for each policy based on:
  - Traditional policyholder info: `Insured_age`, `Insured_sex`, `Car_age`, `Marital`, `Car_use`, `Credit_score`, `Region`, `Annual_miles_drive`, `Years_noclaims`, `Territory`, `Duration`.
  - Telematics features:
    - Driving time and distribution (`Annual_pct_driven`, `Total_miles_driven`, `Pct_drive_mon...sun`, `Pct_drive_wkday/wkend`, `Pct_drive_rush am/pm`, `Avgdays_week`).
    - Driving style: `Accel_xxmiles`, `Brake_xxmiles`, left/right turn intensities.

- **Source**: Synthetic dataset from  
  *So, Boucher, Valdez – "Synthetic Dataset Generation of Driver Telematics" (2021)*.

- **Scale**:
  - ~100,000 policies.
  - Very low claim rate: only ~4–5% of rows have `NB_Claim > 0`, mirroring real-world low claim frequency.

---

## Key Technical Choices

### 1. Feature engineering

- **Column normalization**:
  - Replace `"."` and spaces with `"_"` for easier programmatic access (e.g., `Pct.drive.mon` → `Pct_drive_mon`).
- **Unique ID generation**:
  - Synthetic `licensePlate` field built using `uuid` + random character substitution to create anonymized identifiers for UI/frontend usage.
- **Binning / grouping (for analysis and potential features)**:
  - `AgeGroup` using decade bins for `Insured_age`.
  - `CSgroup` by credit score ranges (e.g., `400s`, `500s`, … `900s`).
  - `ClaimGroup` bins for `AMT_Claim` for exploratory analysis of severity.

Technologies: `pandas`, `numpy`, `uuid`, `string`, `random`.

---

### 2. Target construction & imbalance framing

- Original target: `NB_Claim` (integer: 0, 1, 2, 3).
- For classification, converted to **binary**:
  - `NB_Claim = 0` → no claim.
  - `NB_Claim >= 1` (1, 2, 3) → claim.
- Imbalance:
  - `Counter({0: 95,728; 1: 4,272})` → highly skewed.
  - Illustrates why naïve models appear to have high accuracy but fail to capture the minority class.

Technologies: `pandas`, `collections.Counter`.

---

### 3. Feature selection (correlation-based)

- Compute full correlation matrix with `.corr()` and take absolute correlations vs `NB_Claim`.
- Separate:
  - **Low-importance features** (`|corr| <= 0.02`): many fine-grained turn intensity and micro-driving pattern variables.
  - **Higher-signal features**:
    - `Duration`, `Insured_age`, `Car_age`, `Credit_score`
    - `Annual_miles_drive`, `Years_noclaims`
    - `Annual_pct_driven`, `Total_miles_driven`
    - `Pct_drive_2hrs`, `Pct_drive_rushpm`
    - `Avgdays_week`, `Accel_06miles`, `Brake_06miles`, `Brake_08miles`.

- Modeling subset `X_df` uses these more informative variables to reduce noise and dimensionality while keeping core telematics signal.

Technologies: `pandas`, `numpy`.

---

### 4. Class imbalance handling (SMOTE and SMOTE-Tomek)

To avoid a model that always predicts “no claim”, the project applies:

1. **SMOTE (Synthetic Minority Oversampling Technique)**  
   - Generates synthetic examples for the minority class by interpolating between nearest neighbors in feature space.
   - Balances `y` such that:
     - Before: `Counter({0: 95,728; 1: 4,272})`
     - After:  `Counter({0: 95,728; 1: 95,728})`

2. **SMOTE-Tomek Links**
   - Combination of SMOTE with Tomek links under-sampling.
   - Over-samples minority and then removes ambiguous majority samples that form Tomek links (borderline pairs).
   - Result: near perfectly balanced dataset with slightly fewer samples than pure SMOTE (around 95k per class).

Technologies:  
- `imblearn.over_sampling.SMOTE`  
- `imblearn.combine.SMOTETomek`  

---

### 5. Models & training approach

The project compares several algorithms rather than relying on a single model:

#### Baseline models (on original imbalanced data)

- **Logistic Regression** (`LogisticRegression(multi_class='ovr')`)
  - With standardized features (`StandardScaler`).
  - Shows very high accuracy but mostly driven by predicting the majority class.

- **Random Forest** (`RandomForestClassifier`)
  - Parameters: ~50 estimators, `max_depth=10`.
  - Good training performance but still biased by class imbalance.

- **SGDClassifier**
  - Linear classifier optimized via stochastic gradient descent.
  - Initially trained on imbalanced data (shows near-zero recall for the minority class).

Technologies:  
- `sklearn.linear_model.LogisticRegression`  
- `sklearn.ensemble.RandomForestClassifier`  
- `sklearn.linear_model.SGDClassifier`  
- `sklearn.preprocessing.StandardScaler`  

#### Models on balanced data (SMOTE / SMOTE-Tomek)

- **SGDClassifier + SMOTE**
  - Uses `loss='log'` (logistic regression).
  - After SMOTE balancing, training accuracy stabilizes around ~0.70 with more reasonable precision/recall trade-offs.
  - Classification reports show both classes being detected (instead of ignoring claims).

- **SGDClassifier + SMOTE-Tomek**
  - Trains on cleaner decision boundary but with different precision/recall dynamics.
  - Demonstrates the impact of removing borderline majority samples.

Technologies:  
- `sklearn.metrics.confusion_matrix`  
- `sklearn.metrics.classification_report`  

#### XGBoost (tree-based gradient boosting)

- **XGBClassifier**
  - Used as a more expressive, non-linear model on the balanced data.
  - Evaluated with:
    - 5-fold cross-validation (`cross_val_score`)
    - 10-fold K-Fold (`KFold`) for a more robust estimate.
  - Confusion matrix indicates strong ability to capture both claim and non-claim classes.

Technologies:  
- `xgboost.XGBClassifier`  
- `sklearn.model_selection.cross_val_score`, `KFold`  

---

### 6. Evaluation methodology

- **Metrics used**:
  - Accuracy for quick comparison.
  - Confusion matrix for understanding false positives/negatives.
  - Classification report:
    - `precision`, `recall`, `f1-score`, `support` for each class.
  - Cross-validation (5-fold + 10-fold K-Fold) for XGBoost.

- **Focus**:
  - Not just raw accuracy, but ability to detect the **minority (claim) class**.
  - Inspecting trade-offs between precision (avoiding false alarms) and recall (capturing true claims).

Technologies:  
- `sklearn.metrics.confusion_matrix`  
- `sklearn.metrics.classification_report`  
- `sklearn.model_selection.cross_val_score`, `KFold`  

---

### 7. UI integration output

To plug the ML backend into a front-end **Insurance Management System**:

- Final DataFrame includes:
  - `licensePlate` (synthetic ID)
  - Telematics + driving style columns (turn intensities, daily distribution, acceleration/braking counts)
  - `predictedClaimValue` (boolean prediction from the selected model)

- Exported artifacts:
  - `telematics_ui.csv` – tabular data for dashboards or REST APIs.
  - `data.json` – small slice example encoded as JSON for front-end prototyping.

Technologies:  
- `pandas.DataFrame.to_csv`  
- Python `json` module.

---

## Tech Stack Summary

- **Language**: Python
- **Data & analysis**: `pandas`, `numpy`, `seaborn`, `matplotlib`
- **ML & evaluation**: `scikit-learn`, `xgboost`
- **Imbalance learning**: `imblearn` (SMOTE, SMOTE-Tomek)
- **Environment**: Jupyter Notebook / Python scripts

---

## How this fits into the IMS project

This module provides:

1. **Risk scoring engine**  
   - Converts raw telematics + policy attributes into `predictedClaimValue` (True/False).

2. **Data products for UI**  
   - Structured CSV/JSON outputs for policy-level dashboards, per-driver risk views, or underwriting tools.

3. **Experimental framework**  
   - A reproducible notebook to try alternative models, resampling strategies, or feature sets as more real-world data becomes available.
