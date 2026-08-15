# 💳 Day 6 — Loan Default Predictor
### *XGBoost + Gradient Boosting | Lending Club Loans Dataset*

![Python](https://img.shields.io/badge/Python-3.10-blue) ![XGBoost](https://img.shields.io/badge/XGBoost-Classifier-red) ![Sklearn](https://img.shields.io/badge/Scikit--Learn-Tuning-orange) ![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## 📋 Table of Contents
- [Overview](#-overview)
- [Dataset](#-dataset)
- [Workflow](#%EF%B8%8F-workflow)
- [Results](#-results)
- [Key Findings](#-key-findings)
- [Output Files](#-output-files)
- [Tech Stack](#%EF%B8%8F-tech-stack)

---

## 🎯 Overview
Built an **XGBoost classifier** to predict loan defaults using the Lending Club dataset, with hyperparameter tuning, SHAP-based interpretability, and full evaluation via ROC-AUC and Precision-Recall analysis.

## 📊 Dataset
| Detail | Value |
|---|---|
| Source | [Kaggle — Lending Club Loan Data](https://www.kaggle.com/datasets/wordsforthewise/lending-club) |
| File used | `accepted_2007_to_2018Q4.csv` |
| Rows used | 300,000 (loaded via chunking for memory efficiency) |
| Columns (after cleaning) | 85 |
| Target | `is_bad` (1 = defaulted/late loan, 0 = good standing) |
| Class balance | 18.26% default rate |

## ⚙️ Workflow

<details>
<summary><b>1️⃣ Memory-Efficient Data Loading</b></summary>

- Full dataset (~1.5GB+, 2.2M rows) caused repeated Colab session crashes due to RAM limits
- Solved using **chunked reading** (`chunksize=100000`) and capping at 300,000 rows
</details>

<details>
<summary><b>2️⃣ Data Cleaning</b></summary>

- Dropped 57 columns with >50% missing values
- Removed irrelevant identifier/date columns (`id`, `member_id`, `url`, `desc`, date fields, etc.)
- Final shape after cleaning: 300,000 × 85
</details>

<details>
<summary><b>3️⃣ Target Engineering</b></summary>

- Created binary target `is_bad` from `loan_status`:
  - Bad loan = `Charged Off`, `Default`, `Late (31-120 days)`, `Late (16-30 days)`
  - Good loan = everything else
- Original `loan_status` column dropped after encoding
</details>

<details>
<summary><b>4️⃣ Encoding & Missing Value Imputation</b></summary>

- 14 categorical columns encoded using `LabelEncoder`
- Remaining missing numeric values filled using `SimpleImputer` (median strategy)
</details>

<details>
<summary><b>5️⃣ Train-Test Split</b></summary>

- 80/20 stratified split to preserve the 18.26% default rate in both sets
</details>

<details>
<summary><b>6️⃣ Model Training & Tuning</b></summary>

- Trained a baseline `XGBClassifier`
- Tuned with `RandomizedSearchCV` (20 candidate combinations, 3-fold CV, scoring = ROC-AUC) across:
  - `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`
</details>

<details>
<summary><b>7️⃣ Evaluation & Interpretability</b></summary>

- Evaluated with Accuracy, F1-score, AUC-ROC, Confusion Matrix
- Plotted ROC Curve and Precision-Recall Curve
- Feature importance via `feature_importances_` and **SHAP summary plots**
- Checked train vs test performance for overfitting
</details>

## 📈 Results

| Model | Accuracy | F1-Score | AUC-ROC |
|---|---|---|---|
| Base XGBoost | 0.9941 | 0.9835 | 0.9992 |
| Tuned XGBoost (final) | **0.9952** | **0.9867** | **0.9994** |

**Best hyperparameters found:** `n_estimators=200`, `max_depth=6`, `subsample=1.0` (via RandomizedSearchCV)

**Train vs Test (overfitting check):**
| Metric | Train | Test |
|---|---|---|
| Accuracy | 0.9972 | 0.9952 |
| AUC-ROC | 0.9999 | 0.9994 |

Gap between train and test is small (<1%), so the model isn't meaningfully overfitting to the training data.

## 🔑 Key Findings

**Top features by importance:** `recoveries`, `collection_recovery_fee`, `last_fico_range_high`, `term`, `last_pymnt_amnt`

**⚠️ Important caveat — likely data leakage:** Several of the top-ranked features (`recoveries`, `collection_recovery_fee`, `total_pymnt`, `last_pymnt_amnt`, `out_prncp`) are only known **after** a loan has already been issued and its outcome is unfolding — they aren't available at the time a lender would actually need to predict default risk. This is almost certainly why AUC-ROC came out near-perfect (0.999+), which is unrealistically high for real-world credit risk models (industry AUC is typically 0.65–0.75). A more realistic version of this project would restrict features to only those known **at loan origination** (e.g., `loan_amnt`, `int_rate`, `grade`, `annual_inc`, `dti`, `emp_length`, `fico_range_low/high`, `purpose`) to get a trustworthy estimate of predictive power.

## 📁 Output Files
- `loan_default_xgboost_model.pkl` — trained final model
- `label_encoders.pkl` — saved categorical encoders
- `imputer.pkl` — saved missing-value imputer
- `feature_importance.csv` — ranked feature importance table

## 🛠️ Tech Stack
`pandas` · `numpy` · `matplotlib` · `seaborn` · `scikit-learn` · `xgboost` · `shap` · `joblib`
