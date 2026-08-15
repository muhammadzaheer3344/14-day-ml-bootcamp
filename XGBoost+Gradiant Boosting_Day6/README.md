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
- [Tech Stack](#%EF%B8%8F-tech-stack)

---

## 🎯 Overview
Built an **XGBoost classifier** (with a Gradient Boosting comparison) to predict loan defaults on the Lending Club dataset — this version fixes a critical **data leakage** issue found in an earlier iteration by restricting features to information known only *at loan origination*.

## 📊 Dataset
| Detail | Value |
|---|---|
| Source | [Kaggle — Lending Club Loan Data](https://www.kaggle.com/datasets/wordsforthewise/lending-club) |
| File used | `accepted_2007_to_2018Q4.csv` |
| Rows loaded | 300,000 (via `usecols` + `nrows` to manage memory) |
| Rows after filtering | 265,777 (loans with a final outcome) |
| Features used | 27 (origination-time only) |
| Target | `default` (1 = Charged Off/Default, 0 = Fully Paid) |
| Class balance | 20.1% default rate |

## ⚙️ Workflow

<details>
<summary><b>1️⃣ Memory-Efficient Loading</b></summary>

- Full file has 151 columns and 2.2M+ rows — inspected column names first with `nrows=0` before loading anything
- Loaded only 27 hand-picked, origination-time-safe columns + target using `usecols`, capped at 300,000 rows
</details>

<details>
<summary><b>2️⃣ Fixing Data Leakage (key improvement over v1)</b></summary>

- Deliberately **excluded** post-origination columns like `total_pymnt`, `recoveries`, `last_pymnt_amnt`, `out_prncp` — these are only known after a loan starts being repaid/defaulting, and caused an unrealistic ~0.999 AUC in an earlier version
- Kept only features a lender would actually have **before** approving the loan: `loan_amnt`, `term`, `int_rate`, `grade`, `sub_grade`, `annual_inc`, `dti`, `fico_range_low/high`, `purpose`, `emp_length`, `home_ownership`, etc.
</details>

<details>
<summary><b>3️⃣ Target Engineering</b></summary>

- Filtered to loans with a resolved outcome only: `Fully Paid`, `Charged Off`, `Default` (dropped `Current`, `Late`, `In Grace Period` — ongoing/ambiguous loans)
- Created binary target `default`: 1 = Charged Off/Default, 0 = Fully Paid
</details>

<details>
<summary><b>4️⃣ Cleaning & Encoding</b></summary>

- Handled missing values: `emp_length` → `'Unknown'`, `revol_util` & `dti` → median
- One-Hot Encoded 10 categorical columns (`drop_first=True`) → 27 raw features became 741 encoded features
- Sanitized column names (removed `[`, `]`, `<`, `>`) for XGBoost compatibility
</details>

<details>
<summary><b>5️⃣ Train-Test Split</b></summary>

- 80/20 stratified split preserving the 20.1% default rate
</details>

<details>
<summary><b>6️⃣ Model Training & Tuning</b></summary>

- Trained a baseline `XGBClassifier`
- Tuned with `RandomizedSearchCV` (8 candidates, 3-fold CV, scoring = F1) across `n_estimators`, `max_depth`, `learning_rate`, `subsample`
- Also trained a `GradientBoostingClassifier` for comparison
</details>

<details>
<summary><b>7️⃣ Evaluation & Interpretability</b></summary>

- Evaluated with Accuracy, Precision, Recall, F1, AUC-ROC on train and test sets
- Plotted Precision-Recall Curve and ROC Curve
- Extracted and plotted top 15 features by importance
</details>

## 📈 Results

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|---|---|---|---|---|---|
| Baseline XGBoost | 80.78% | 59.04% | 14.77% | 23.63% | 0.7446 |
| **Tuned XGBoost** | 80.74% | 58.03% | 15.56% | 24.54% | 0.7435 |
| Gradient Boosting | 80.86% | 59.58% | 15.37% | 24.44% | 0.7445 |

**Train vs. test gap (Tuned XGBoost):** Train accuracy 81.34% vs. test accuracy 80.74% — a ~0.6% gap, showing the model generalizes well and isn't overfitting.

## 🔑 Key Findings

- **AUC-ROC of ~0.74 is a far more realistic and trustworthy number** than the ~0.999 seen in an earlier leaky version — it falls right in the industry-typical range (0.65–0.75) for credit risk models built on legitimate, pre-approval data
- **Recall is low (~15%) despite decent AUC** — the model struggles to catch most actual defaulters at the default 0.5 threshold. This is expected given the class imbalance (only 20% defaults) and is a strong candidate for threshold tuning or techniques like SMOTE/class weighting in a future iteration
- **Top predictive features are all origination-time signals that make business sense:** loan `grade` (especially lower grades F/G), `term` (60-month loans riskier than 36-month), `int_rate`, and `fico_range_high` — exactly what a credit analyst would expect to matter
- **Hyperparameter tuning gave a marginal F1 improvement** (23.6% → 24.5%) but didn't meaningfully move AUC — suggesting the ceiling here is more about feature richness than further tuning
- **XGBoost and Gradient Boosting performed almost identically**, reinforcing that model choice mattered far less than fixing the data leakage issue in the first place

## 🛠️ Tech Stack
`pandas` · `numpy` · `matplotlib` · `seaborn` · `scikit-learn` · `xgboost`
