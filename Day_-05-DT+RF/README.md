# 📞 Day 5 — Churn Predictor
### *Decision Trees + Random Forest | Telco Customer Churn Dataset*

![Python](https://img.shields.io/badge/Python-3.10-blue) ![Sklearn](https://img.shields.io/badge/Scikit--Learn-Ensemble-orange) ![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

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
Built and compared **Decision Tree** and **Random Forest** classifiers to predict customer churn for a telecom company, with hyperparameter experimentation and feature importance analysis.

## 📊 Dataset
| Detail | Value |
|---|---|
| Source | [Kaggle — Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) |
| Rows | 7,043 |
| Columns | 21 |
| Target | `Churn` (Yes/No → encoded 1/0) |
| Class balance | 26.5% churned / 73.5% retained |

## ⚙️ Workflow

<details>
<summary><b>1️⃣ EDA — Churn Distribution</b></summary>

- Checked churn counts and percentage split (5,174 No / 1,869 Yes)
- Visualized class imbalance with a countplot
</details>

<details>
<summary><b>2️⃣ Data Cleaning</b></summary>

- `TotalCharges` was stored as text (`object`) with 11 blank/whitespace entries
- Converted to numeric using `pd.to_numeric(errors='coerce')`
- Filled the 11 resulting missing values with `0` (all corresponded to customers with `tenure = 0`, i.e. brand-new customers)
- Confirmed zero missing values across the full dataset afterward
</details>

<details>
<summary><b>3️⃣ Feature/Target Split & Encoding</b></summary>

- Dropped `customerID` (identifier, not predictive) and separated `Churn` as target
- One-Hot Encoded 15 categorical features (`drop_first=True`) → 19 raw features became 30 encoded features
- Mapped target `Churn`: `No → 0`, `Yes → 1`
</details>

<details>
<summary><b>4️⃣ Train-Test Split</b></summary>

- 80/20 **stratified** split to preserve the 26.5% churn rate in both sets
</details>

<details>
<summary><b>5️⃣ Decision Tree</b></summary>

- Trained `DecisionTreeClassifier(max_depth=5)`
- Visualized the first 3 levels of the tree with `plot_tree()`
- Evaluated on train and test sets
</details>

<details>
<summary><b>6️⃣ Random Forest + Hyperparameter Exploration</b></summary>

- Trained a baseline `RandomForestClassifier(n_estimators=100, max_depth=5)`
- Manually tested 9 configurations (`n_estimators`: 50/100/200 × `max_depth`: 3/5/8), ranked by F1-score
- Selected the best-performing config: `n_estimators=50, max_depth=8`
</details>

<details>
<summary><b>7️⃣ Feature Importance</b></summary>

- Extracted and ranked feature importances from the tuned Random Forest
- Plotted the top 10 most influential features
</details>

<details>
<summary><b>8️⃣ Model Comparison & Overfitting Check</b></summary>

- Compared Decision Tree vs. tuned Random Forest on test metrics
- Checked train vs. test accuracy gap to assess overfitting
</details>

<details>
<summary><b>9️⃣ Model Saving</b></summary>

- Saved the final model bundle (model + feature columns + target mapping) using `joblib`
</details>

## 📈 Results

**Model Comparison (Test Set):**
| Model | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| Decision Tree | 79.42% | 63.12% | 54.01% | 58.21% |
| **Random Forest (tuned)** | **80.77%** | **67.82%** | 52.41% | **59.13%** |

**Overfitting Check (Train vs. Test Accuracy):**
| Model | Train Accuracy | Test Accuracy | Gap |
|---|---|---|---|
| Decision Tree | 80.23% | 79.42% | 0.81% |
| Random Forest | 83.16% | 80.77% | 2.39% |

Both models show a small train-test gap, indicating minimal overfitting — Random Forest's slightly larger gap is expected given its higher capacity, but still well within a healthy range.

## 🔑 Key Findings

**Top 5 churn drivers (Random Forest feature importance):**
1. 🥇 `tenure` — how long a customer has stayed is the single strongest signal
2. 🥈 `TotalCharges`
3. 🥉 `MonthlyCharges`
4. `InternetService_Fiber optic` — fiber customers churn more than DSL/no-internet customers
5. `Contract_Two year` — long-term contracts strongly reduce churn risk

**Trade-off observed:** Random Forest improved precision noticeably over the Decision Tree (67.8% vs. 63.1%) but recall stayed roughly flat — meaning it's more conservative about flagging churn, so a business focused on catching every at-risk customer may want to adjust the decision threshold rather than rely on default 0.5.

## 📁 Output Files
- `telco_churn_random_forest.pkl` — saved model bundle (model + feature columns + target encoding)

## 🛠️ Tech Stack
`pandas` · `numpy` · `matplotlib` · `seaborn` · `scikit-learn` · `joblib`
