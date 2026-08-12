# 🏠 Day 2 — House Price Features
### *Feature Engineering | Ames Housing Dataset*

![Python](https://img.shields.io/badge/Python-3.10-blue) ![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-green) ![Sklearn](https://img.shields.io/badge/Scikit--Learn-Preprocessing-orange) ![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## 📋 Table of Contents
- [Overview](#-overview)
- [Dataset](#-dataset)
- [Workflow](#%EF%B8%8F-workflow)
- [Key Findings](#-key-findings)
- [Output Files](#-output-files)
- [Tech Stack](#%EF%B8%8F-tech-stack)

---

## 🎯 Overview
Feature engineering pipeline on the **Ames Housing dataset** to clean, transform, encode, and scale data — prepping it for house price prediction models.

## 📊 Dataset
| Detail | Value |
|---|---|
| Source | [Kaggle — House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) |
| Rows | 1,460 |
| Columns | 81 |
| Target | `SalePrice` |

## ⚙️ Workflow

<details>
<summary><b>1️⃣ Data Overview & Missing Value Analysis</b></summary>

- Explored dataset with `head()`, `tail()`, `shape`, `info()`, `describe()`
- Separated numerical vs categorical columns
- Mapped missing value % per column (`isnull().mean()`)
</details>

<details>
<summary><b>2️⃣ Missing Value Imputation</b></summary>

| Column Type | Strategy |
|---|---|
| "Feature absent" categoricals (`PoolQC`, `Alley`, `FireplaceQu`...) | Filled with `'None'` |
| "Feature absent" numerics (`GarageYrBlt`, `MasVnrArea`) | Filled with `0` |
| `LotFrontage` | Neighborhood-wise median |
| `Electrical` | Mode |
</details>

<details>
<summary><b>3️⃣ Feature Engineering</b></summary>

- `HouseAge` = `YrSold` − `YearBuilt`
- `TotalSF` = `1stFlrSF` + `2ndFlrSF` + `TotalBsmtSF`
</details>

<details>
<summary><b>4️⃣ Encoding & Scaling</b></summary>

- One-Hot Encoding (`pd.get_dummies`, `drop_first=True`) for categoricals
- `StandardScaler` applied to numeric features
</details>

<details>
<summary><b>5️⃣ Correlation & Transformation</b></summary>

- Correlation analysis against `SalePrice`
- Log-transformation (`log1p`) applied to skewed features + target
</details>

<details>
<summary><b>6️⃣ Split & Save</b></summary>

- 80/20 train-test split
- Saved processed dataset + splits
</details>

## 🔑 Key Findings

**Top positive correlations with `SalePrice`:**
- 🥇 `OverallQual` — 0.79
- 🥈 `TotalSF` — 0.78 *(engineered feature!)*
- 🥉 `GrLivArea` — 0.71

**Top negative correlations:**
- `ExterQual_TA` — -0.59
- `HouseAge` — -0.52

**Skewness reduction:** `SalePrice` skew dropped from **1.88 → 1.21** after log-transform ✅

## 📁 Output Files
- `ames_housing_processed.csv` — full processed dataset
- `ames_train_test_split.pkl` — pickled `X_train`, `X_test`, `y_train`, `y_test`

## 🛠️ Tech Stack
`pandas` · `numpy` · `matplotlib` · `seaborn` · `scikit-learn` · `scipy`
