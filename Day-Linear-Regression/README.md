# 🏡 Day 3 — Home Price Predictor
### *Linear Regression | California Housing Dataset*

![Python](https://img.shields.io/badge/Python-3.10-blue) ![Sklearn](https://img.shields.io/badge/Scikit--Learn-Regression-orange) ![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

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
Built a **Linear Regression** model to predict median house values in California using demographic, geographic, and housing features.

## 📊 Dataset
| Detail | Value |
|---|---|
| Source | [Kaggle — California Housing Prices](https://www.kaggle.com/datasets/camnugent/california-housing-prices) |
| Rows | 20,640 |
| Columns | 10 |
| Target | `median_house_value` |

## ⚙️ Workflow

<details>
<summary><b>1️⃣ EDA & Data Cleaning</b></summary>

- Explored dataset with `head()`, `info()`, `describe()`
- Found missing values in `total_bedrooms` (207 rows) → filled with median
- Encoded `ocean_proximity` (categorical) using One-Hot Encoding
</details>

<details>
<summary><b>2️⃣ Feature-Target Split</b></summary>

- Separated features (`X`) from target (`y = median_house_value`)
</details>

<details>
<summary><b>3️⃣ Train-Test Split & Scaling</b></summary>

- 80/20 train-test split
- Applied `StandardScaler` to normalize feature ranges
</details>

<details>
<summary><b>4️⃣ Model Training</b></summary>

- Trained a `LinearRegression` model on scaled training data
</details>

<details>
<summary><b>5️⃣ Evaluation</b></summary>

- Generated predictions on test set
- Evaluated using R², MAE, RMSE
- Visualized Actual vs Predicted values and Residual plot
- Analyzed model coefficients to identify key price drivers
</details>

## 📈 Results

| Metric | Score |
|---|---|
| R² Score | 0.625 |
| MAE | $50,670 |
| RMSE | $70,060 |

## 🔑 Key Findings

**Top positive drivers of price:**
- 🥇 `median_income` — strongest predictor by far
- 🥈 `total_bedrooms`
- 🥉 `households`

**Top negative drivers:**
- `latitude` / `longitude` — location strongly affects price (coastal vs inland)
- `population`
- `ocean_proximity_INLAND` — inland homes valued lower than coastal ones

**Limitations noticed:**
- Dataset has house values **capped at $500,001**, causing a visible cluster in Actual vs Predicted plot
- Residual plot shows a structured (non-random) pattern → Linear Regression struggles with non-linear relationships in this data
- Sets up the case for tree-based models (Random Forest, XGBoost) in later days

## 🛠️ Tech Stack
`pandas` · `numpy` · `matplotlib` · `seaborn` · `scikit-learn`
