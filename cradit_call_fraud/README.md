# 💳 Day 11 — Fraud Detector
### *Imbalanced Data (SMOTE) | Credit Card Fraud Dataset*

![Python](https://img.shields.io/badge/Python-3.10-blue) ![Sklearn](https://img.shields.io/badge/Scikit--Learn-Classification-orange) ![XGBoost](https://img.shields.io/badge/XGBoost-Comparison-red) ![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

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
Built a fraud detection model on one of the most extreme class-imbalance problems in ML (0.17% fraud rate), then ran a controlled experiment comparing SMOTE vs. no-SMOTE to test whether the "obvious" imbalance fix actually helps.

## 📊 Dataset
| Detail | Value |
|---|---|
| Source | [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |
| Rows | 284,807 |
| Columns | 31 (`Time`, `Amount`, `V1`–`V28` PCA-anonymized features, `Class`) |
| Target | `Class` (1 = fraud, 0 = legitimate) |
| Class balance | **577.9 : 1** — only 492 fraud cases out of 284,807 transactions (0.17%) |

## ⚙️ Workflow

<details>
<summary><b>1️⃣ EDA — Understanding the Imbalance</b></summary>

- Confirmed 0 missing values across all columns
- Quantified the imbalance: 284,315 legitimate vs. 492 fraudulent transactions (577.9:1 ratio)
- Flagged that accuracy alone would be a meaningless metric here — a model predicting "not fraud" every time would score 99.8% accuracy while catching zero fraud
</details>

<details>
<summary><b>2️⃣ Time & Amount Analysis</b></summary>

- `Time` spans 48 hours of transactions; `Amount` ranges from $0 to $25,691 (mean $88.35, median just $22)
- Compared fraud vs. legitimate transaction amounts: fraud transactions actually had a **higher mean** ($122.21 vs. $88.29) but **lower median** ($9.25 vs. $22) — fraud tends toward either very small "test" charges or occasional large ones
</details>

<details>
<summary><b>3️⃣ Scaling</b></summary>

- `Time` and `Amount` were the only unscaled raw features (V1–V28 already PCA-transformed and scaled)
- Used `RobustScaler` instead of `StandardScaler` specifically because `Amount` has extreme outliers (up to $25,691) that would distort a mean/std-based scaler
</details>

<details>
<summary><b>4️⃣ Stratified Train-Test Split</b></summary>

- 80/20 split, stratified to preserve the 0.17% fraud rate in both sets exactly
</details>

<details>
<summary><b>5️⃣ SMOTE — Applied Correctly</b></summary>

- Applied SMOTE **only to the training set** (227,451 vs. 394 → 227,451 vs. 227,451), explicitly verified the test set was left untouched to avoid data leakage
- Generated 227,057 synthetic fraud examples
</details>

<details>
<summary><b>6️⃣ Model Training & Comparison</b></summary>

- Trained Logistic Regression, Random Forest, and XGBoost on the SMOTE-balanced data
- Evaluated all three on the original (untouched) imbalanced test set using Precision, Recall, F1, and AUC-ROC — not accuracy
</details>

<details>
<summary><b>7️⃣ The Critical Experiment — SMOTE vs. No-SMOTE</b></summary>

- Re-trained the best model (Random Forest) **without** SMOTE, on the original imbalanced training data
- Directly compared both versions on the same test set
</details>

<details>
<summary><b>8️⃣ Final Model Selection & Saving</b></summary>

- Selected Random Forest **without** SMOTE as the final model, based on the head-to-head comparison
- Saved model + both scalers with `joblib`
</details>

## 📈 Results

**All models trained on SMOTE-balanced data (evaluated on real imbalanced test set):**
| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|---|---|---|---|---|---|
| Logistic Regression | 97.47% | 5.91% | 91.84% | 11.10% | 0.9712 |
| **Random Forest** | 99.95% | 86.96% | 81.63% | 84.21% | 0.9778 |
| XGBoost | 99.92% | 73.28% | 86.73% | 79.44% | 0.9775 |

**The key experiment — Random Forest, SMOTE vs. No-SMOTE (same test set):**
| Version | Precision | Recall | F1-Score | AUC-ROC | False Positives | False Negatives |
|---|---|---|---|---|---|---|
| With SMOTE | 86.96% | 81.63% | 84.21% | 0.9778 | 31 | 18 |
| **Without SMOTE** | **95.24%** | 81.63% | **87.91%** | 0.9528 | **4** | 18 |

## 🔑 Key Findings

- **Logistic Regression trained on SMOTE data was nearly useless in practice** — despite catching 91.8% of fraud (high recall), only 5.9% of its fraud alerts were real, meaning it flagged over 1,400 legitimate transactions as fraud. High recall alone is meaningless without checking precision
- **SMOTE didn't actually improve the best model — it made it worse.** Random Forest without SMOTE beat Random Forest with SMOTE on precision (95.2% vs. 87.0%) and F1 (87.9% vs. 84.2%), while matching it exactly on recall (81.6%). SMOTE's synthetic fraud examples pushed the model toward over-flagging, nearly 8x more false positives (31 vs. 4) for zero recall gain
- **This contradicts the "always use SMOTE for imbalanced data" assumption** — Random Forest's built-in ability to handle imbalance (via bootstrapping and ensemble averaging) was already good enough here; adding synthetic oversampling only introduced noise
- **The final model catches 82% of fraud with only 4 false positives out of 56,864 legitimate transactions** — a precision/recall balance realistic for a production fraud system, where false positives mean real customer friction
- **Takeaway for future imbalanced-data problems:** always run the no-SMOTE baseline as a control, rather than assuming resampling techniques are automatically beneficial

## 📁 Output Files
- `fraud_detector_model.pkl` — final Random Forest model (trained without SMOTE)
- `scaler_time.pkl` — fitted RobustScaler for `Time`
- `scaler_amount.pkl` — fitted RobustScaler for `Amount`

## 🛠️ Tech Stack
`pandas` · `numpy` · `matplotlib` · `seaborn` · `scikit-learn` · `xgboost` · `imbalanced-learn` · `joblib`
