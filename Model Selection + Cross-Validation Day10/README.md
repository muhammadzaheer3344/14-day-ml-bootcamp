# 🩺 Day 10 — Diabetes Predictor
### *Model Selection + Cross-Validation | Diabetes Dataset*

![Python](https://img.shields.io/badge/Python-3.10-blue) ![Sklearn](https://img.shields.io/badge/Scikit--Learn-Model_Selection-orange) ![XGBoost](https://img.shields.io/badge/XGBoost-Comparison-red) ![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

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
Compared 5 classification algorithms using systematic cross-validation and hyperparameter tuning to predict diabetes onset, then diagnosed and fixed a significant overfitting problem in the "best" model before finalizing.

## 📊 Dataset
| Detail | Value |
|---|---|
| Source | [Kaggle — Pima Indians Diabetes Database](https://www.kaggle.com/datasets/saurabh00007/diabetescsv) |
| Rows | 768 |
| Columns | 9 |
| Target | `Outcome` (1 = diabetic, 0 = non-diabetic) |
| Class balance | 65.1% non-diabetic / 34.9% diabetic |

## ⚙️ Workflow

<details>
<summary><b>1️⃣ Data Cleaning — the "hidden missing values" problem</b></summary>

- No literal `NaN` values, but 5 medical columns had biologically impossible **zeros** (e.g., 0 Glucose, 0 BMI) — a common gotcha in this dataset
- `Insulin`: 48.7% zeros | `SkinThickness`: 29.6% zeros | `BloodPressure`: 4.6% zeros | `BMI`: 1.4% zeros | `Glucose`: 0.65% zeros
- Replaced zeros with `NaN`, then imputed using column median
</details>

<details>
<summary><b>2️⃣ Scaling & Split</b></summary>

- Applied `StandardScaler` to all features
- 80/20 stratified split, preserving the 34.9% diabetic rate in both sets
</details>

<details>
<summary><b>3️⃣ Baseline Model Comparison (5-Fold CV)</b></summary>

- Compared Logistic Regression, Random Forest, SVM, and KNN with default settings using 5-fold cross-validation (not just a single train/test check)
</details>

<details>
<summary><b>4️⃣ Hyperparameter Tuning</b></summary>

- `GridSearchCV` for Logistic Regression, SVM, and KNN
- `RandomizedSearchCV` (30 iterations) for Random Forest and XGBoost
- Measured improvement of each tuned model over its own default baseline
</details>

<details>
<summary><b>5️⃣ Handling Class Imbalance</b></summary>

- Applied **SMOTE** to the training set to balance classes (400 vs. 214 → 400 vs. 400)
- Re-evaluated all 5 tuned models on the SMOTE-balanced data
- Also tried **class-weighted** Logistic Regression and Random Forest as a lighter-weight alternative to SMOTE
</details>

<details>
<summary><b>6️⃣ Ensemble Method</b></summary>

- Built a soft-voting `VotingClassifier` combining tuned Logistic Regression, Random Forest, and KNN
</details>

<details>
<summary><b>7️⃣ Overfitting Diagnosis & Fix (the core lesson of this day)</b></summary>

- Checked train vs. test performance on the top SMOTE model (Random Forest) → found a **22.6% accuracy gap** — severe overfitting
- Rebuilt with regularization: fewer/shallower trees, higher `min_samples_split`/`min_samples_leaf`, `max_samples=0.8` → gap dropped to **10.5%**
- Also tested a heavily-regularized Logistic Regression (`C=0.1`) as a lower-variance alternative → gap of just **4.5%**, at the cost of some accuracy
</details>

<details>
<summary><b>8️⃣ Final Model Selection & Saving</b></summary>

- Selected the **regularized Random Forest** as the final model — best balance of accuracy, AUC, and generalization
- Saved model + scaler with `joblib`
</details>

## 📈 Results

**5-Fold CV comparison (default vs. tuned, original data):**
| Model | Default CV | Tuned CV | Improvement |
|---|---|---|---|
| KNN | 75.25% | **78.34%** | +3.09% |
| Logistic Regression | 78.18% | 78.18% | +0.00% |
| Random Forest | 77.20% | 77.69% | +0.49% |
| SVM | 76.55% | 77.37% | +0.82% |
| XGBoost | 72.80% | 77.53% | +4.73% |

**With SMOTE (class-balanced training data):**
| Model | CV Accuracy | vs. Original |
|---|---|---|
| **Random Forest** | **82.25%** | +4.56% |
| KNN | 81.50% | +3.16% |
| XGBoost | 80.50% | +2.70% |
| SVM | 75.37% | -1.99% |
| Logistic Regression | 73.63% | -4.56% |

**Overfitting fix — Random Forest, train vs. test:**
| Version | Train Acc | Test Acc | Gap | Test AUC |
|---|---|---|---|---|
| SMOTE RF (unregularized) | 97.88% | 75.32% | 22.56% ❌ | 0.8246 |
| **Regularized RF (final)** | 87.13% | **76.62%** | **10.51%** ✅ | **0.8289** |
| Regularized Logistic Regression | 76.55% | 72.08% | 4.47% ✅ | 0.8100 |

## 🔑 Key Findings

- **This dataset's real trap wasn't missing values — it was disguised zeros.** Nearly half of all `Insulin` readings were literally `0`, which is medically impossible and would have quietly wrecked any model trained on raw data
- **Hyperparameter tuning gave modest gains on most models (0–5%)**, but the highest-leverage move was SMOTE — rebalancing the training classes lifted Random Forest and KNN by 3–5 points, though it *hurt* Logistic Regression and SVM, showing that imbalance-handling techniques aren't universally beneficial
- **The single most important step was checking train-vs-test gap, not chasing the highest CV score.** SMOTE-boosted Random Forest looked like the winner at 82% CV accuracy, but a 22.6% train-test gap on the actual test set revealed it had badly overfit
- **Regularizing traded a few points of accuracy for real generalization** — capping tree depth, requiring more samples per split, and subsampling per tree cut the overfitting gap by more than half (22.6% → 10.5%) while keeping AUC essentially unchanged (0.8246 → 0.8289)
- **Final model: Regularized Random Forest** — 76.6% test accuracy, 0.829 AUC-ROC, and a gap small enough to trust on unseen patients

## 📁 Output Files
- `diabetes_final_model.pkl` — final regularized Random Forest model
- `scaler.pkl` — fitted StandardScaler

## 🛠️ Tech Stack
`pandas` · `numpy` · `matplotlib` · `seaborn` · `scikit-learn` · `xgboost` · `imbalanced-learn` · `joblib`
