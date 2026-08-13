# 🚢 Day 4 — Survival Classifier
### *Logistic Regression | Titanic Dataset*

![Python](https://img.shields.io/badge/Python-3.10-blue) ![Sklearn](https://img.shields.io/badge/Scikit--Learn-Classification-orange) ![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

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
Built a **Logistic Regression** model to predict passenger survival on the Titanic using demographic, family, and ticket-class features.

## 📊 Dataset
| Detail | Value |
|---|---|
| Source | [Kaggle — Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic) |
| Rows | 891 |
| Columns | 12 |
| Target | `Survived` (0 = died, 1 = survived) |
| Class balance | 38.4% survived / 61.6% died |

## ⚙️ Workflow

<details>
<summary><b>1️⃣ EDA — Survival Distribution</b></summary>

- Checked survival counts and rate with `value_counts()`
- Visualized class balance with a countplot
</details>

<details>
<summary><b>2️⃣ Missing Value Handling</b></summary>

| Column | Missing | Strategy |
|---|---|---|
| `Age` | 177 | Filled using group-wise median (`Pclass` + `Sex`) |
| `Cabin` | 687 (77%) | Dropped entirely — too sparse to be useful |
| `Embarked` | 2 | Filled with mode |
</details>

<details>
<summary><b>3️⃣ Feature Engineering</b></summary>

- `FamilySize` = `SibSp` + `Parch` + 1
- `Title` extracted from `Name` (e.g., Mr, Mrs, Miss, Master) using regex
- Rare titles (Dr, Rev, Col, Major, Don, Lady, Sir, Capt, Countess, Jonkheer) grouped into a single `Rare` category
- Final title distribution: `Mr` (517), `Miss` (185), `Mrs` (126), `Master` (40), `Rare` (23)
</details>

<details>
<summary><b>4️⃣ Cleanup & Encoding</b></summary>

- Dropped irrelevant columns: `Name`, `Ticket`, `PassengerId`
- One-Hot Encoded `Sex`, `Embarked`, `Title` (`drop_first=True`)
</details>

<details>
<summary><b>5️⃣ Train-Test Split & Scaling</b></summary>

- 80/20 train-test split
- Applied `StandardScaler` to normalize feature ranges
</details>

<details>
<summary><b>6️⃣ Model Training & Evaluation</b></summary>

- Trained `LogisticRegression`
- Generated both class predictions (`predict()`) and probability scores (`predict_proba()`)
- Evaluated with Confusion Matrix, Accuracy, Precision, Recall, F1-score
- Plotted ROC Curve and calculated AUC score
</details>

## 📈 Results

| Metric | Score |
|---|---|
| Accuracy | 81.6% |
| Precision | 78.1% |
| Recall | 77.0% |
| F1-Score | 77.6% |
| **AUC-ROC** | **0.890** |

**Classification Report:**
| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| 0 (Died) | 0.84 | 0.85 | 0.84 | 105 |
| 1 (Survived) | 0.78 | 0.77 | 0.78 | 74 |

## 🔑 Key Findings

- **AUC = 0.890** indicates strong discriminative power — the model correctly ranks a random survivor above a random non-survivor ~89% of the time
- Class 0 (died) predictions are slightly more reliable than class 1 (survived), likely due to more training examples (105 vs 74 in test set)
- Engineered features (`Title`, `FamilySize`) captured meaningful social/demographic signal beyond raw `Sex` and `Age` alone
- Model performs well as a baseline; ensemble methods (Day 5–6) are expected to push performance further by capturing non-linear interactions

## 🛠️ Tech Stack
`pandas` · `numpy` · `matplotlib` · `seaborn` · `scikit-learn`
