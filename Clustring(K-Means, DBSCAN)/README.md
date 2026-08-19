# 📊 Day 9 — Clustering (K-Means, DBSCAN)
### *Customer Segmentation using Mall Customers Dataset*

![Python](https://img.shields.io/badge/Python-3.10-blue) ![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Clustering-orange) ![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## 📋 Table of Contents
- [Overview](#-overview)
- [Dataset](#-dataset)
- [Workflow](#%EF%B8%8F-workflow)
- [Key Findings](#-key-findings)
- [Output Files](#-output-files)
- [Tools & Libraries](#%EF%B8%8F-tools--libraries)

---

## 🎯 Overview
This project focuses on unsupervised learning through customer segmentation. The notebook applies clustering techniques to group mall customers based on purchasing behavior and demographic factors, helping uncover patterns in spending habits and income levels.

The main goal is to identify meaningful customer segments that can support business decisions such as targeted marketing, loyalty strategies, and customer retention planning.

## 📊 Dataset
| Detail | Value |
|---|---|
| Source | Mall Customers Dataset |
| Rows | 200 |
| Columns | 5 |
| Features | `CustomerID`, `Gender`, `Age`, `Annual Income (k$)`, `Spending Score (1-100)` |
| Objective | Segment customers into clusters based on spending and income behavior |

## ⚙️ Workflow

<details>
<summary><b>1️⃣ Data Exploration</b></summary>

- Loaded the dataset and checked its structure, missing values, and feature distributions
- Examined relationships between `Annual Income`, `Spending Score`, and `Age`
- Explored customer patterns by gender and spending behavior
</details>

<details>
<summary><b>2️⃣ Data Preprocessing</b></summary>

- Selected the most relevant clustering features
- Standardized numerical features before model training
- Prepared the data for distance-based clustering methods
</details>

<details>
<summary><b>3️⃣ K-Means Clustering</b></summary>

- Applied the K-Means algorithm to cluster customers
- Used the elbow method to estimate the optimal number of clusters
- Evaluated cluster separation and interpretability
</details>

<details>
<summary><b>4️⃣ DBSCAN Clustering</b></summary>

- Tested DBSCAN to identify clusters based on density instead of fixed centroids
- Tuned the `eps` and `min_samples` parameters
- Compared DBSCAN results with K-Means to understand the strengths and limitations of each method
</details>

<details>
<summary><b>5️⃣ Cluster Analysis & Interpretation</b></summary>

- Examined cluster profiles based on age, income, and spending score
- Identified customer groups such as high-income/high-spending, low-income/low-spending, and moderate segments
- Used cluster summaries to describe customer personas and business insights
</details>

## 🔑 Key Findings

- K-Means is effective when the number of clusters is known or can be estimated using the elbow method
- DBSCAN is useful for discovering irregular or noise-containing clusters, especially when data is not evenly distributed
- The most informative features for segmentation were `Annual Income` and `Spending Score`
- Customers could be grouped into distinct behavioral segments, enabling better marketing and targeting decisions
- Clustering revealed that not all customers behave similarly, and segmentation can help identify high-value and low-value customer groups

## 📁 Output Files
- `Mall_Customers.csv` — customer dataset used for clustering
- `cluster_profiles.txt` — text summary of identified clusters and their characteristics
- `kmeans_model.pkl` — trained K-Means model
- `scaler.pkl` — fitted scaler used for standardizing features
- `Clustering (K-Means, DBSCAN).ipynb` — full notebook with code, analysis, and visualizations

## 🛠️ Tools & Libraries
`pandas` · `numpy` · `matplotlib` · `seaborn` · `scikit-learn` · `joblib`

---

## ✅ Summary
This day introduced unsupervised machine learning through clustering. The notebook demonstrates how K-Means and DBSCAN can be used to segment customers into meaningful groups, turning raw customer data into actionable business insights.
