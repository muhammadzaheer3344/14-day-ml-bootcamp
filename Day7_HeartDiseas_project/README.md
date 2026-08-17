# Heart Disease Detector — Day 7

Project notebook: Heart Disease Detector_.ipynb

## Overview
This project trains and evaluates models to predict heart disease using the UCI Heart Disease dataset. The notebook covers data cleaning, feature engineering, model training, hyperparameter tuning, evaluation, and model export.

## Files
- `Heart Disease Detector_.ipynb` — Jupyter notebook with end-to-end workflow.
- `heart_disease_uci.csv` — Raw dataset used for analysis and modeling.
- `feature_names.txt` — Saved list of feature names produced by the notebook (after training).
- `heart_disease_model.pkl` — Trained model (saved by the notebook after training).
- `scaler.pkl` — StandardScaler fitted on training data.

## Notebook Outline
1. Import libraries and load data
2. Initial EDA: shapes, datatypes, missing values
3. Data cleaning and imputation
4. Encoding categorical variables
5. Train/test split and scaling
6. Train baseline models (Logistic Regression, Random Forest, XGBoost)
7. Cross-validation and model comparison
8. Grid search for Random Forest hyperparameters
9. Create binary target variants and handle class imbalance (SMOTE)
10. Feature engineering and enhanced models
11. Final model selection, evaluation, and saving artifacts

## How to run
1. Open the notebook `Heart Disease Detector_.ipynb` in Jupyter or VS Code.
2. Ensure dependencies are installed. A minimal `requirements.txt` should include:

```
pandas
numpy
matplotlib
seaborn
scikit-learn
joblib
scikit-learn
imbalanced-learn
xgboost
```

3. Update file paths in the notebook (if needed) to point to `heart_disease_uci.csv` and desired save locations.
4. Run cells sequentially. The notebook saves the best model and scaler to the folder.

## Notes
- The notebook creates a binary/3-class variant of the target for some experiments.
- Some columns with many missing values were dropped during preprocessing (e.g., `ca`, `thal`, `slope`).
- Feature engineering steps add interaction features and ratios such as `bp_ratio`, `age_bp`, and `chol_bp_ratio`.

## Contact
If you have questions or suggestions, open an issue or contact the author.
