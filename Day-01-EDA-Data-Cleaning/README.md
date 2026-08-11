# Day 1 — Taxi Fare Analysis (EDA + Data Cleaning)

## Overview
Exploratory Data Analysis and data cleaning on the NYC Taxi Trip Duration dataset to understand trip patterns and prepare clean data for modeling.

## Dataset
- **Source:** [NYC Taxi Trip Duration (Kaggle)](https://www.kaggle.com/c/nyc-taxi-trip-duration)
- **Size:** 1,458,644 rows × 11 columns (before cleaning)
- **Target variable:** `trip_duration` (in seconds)

## Steps Performed
1. Loaded dataset and explored with `head()`, `tail()`, `shape`, `info()`, `describe()`
2. Checked for missing values and duplicates (none found)
3. Converted `pickup_datetime` and `dropoff_datetime` to datetime format
4. Engineered new features: `hour_of_day`, `day_of_week`, `month`
5. Calculated `distance_km` from pickup/dropoff coordinates using the Haversine formula
6. Identified and removed outliers:
   - Zero passenger trips
   - Trips under 60 seconds or over 2 hours
   - Zero or excessive distance (>100 km)
   - GPS coordinates outside NYC bounds
7. Performed univariate analysis (histograms, boxplots) on `trip_duration` and `distance_km`
8. Performed bivariate analysis: distance vs. duration, average duration by hour of day
9. Built a correlation heatmap — `distance_km` showed the strongest correlation (0.77) with `trip_duration`
10. Saved the cleaned dataset to CSV

## Key Findings
- Trip duration is heavily right-skewed; most trips are short.
- Average trip duration peaks around 3 PM (rush hour) and is lowest around 6 AM.
- Distance is the strongest predictor of trip duration.
- ~0.9% of rows were removed as outliers/data errors.

## Output
- `nyc_taxi_cleaned.csv` — cleaned dataset ready for feature engineering and modeling

## Tools Used
`pandas`, `numpy`, `matplotlib`, `seaborn`
