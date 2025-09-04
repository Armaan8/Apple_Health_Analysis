# Apple Health Personal Analytics Project

An end-to-end data analytics & machine learning project using my personal Apple Health data (2022–2024).  
Built with Python (pandas, matplotlib, scikit-learn) for analysis and modeling.  

---

## Motivation

I wanted to explore how daily activity, mobility, lifestyle, and environmental factors impact my cardio health, and whether I could build a model to predict Resting Heart Rate (RHR).  

This project demonstrates:
- End-to-end data cleaning, feature engineering, analysis, and modeling
- Translating raw sensor data into meaningful insights
- Applying business analyst + data science skills on a real-world dataset  

---

## Data Processing

- Raw Apple Health export (XML) converted to CSV files  
- Combined 20+ metrics into a daily dataset (1127 rows × 30 columns)  
- Added calendar features (`year`, `month`, `week`, `weekday`)  
- Derived new indices:
  - EnergyBalance = Active / (Active + Basal)  
  - MobilityIndex = mean(WalkingSpeed, StepLength, SixMinuteWalkTest)  
  - StabilityIndex = mean(WalkingAsymmetry%, DoubleSupport%)  
  - HRFitnessIndex = HRV / RestingHR  


---

## Analysis Workflow

1. Overview: QA, missing values, daily/weekly/monthly steps analysis  
2. Activity & Energy: Exercise Time, Stand Hours, Active vs Basal Energy, Flights Climbed  
3. Mobility & Stability: Walking Speed, Step Length, Asymmetry, Stability Index  
4. Cardio Health: Resting HR, HRV (SDNN), Walking HR, HR Fitness Index  
5. Lifestyle: Time in Daylight vs Steps, Headphone Audio Exposure  
6. Multivariate: Correlation heatmap, PCA, KMeans clustering of daily patterns  
7. Modeling: Predicting Resting HR (train: 2023, test: 2024)

---

## Key Insights

- Activity cluster: StepCount, DistanceWalkingRunning, ExerciseTime, EnergyBurned strongly correlated  
- Mobility cluster: WalkingSpeed, StepLength, SixMinuteWalkDistance move together  
- Cardio link: Higher HRV ↔ Lower Resting HR (classic fitness marker)  
- Lifestyle: More daylight exposure linked to higher step counts; headphone audio peaked on commute/work days  
- Modeling: Random Forest explained significant variance in RHR  
  - Top drivers: Exercise Time, HRV, Step Count, Walking Speed  

---

## Predictive Modeling

- Target: Resting Heart Rate (bpm)  
- Features: ~20 activity, mobility, cardio, lifestyle metrics + calendar effects  
- Split: Train on 2023 → Test on 2024  
- Model: Random Forest Regressor  

**Results (2023→2024 test set):**
- Random Forest achieved strong performance with low MAE
- Predictions tracked long-term RHR trends  

---

## Tools & Libraries

- Python: pandas, numpy, matplotlib  
- Machine Learning: scikit-learn (Random Forest, PCA, KMeans)  
- Visualization: matplotlib (heatmaps, scatter, violin, bar/line plots) 

---

## Author

**Armaan Sharma**  
Data Science & AI | Business Analytics | Applied ML Projects  
