# Apple Health Personal Analytics Project & Dashboard

An end-to-end **data analytics and machine learning project** using my personal Apple Health data (2022–2024).  
Includes offline analysis in Python and an interactive Streamlit dashboard.

---

## Motivation
Explore how **daily activity, mobility, lifestyle, and environment** affect cardio health, and build a model to predict **Resting Heart Rate (RHR)**.

---

## Data
- Source: Apple Health XML exports → structured CSV  
- Derived indices:  
  - **EnergyBalance** = Active / (Active + Basal)  
  - **MobilityIndex** = mean(WalkingSpeed, StepLength, SixMinuteWalkTest)  
  - **StabilityIndex** = mean(WalkingAsymmetry%, DoubleSupport%)  
  - **HRFitnessIndex** = HRV / RestingHR  

---

## Workflow
1. **Preprocessing**: cleaning, missing values, feature engineering  
2. **Analysis**: activity, mobility, cardio, lifestyle metrics  
3. **Multivariate**: correlation heatmaps, PCA, clustering  
4. **Modeling**: Random Forest regression to predict RHR  
5. **Validation**: Train on 2023 → Test on 2024  

---

## Key Insights
- Activity cluster: Steps, Distance, ExerciseTime, EnergyBurned strongly correlated  
- Mobility cluster: WalkingSpeed, StepLength, SixMinuteWalkDistance move together  
- Cardio: Higher HRV ↔ Lower Resting HR  
- Lifestyle: More daylight linked to higher activity; audio exposure peaked on commute days  
- Top RHR drivers: **Exercise Time, HRV, Step Count, Walking Speed**

---

## Dashboard Features (Streamlit)
- **Overview**: KPIs and high-level metrics  
- **Activity Analysis**: steps, distance, energy trends  
- **Heart Rate Analysis**: Resting HR, HRV, Walking HR, Fitness Index  
- **Lifestyle Analysis**: daylight exposure, audio patterns  
- **Predictive Models**: automated RHR forecasting  
- **Relationship Analysis**: correlations, PCA, clustering  
- **Walking Analysis**: gait stability and mobility indices  

---

## Tech Stack
- **Data**: pandas, numpy  
- **Visualization**: matplotlib, plotly  
- **ML**: scikit-learn (RandomForest, PCA, KMeans, LinearRegression)  
- **App**: Streamlit  

---

## Results
- Random Forest explained significant variance in RHR  
- Achieved low MAE and strong R² on 2024 validation  
- Predictions tracked long-term RHR trends  

---

## Author
**Armaan Sharma**  
