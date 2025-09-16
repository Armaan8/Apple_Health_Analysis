import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# Create date range from Jan 2022 to Dec 2024 (3 years of data)
start_date = datetime(2022, 1, 1)
end_date = datetime(2024, 12, 31)
dates = pd.date_range(start=start_date, end=end_date, freq='D')

n_days = len(dates)
print(f"Creating {n_days} days of health data...")

# Create base data
data = {
    'date': dates,
    'year': dates.year,
    'month': dates.month,
    'week': dates.isocalendar().week,
    'weekday': dates.day_name(),
}

# Add year_month column
data['year_month'] = dates.to_period('M')

# Activity metrics with realistic patterns
base_steps = 8000 + np.random.normal(0, 2000, n_days)
data['StepCount'] = np.maximum(0, base_steps + 2000 * np.sin(2 * np.pi * np.arange(n_days) / 365))  # Seasonal pattern

# Distance proportional to steps with some noise
data['DistanceWalkingRunning'] = data['StepCount'] * 0.0008 + np.random.normal(0, 1, n_days)
data['DistanceWalkingRunning'] = np.maximum(0, data['DistanceWalkingRunning'])

# Cycling distance (occasional, mostly zeros)
cycling_prob = 0.05
data['DistanceCycling'] = np.where(np.random.random(n_days) < cycling_prob, 
                                   np.random.exponential(2, n_days), 0)

# Energy metrics
data['ActiveEnergyBurned'] = 200 + data['StepCount'] * 0.03 + np.random.normal(0, 50, n_days)
data['ActiveEnergyBurned'] = np.maximum(0, data['ActiveEnergyBurned'])

data['BasalEnergyBurned'] = 1600 + np.random.normal(0, 100, n_days)
data['BasalEnergyBurned'] = np.maximum(1000, data['BasalEnergyBurned'])

# Exercise time (correlated with active energy)
data['AppleExerciseTime'] = np.maximum(0, (data['ActiveEnergyBurned'] - 200) / 10 + np.random.normal(0, 5, n_days))

# Stand time and hours
data['AppleStandTime'] = 120 + np.random.normal(0, 60, n_days)
data['AppleStandTime'] = np.maximum(0, data['AppleStandTime'])
data['AppleStandHour'] = data['AppleStandTime'] / 60

# Flights climbed
data['FlightsClimbed'] = np.maximum(0, np.random.poisson(5, n_days))

# Heart rate metrics
base_rhr = 65 + np.random.normal(0, 8, n_days)
fitness_trend = -0.1 * np.arange(n_days) / 365  # Slight improvement over time
data['RestingHeartRate'] = np.maximum(45, base_rhr + fitness_trend)

# Heart rate variability (inversely correlated with resting HR)
data['HeartRateVariabilitySDNN'] = np.maximum(20, 80 - 0.5 * data['RestingHeartRate'] + np.random.normal(0, 10, n_days))

# General heart rate
data['HeartRate'] = data['RestingHeartRate'] + 20 + np.random.normal(0, 15, n_days)

# Walking heart rate (higher than resting)
data['WalkingHeartRateAverage'] = data['RestingHeartRate'] + 25 + np.random.normal(0, 10, n_days)

# Fitness index (derived metric)
data['HRFitnessIndex'] = (data['HeartRateVariabilitySDNN'] / data['RestingHeartRate']) * 100

# Walking metrics
data['WalkingSpeed'] = 3.5 + np.random.normal(0, 0.5, n_days)
data['WalkingSpeed'] = np.maximum(2.0, data['WalkingSpeed'])

data['WalkingStepLength'] = 65 + np.random.normal(0, 8, n_days)
data['WalkingStepLength'] = np.maximum(50, data['WalkingStepLength'])

data['WalkingAsymmetryPercentage'] = np.maximum(0, np.random.exponential(0.05, n_days))
data['WalkingDoubleSupportPercentage'] = 0.25 + np.random.normal(0, 0.05, n_days)
data['WalkingDoubleSupportPercentage'] = np.clip(data['WalkingDoubleSupportPercentage'], 0.1, 0.4)

# Advanced assessments (sparser data)
six_min_prob = 0.1
data['SixMinuteWalkTestDistance'] = np.where(np.random.random(n_days) < six_min_prob,
                                           400 + np.random.normal(0, 50, n_days), np.nan)

walking_steadiness_prob = 0.2
data['AppleWalkingSteadiness'] = np.where(np.random.random(n_days) < walking_steadiness_prob,
                                        0.8 + np.random.normal(0, 0.1, n_days), np.nan)

# Derived indices
data['MobilityIndex'] = 30 + data['WalkingSpeed'] * 5 + np.random.normal(0, 5, n_days)
data['StabilityIndex'] = 0.3 - data['WalkingAsymmetryPercentage'] * 2 + np.random.normal(0, 0.1, n_days)
data['StabilityIndex'] = np.maximum(0, data['StabilityIndex'])

# Lifestyle metrics
# Time in daylight (seasonal pattern, missing for early data)
daylight_start_idx = n_days // 3  # Start recording after 1/3 of the period
daylight_data = np.full(n_days, np.nan)
seasonal_daylight = 60 + 40 * np.sin(2 * np.pi * (np.arange(n_days) + 80) / 365)  # Peak in summer
daylight_data[daylight_start_idx:] = seasonal_daylight[daylight_start_idx:] + np.random.normal(0, 20, n_days - daylight_start_idx)
data['TimeInDaylight'] = np.maximum(0, daylight_data)

# Audio exposure (frequent but variable)
audio_prob = 0.6
data['HeadphoneAudioExposure'] = np.where(np.random.random(n_days) < audio_prob,
                                        70 + np.random.exponential(10, n_days), np.nan)

# Energy balance (derived)
data['EnergyBalance'] = (data['ActiveEnergyBurned'] + data['BasalEnergyBurned']) / 2500

# Create DataFrame
df = pd.DataFrame(data)

# Add some missing values to make it realistic
missing_cols = ['RestingHeartRate', 'HeartRateVariabilitySDNN', 'AppleExerciseTime']
for col in missing_cols:
    missing_mask = np.random.random(n_days) < 0.1  # 10% missing
    df.loc[missing_mask, col] = np.nan

print("Sample data created successfully!")
print(f"Data shape: {df.shape}")
print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
print(f"Columns: {list(df.columns)}")

# Save to CSV
df.to_csv('health_armaan.csv', index=False)
print("Data saved as 'health_armaan.csv'")