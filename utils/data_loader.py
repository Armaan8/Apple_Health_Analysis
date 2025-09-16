import pandas as pd
import numpy as np
import streamlit as st
import os

@st.cache_data
def load_health_data():
    """Load and preprocess health data."""
    try:
        # Try different possible locations for the data file
        possible_paths = [
            "health_armaan.csv",
            "data/health_armaan.csv",
            "../health_armaan.csv"
        ]
        
        df = None
        for path in possible_paths:
            if os.path.exists(path):
                df = pd.read_csv(path, parse_dates=["date"])
                break
        
        if df is None:
            st.warning("Health data file not found in expected locations. Please upload health_armaan.csv")
            return None
            
        # Sort by date and reset index
        df = df.sort_values("date").reset_index(drop=True)
        
        # Add derived columns if not present
        if 'year' not in df.columns:
            df['year'] = df['date'].dt.year
        if 'month' not in df.columns:
            df['month'] = df['date'].dt.month
        if 'week' not in df.columns:
            df['week'] = df['date'].dt.isocalendar().week
        if 'weekday' not in df.columns:
            df['weekday'] = df['date'].dt.day_name()
        if 'year_month' not in df.columns:
            df['year_month'] = df['date'].dt.to_period('M')
            
        return df
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def get_data_summary(df):
    """Get summary statistics of the dataset."""
    if df is None:
        return {}
    
    summary = {
        'total_days': len(df),
        'date_range': {
            'start': df['date'].min().strftime('%Y-%m-%d'),
            'end': df['date'].max().strftime('%Y-%m-%d')
        },
        'complete_records': df.dropna(subset=['StepCount', 'RestingHeartRate']).shape[0],
        'missing_data_pct': (df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100)
    }
    
    return summary

def get_numeric_columns(df):
    """Get list of numeric columns for analysis."""
    if df is None:
        return []
    return df.select_dtypes(include=[np.number]).columns.tolist()

def filter_data_by_date_range(df, start_date, end_date):
    """Filter dataframe by date range."""
    if df is None:
        return None
    return df[(df['date'] >= start_date) & (df['date'] <= end_date)]
