import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
from utils.data_loader import load_health_data, get_data_summary
from utils.analysis_functions import calculate_kpis, remove_outliers_iqr

# Page configuration
st.set_page_config(
    page_title="Health Data Analytics Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main dashboard page
def main():
    st.title("Personal Health Data Analytics Dashboard")
    st.markdown("---")
    
    # Load data
    try:
        df = load_health_data()
        if df is not None:
            st.success(f"Health data loaded successfully!")
            
            # Data summary in sidebar
            st.sidebar.header("Data Overview")
            data_summary = get_data_summary(df)
            
            col1, col2 = st.sidebar.columns(2)
            with col1:
                st.metric("Total Days", data_summary['total_days'])
                st.metric("Date Range", f"{data_summary['date_range']['start']} to")
            with col2:
                st.metric("Complete Records", data_summary['complete_records'])
                st.metric("", data_summary['date_range']['end'])
            
            # Main dashboard content
            col1, col2, col3, col4 = st.columns(4)
            
            # Calculate KPIs
            kpis = calculate_kpis(df)
            
            with col1:
                st.metric(
                    "Average Daily Steps", 
                    f"{kpis['avg_steps']:,.0f}",
                    delta=f"{kpis['steps_trend']:+.1f}%" if kpis['steps_trend'] is not None else None
                )
                
            with col2:
                st.metric(
                    "Average Resting HR", 
                    f"{kpis['avg_resting_hr']:.0f} bpm",
                    delta=f"{kpis['hr_trend']:+.1f} bpm" if kpis['hr_trend'] is not None else None
                )
                
            with col3:
                st.metric(
                    "Average Active Energy", 
                    f"{kpis['avg_active_energy']:.0f} cal",
                    delta=f"{kpis['energy_trend']:+.1f}%" if kpis['energy_trend'] is not None else None
                )
                
            with col4:
                st.metric(
                    "Average Exercise Time", 
                    f"{kpis['avg_exercise_time']:.0f} min",
                    delta=f"{kpis['exercise_trend']:+.1f}%" if kpis['exercise_trend'] is not None else None
                )
            
            st.markdown("---")
            
            # Recent trends section
            st.header("Activity Trends in Dec 2024")
            
            # Get last 30 days of data
            recent_df = df.tail(30).copy()
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Steps trend
                fig_steps = px.line(
                    recent_df, 
                    x='date', 
                    y='StepCount',
                    title="Daily Steps",
                    markers=True
                )
                fig_steps.update_layout(height=300)
                st.plotly_chart(fig_steps, use_container_width=True)
                
            with col2:
                # Resting HR trend
                fig_hr = px.line(
                    recent_df, 
                    x='date', 
                    y='RestingHeartRate',
                    title="Resting Heart Rate",
                    markers=True,
                    color_discrete_sequence=['red']
                )
                fig_hr.update_layout(height=300)
                st.plotly_chart(fig_hr, use_container_width=True)
            
            # Navigation guide
            st.markdown("---")
            st.header("Dashboard Navigation")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("""
                **Activity Analysis**
                - Step count patterns
                - Energy expenditure
                - Exercise time trends
                """)
                
                st.markdown("""
                **Heart Rate Analysis**
                - Resting HR trends
                - HRV analysis
                - Fitness indicators
                """)
                
            with col2:
                st.markdown("""
                **Lifestyle Analysis**
                - Daylight exposure
                - Seasonal patterns
                - Audio exposure
                """)
                
                st.markdown("""
                **Predictive Models**
                - ML-based predictions
                - Model performance
                - Future trends
                """)
                
            with col3:
                st.markdown("""
                **Relationship Analysis**
                - Metric correlations
                - PCA analysis
                - Clustering insights
                """)
                
                st.markdown("""
                **Walking Analysis**
                - Gait stability
                - Mobility metrics
                - Asymmetry detection
                """)
            
            st.info("Use the sidebar to navigate between different analysis sections!")
            
        else:
            st.error("Could not load health data. Please ensure 'health_armaan.csv' is available.")
            
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.info("Please ensure the 'health_armaan.csv' file is in the correct location.")

if __name__ == "__main__":
    main()
