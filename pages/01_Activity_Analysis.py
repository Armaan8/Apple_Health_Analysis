import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
from utils.data_loader import load_health_data, filter_data_by_date_range
from utils.analysis_functions import create_time_series_plot

st.set_page_config(page_title="Activity Analysis", page_icon="📊", layout="wide")

def main():
    st.title("📊 Activity Analysis Dashboard")
    st.markdown("---")
    
    # Load data
    df = load_health_data()
    if df is None:
        st.error("Could not load health data. Please ensure the data file is available.")
        return
    
    # Date range selector
    st.sidebar.header("📅 Date Range Filter")
    min_date = df['date'].min().date()
    max_date = df['date'].max().date()
    
    start_date = st.sidebar.date_input(
        "Start Date", 
        value=max_date - timedelta(days=90),
        min_value=min_date,
        max_value=max_date
    )
    
    end_date = st.sidebar.date_input(
        "End Date", 
        value=max_date,
        min_value=min_date,
        max_value=max_date
    )
    
    # Filter data
    filtered_df = filter_data_by_date_range(df, pd.Timestamp(start_date), pd.Timestamp(end_date))
    
    if filtered_df.empty:
        st.warning("No data available for the selected date range.")
        return
    
    # Key metrics
    st.header("🎯 Key Activity Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_steps = filtered_df['StepCount'].mean() if 'StepCount' in filtered_df.columns else 0
        total_steps = filtered_df['StepCount'].sum() if 'StepCount' in filtered_df.columns else 0
        st.metric(
            "Average Daily Steps", 
            f"{avg_steps:,.0f}",
            help=f"Total steps in period: {total_steps:,.0f}"
        )
        
    with col2:
        avg_distance = filtered_df['DistanceWalkingRunning'].mean() if 'DistanceWalkingRunning' in filtered_df.columns else 0
        st.metric(
            "Average Distance", 
            f"{avg_distance:.1f} km"
        )
        
    with col3:
        avg_active_energy = filtered_df['ActiveEnergyBurned'].mean() if 'ActiveEnergyBurned' in filtered_df.columns else 0
        st.metric(
            "Average Active Energy", 
            f"{avg_active_energy:.0f} cal"
        )
        
    with col4:
        avg_exercise_time = filtered_df['AppleExerciseTime'].mean() if 'AppleExerciseTime' in filtered_df.columns else 0
        st.metric(
            "Average Exercise Time", 
            f"{avg_exercise_time:.0f} min"
        )
    
    st.markdown("---")
    
    # Time series plots
    st.header("📈 Activity Trends")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Daily steps
        if 'StepCount' in filtered_df.columns:
            fig_steps = create_time_series_plot(filtered_df, 'StepCount', "Daily Step Count")
            if fig_steps:
                st.plotly_chart(fig_steps, use_container_width=True)
        
        # Active energy
        if 'ActiveEnergyBurned' in filtered_df.columns:
            fig_energy = create_time_series_plot(filtered_df, 'ActiveEnergyBurned', "Daily Active Energy Burned")
            if fig_energy:
                st.plotly_chart(fig_energy, use_container_width=True)
    
    with col2:
        # Walking/running distance
        if 'DistanceWalkingRunning' in filtered_df.columns:
            fig_distance = create_time_series_plot(filtered_df, 'DistanceWalkingRunning', "Daily Walking/Running Distance")
            if fig_distance:
                st.plotly_chart(fig_distance, use_container_width=True)
        
        # Exercise time
        if 'AppleExerciseTime' in filtered_df.columns:
            fig_exercise = create_time_series_plot(filtered_df, 'AppleExerciseTime', "Daily Exercise Time")
            if fig_exercise:
                st.plotly_chart(fig_exercise, use_container_width=True)
    
    # Weekly patterns
    st.header("📅 Weekly Activity Patterns")
    
    if 'weekday' in filtered_df.columns and 'StepCount' in filtered_df.columns:
        # Calculate average by weekday
        weekday_avg = filtered_df.groupby('weekday')['StepCount'].mean().reindex([
            'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
        ])
        
        fig_weekday = px.bar(
            x=weekday_avg.index, 
            y=weekday_avg.values,
            title="Average Steps by Day of Week",
            labels={'x': 'Day of Week', 'y': 'Average Steps'}
        )
        fig_weekday.update_layout(height=400)
        st.plotly_chart(fig_weekday, use_container_width=True)
    
    # Monthly summary
    st.header("📊 Monthly Activity Summary")
    
    if 'year_month' in filtered_df.columns:
        monthly_summary = filtered_df.groupby('year_month').agg({
            'StepCount': ['mean', 'sum'] if 'StepCount' in filtered_df.columns else lambda x: np.nan,
            'ActiveEnergyBurned': 'mean' if 'ActiveEnergyBurned' in filtered_df.columns else lambda x: np.nan,
            'AppleExerciseTime': 'mean' if 'AppleExerciseTime' in filtered_df.columns else lambda x: np.nan,
            'DistanceWalkingRunning': 'mean' if 'DistanceWalkingRunning' in filtered_df.columns else lambda x: np.nan
        }).round(2)
        
        monthly_summary.columns = ['Avg Daily Steps', 'Total Steps', 'Avg Active Energy', 'Avg Exercise Time', 'Avg Distance']
        st.dataframe(monthly_summary, use_container_width=True)
    
    # Activity distribution
    st.header("📊 Activity Distribution Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'StepCount' in filtered_df.columns:
            fig_hist_steps = px.histogram(
                filtered_df, 
                x='StepCount', 
                nbins=30,
                title="Step Count Distribution",
                labels={'StepCount': 'Daily Steps', 'count': 'Frequency'}
            )
            st.plotly_chart(fig_hist_steps, use_container_width=True)
    
    with col2:
        if 'ActiveEnergyBurned' in filtered_df.columns:
            fig_hist_energy = px.histogram(
                filtered_df, 
                x='ActiveEnergyBurned', 
                nbins=30,
                title="Active Energy Distribution",
                labels={'ActiveEnergyBurned': 'Daily Active Energy (cal)', 'count': 'Frequency'}
            )
            st.plotly_chart(fig_hist_energy, use_container_width=True)
    
    # Data insights
    st.header("💡 Activity Insights")
    
    insights = []
    
    if 'StepCount' in filtered_df.columns:
        step_data = filtered_df['StepCount'].dropna()
        if not step_data.empty:
            max_steps = step_data.max()
            max_steps_date = filtered_df.loc[filtered_df['StepCount'] == max_steps, 'date'].iloc[0].strftime('%Y-%m-%d')
            insights.append(f"🏆 **Highest step day:** {max_steps:,.0f} steps on {max_steps_date}")
            
            days_above_10k = (step_data >= 10000).sum()
            total_days = len(step_data)
            pct_above_10k = (days_above_10k / total_days) * 100
            insights.append(f"🎯 **10,000+ step days:** {days_above_10k}/{total_days} days ({pct_above_10k:.1f}%)")
    
    if 'ActiveEnergyBurned' in filtered_df.columns:
        energy_data = filtered_df['ActiveEnergyBurned'].dropna()
        if not energy_data.empty:
            avg_weekly_energy = energy_data.mean() * 7
            insights.append(f"⚡ **Average weekly active energy:** {avg_weekly_energy:,.0f} calories")
    
    if insights:
        for insight in insights:
            st.markdown(insight)
    else:
        st.info("Enable more data points to see detailed insights.")

if __name__ == "__main__":
    main()
