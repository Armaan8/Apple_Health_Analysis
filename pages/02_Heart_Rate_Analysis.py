import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
from utils.data_loader import load_health_data, filter_data_by_date_range
from utils.analysis_functions import create_time_series_plot, create_scatter_plot

st.set_page_config(page_title="Heart Rate Analysis", page_icon="❤️", layout="wide")

def main():
    st.title("❤️ Heart Rate Analysis Dashboard")
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
    
    # Key heart rate metrics
    st.header("❤️ Heart Rate Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_resting_hr = filtered_df['RestingHeartRate'].mean() if 'RestingHeartRate' in filtered_df.columns else 0
        st.metric(
            "Average Resting HR", 
            f"{avg_resting_hr:.0f} bpm"
        )
        
    with col2:
        avg_hr = filtered_df['HeartRate'].mean() if 'HeartRate' in filtered_df.columns else 0
        st.metric(
            "Average Heart Rate", 
            f"{avg_hr:.0f} bpm"
        )
        
    with col3:
        avg_walking_hr = filtered_df['WalkingHeartRateAverage'].mean() if 'WalkingHeartRateAverage' in filtered_df.columns else 0
        st.metric(
            "Average Walking HR", 
            f"{avg_walking_hr:.0f} bpm"
        )
        
    with col4:
        avg_hrv = filtered_df['HeartRateVariabilitySDNN'].mean() if 'HeartRateVariabilitySDNN' in filtered_df.columns else 0
        st.metric(
            "Average HRV (SDNN)", 
            f"{avg_hrv:.1f} ms"
        )
    
    st.markdown("---")
    
    # Heart rate trends
    st.header("📈 Heart Rate Trends")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Resting heart rate trend
        if 'RestingHeartRate' in filtered_df.columns:
            fig_resting = create_time_series_plot(filtered_df, 'RestingHeartRate', "Resting Heart Rate Over Time")
            if fig_resting:
                fig_resting.update_traces(line_color='red')
                st.plotly_chart(fig_resting, use_container_width=True)
        
        # Heart rate variability
        if 'HeartRateVariabilitySDNN' in filtered_df.columns:
            fig_hrv = create_time_series_plot(filtered_df, 'HeartRateVariabilitySDNN', "Heart Rate Variability (SDNN)")
            if fig_hrv:
                fig_hrv.update_traces(line_color='green')
                st.plotly_chart(fig_hrv, use_container_width=True)
    
    with col2:
        # General heart rate trend
        if 'HeartRate' in filtered_df.columns:
            fig_hr = create_time_series_plot(filtered_df, 'HeartRate', "Heart Rate Over Time")
            if fig_hr:
                fig_hr.update_traces(line_color='orange')
                st.plotly_chart(fig_hr, use_container_width=True)
        
        # Walking heart rate
        if 'WalkingHeartRateAverage' in filtered_df.columns:
            fig_walking_hr = create_time_series_plot(filtered_df, 'WalkingHeartRateAverage', "Walking Heart Rate Average")
            if fig_walking_hr:
                fig_walking_hr.update_traces(line_color='purple')
                st.plotly_chart(fig_walking_hr, use_container_width=True)
    
    # Heart rate vs activity correlation
    st.header("🔗 Heart Rate vs Activity Correlations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Resting HR vs Steps
        if 'RestingHeartRate' in filtered_df.columns and 'StepCount' in filtered_df.columns:
            fig_hr_steps = create_scatter_plot(
                filtered_df, 
                'StepCount', 
                'RestingHeartRate',
                "Daily Steps vs Resting Heart Rate"
            )
            if fig_hr_steps:
                st.plotly_chart(fig_hr_steps, use_container_width=True)
    
    with col2:
        # HRV vs Active Energy
        if 'HeartRateVariabilitySDNN' in filtered_df.columns and 'ActiveEnergyBurned' in filtered_df.columns:
            fig_hrv_energy = create_scatter_plot(
                filtered_df, 
                'ActiveEnergyBurned', 
                'HeartRateVariabilitySDNN',
                "Active Energy vs Heart Rate Variability"
            )
            if fig_hrv_energy:
                st.plotly_chart(fig_hrv_energy, use_container_width=True)
    
    # HRV and Fitness Analysis
    st.header("🏃 HRV and Fitness Analysis")
    
    if 'HRFitnessIndex' in filtered_df.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            # HRV vs Fitness Index
            if 'HeartRateVariabilitySDNN' in filtered_df.columns:
                fig_hrv_fitness = create_scatter_plot(
                    filtered_df, 
                    'HeartRateVariabilitySDNN', 
                    'HRFitnessIndex',
                    "HRV vs Fitness Index"
                )
                if fig_hrv_fitness:
                    st.plotly_chart(fig_hrv_fitness, use_container_width=True)
        
        with col2:
            # Fitness index trend
            fig_fitness = create_time_series_plot(filtered_df, 'HRFitnessIndex', "Fitness Index Over Time")
            if fig_fitness:
                fig_fitness.update_traces(line_color='darkgreen')
                st.plotly_chart(fig_fitness, use_container_width=True)
    
    # Heart rate distribution analysis
    st.header("📊 Heart Rate Distribution Analysis")
    
    hr_cols = ['RestingHeartRate', 'HeartRate', 'WalkingHeartRateAverage']
    available_hr_cols = [col for col in hr_cols if col in filtered_df.columns]
    
    if available_hr_cols:
        # Create violin plots for heart rate distributions
        fig_violin = go.Figure()
        
        for i, col in enumerate(available_hr_cols):
            data = filtered_df[col].dropna()
            if not data.empty:
                fig_violin.add_trace(go.Violin(
                    y=data,
                    name=col.replace('HeartRate', 'HR').replace('Average', 'Avg'),
                    box_visible=True,
                    meanline_visible=True
                ))
        
        fig_violin.update_layout(
            title="Heart Rate Distribution Comparison",
            yaxis_title="Heart Rate (bpm)",
            height=400
        )
        
        st.plotly_chart(fig_violin, use_container_width=True)
    
    # Monthly heart rate summary
    st.header("📅 Monthly Heart Rate Summary")
    
    if 'year_month' in filtered_df.columns:
        hr_summary_cols = ['RestingHeartRate', 'HeartRate', 'HeartRateVariabilitySDNN', 'WalkingHeartRateAverage']
        available_summary_cols = [col for col in hr_summary_cols if col in filtered_df.columns]
        
        if available_summary_cols:
            monthly_hr_summary = filtered_df.groupby('year_month')[available_summary_cols].mean().round(1)
            monthly_hr_summary.columns = [col.replace('HeartRate', 'HR').replace('Variability', 'Var') for col in monthly_hr_summary.columns]
            st.dataframe(monthly_hr_summary, use_container_width=True)
    
    # Heart rate insights
    st.header("💡 Heart Rate Insights")
    
    insights = []
    
    if 'RestingHeartRate' in filtered_df.columns:
        rhr_data = filtered_df['RestingHeartRate'].dropna()
        if not rhr_data.empty:
            min_rhr = rhr_data.min()
            max_rhr = rhr_data.max()
            rhr_range = max_rhr - min_rhr
            insights.append(f"🫀 **Resting HR range:** {min_rhr:.0f} - {max_rhr:.0f} bpm (range: {rhr_range:.0f} bpm)")
            
            # Trend analysis
            if len(rhr_data) > 7:
                recent_avg = rhr_data.tail(7).mean()
                earlier_avg = rhr_data.head(7).mean()
                trend = recent_avg - earlier_avg
                trend_direction = "📈" if trend > 0 else "📉" if trend < -1 else "➡️"
                insights.append(f"{trend_direction} **Recent trend:** {trend:+.1f} bpm vs period start")
    
    if 'HeartRateVariabilitySDNN' in filtered_df.columns:
        hrv_data = filtered_df['HeartRateVariabilitySDNN'].dropna()
        if not hrv_data.empty:
            avg_hrv = hrv_data.mean()
            if avg_hrv > 40:
                hrv_status = "Excellent"
            elif avg_hrv > 30:
                hrv_status = "Good"
            elif avg_hrv > 20:
                hrv_status = "Fair"
            else:
                hrv_status = "Needs attention"
            insights.append(f"💚 **HRV status:** {hrv_status} (avg: {avg_hrv:.1f} ms)")
    
    if insights:
        for insight in insights:
            st.markdown(insight)
    else:
        st.info("Enable more data points to see detailed insights.")

if __name__ == "__main__":
    main()
