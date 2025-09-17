import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
from utils.data_loader import load_health_data, filter_data_by_date_range
from utils.analysis_functions import create_time_series_plot, create_scatter_plot

st.set_page_config(page_title="Lifestyle Analysis", page_icon="", layout="wide")

def main():
    st.title("Lifestyle Analysis Dashboard")
    st.markdown("---")
    
    # Load data
    df = load_health_data()
    if df is None:
        st.error("Could not load health data. Please ensure the data file is available.")
        return
    
    # Date range selector
    st.sidebar.header("Date Range Filter")
    min_date = df['date'].min().date()
    max_date = df['date'].max().date()
    
    start_date = st.sidebar.date_input(
        "Start Date", 
        value=max_date - timedelta(days=180),
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
    
    # Key lifestyle metrics
    st.header("Lifestyle Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_daylight = filtered_df['TimeInDaylight'].mean() if 'TimeInDaylight' in filtered_df.columns else 0
        st.metric(
            "Average Daylight", 
            f"{avg_daylight:.0f} min/day"
        )
        
    with col2:
        avg_stand_time = filtered_df['AppleStandTime'].mean() if 'AppleStandTime' in filtered_df.columns else 0
        st.metric(
            "Average Stand Time", 
            f"{avg_stand_time:.0f} min/day"
        )
        
    with col3:
        avg_audio_exposure = filtered_df['HeadphoneAudioExposure'].mean() if 'HeadphoneAudioExposure' in filtered_df.columns else 0
        st.metric(
            "Avg Audio Exposure", 
            f"{avg_audio_exposure:.0f} dB"
        )
        
    with col4:
        avg_stand_hours = filtered_df['AppleStandHour'].mean() if 'AppleStandHour' in filtered_df.columns else 0
        st.metric(
            "Average Stand Hours", 
            f"{avg_stand_hours:.1f} hrs/day"
        )
    
    st.markdown("---")
    
    # Daylight exposure analysis
    st.header("Daylight Exposure Analysis")
    
    if 'TimeInDaylight' in filtered_df.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            # Daily daylight trend
            fig_daylight = create_time_series_plot(filtered_df, 'TimeInDaylight', "Daily Time in Daylight")
            if fig_daylight:
                fig_daylight.update_traces(line_color='gold')
                st.plotly_chart(fig_daylight, use_container_width=True)
        
        with col2:
            # Monthly daylight averages
            if 'year_month' in filtered_df.columns:
                monthly_daylight = filtered_df.groupby('year_month')['TimeInDaylight'].mean().reset_index()
                monthly_daylight['year_month_str'] = monthly_daylight['year_month'].astype(str)
                
                fig_monthly_daylight = px.bar(
                    monthly_daylight, 
                    x='year_month_str', 
                    y='TimeInDaylight',
                    title="Average Daylight Exposure by Month",
                    labels={'year_month_str': 'Month', 'TimeInDaylight': 'Minutes in Daylight'}
                )
                fig_monthly_daylight.update_layout(height=400)
                st.plotly_chart(fig_monthly_daylight, use_container_width=True)
        
        # Daylight vs activity correlation
        st.subheader("Daylight vs Activity Correlation")
        
        if 'StepCount' in filtered_df.columns:
            fig_daylight_steps = create_scatter_plot(
                filtered_df, 
                'TimeInDaylight', 
                'StepCount',
                "Daylight Exposure vs Daily Steps"
            )
            if fig_daylight_steps:
                st.plotly_chart(fig_daylight_steps, use_container_width=True)
    
    # Stand time analysis
    st.header("Standing and Movement Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'AppleStandTime' in filtered_df.columns:
            fig_stand_time = create_time_series_plot(filtered_df, 'AppleStandTime', "Daily Stand Time")
            if fig_stand_time:
                fig_stand_time.update_traces(line_color='purple')
                st.plotly_chart(fig_stand_time, use_container_width=True)
    
    with col2:
        if 'AppleStandHour' in filtered_df.columns:
            fig_stand_hours = create_time_series_plot(filtered_df, 'AppleStandHour', "Daily Stand Hours")
            if fig_stand_hours:
                fig_stand_hours.update_traces(line_color='darkblue')
                st.plotly_chart(fig_stand_hours, use_container_width=True)
    
    # Audio exposure analysis
    st.header("Audio Exposure Analysis")
    
    if 'HeadphoneAudioExposure' in filtered_df.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            # Daily audio exposure trend
            fig_audio = create_time_series_plot(filtered_df, 'HeadphoneAudioExposure', "Daily Headphone Audio Exposure")
            if fig_audio:
                fig_audio.update_traces(line_color='red')
                st.plotly_chart(fig_audio, use_container_width=True)
        
        with col2:
            # Audio exposure distribution
            audio_data = filtered_df['HeadphoneAudioExposure'].dropna()
            if not audio_data.empty:
                fig_audio_hist = px.histogram(
                    audio_data, 
                    nbins=30,
                    title="Audio Exposure Distribution",
                    labels={'value': 'Audio Exposure (dB)', 'count': 'Frequency'}
                )
                fig_audio_hist.update_layout(height=400)
                st.plotly_chart(fig_audio_hist, use_container_width=True)
    
    # Seasonal analysis
    st.header("Seasonal Pattern Analysis")
    
    if 'month' in filtered_df.columns:
        # Create seasonal groupings
        filtered_df_copy = filtered_df.copy()
        filtered_df_copy['season'] = filtered_df_copy['month'].map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Fall', 10: 'Fall', 11: 'Fall'
        })
        
        seasonal_metrics = ['StepCount', 'TimeInDaylight', 'ActiveEnergyBurned']
        available_seasonal_metrics = [col for col in seasonal_metrics if col in filtered_df_copy.columns]
        
        if available_seasonal_metrics and 'season' in filtered_df_copy.columns:
            seasonal_summary = filtered_df_copy.groupby('season')[available_seasonal_metrics].mean().round(1)
            
            # Create seasonal comparison charts
            for metric in available_seasonal_metrics:
                fig_seasonal = px.bar(
                    x=seasonal_summary.index, 
                    y=seasonal_summary[metric],
                    title=f"Average {metric} by Season",
                    labels={'x': 'Season', 'y': metric}
                )
                fig_seasonal.update_layout(height=300)
                st.plotly_chart(fig_seasonal, use_container_width=True)
    
    # Weekly lifestyle patterns
    st.header("Weekly Lifestyle Patterns")
    
    if 'weekday' in filtered_df.columns:
        weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        lifestyle_metrics = ['TimeInDaylight', 'AppleStandTime', 'HeadphoneAudioExposure']
        available_lifestyle_metrics = [col for col in lifestyle_metrics if col in filtered_df.columns]
        
        if available_lifestyle_metrics:
            for metric in available_lifestyle_metrics:
                weekday_avg = filtered_df.groupby('weekday')[metric].mean().reindex(weekday_order)
                
                fig_weekday = px.bar(
                    x=weekday_avg.index, 
                    y=weekday_avg.values,
                    title=f"Average {metric} by Day of Week",
                    labels={'x': 'Day of Week', 'y': metric}
                )
                fig_weekday.update_layout(height=300)
                st.plotly_chart(fig_weekday, use_container_width=True)
    
    # Lifestyle insights
    st.header("Lifestyle Insights")
    
    insights = []
    
    if 'TimeInDaylight' in filtered_df.columns:
        daylight_data = filtered_df['TimeInDaylight'].dropna()
        if not daylight_data.empty:
            avg_daylight = daylight_data.mean()
            max_daylight = daylight_data.max()
            max_daylight_date = filtered_df.loc[filtered_df['TimeInDaylight'] == max_daylight, 'date'].iloc[0].strftime('%Y-%m-%d')
            
            insights.append(f"**Best daylight day:** {max_daylight:.0f} minutes on {max_daylight_date}")
            
            # Daylight recommendations
            if avg_daylight < 30:
                insights.append("**Recommendation:** Consider increasing outdoor time for better circadian rhythm")
            elif avg_daylight > 120:
                insights.append("**Great job:** Excellent daylight exposure supporting healthy sleep patterns")
    
    if 'AppleStandTime' in filtered_df.columns:
        stand_data = filtered_df['AppleStandTime'].dropna()
        if not stand_data.empty:
            avg_stand = stand_data.mean()
            stand_goal_days = (stand_data >= 720).sum()  # 12 hours * 60 minutes
            total_stand_days = len(stand_data)
            
            insights.append(f"**Stand goal achievement:** {stand_goal_days}/{total_stand_days} days reached 12+ hours")
    
    if 'HeadphoneAudioExposure' in filtered_df.columns:
        audio_data = filtered_df['HeadphoneAudioExposure'].dropna()
        if not audio_data.empty:
            high_exposure_days = (audio_data > 85).sum()  # WHO recommended limit
            total_audio_days = len(audio_data)
            
            if high_exposure_days > 0:
                insights.append(f"**Audio safety:** {high_exposure_days}/{total_audio_days} days exceeded 85dB (consider volume reduction)")
    
    if insights:
        for insight in insights:
            st.markdown(insight)
    else:
        st.info("Enable more data points to see detailed insights.")

if __name__ == "__main__":
    main()
