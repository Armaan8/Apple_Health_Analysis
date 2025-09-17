import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
from utils.data_loader import load_health_data, filter_data_by_date_range
from utils.analysis_functions import remove_outliers_iqr, create_time_series_plot, create_scatter_plot

st.set_page_config(page_title="Walking Analysis", layout="wide")

def main():
    st.title("Walking Stability & Mobility Analysis")
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
    
    # Outlier removal option
    remove_outliers = st.sidebar.checkbox("Remove Outliers (IQR method)", value=True)
    
    # Filter data
    filtered_df = filter_data_by_date_range(df, pd.Timestamp(start_date), pd.Timestamp(end_date))
    
    if filtered_df.empty:
        st.warning("No data available for the selected date range.")
        return
    
    # Apply outlier removal if selected
    if remove_outliers:
        mobility_cols = [
            "WalkingSpeed", "WalkingStepLength", "WalkingAsymmetryPercentage",
            "WalkingDoubleSupportPercentage", "SixMinuteWalkTestDistance",
            "AppleWalkingSteadiness", "MobilityIndex", "StabilityIndex"
        ]
        
        for col in mobility_cols:
            if col in filtered_df.columns:
                filtered_df[col] = remove_outliers_iqr(filtered_df[col])
    
    # Key walking metrics
    st.header("Key Walking Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_speed = filtered_df['WalkingSpeed'].mean() if 'WalkingSpeed' in filtered_df.columns else 0
        st.metric(
            "Average Walking Speed", 
            f"{avg_speed:.2f} m/s"
        )
        
    with col2:
        avg_step_length = filtered_df['WalkingStepLength'].mean() if 'WalkingStepLength' in filtered_df.columns else 0
        st.metric(
            "Average Step Length", 
            f"{avg_step_length:.0f} cm"
        )
        
    with col3:
        avg_asymmetry = filtered_df['WalkingAsymmetryPercentage'].mean() if 'WalkingAsymmetryPercentage' in filtered_df.columns else 0
        st.metric(
            "Average Asymmetry", 
            f"{avg_asymmetry:.2f}%"
        )
        
    with col4:
        avg_double_support = filtered_df['WalkingDoubleSupportPercentage'].mean() if 'WalkingDoubleSupportPercentage' in filtered_df.columns else 0
        st.metric(
            "Avg Double Support", 
            f"{avg_double_support:.1f}%"
        )
    
    st.markdown("---")
    
    # Walking speed and step length analysis
    st.header("Speed and Step Length Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'WalkingSpeed' in filtered_df.columns:
            fig_speed = create_time_series_plot(filtered_df, 'WalkingSpeed', "Daily Walking Speed")
            if fig_speed:
                fig_speed.update_traces(line_color='blue')
                st.plotly_chart(fig_speed, use_container_width=True)
    
    with col2:
        if 'WalkingStepLength' in filtered_df.columns:
            fig_step_length = create_time_series_plot(filtered_df, 'WalkingStepLength', "Daily Walking Step Length")
            if fig_step_length:
                fig_step_length.update_traces(line_color='green')
                st.plotly_chart(fig_step_length, use_container_width=True)
    
    # Speed vs step length correlation
    if 'WalkingSpeed' in filtered_df.columns and 'WalkingStepLength' in filtered_df.columns:
        st.subheader("🔗 Walking Speed vs Step Length Correlation")
        
        fig_speed_length = create_scatter_plot(
            filtered_df, 
            'WalkingSpeed', 
            'WalkingStepLength',
            "Walking Speed vs Step Length"
        )
        if fig_speed_length:
            st.plotly_chart(fig_speed_length, use_container_width=True)
    
    # Gait stability analysis
    st.header("Gait Stability Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'WalkingAsymmetryPercentage' in filtered_df.columns:
            fig_asymmetry = create_time_series_plot(filtered_df, 'WalkingAsymmetryPercentage', "Walking Asymmetry Percentage")
            if fig_asymmetry:
                fig_asymmetry.update_traces(line_color='red')
                st.plotly_chart(fig_asymmetry, use_container_width=True)
    
    with col2:
        if 'WalkingDoubleSupportPercentage' in filtered_df.columns:
            fig_double_support = create_time_series_plot(filtered_df, 'WalkingDoubleSupportPercentage', "Walking Double Support Percentage")
            if fig_double_support:
                fig_double_support.update_traces(line_color='purple')
                st.plotly_chart(fig_double_support, use_container_width=True)
    
    # Mobility and stability indices
    st.header("Mobility and Stability Indices")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'MobilityIndex' in filtered_df.columns:
            fig_mobility = create_time_series_plot(filtered_df, 'MobilityIndex', "Mobility Index Over Time")
            if fig_mobility:
                fig_mobility.update_traces(line_color='orange')
                st.plotly_chart(fig_mobility, use_container_width=True)
    
    with col2:
        if 'StabilityIndex' in filtered_df.columns:
            fig_stability = create_time_series_plot(filtered_df, 'StabilityIndex', "Stability Index Over Time")
            if fig_stability:
                fig_stability.update_traces(line_color='darkgreen')
                st.plotly_chart(fig_stability, use_container_width=True)
    
    # Mobility vs stability correlation
    if 'MobilityIndex' in filtered_df.columns and 'StabilityIndex' in filtered_df.columns:
        st.subheader("Mobility vs Stability Index Correlation")
        
        fig_mobility_stability = create_scatter_plot(
            filtered_df, 
            'MobilityIndex', 
            'StabilityIndex',
            "Mobility Index vs Stability Index"
        )
        if fig_mobility_stability:
            st.plotly_chart(fig_mobility_stability, use_container_width=True)
    
    # Advanced walking metrics
    st.header("Advanced Walking Assessments")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'SixMinuteWalkTestDistance' in filtered_df.columns:
            fig_6mwt = create_time_series_plot(filtered_df, 'SixMinuteWalkTestDistance', "Six Minute Walk Test Distance")
            if fig_6mwt:
                fig_6mwt.update_traces(line_color='darkblue')
                st.plotly_chart(fig_6mwt, use_container_width=True)
    
    with col2:
        if 'AppleWalkingSteadiness' in filtered_df.columns:
            fig_steadiness = create_time_series_plot(filtered_df, 'AppleWalkingSteadiness', "Apple Walking Steadiness")
            if fig_steadiness:
                fig_steadiness.update_traces(line_color='darkred')
                st.plotly_chart(fig_steadiness, use_container_width=True)
    
    # Walking pattern distribution
    st.header("Walking Pattern Distribution")
    
    walking_cols = ['WalkingSpeed', 'WalkingStepLength', 'WalkingAsymmetryPercentage', 'WalkingDoubleSupportPercentage']
    available_walking_cols = [col for col in walking_cols if col in filtered_df.columns]
    
    if available_walking_cols:
        selected_metric = st.selectbox(
            "Select metric for distribution analysis:",
            available_walking_cols,
            index=0
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Histogram
            fig_hist = px.histogram(
                filtered_df,
                x=selected_metric,
                nbins=30,
                title=f"{selected_metric} Distribution",
                labels={selected_metric: selected_metric, 'count': 'Frequency'}
            )
            fig_hist.update_layout(height=400)
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            # Box plot by year
            if 'year' in filtered_df.columns:
                fig_box = px.box(
                    filtered_df,
                    x='year',
                    y=selected_metric,
                    title=f"{selected_metric} Distribution by Year"
                )
                fig_box.update_layout(height=400)
                st.plotly_chart(fig_box, use_container_width=True)
    
    # Walking performance summary
    st.header("Walking Performance Summary")
    
    if 'year_month' in filtered_df.columns:
        walking_summary_cols = [
            'WalkingSpeed', 'WalkingStepLength', 'WalkingAsymmetryPercentage',
            'WalkingDoubleSupportPercentage', 'MobilityIndex', 'StabilityIndex'
        ]
        available_summary_cols = [col for col in walking_summary_cols if col in filtered_df.columns]
        
        if available_summary_cols:
            monthly_walking_summary = filtered_df.groupby('year_month')[available_summary_cols].mean().round(2)
            st.dataframe(monthly_walking_summary, use_container_width=True)
    
    # Walking insights and recommendations
    st.header("Walking Analysis Insights")
    
    insights = []
    
    # Speed analysis
    if 'WalkingSpeed' in filtered_df.columns:
        speed_data = filtered_df['WalkingSpeed'].dropna()
        if not speed_data.empty:
            avg_speed = speed_data.mean()
            speed_std = speed_data.std()
            
            if avg_speed > 1.2:
                insights.append("**Excellent walking speed:** Above average for healthy adults")
            elif avg_speed > 1.0:
                insights.append("**Good walking speed:** Within healthy adult range")
            else:
                insights.append("**Walking speed below average:** Consider mobility assessment")
            
            if speed_std > 0.3:
                insights.append("**High speed variability:** Walking patterns show significant variation")
    
    # Asymmetry analysis
    if 'WalkingAsymmetryPercentage' in filtered_df.columns:
        asymmetry_data = filtered_df['WalkingAsymmetryPercentage'].dropna()
        if not asymmetry_data.empty:
            avg_asymmetry = asymmetry_data.mean()
            
            if avg_asymmetry < 0.03:  # 3%
                insights.append("**Excellent gait symmetry:** Very low asymmetry levels")
            elif avg_asymmetry < 0.05:  # 5%
                insights.append("**Good gait symmetry:** Asymmetry within normal limits")
            else:
                insights.append("**Elevated asymmetry:** May indicate gait irregularities")
    
    # Double support analysis
    if 'WalkingDoubleSupportPercentage' in filtered_df.columns:
        ds_data = filtered_df['WalkingDoubleSupportPercentage'].dropna()
        if not ds_data.empty:
            avg_ds = ds_data.mean()
            
            if avg_ds < 0.20:  # 20%
                insights.append("🏃 **Efficient gait pattern:** Low double support time")
            elif avg_ds > 0.30:  # 30%
                insights.append("🚶 **Cautious walking pattern:** Higher double support may indicate stability concerns")
    
    # Mobility vs stability insight
    if 'MobilityIndex' in filtered_df.columns and 'StabilityIndex' in filtered_df.columns:
        mobility_data = filtered_df['MobilityIndex'].dropna()
        stability_data = filtered_df['StabilityIndex'].dropna()
        
        if not mobility_data.empty and not stability_data.empty:
            mobility_avg = mobility_data.mean()
            stability_avg = stability_data.mean()
            
            if mobility_avg > 35 and stability_avg < 0.2:
                insights.append("**Optimal walking pattern:** High mobility with good stability")
            elif mobility_avg < 25:
                insights.append("**Mobility improvement opportunity:** Consider increasing walking activity")
    
    # Data quality insight
    walking_metrics = ['WalkingSpeed', 'WalkingStepLength', 'WalkingAsymmetryPercentage', 'WalkingDoubleSupportPercentage']
    available_metrics = [col for col in walking_metrics if col in filtered_df.columns]
    data_completeness = len(available_metrics) / len(walking_metrics) * 100
    
    insights.append(f"**Data completeness:** {data_completeness:.0f}% of walking metrics available")
    
    if remove_outliers:
        insights.append("**Data processing:** Outliers removed using IQR method for cleaner analysis")
    
    for insight in insights:
        st.markdown(insight)
    
    # Recommendations based on data
    st.header("Walking Performance Recommendations")
    
    recommendations = []
    
    if 'WalkingSpeed' in filtered_df.columns:
        speed_data = filtered_df['WalkingSpeed'].dropna()
        if not speed_data.empty and speed_data.mean() < 1.0:
            recommendations.append("🏃 **Speed improvement:** Incorporate brisk walking intervals to increase average speed")
    
    if 'WalkingAsymmetryPercentage' in filtered_df.columns:
        asymmetry_data = filtered_df['WalkingAsymmetryPercentage'].dropna()
        if not asymmetry_data.empty and asymmetry_data.mean() > 0.05:
            recommendations.append("**Gait symmetry:** Consider balance exercises to improve walking symmetry")
    
    if 'StabilityIndex' in filtered_df.columns:
        stability_data = filtered_df['StabilityIndex'].dropna()
        if not stability_data.empty and stability_data.mean() > 0.25:
            recommendations.append("**Stability training:** Include stability exercises to improve walking confidence")
    
    recommendations.append("**Continue monitoring:** Regular walking assessments help track improvements")
    recommendations.append("**Professional consultation:** Discuss significant changes with healthcare provider")
    
    for recommendation in recommendations:
        st.markdown(recommendation)

if __name__ == "__main__":
    main()
