import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
from utils.data_loader import load_health_data, filter_data_by_date_range
from utils.analysis_functions import train_predictive_model

st.set_page_config(page_title="Predictive Models", page_icon="🔮", layout="wide")

def main():
    st.title("🔮 Predictive Health Models Dashboard")
    st.markdown("---")
    
    # Load data
    df = load_health_data()
    if df is None:
        st.error("Could not load health data. Please ensure the data file is available.")
        return
    
    st.sidebar.header("🎯 Model Configuration")
    
    # Target selection
    target_options = [
        'RestingHeartRate',
        'HeartRateVariabilitySDNN', 
        'StepCount',
        'ActiveEnergyBurned',
        'AppleExerciseTime'
    ]
    
    available_targets = [col for col in target_options if col in df.columns]
    
    if not available_targets:
        st.error("No suitable target variables found in the data.")
        return
    
    selected_target = st.sidebar.selectbox(
        "Select Target Variable",
        available_targets,
        index=0
    )
    
    # Feature selection
    feature_options = [
        # Activity & Energy
        'StepCount', 'DistanceWalkingRunning', 'DistanceCycling',
        'AppleExerciseTime', 'AppleStandTime', 'FlightsClimbed',
        'ActiveEnergyBurned', 'BasalEnergyBurned',
        
        # Mobility & Stability
        'WalkingSpeed', 'WalkingStepLength',
        'WalkingAsymmetryPercentage', 'WalkingDoubleSupportPercentage',
        'MobilityIndex', 'StabilityIndex',
        
        # Cardio context
        'HeartRate', 'RestingHeartRate', 'HeartRateVariabilitySDNN', 
        'WalkingHeartRateAverage', 'HRFitnessIndex',
        
        # Lifestyle
        'TimeInDaylight', 'HeadphoneAudioExposure',
        
        # Calendar
        'year', 'month', 'week'
    ]
    
    # Remove the selected target from features
    available_features = [col for col in feature_options if col in df.columns and col != selected_target]
    
    selected_features = st.sidebar.multiselect(
        "Select Features",
        available_features,
        default=available_features[:10]  # Select first 10 by default
    )
    
    if not selected_features:
        st.warning("Please select at least one feature for modeling.")
        return
    
    # Date range for training
    st.sidebar.subheader("📅 Training Data Range")
    min_date = df['date'].min().date()
    max_date = df['date'].max().date()
    
    # Use data from December 2022 onwards (as per notebook)
    default_start = max(min_date, datetime(2022, 12, 1).date())
    
    train_start_date = st.sidebar.date_input(
        "Training Start Date", 
        value=default_start,
        min_value=min_date,
        max_value=max_date
    )
    
    train_end_date = st.sidebar.date_input(
        "Training End Date", 
        value=max_date,
        min_value=min_date,
        max_value=max_date
    )
    
    # Filter training data
    train_df = filter_data_by_date_range(df, pd.Timestamp(train_start_date), pd.Timestamp(train_end_date))
    
    if train_df.empty:
        st.warning("No training data available for the selected date range.")
        return
    
    # Train model button
    if st.sidebar.button("🚀 Train Model", type="primary"):
        with st.spinner("Training predictive model..."):
            model, results, used_features = train_predictive_model(
                train_df, 
                selected_target, 
                selected_features
            )
            
            if model is None:
                st.error("Could not train model. Please check your data and feature selection.")
                return
            
            # Store results in session state
            st.session_state['model_results'] = results
            st.session_state['model_features'] = used_features
            st.session_state['model_target'] = selected_target
    
    # Display results if available
    if 'model_results' in st.session_state:
        results = st.session_state['model_results']
        used_features = st.session_state['model_features']
        target = st.session_state['model_target']
        
        st.success("✅ Model trained successfully!")
        
        # Model performance metrics
        st.header("📊 Model Performance")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Mean Absolute Error", f"{results['mae']:.2f}")
        
        with col2:
            st.metric("R² Score", f"{results['r2']:.3f}")
        
        with col3:
            rmse = np.sqrt(np.mean((results['actual'] - results['predictions'])**2))
            st.metric("RMSE", f"{rmse:.2f}")
        
        with col4:
            mape = np.mean(np.abs((results['actual'] - results['predictions']) / results['actual'])) * 100
            st.metric("MAPE", f"{mape:.1f}%")
        
        st.markdown("---")
        
        # Prediction vs actual plot
        st.header("🎯 Predictions vs Actual")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Time series plot
            fig_ts = go.Figure()
            
            fig_ts.add_trace(go.Scatter(
                x=pd.to_datetime(results['dates']),
                y=results['actual'],
                mode='lines+markers',
                name='Actual',
                line=dict(color='black', width=2)
            ))
            
            fig_ts.add_trace(go.Scatter(
                x=pd.to_datetime(results['dates']),
                y=results['predictions'],
                mode='lines+markers',
                name='Predicted',
                line=dict(color='red', width=2)
            ))
            
            fig_ts.update_layout(
                title=f"{target} - Actual vs Predicted",
                xaxis_title="Date",
                yaxis_title=target,
                height=400
            )
            
            st.plotly_chart(fig_ts, use_container_width=True)
        
        with col2:
            # Scatter plot
            fig_scatter = px.scatter(
                x=results['actual'],
                y=results['predictions'],
                title="Predicted vs Actual Values",
                labels={'x': f'Actual {target}', 'y': f'Predicted {target}'}
            )
            
            # Add diagonal line for perfect predictions
            min_val = min(results['actual'].min(), results['predictions'].min())
            max_val = max(results['actual'].max(), results['predictions'].max())
            fig_scatter.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='gray', dash='dash')
            ))
            
            fig_scatter.update_layout(height=400)
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Residuals analysis
        st.header("📈 Residuals Analysis")
        
        residuals = results['actual'] - results['predictions']
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Residuals over time
            fig_residuals = px.line(
                x=pd.to_datetime(results['dates']),
                y=residuals,
                title="Residuals Over Time",
                labels={'x': 'Date', 'y': 'Residual (Actual - Predicted)'}
            )
            fig_residuals.add_hline(y=0, line_dash="dash", line_color="gray")
            fig_residuals.update_layout(height=400)
            st.plotly_chart(fig_residuals, use_container_width=True)
        
        with col2:
            # Residuals distribution
            fig_residuals_hist = px.histogram(
                x=residuals,
                title="Residuals Distribution",
                labels={'x': 'Residual', 'y': 'Frequency'},
                nbins=30
            )
            fig_residuals_hist.update_layout(height=400)
            st.plotly_chart(fig_residuals_hist, use_container_width=True)
        
        # Feature importance
        if results['feature_importance'] is not None:
            st.header("🎯 Feature Importance")
            
            importance_df = pd.DataFrame(
                list(results['feature_importance'].items()),
                columns=['Feature', 'Importance']
            ).sort_values('Importance', ascending=False)
            
            fig_importance = px.bar(
                importance_df,
                x='Importance',
                y='Feature',
                orientation='h',
                title="Feature Importance Ranking"
            )
            fig_importance.update_layout(height=400)
            st.plotly_chart(fig_importance, use_container_width=True)
            
            # Show importance table
            st.subheader("📋 Feature Importance Table")
            st.dataframe(importance_df, use_container_width=True)
        
        # Model insights
        st.header("💡 Model Insights")
        
        insights = []
        
        # Performance insights
        if results['r2'] > 0.7:
            insights.append("🎉 **Excellent model performance** - R² > 0.7 indicates strong predictive power")
        elif results['r2'] > 0.5:
            insights.append("✅ **Good model performance** - R² > 0.5 shows reasonable predictive ability")
        elif results['r2'] > 0.3:
            insights.append("⚠️ **Moderate model performance** - Consider adding more features or data")
        else:
            insights.append("❌ **Poor model performance** - Model may not be suitable for this prediction task")
        
        # Residual insights
        residual_std = np.std(residuals)
        mean_actual = np.mean(results['actual'])
        cv = (residual_std / mean_actual) * 100 if mean_actual != 0 else 0
        
        insights.append(f"📊 **Prediction variability:** {cv:.1f}% coefficient of variation")
        
        # Feature insights
        if results['feature_importance'] is not None:
            top_feature = max(results['feature_importance'], key=results['feature_importance'].get)
            top_importance = results['feature_importance'][top_feature]
            insights.append(f"🏆 **Most important feature:** {top_feature} ({top_importance:.3f} importance)")
        
        for insight in insights:
            st.markdown(insight)
    
    else:
        # Instructions when no model is trained
        st.header("🚀 Getting Started with Predictive Modeling")
        
        st.markdown("""
        ### How to use this dashboard:
        
        1. **Select Target Variable**: Choose what you want to predict (e.g., Resting Heart Rate)
        2. **Choose Features**: Select health metrics that might influence your target
        3. **Set Date Range**: Define the training period for your model
        4. **Train Model**: Click the "Train Model" button to build your prediction model
        
        ### Available Models:
        - **Random Forest Regression**: Robust ensemble method that handles non-linear relationships
        - **Time Series Cross-Validation**: Prevents data leakage by respecting temporal order
        
        ### Model Evaluation Metrics:
        - **MAE (Mean Absolute Error)**: Average prediction error in original units
        - **R² Score**: Proportion of variance explained by the model (higher is better)
        - **RMSE**: Root mean squared error, penalizes large errors more heavily
        - **MAPE**: Mean absolute percentage error, relative error measure
        """)
        
        # Show sample predictions based on notebook analysis
        st.subheader("📊 Expected Model Performance")
        
        performance_data = {
            'Target Variable': ['Resting Heart Rate', 'Heart Rate Variability', 'Step Count'],
            'Expected R² Range': ['0.3 - 0.6', '0.4 - 0.7', '0.5 - 0.8'],
            'Typical MAE': ['5-15 bpm', '10-25 ms', '1000-3000 steps'],
            'Key Features': [
                'HRV, Exercise Time, Steps',
                'Resting HR, Activity Level',
                'Exercise Time, Distance, Energy'
            ]
        }
        
        st.dataframe(pd.DataFrame(performance_data), use_container_width=True)

if __name__ == "__main__":
    main()
