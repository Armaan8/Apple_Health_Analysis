import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
import streamlit as st

def calculate_kpis(df):
    """Calculate key performance indicators."""
    if df is None or df.empty:
        return {}
    
    kpis = {}
    
    # Average metrics
    kpis['avg_steps'] = df['StepCount'].mean() if 'StepCount' in df.columns else 0
    kpis['avg_resting_hr'] = df['RestingHeartRate'].mean() if 'RestingHeartRate' in df.columns else 0
    kpis['avg_active_energy'] = df['ActiveEnergyBurned'].mean() if 'ActiveEnergyBurned' in df.columns else 0
    kpis['avg_exercise_time'] = df['AppleExerciseTime'].mean() if 'AppleExerciseTime' in df.columns else 0
    
    # Calculate trends (last 30 days vs previous 30 days)
    if len(df) >= 60:
        last_30 = df.tail(30)
        prev_30 = df.tail(60).head(30)
        
        # Steps trend
        if 'StepCount' in df.columns:
            last_avg = last_30['StepCount'].mean()
            prev_avg = prev_30['StepCount'].mean()
            kpis['steps_trend'] = ((last_avg - prev_avg) / prev_avg * 100) if prev_avg > 0 else None
        else:
            kpis['steps_trend'] = None
            
        # HR trend (absolute difference)
        if 'RestingHeartRate' in df.columns:
            last_hr = last_30['RestingHeartRate'].mean()
            prev_hr = prev_30['RestingHeartRate'].mean()
            kpis['hr_trend'] = last_hr - prev_hr if not (pd.isna(last_hr) or pd.isna(prev_hr)) else None
        else:
            kpis['hr_trend'] = None
            
        # Energy trend
        if 'ActiveEnergyBurned' in df.columns:
            last_energy = last_30['ActiveEnergyBurned'].mean()
            prev_energy = prev_30['ActiveEnergyBurned'].mean()
            kpis['energy_trend'] = ((last_energy - prev_energy) / prev_energy * 100) if prev_energy > 0 else None
        else:
            kpis['energy_trend'] = None
            
        # Exercise trend
        if 'AppleExerciseTime' in df.columns:
            last_exercise = last_30['AppleExerciseTime'].mean()
            prev_exercise = prev_30['AppleExerciseTime'].mean()
            kpis['exercise_trend'] = ((last_exercise - prev_exercise) / prev_exercise * 100) if prev_exercise > 0 else None
        else:
            kpis['exercise_trend'] = None
    else:
        kpis['steps_trend'] = None
        kpis['hr_trend'] = None
        kpis['energy_trend'] = None
        kpis['exercise_trend'] = None
    
    return kpis

def remove_outliers_iqr(series):
    """Remove outliers using IQR method."""
    if series.empty or series.isna().all():
        return series
    
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return series.where((series >= lower) & (series <= upper))

def create_correlation_heatmap(df, columns=None):
    """Create correlation heatmap."""
    if df is None:
        return None
    
    if columns is None:
        numeric_df = df.select_dtypes(include=[np.number])
    else:
        numeric_df = df[columns].select_dtypes(include=[np.number])
    
    if numeric_df.empty:
        return None
    
    corr = numeric_df.corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.index,
        colorscale='RdBu',
        zmid=0,
        zmin=-1,
        zmax=1
    ))
    
    fig.update_layout(
        title="Correlation Heatmap",
        height=600
    )
    
    return fig

def perform_pca_analysis(df, columns=None):
    """Perform PCA analysis."""
    if df is None:
        return None, None
    
    if columns is None:
        numeric_df = df.select_dtypes(include=[np.number])
    else:
        numeric_df = df[columns].select_dtypes(include=[np.number])
    
    # Remove columns with too many NaN values
    numeric_df = numeric_df.loc[:, numeric_df.isnull().mean() < 0.5]
    
    if numeric_df.empty or numeric_df.shape[1] < 2:
        return None, None
    
    # Fill NaN values with median
    numeric_df = numeric_df.fillna(numeric_df.median())
    
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_df)
    
    # Perform PCA
    pca = PCA(n_components=min(2, numeric_df.shape[1]))
    pca_result = pca.fit_transform(scaled_data)
    
    # Create DataFrame with PCA results
    pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'] if pca_result.shape[1] >= 2 else ['PC1'])
    
    explained_variance = pca.explained_variance_ratio_.sum()
    
    return pca_df, explained_variance

def perform_clustering(df, columns=None, n_clusters=3):
    """Perform K-means clustering."""
    if df is None:
        return None
    
    if columns is None:
        cluster_df = df.select_dtypes(include=[np.number])
    else:
        cluster_df = df[columns].select_dtypes(include=[np.number])
    
    # Remove columns with too many NaN values
    cluster_df = cluster_df.loc[:, cluster_df.isnull().mean() < 0.5]
    
    if cluster_df.empty or cluster_df.shape[1] < 2:
        return None
    
    # Fill NaN values with median
    cluster_df = cluster_df.fillna(cluster_df.median())
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(cluster_df)
    
    return labels

def create_time_series_plot(df, column, title=None):
    """Create a time series plot for a specific column."""
    if df is None or column not in df.columns:
        return None
    
    fig = px.line(
        df, 
        x='date', 
        y=column,
        title=title or f"{column} Over Time",
        markers=True
    )
    
    fig.update_layout(
        height=400,
        showlegend=False
    )
    
    return fig

def create_scatter_plot(df, x_col, y_col, title=None):
    """Create scatter plot between two columns."""
    if df is None or x_col not in df.columns or y_col not in df.columns:
        return None
    
    # Calculate correlation
    mask = df[x_col].notna() & df[y_col].notna()
    if mask.sum() < 2:
        return None
    
    correlation = df.loc[mask, [x_col, y_col]].corr().iloc[0, 1]
    
    fig = px.scatter(
        df, 
        x=x_col, 
        y=y_col,
        title=title or f"{x_col} vs {y_col} (r = {correlation:.2f})",
        opacity=0.6
    )
    
    fig.update_layout(height=400)
    
    return fig

def train_predictive_model(df, target_column, feature_columns):
    """Train a predictive model for health metrics."""
    if df is None or target_column not in df.columns:
        return None, None, None
    
    # Prepare features
    available_features = [col for col in feature_columns if col in df.columns]
    if not available_features:
        return None, None, None
    
    X = df[available_features].copy()
    y = df[target_column].copy()
    
    # Remove rows where target is NaN
    mask = y.notna()
    X = X.loc[mask].fillna(X.median(numeric_only=True))
    y = y.loc[mask]
    
    if len(y) < 10:  # Need minimum data for modeling
        return None, None, None
    
    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Random Forest model
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
    ])
    
    predictions = np.zeros(len(y))
    
    for train_idx, test_idx in tscv.split(X):
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        predictions[test_idx] = model.predict(X.iloc[test_idx])
    
    # Calculate metrics
    mae = mean_absolute_error(y, predictions)
    r2 = r2_score(y, predictions)
    
    results = {
        'predictions': predictions,
        'actual': y.values,
        'dates': df.loc[mask, 'date'].values,
        'mae': mae,
        'r2': r2,
        'feature_importance': None
    }
    
    # Get feature importance from the last trained model
    if hasattr(model.named_steps['model'], 'feature_importances_'):
        results['feature_importance'] = dict(zip(
            available_features,
            model.named_steps['model'].feature_importances_
        ))
    
    return model, results, available_features
