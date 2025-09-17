import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
from utils.data_loader import load_health_data, get_numeric_columns
from utils.analysis_functions import (
    create_correlation_heatmap, 
    perform_pca_analysis, 
    perform_clustering,
    create_scatter_plot
)

st.set_page_config(page_title="Relationship Analysis", layout="wide")

def main():
    st.title("Health Metrics Relationship Analysis")
    st.markdown("---")
    
    # Load data
    df = load_health_data()
    if df is None:
        st.error("Could not load health data. Please ensure the data file is available.")
        return
    
    # Get numeric columns for analysis
    numeric_cols = get_numeric_columns(df)
    
    if not numeric_cols:
        st.error("No numeric columns found for analysis.")
        return
    
    # Analysis selection
    st.sidebar.header("Analysis Configuration")
    
    analysis_type = st.sidebar.selectbox(
        "Select Analysis Type",
        ["Correlation Analysis", "PCA Analysis", "Clustering Analysis", "Custom Scatter Plots"]
    )
    
    # Column selection for focused analysis
    focus_cols = st.sidebar.multiselect(
        "Select Columns for Analysis",
        numeric_cols,
        default=[col for col in [
            'StepCount', 'ActiveEnergyBurned', 'AppleExerciseTime',
            'RestingHeartRate', 'HeartRateVariabilitySDNN', 'HRFitnessIndex',
            'WalkingSpeed', 'WalkingStepLength', 'MobilityIndex', 'StabilityIndex'
        ] if col in numeric_cols][:10]
    )
    
    if not focus_cols:
        st.warning("Please select at least one column for analysis.")
        return
    
    # Analysis implementations
    if analysis_type == "Correlation Analysis":
        correlation_analysis(df, focus_cols)
    elif analysis_type == "PCA Analysis":
        pca_analysis(df, focus_cols)
    elif analysis_type == "Clustering Analysis":
        clustering_analysis(df, focus_cols)
    else:
        custom_scatter_analysis(df, focus_cols)

def correlation_analysis(df, columns):
    st.header("Correlation Analysis")
    
    # Create correlation heatmap
    fig_corr = create_correlation_heatmap(df, columns)
    if fig_corr:
        st.plotly_chart(fig_corr, use_container_width=True)
    
    # Calculate and display correlation matrix
    corr_data = df[columns].corr()
    
    st.subheader("Correlation Matrix")
    st.dataframe(corr_data.round(3), use_container_width=True)
    
    # Find strongest correlations
    st.subheader("Strongest Correlations")
    
    # Get upper triangle of correlation matrix
    mask = np.triu(np.ones_like(corr_data, dtype=bool), k=1)
    corr_pairs = []
    
    for i in range(len(corr_data.columns)):
        for j in range(i+1, len(corr_data.columns)):
            if not pd.isna(corr_data.iloc[i, j]):
                corr_pairs.append({
                    'Variable 1': corr_data.columns[i],
                    'Variable 2': corr_data.columns[j],
                    'Correlation': corr_data.iloc[i, j]
                })
    
    if corr_pairs:
        corr_df = pd.DataFrame(corr_pairs)
        corr_df['Abs_Correlation'] = abs(corr_df['Correlation'])
        top_correlations = corr_df.nlargest(10, 'Abs_Correlation')[['Variable 1', 'Variable 2', 'Correlation']]
        
        st.dataframe(top_correlations.round(3), use_container_width=True)
        
        # Visualize top correlations
        col1, col2 = st.columns(2)
        
        for idx, row in top_correlations.head(4).iterrows():
            col = col1 if idx % 2 == 0 else col2
            with col:
                fig_scatter = create_scatter_plot(
                    df, 
                    row['Variable 1'], 
                    row['Variable 2'],
                    f"{row['Variable 1']} vs {row['Variable 2']}"
                )
                if fig_scatter:
                    st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Correlation insights
    st.subheader("Correlation Insights")
    
    insights = []
    
    if corr_pairs:
        # Find strongest positive correlation
        strongest_pos = max(corr_pairs, key=lambda x: x['Correlation'] if x['Correlation'] > 0 else -1)
        if strongest_pos['Correlation'] > 0.5:
            insights.append(f"**Strongest positive relationship:** {strongest_pos['Variable 1']} and {strongest_pos['Variable 2']} (r = {strongest_pos['Correlation']:.2f})")
        
        # Find strongest negative correlation
        strongest_neg = min(corr_pairs, key=lambda x: x['Correlation'])
        if strongest_neg['Correlation'] < -0.3:
            insights.append(f"**Strongest negative relationship:** {strongest_neg['Variable 1']} and {strongest_neg['Variable 2']} (r = {strongest_neg['Correlation']:.2f})")
        
        # Count strong correlations
        strong_corrs = [p for p in corr_pairs if abs(p['Correlation']) > 0.5]
        insights.append(f"**Strong correlations found:** {len(strong_corrs)} pairs with |r| > 0.5")
    
    for insight in insights:
        st.markdown(insight)

def pca_analysis(df, columns):
    st.header("Principal Component Analysis (PCA)")
    
    # Perform PCA
    pca_df, explained_variance = perform_pca_analysis(df, columns)
    
    if pca_df is None:
        st.error("Could not perform PCA. Please check your data and column selection.")
        return
    
    st.success(f"PCA completed! Explained variance: {explained_variance:.1%}")
    
    # PCA visualization
    if pca_df.shape[1] >= 2:
        col1, col2 = st.columns(2)
        
        with col1:
            # PCA scatter plot
            fig_pca = px.scatter(
                pca_df,
                x='PC1',
                y='PC2',
                title=f"PCA Projection (Explained Variance: {explained_variance:.1%})",
                opacity=0.6
            )
            fig_pca.update_layout(height=500)
            st.plotly_chart(fig_pca, use_container_width=True)
        
        with col2:
            # PCA with color by time if date available
            if 'date' in df.columns and len(pca_df) == len(df):
                pca_with_date = pca_df.copy()
                pca_with_date['year'] = df['year']
                
                fig_pca_time = px.scatter(
                    pca_with_date,
                    x='PC1',
                    y='PC2',
                    color='year',
                    title="PCA Projection Colored by Year",
                    opacity=0.6
                )
                fig_pca_time.update_layout(height=500)
                st.plotly_chart(fig_pca_time, use_container_width=True)
    
    # PCA component loadings (if we can recreate them)
    st.subheader("PCA Summary")
    
    pca_summary = pd.DataFrame({
        'Component': ['PC1', 'PC2'][:pca_df.shape[1]],
        'Data Points': [len(pca_df)] * pca_df.shape[1],
        'Explained Variance': [f"{explained_variance:.1%}"] * pca_df.shape[1]
    })
    
    st.dataframe(pca_summary, use_container_width=True)
    
    # PCA insights
    st.subheader("PCA Insights")
    
    insights = [
        f"**Dimensionality reduction:** {len(columns)} variables compressed to 2 components",
        f"**Information retained:** {explained_variance:.1%} of original variance preserved",
    ]
    
    if explained_variance > 0.7:
        insights.append("**Excellent compression:** Most health patterns captured in 2 dimensions")
    elif explained_variance > 0.5:
        insights.append("**Good compression:** Major health patterns captured")
    else:
        insights.append("**Complex patterns:** Health data has high dimensionality")
    
    for insight in insights:
        st.markdown(insight)

def clustering_analysis(df, columns):
    st.header("Health Pattern Clustering Analysis")
    
    # Clustering configuration
    n_clusters = st.sidebar.slider("Number of Clusters", 2, 8, 3)
    
    # Perform clustering
    cluster_labels = perform_clustering(df, columns, n_clusters)
    
    if cluster_labels is None:
        st.error("Could not perform clustering. Please check your data and column selection.")
        return
    
    # Add cluster labels to dataframe
    df_with_clusters = df[columns].copy()
    df_with_clusters['Cluster'] = cluster_labels
    df_with_clusters['Date'] = df['date'] if 'date' in df.columns else range(len(df))
    
    st.success(f"Clustering completed! Found {n_clusters} health pattern groups")
    
    # Cluster summary
    st.subheader("Cluster Summary")
    
    cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
    cluster_summary = pd.DataFrame({
        'Cluster': [f"Cluster {i}" for i in range(n_clusters)],
        'Size': cluster_counts.values,
        'Percentage': (cluster_counts.values / len(cluster_labels) * 100).round(1)
    })
    
    st.dataframe(cluster_summary, use_container_width=True)
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Cluster size pie chart
        fig_pie = px.pie(
            values=cluster_counts.values,
            names=[f"Cluster {i}" for i in range(n_clusters)],
            title="Cluster Size Distribution"
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Cluster timeline
        if 'date' in df.columns:
            timeline_df = pd.DataFrame({
                'Date': df['date'],
                'Cluster': [f"Cluster {label}" for label in cluster_labels]
            })
            
            fig_timeline = px.scatter(
                timeline_df,
                x='Date',
                y='Cluster',
                color='Cluster',
                title="Cluster Timeline",
                height=400
            )
            st.plotly_chart(fig_timeline, use_container_width=True)
    
    # Cluster characteristics
    st.subheader("Cluster Characteristics")
    
    cluster_means = df_with_clusters.groupby('Cluster')[columns].mean().round(2)
    cluster_means.index = [f"Cluster {i}" for i in range(n_clusters)]
    
    st.dataframe(cluster_means.T, use_container_width=True)
    
    # Radar chart for cluster comparison
    st.subheader("Cluster Profile Comparison")
    
    # Normalize data for radar chart
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    normalized_means = pd.DataFrame(
        scaler.fit_transform(cluster_means.T).T,
        columns=cluster_means.columns,
        index=cluster_means.index
    )
    
    # Create radar chart
    fig_radar = go.Figure()
    
    for cluster in normalized_means.index:
        fig_radar.add_trace(go.Scatterpolar(
            r=normalized_means.loc[cluster].values,
            theta=normalized_means.columns,
            fill='toself',
            name=cluster,
            opacity=0.6
        ))
    
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[-2, 2]
            )
        ),
        showlegend=True,
        title="Cluster Profiles (Standardized)",
        height=500
    )
    
    st.plotly_chart(fig_radar, use_container_width=True)
    
    # Clustering insights
    st.subheader("Clustering Insights")
    
    insights = []
    
    # Find most distinct clusters
    cluster_ranges = cluster_means.max(axis=0) - cluster_means.min(axis=0)
    most_variable_metric = cluster_ranges.idxmax()
    
    insights.append(f"**Most distinctive metric:** {most_variable_metric} shows largest variation between clusters")
    
    # Identify cluster patterns
    for i, cluster_name in enumerate(cluster_means.index):
        cluster_data = cluster_means.loc[cluster_name]
        top_metric = cluster_data.idxmax()
        insights.append(f"**{cluster_name} pattern:** Highest in {top_metric} ({cluster_data[top_metric]:.1f} avg)")
    
    for insight in insights:
        st.markdown(insight)

def custom_scatter_analysis(df, columns):
    st.header("Custom Scatter Plot Analysis")
    
    # Column selection for scatter plots
    col1, col2 = st.columns(2)
    
    with col1:
        x_col = st.selectbox("Select X-axis variable", columns, index=0)
    
    with col2:
        y_col = st.selectbox("Select Y-axis variable", columns, index=1 if len(columns) > 1 else 0)
    
    if x_col == y_col:
        st.warning("Please select different variables for X and Y axes.")
        return
    
    # Create scatter plot
    fig_scatter = create_scatter_plot(df, x_col, y_col, f"{x_col} vs {y_col}")
    if fig_scatter:
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Additional scatter plot options
    st.subheader("Enhanced Visualizations")
    
    # Color by another variable
    color_options = ['None'] + [col for col in columns if col not in [x_col, y_col]]
    color_by = st.selectbox("Color points by:", color_options)
    
    if color_by != 'None':
        fig_colored = px.scatter(
            df,
            x=x_col,
            y=y_col,
            color=color_by,
            title=f"{x_col} vs {y_col} (colored by {color_by})",
            opacity=0.6
        )
        st.plotly_chart(fig_colored, use_container_width=True)
    
    # Time series colored scatter if date available
    if 'date' in df.columns:
        st.subheader("Time-Based Analysis")
        
        # Add year for coloring
        df_with_year = df.copy()
        if 'year' not in df_with_year.columns:
            df_with_year['year'] = df_with_year['date'].dt.year
        
        fig_time = px.scatter(
            df_with_year,
            x=x_col,
            y=y_col,
            color='year',
            title=f"{x_col} vs {y_col} over time",
            opacity=0.6
        )
        st.plotly_chart(fig_time, use_container_width=True)
    
    # Statistical summary
    st.subheader("Statistical Summary")
    
    summary_stats = df[[x_col, y_col]].describe().round(2)
    st.dataframe(summary_stats, use_container_width=True)
    
    # Correlation coefficient
    correlation = df[[x_col, y_col]].corr().iloc[0, 1]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Correlation Coefficient", f"{correlation:.3f}")
    with col2:
        st.metric("Sample Size", f"{len(df[[x_col, y_col]].dropna())}")
    with col3:
        correlation_strength = "Strong" if abs(correlation) > 0.7 else "Moderate" if abs(correlation) > 0.3 else "Weak"
        st.metric("Relationship Strength", correlation_strength)

if __name__ == "__main__":
    main()
