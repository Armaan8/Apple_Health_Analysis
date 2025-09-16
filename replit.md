# Health Data Analytics Dashboard

## Overview

This is a comprehensive personal health data analytics dashboard built with Streamlit that provides multi-dimensional analysis of health and fitness metrics. The application processes personal health data exported from Apple Health or similar fitness tracking platforms and presents interactive visualizations and insights across six specialized analysis pages: main dashboard overview, activity analysis, heart rate monitoring, lifestyle patterns, predictive modeling, and relationship analysis between different health metrics.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
The application uses **Streamlit** as the primary web framework with a multi-page architecture. The main entry point is `app.py` which serves as the dashboard homepage, while specialized analysis pages are organized in the `/pages` directory using Streamlit's native page routing system. Each page is self-contained with its own imports and functionality, following a consistent layout pattern with sidebar controls and multi-column layouts for optimal data presentation.

### Data Processing Architecture
The system implements a **utility-based modular design** with two core utility modules:
- `utils/data_loader.py` - Handles CSV data ingestion, preprocessing, and caching using Streamlit's `@st.cache_data` decorator
- `utils/analysis_functions.py` - Contains statistical analysis functions, visualization creators, and machine learning utilities

**Data Flow**: Raw CSV health data → Preprocessing & validation → Cached DataFrame → Analysis functions → Interactive visualizations

### Visualization Strategy
The application leverages **Plotly** (both Express and Graph Objects) for all interactive visualizations, providing zoom, pan, and hover capabilities. Chart types include time series plots, correlation heatmaps, scatter plots, and statistical distribution charts. The visualization approach emphasizes interactivity and responsiveness across different screen sizes.

### Analysis Capabilities
The system supports multiple analytical approaches:
- **Time Series Analysis**: Trend analysis, moving averages, seasonal pattern detection
- **Statistical Analysis**: KPI calculations, outlier detection using IQR method, correlation analysis
- **Machine Learning**: Predictive modeling using RandomForest and Linear Regression with time-series cross-validation
- **Dimensionality Reduction**: PCA analysis for feature relationships
- **Clustering**: K-means clustering for pattern recognition in health metrics

### Data Management
The application expects health data in CSV format with date-indexed records. It implements flexible data loading that searches multiple file locations and gracefully handles missing data. The data preprocessing pipeline adds derived temporal features (year, month, week, weekday) and handles data type conversions automatically.

### State Management
Streamlit's native session state and caching mechanisms manage application state. Date range filters and analysis parameters are maintained through sidebar controls, with automatic data filtering and re-computation when parameters change.

## External Dependencies

### Core Framework Dependencies
- **Streamlit** - Web application framework and UI components
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computations and array operations

### Visualization Dependencies
- **Plotly Express & Graph Objects** - Interactive plotting and charting
- Charts support zooming, panning, hover tooltips, and responsive design

### Machine Learning Dependencies
- **Scikit-learn** - Machine learning algorithms, preprocessing, and model evaluation
  - RandomForestRegressor and LinearRegression for predictive modeling
  - StandardScaler and SimpleImputer for data preprocessing
  - PCA for dimensionality reduction
  - KMeans for clustering analysis
  - TimeSeriesSplit for temporal cross-validation

### Data Processing Dependencies
- **datetime** - Date and time manipulation for temporal analysis
- **os** - File system operations for data loading

### Data Source Requirements
The application expects personal health data exported as CSV format, typically from Apple Health or similar fitness tracking platforms. The data should include metrics such as step count, heart rate, energy expenditure, sleep data, and walking stability measurements with daily timestamp indexing.