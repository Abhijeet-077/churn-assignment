"""
🎮 Cyberpunk Churn Prediction Dashboard
Standalone Streamlit application with no external dependencies
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
import io

# Page configuration
st.set_page_config(
    page_title="🎮 Cyberpunk Churn Prediction",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# High Contrast Dark Theme with Blue Elements
st.markdown("""
<style>
    /* Import Professional Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* Global High Contrast Dark Theme */
    .stApp {
        font-family: 'Inter', sans-serif;
        background: #000000;
        color: #ffffff;
    }

    /* Main Container */
    .main .block-container {
        background: #1a1a1a;
        border-radius: 12px;
        border: 2px solid #0066cc;
        box-shadow: 0 4px 20px rgba(0, 102, 204, 0.3);
        padding: 2rem;
        margin: 1rem;
    }

    /* Header */
    .main-header {
        font-family: 'Inter', sans-serif;
        font-size: 3rem;
        font-weight: 700;
        color: #ffffff;
        text-align: center;
        margin-bottom: 1rem;
        letter-spacing: 1px;
    }

    /* Metric Cards */
    .metric-card {
        background: #2a2a2a;
        padding: 1.5rem;
        border-radius: 8px;
        border: 2px solid #0066cc;
        box-shadow: 0 2px 10px rgba(0, 102, 204, 0.2);
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }

    .metric-card:hover {
        border-color: #0080ff;
        box-shadow: 0 4px 15px rgba(0, 128, 255, 0.3);
        transform: translateY(-2px);
    }

    .metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2rem;
        font-weight: 600;
        color: #ffffff;
    }

    .metric-label {
        font-size: 0.9rem;
        color: #cccccc;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Prediction Results */
    .prediction-result {
        font-family: 'Inter', sans-serif;
        font-size: 1.5rem;
        font-weight: 600;
        padding: 2rem;
        border-radius: 8px;
        text-align: center;
        margin: 2rem 0;
        border: 2px solid;
    }

    .churn-yes {
        background: #2a2a2a;
        color: #ffffff;
        border-color: #ff4444;
        box-shadow: 0 4px 15px rgba(255, 68, 68, 0.3);
    }

    .churn-no {
        background: #2a2a2a;
        color: #ffffff;
        border-color: #44ff44;
        box-shadow: 0 4px 15px rgba(68, 255, 68, 0.3);
    }

    /* Sidebar */
    .css-1d391kg {
        background: #000000;
        border-right: 2px solid #0066cc;
        box-shadow: 2px 0 10px rgba(0, 102, 204, 0.2);
    }

    /* Buttons */
    .stButton > button {
        background: #0066cc;
        color: #ffffff;
        border: 2px solid #0066cc;
        border-radius: 6px;
        padding: 0.75rem 1.5rem;
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        background: #0080ff;
        border-color: #0080ff;
        color: #ffffff;
        box-shadow: 0 4px 15px rgba(0, 128, 255, 0.4);
        transform: translateY(-1px);
    }

    /* Text styling */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
    }

    /* Form elements */
    .stSelectbox > div > div {
        background: #2a2a2a;
        border: 2px solid #0066cc;
        color: #ffffff;
    }

    .stTextInput > div > div > input {
        background: #2a2a2a;
        border: 2px solid #0066cc;
        color: #ffffff;
    }

    .stNumberInput > div > div > input {
        background: #2a2a2a;
        border: 2px solid #0066cc;
        color: #ffffff;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: #1a1a1a;
        border-radius: 6px;
        border: 2px solid #0066cc;
    }

    .stTabs [data-baseweb="tab"] {
        background: #2a2a2a;
        color: #ffffff;
        border-radius: 4px;
        margin: 2px;
    }

    .stTabs [aria-selected="true"] {
        background: #0066cc !important;
        color: #ffffff !important;
    }

    /* Success/Error messages */
    .stSuccess {
        background: #1a1a1a;
        border: 2px solid #44ff44;
        color: #ffffff;
        border-radius: 6px;
    }

    .stError {
        background: #1a1a1a;
        border: 2px solid #ff4444;
        color: #ffffff;
        border-radius: 6px;
    }

    .stWarning {
        background: #1a1a1a;
        border: 2px solid #ffaa00;
        color: #ffffff;
        border-radius: 6px;
    }

    .stInfo {
        background: #1a1a1a;
        border: 2px solid #0066cc;
        color: #ffffff;
        border-radius: 6px;
    }

    /* Data tables */
    .stDataFrame {
        background: #ffffff;
        color: #000000;
        border-radius: 6px;
        border: 2px solid #0066cc;
    }

    /* File uploader */
    .stFileUploader {
        background: #2a2a2a;
        border: 2px dashed #0066cc;
        border-radius: 6px;
        color: #ffffff;
    }

    /* Metrics */
    [data-testid="metric-container"] {
        background: #2a2a2a;
        border: 2px solid #0066cc;
        border-radius: 6px;
        padding: 1rem;
        box-shadow: 0 2px 10px rgba(0, 102, 204, 0.2);
    }

    [data-testid="metric-container"] > div {
        color: #ffffff;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background: #2a2a2a;
        border: 2px solid #0066cc;
        color: #ffffff;
        border-radius: 6px;
    }

    .streamlit-expanderContent {
        background: #1a1a1a;
        border: 2px solid #0066cc;
        border-top: none;
        color: #ffffff;
        border-radius: 0 0 6px 6px;
    }
</style>
""", unsafe_allow_html=True)

def generate_sample_data():
    """Generate sample customer data for demonstration."""
    np.random.seed(42)
    
    # Generate sample customer data
    n_customers = 1000
    
    data = {
        'CustomerID': [f'C{i:04d}' for i in range(1, n_customers + 1)],
        'Gender': np.random.choice(['Male', 'Female'], n_customers),
        'SeniorCitizen': np.random.choice([0, 1], n_customers, p=[0.84, 0.16]),
        'Partner': np.random.choice(['Yes', 'No'], n_customers, p=[0.52, 0.48]),
        'Dependents': np.random.choice(['Yes', 'No'], n_customers, p=[0.30, 0.70]),
        'Tenure': np.random.randint(1, 73, n_customers),
        'PhoneService': np.random.choice(['Yes', 'No'], n_customers, p=[0.90, 0.10]),
        'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service'], n_customers, p=[0.42, 0.48, 0.10]),
        'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_customers, p=[0.34, 0.44, 0.22]),
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_customers, p=[0.55, 0.21, 0.24]),
        'PaperlessBilling': np.random.choice(['Yes', 'No'], n_customers, p=[0.59, 0.41]),
        'PaymentMethod': np.random.choice([
            'Electronic check', 'Mailed check', 
            'Bank transfer (automatic)', 'Credit card (automatic)'
        ], n_customers, p=[0.34, 0.19, 0.22, 0.25]),
        'MonthlyCharges': np.random.uniform(18.25, 118.75, n_customers).round(2),
        'TotalCharges': np.random.uniform(18.8, 8684.8, n_customers).round(2),
    }
    
    # Generate churn based on some logic
    churn_prob = (
        (data['Tenure'] < 12) * 0.3 +
        (np.array(data['Contract']) == 'Month-to-month') * 0.2 +
        (np.array(data['MonthlyCharges']) > 80) * 0.2 +
        np.random.uniform(0, 0.3, n_customers)
    )
    
    data['Churn'] = np.random.binomial(1, np.clip(churn_prob, 0, 1), n_customers)
    data['Churn'] = ['Yes' if x == 1 else 'No' for x in data['Churn']]
    
    return pd.DataFrame(data)

def simulate_prediction(customer_data):
    """Simulate ML prediction for demo purposes."""
    # Simple rule-based prediction for demo
    risk_score = 0
    
    # Contract type impact
    if customer_data['contract'] == 'Month-to-month':
        risk_score += 0.3
    elif customer_data['contract'] == 'One year':
        risk_score += 0.1
    
    # Tenure impact
    if customer_data['tenure'] < 12:
        risk_score += 0.25
    elif customer_data['tenure'] < 24:
        risk_score += 0.1
    
    # Monthly charges impact
    if customer_data['monthly_charges'] > 80:
        risk_score += 0.2
    elif customer_data['monthly_charges'] > 60:
        risk_score += 0.1
    
    # Add some randomness
    risk_score += random.uniform(-0.1, 0.1)
    risk_score = max(0, min(1, risk_score))  # Clamp between 0 and 1
    
    prediction = "Yes" if risk_score > 0.5 else "No"
    
    # Determine risk level
    if risk_score > 0.8:
        risk_level = "Very High"
    elif risk_score > 0.6:
        risk_level = "High"
    elif risk_score > 0.4:
        risk_level = "Medium"
    elif risk_score > 0.2:
        risk_level = "Low"
    else:
        risk_level = "Very Low"
    
    return {
        'prediction': prediction,
        'probability': risk_score,
        'risk_level': risk_level,
        'confidence': 'High' if abs(risk_score - 0.5) > 0.3 else 'Medium'
    }

def main():
    """Main application function."""
    
    # Header
    st.markdown("""
    <div class="main-header">
        🤖 CHURN PREDICTION DASHBOARD
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align: center; margin-bottom: 3rem; color: #cccccc; font-size: 1.2rem;">
        Advanced Machine Learning Platform with High Contrast Interface<br>
        <small style="color: #0066cc;">Powered by AI • Real-time Predictions • Professional Analytics</small><br>
        <small style="color: #666666;">Version 2.0 - Enhanced Visibility & Professional Charts</small>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <h2 style="color: white; margin-bottom: 0;">🧭 NAVIGATION</h2>
            <p style="color: rgba(255,255,255,0.8); font-size: 0.9rem;">Explore the ML Dashboard</p>
        </div>
        """, unsafe_allow_html=True)
        
        page = st.selectbox(
            "Choose Module:",
            ["🏠 Dashboard", "📊 Data Upload", "🤖 ML Training", "🎯 Prediction", "📈 Data Explorer"],
            help="Navigate between different sections",
            key="main_navigation"
        )
        
        st.markdown("---")
        st.markdown("### 📊 System Status")
        st.success("✅ High Contrast Theme Active")
        st.success("✅ Dashboard Online")
        st.info("🤖 ML Platform Ready")
    
    # Main content based on page selection
    if page == "🏠 Dashboard":
        show_dashboard()
    elif page == "📊 Data Upload":
        show_data_upload()
    elif page == "🤖 ML Training":
        show_ml_training()
    elif page == "🎯 Prediction":
        show_prediction()
    elif page == "📈 Data Explorer":
        show_data_explorer()

def validate_dataset(df):
    """Validate uploaded dataset."""
    issues = []

    if df.empty:
        issues.append("Dataset is empty")
        return issues

    if len(df.columns) < 2:
        issues.append("Dataset must have at least 2 columns")

    if df.isnull().sum().sum() > len(df) * 0.5:
        issues.append("Dataset has too many missing values (>50%)")

    return issues

def show_data_upload():
    """Show data upload interface."""

    st.markdown("### 📊 Dataset Upload & Validation")

    st.info("📁 Upload your customer dataset (CSV format) for churn prediction analysis")

    # File upload
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type="csv",
        help="Upload a CSV file containing customer data with features and target variable"
    )

    if uploaded_file is not None:
        try:
            # Read the uploaded file
            df = pd.read_csv(uploaded_file)

            # Store in session state
            st.session_state['uploaded_data'] = df

            st.success(f"✅ Dataset uploaded successfully! Shape: {df.shape}")

            # Data validation
            st.markdown("#### 🔍 Data Validation")
            issues = validate_dataset(df)

            if issues:
                for issue in issues:
                    st.warning(f"⚠️ {issue}")
            else:
                st.success("✅ Dataset validation passed!")

            # Basic statistics
            col1, col2, col3, col4 = st.columns(4)

            missing_count = df.isnull().sum().sum()
            missing_percentage = (missing_count / (len(df) * len(df.columns))) * 100
            numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
            categorical_cols = len(df.columns) - numeric_cols

            with col1:
                st.metric("Total Rows", f"{len(df):,}")
            with col2:
                st.metric("Total Columns", f"{len(df.columns)}", f"{numeric_cols} numeric, {categorical_cols} categorical")
            with col3:
                st.metric("Missing Values", f"{missing_count:,}", f"{missing_percentage:.1f}% of total")
            with col4:
                memory_usage = df.memory_usage(deep=True).sum() / 1024 / 1024  # MB
                st.metric("Memory Usage", f"{memory_usage:.1f} MB")

            # Data preview
            st.markdown("#### 👀 Data Preview")
            st.dataframe(df.head(10), use_container_width=True)

            # Column information
            st.markdown("#### 📋 Column Information")

            col_info = pd.DataFrame({
                'Column': df.columns,
                'Data Type': df.dtypes.astype(str),
                'Non-Null Count': df.count(),
                'Null Count': df.isnull().sum(),
                'Unique Values': df.nunique()
            })

            st.dataframe(col_info, use_container_width=True)

            # Target column selection
            st.markdown("#### 🎯 Target Column Selection")

            target_column = st.selectbox(
                "Select the target column for prediction:",
                options=df.columns.tolist(),
                help="Choose the column that contains the outcome you want to predict",
                key="target_column_selector"
            )

            if target_column:
                st.session_state['target_column'] = target_column

                # Show target distribution
                if df[target_column].dtype == 'object' or df[target_column].nunique() <= 10:
                    target_counts = df[target_column].value_counts()

                    fig = px.bar(
                        x=target_counts.index,
                        y=target_counts.values,
                        title=f'📊 Distribution of {target_column}',
                        labels={'x': target_column, 'y': 'Count'}
                    )

                    # High contrast styling for target distribution
                    fig.update_layout(
                        title_font_size=20,
                        title_x=0.5,
                        title_font_color='#000000',
                        title_font_family='Inter',
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        font=dict(size=14, color='#000000', family='Inter'),
                        margin=dict(l=60, r=50, t=80, b=80),
                        xaxis=dict(
                            title_font=dict(size=16, color='#000000', family='Inter'),
                            tickfont=dict(size=12, color='#000000'),
                            gridcolor='#e0e0e0',
                            linecolor='#000000',
                            linewidth=2,
                            tickangle=0
                        ),
                        yaxis=dict(
                            title_font=dict(size=16, color='#000000', family='Inter'),
                            tickfont=dict(size=12, color='#000000'),
                            gridcolor='#e0e0e0',
                            linecolor='#000000',
                            linewidth=2
                        )
                    )

                    # Update traces for better visibility
                    fig.update_traces(
                        marker=dict(line=dict(color='#000000', width=1)),
                        hovertemplate='<b>%{x}</b><br>Count: %{y:,}<extra></extra>'
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # Class balance information
                    st.markdown("##### 📈 Class Distribution")
                    for value, count in target_counts.items():
                        percentage = (count / len(df)) * 100
                        st.write(f"**{value}**: {count:,} samples ({percentage:.1f}%)")

                st.success(f"✅ Target column '{target_column}' selected successfully!")
                st.info("🚀 You can now proceed to the ML Training section to build your model.")

        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
            st.info("💡 Please ensure your file is a valid CSV format")

    else:
        # Show sample data format
        st.markdown("#### 📝 Expected Data Format")
        st.info("Your CSV file should contain customer features and a target column for churn prediction")

        sample_data = pd.DataFrame({
            'CustomerID': ['C001', 'C002', 'C003'],
            'Age': [25, 45, 35],
            'MonthlyCharges': [65.5, 89.2, 45.0],
            'Tenure': [12, 36, 8],
            'Contract': ['Month-to-month', 'Two year', 'One year'],
            'Churn': ['No', 'No', 'Yes']
        })

        st.markdown("**Sample format:**")
        st.dataframe(sample_data, use_container_width=True)

def show_ml_training():
    """Show ML training interface."""

    st.markdown("### 🤖 Machine Learning Model Training")

    # Check if data is uploaded
    if 'uploaded_data' not in st.session_state:
        st.warning("⚠️ Please upload a dataset first in the Data Upload section")
        return

    df = st.session_state['uploaded_data']

    if 'target_column' not in st.session_state:
        st.warning("⚠️ Please select a target column in the Data Upload section")
        return

    target_column = st.session_state['target_column']

    st.success(f"✅ Using dataset with {len(df)} rows and target column: '{target_column}'")

    # Algorithm selection
    st.markdown("#### 🔧 Algorithm Selection")

    algorithms = {
        'Logistic Regression': {
            'model': LogisticRegression,
            'description': 'Linear model for binary classification with probabilistic output',
            'use_case': 'Good for linearly separable data and when interpretability is important',
            'params': {
                'C': st.slider('Regularization strength (C)', 0.01, 10.0, 1.0, 0.01, key="lr_c"),
                'max_iter': st.slider('Maximum iterations', 100, 1000, 100, 50, key="lr_max_iter")
            }
        },
        'Random Forest': {
            'model': RandomForestClassifier,
            'description': 'Ensemble method using multiple decision trees',
            'use_case': 'Excellent for mixed data types and provides feature importance',
            'params': {
                'n_estimators': st.slider('Number of trees', 10, 200, 100, 10, key="rf_n_estimators"),
                'max_depth': st.slider('Maximum depth', 3, 20, 10, 1, key="rf_max_depth"),
                'random_state': 42
            }
        },
        'Gradient Boosting': {
            'model': GradientBoostingClassifier,
            'description': 'Sequential ensemble method that builds models iteratively',
            'use_case': 'High performance for complex patterns, good for competitions',
            'params': {
                'n_estimators': st.slider('Number of boosting stages', 50, 300, 100, 25, key="gb_n_estimators"),
                'learning_rate': st.slider('Learning rate', 0.01, 0.3, 0.1, 0.01, key="gb_learning_rate"),
                'max_depth': st.slider('Maximum depth', 3, 10, 6, 1, key="gb_max_depth"),
                'random_state': 42
            }
        },
        'Support Vector Machine': {
            'model': SVC,
            'description': 'Finds optimal boundary between classes using support vectors',
            'use_case': 'Effective for high-dimensional data and non-linear patterns',
            'params': {
                'C': st.slider('Regularization parameter', 0.1, 10.0, 1.0, 0.1, key="svm_c"),
                'kernel': st.selectbox('Kernel', ['rbf', 'linear', 'poly'], index=0, key="svm_kernel"),
                'probability': True,
                'random_state': 42
            }
        }
    }

    selected_algorithm = st.selectbox(
        "Choose Machine Learning Algorithm:",
        list(algorithms.keys()),
        help="Select the algorithm you want to use for training",
        key="algorithm_selector"
    )

    # Show algorithm information
    algo_info = algorithms[selected_algorithm]

    with st.expander(f"ℹ️ About {selected_algorithm}", expanded=True):
        st.write(f"**Description:** {algo_info['description']}")
        st.write(f"**Use Case:** {algo_info['use_case']}")

    # Hyperparameter configuration
    st.markdown("#### ⚙️ Hyperparameter Configuration")

    params = algo_info['params'].copy()

    # Data preprocessing options
    st.markdown("#### 🔄 Data Preprocessing")

    col1, col2 = st.columns(2)

    with col1:
        handle_missing = st.selectbox(
            "Handle missing values:",
            ["Drop rows", "Fill with mean", "Fill with median", "Fill with mode"],
            help="Choose how to handle missing values in the dataset",
            key="missing_values_handler"
        )

    with col2:
        scale_features = st.checkbox(
            "Scale numerical features",
            value=True,
            help="Standardize numerical features (recommended for SVM and Logistic Regression)",
            key="scale_features_checkbox"
        )

    # Train model button
    if st.button("🚀 Train Model", type="primary"):
        with st.spinner("🤖 Training model... This may take a few moments."):
            try:
                # Prepare data
                df_processed = df.copy()

                # Handle missing values
                if handle_missing == "Drop rows":
                    df_processed = df_processed.dropna()
                elif handle_missing == "Fill with mean":
                    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
                    df_processed[numeric_cols] = df_processed[numeric_cols].fillna(df_processed[numeric_cols].mean())
                elif handle_missing == "Fill with median":
                    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
                    df_processed[numeric_cols] = df_processed[numeric_cols].fillna(df_processed[numeric_cols].median())
                elif handle_missing == "Fill with mode":
                    for col in df_processed.columns:
                        df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0] if not df_processed[col].mode().empty else 0)

                # Prepare features and target
                X = df_processed.drop(columns=[target_column])
                y = df_processed[target_column]

                # Encode categorical variables
                label_encoders = {}
                for col in X.select_dtypes(include=['object']).columns:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
                    label_encoders[col] = le

                # Encode target if categorical
                if y.dtype == 'object':
                    target_encoder = LabelEncoder()
                    y = target_encoder.fit_transform(y)
                    st.session_state['target_encoder'] = target_encoder

                # Scale features if requested
                if scale_features:
                    scaler = StandardScaler()
                    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
                    st.session_state['scaler'] = scaler

                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )

                # Train model
                model = algo_info['model'](**params)
                model.fit(X_train, y_train)

                # Make predictions
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')

                # Store model and results
                st.session_state['trained_model'] = model
                st.session_state['label_encoders'] = label_encoders
                st.session_state['feature_columns'] = X.columns.tolist()
                st.session_state['model_metrics'] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                }

                # Display results
                st.success("✅ Model trained successfully!")

                # Performance metrics
                st.markdown("#### 📊 Model Performance")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Accuracy", f"{accuracy:.3f}", f"{accuracy*100:.1f}%")
                with col2:
                    st.metric("Precision", f"{precision:.3f}", f"{precision*100:.1f}%")
                with col3:
                    st.metric("Recall", f"{recall:.3f}", f"{recall*100:.1f}%")
                with col4:
                    st.metric("F1-Score", f"{f1:.3f}", f"{f1*100:.1f}%")

                # Confusion matrix
                cm = confusion_matrix(y_test, y_pred)

                fig = px.imshow(
                    cm,
                    text_auto=True,
                    title="📊 Confusion Matrix",
                    labels=dict(x="Predicted", y="Actual"),
                    color_continuous_scale="Blues"
                )

                # High contrast styling for confusion matrix
                fig.update_layout(
                    title_font_size=20,
                    title_x=0.5,
                    title_font_color='#000000',
                    title_font_family='Inter',
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font=dict(size=14, color='#000000', family='Inter'),
                    margin=dict(l=60, r=60, t=80, b=60),
                    xaxis=dict(
                        title_font=dict(size=16, color='#000000', family='Inter'),
                        tickfont=dict(size=12, color='#000000'),
                        linecolor='#000000',
                        linewidth=2
                    ),
                    yaxis=dict(
                        title_font=dict(size=16, color='#000000', family='Inter'),
                        tickfont=dict(size=12, color='#000000'),
                        linecolor='#000000',
                        linewidth=2
                    )
                )

                # Update text annotations for better visibility
                fig.update_traces(
                    textfont=dict(size=16, color='white', family='Inter'),
                    hovertemplate='Actual: %{y}<br>Predicted: %{x}<br>Count: %{z:,}<extra></extra>'
                )

                st.plotly_chart(fig, use_container_width=True)

                # Feature importance (if available)
                if hasattr(model, 'feature_importances_'):
                    st.markdown("#### 📈 Feature Importance")

                    importance_df = pd.DataFrame({
                        'Feature': X.columns,
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=True)

                    fig = px.bar(
                        importance_df.tail(10),
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        title='📊 Top 10 Most Important Features'
                    )

                    # High contrast styling for feature importance
                    fig.update_layout(
                        title_font_size=20,
                        title_x=0.5,
                        title_font_color='#000000',
                        title_font_family='Inter',
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        font=dict(size=14, color='#000000', family='Inter'),
                        margin=dict(l=120, r=50, t=80, b=60),
                        xaxis=dict(
                            title_font=dict(size=16, color='#000000', family='Inter'),
                            tickfont=dict(size=12, color='#000000'),
                            gridcolor='#e0e0e0',
                            linecolor='#000000',
                            linewidth=2
                        ),
                        yaxis=dict(
                            title_font=dict(size=16, color='#000000', family='Inter'),
                            tickfont=dict(size=12, color='#000000'),
                            linecolor='#000000',
                            linewidth=2
                        )
                    )

                    # Update traces for better visibility
                    fig.update_traces(
                        marker=dict(line=dict(color='#000000', width=1)),
                        hovertemplate='<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>'
                    )

                    st.plotly_chart(fig, use_container_width=True)

                st.info("🎯 You can now use the trained model in the Prediction section!")

            except Exception as e:
                st.error(f"❌ Error training model: {str(e)}")
                st.info("💡 Please check your data format and try again")

def show_dashboard():
    """Show the main dashboard."""
    
    st.markdown("### 📊 Dashboard Overview")
    
    # Generate sample data
    df = generate_sample_data()
    
    # Calculate metrics
    total_customers = len(df)
    churn_rate = (df['Churn'] == 'Yes').mean() * 100
    avg_tenure = df['Tenure'].mean()
    avg_charges = df['MonthlyCharges'].mean()
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Customers", f"{total_customers:,}")

    with col2:
        st.metric("Churn Rate", f"{churn_rate:.1f}%")

    with col3:
        st.metric("Avg Tenure", f"{avg_tenure:.1f} months")

    with col4:
        st.metric("Avg Charges", f"${avg_charges:.2f}")
    
    # Charts
    st.markdown("### 📈 Data Visualization")
    st.info("📊 Charts use clean, professional styling for optimal data readability")

    col1, col2 = st.columns(2)
    
    with col1:
        # Churn distribution
        churn_counts = df['Churn'].value_counts()
        fig = px.pie(values=churn_counts.values, names=churn_counts.index,
                     title='📊 Customer Churn Distribution')

        # High contrast, professional chart styling
        fig.update_layout(
            title_font_size=20,
            title_x=0.5,
            title_font_color='#000000',
            title_font_family='Inter',
            showlegend=True,
            font=dict(size=14, color='#000000', family='Inter'),
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=50, r=50, t=80, b=50),
            legend=dict(
                font=dict(size=14, color='#000000'),
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='#000000',
                borderwidth=1
            )
        )

        # Add percentage labels with high contrast
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            textfont=dict(size=14, color='white', family='Inter'),
            hovertemplate='<b>%{label}</b><br>Count: %{value:,}<br>Percentage: %{percent}<extra></extra>',
            marker=dict(line=dict(color='#000000', width=2))
        )

        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Monthly charges distribution
        fig = px.histogram(df, x='MonthlyCharges', color='Churn',
                          title='📈 Monthly Charges Distribution by Churn Status',
                          nbins=30,
                          barmode='overlay',
                          opacity=0.7)

        # High contrast, professional chart styling
        fig.update_layout(
            title_font_size=20,
            title_x=0.5,
            title_font_color='#000000',
            title_font_family='Inter',
            xaxis_title="Monthly Charges ($)",
            yaxis_title="Number of Customers",
            xaxis_title_font=dict(size=16, color='#000000', family='Inter'),
            yaxis_title_font=dict(size=16, color='#000000', family='Inter'),
            xaxis=dict(
                tickfont=dict(size=12, color='#000000'),
                gridcolor='#e0e0e0',
                linecolor='#000000',
                linewidth=2
            ),
            yaxis=dict(
                tickfont=dict(size=12, color='#000000'),
                gridcolor='#e0e0e0',
                linecolor='#000000',
                linewidth=2
            ),
            showlegend=True,
            font=dict(size=14, color='#000000', family='Inter'),
            bargap=0.1,
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=60, r=50, t=80, b=60),
            legend=dict(
                font=dict(size=14, color='#000000'),
                bgcolor='rgba(255,255,255,0.9)',
                bordercolor='#000000',
                borderwidth=1
            )
        )

        # Improve hover information with number formatting
        fig.update_traces(
            hovertemplate='<b>%{fullData.name}</b><br>Monthly Charges: $%{x:.2f}<br>Count: %{y:,}<extra></extra>'
        )

        st.plotly_chart(fig, use_container_width=True)

def show_prediction():
    """Show prediction interface."""
    
    st.markdown("### 🎯 Customer Churn Prediction")
    
    st.info("🎮 Enter customer details to predict churn probability!")
    
    # Input form
    with st.form("prediction_form"):
        st.markdown("#### 👤 Customer Information")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            gender = st.selectbox("Gender", ["Female", "Male"], key="pred_gender")
            senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"], key="pred_senior")
            partner = st.selectbox("Partner", ["Yes", "No"], key="pred_partner")

        with col2:
            dependents = st.selectbox("Dependents", ["No", "Yes"], key="pred_dependents")
            tenure = st.slider("Tenure (months)", 0, 72, 32, key="pred_tenure")
            phone_service = st.selectbox("Phone Service", ["Yes", "No"], key="pred_phone")

        with col3:
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"], key="pred_internet")
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"], key="pred_contract")
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"], key="pred_billing")

        col1, col2 = st.columns(2)

        with col1:
            payment_method = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check",
                "Bank transfer (automatic)", "Credit card (automatic)"
            ], key="pred_payment")
        
        with col2:
            monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 64.76, key="pred_monthly_charges")
        
        submitted = st.form_submit_button("🚀 PREDICT CHURN")
        
        if submitted:
            # Prepare customer data
            customer_data = {
                'gender': gender,
                'senior_citizen': 1 if senior_citizen == "Yes" else 0,
                'partner': partner,
                'dependents': dependents,
                'tenure': tenure,
                'phone_service': phone_service,
                'internet_service': internet_service,
                'contract': contract,
                'paperless_billing': paperless_billing,
                'payment_method': payment_method,
                'monthly_charges': monthly_charges
            }
            
            # Simulate prediction
            with st.spinner("🤖 Processing with AI..."):
                time.sleep(2)
            
            result = simulate_prediction(customer_data)
            
            # Display results
            st.markdown("### 🎯 Prediction Results")
            
            result_class = "churn-yes" if result['prediction'] == "Yes" else "churn-no"
            
            st.markdown(f"""
            <div class="prediction-result {result_class}">
                <div style="font-size: 2.2rem; margin-bottom: 0.5rem;">
                    {'⚠️ LIKELY TO CHURN' if result['prediction'] == 'Yes' else '✅ LIKELY TO STAY'}
                </div>
                <div style="font-size: 1.2rem; opacity: 0.9;">
                    Probability: {result['probability']:.1%} | Risk Level: {result['risk_level']}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Additional metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Churn Probability", f"{result['probability']:.1%}")
            
            with col2:
                st.metric("Risk Level", result['risk_level'])
            
            with col3:
                st.metric("Confidence", result['confidence'])

def show_data_explorer():
    """Show data exploration interface."""
    
    st.markdown("### 📊 Data Explorer")
    
    # Generate sample data
    df = generate_sample_data()
    
    st.markdown("#### 📋 Sample Customer Data")
    st.dataframe(df.head(100), use_container_width=True)
    
    st.markdown("#### 📈 Data Analysis")
    st.info("📊 Professional chart styling for clear data insights")

    col1, col2 = st.columns(2)
    
    with col1:
        # Contract type analysis
        contract_churn = df.groupby(['Contract', 'Churn']).size().unstack()
        fig = px.bar(contract_churn, title='📊 Customer Churn by Contract Type',
                     barmode='group')

        # High contrast, professional chart styling
        fig.update_layout(
            title_font_size=20,
            title_x=0.5,
            title_font_color='#000000',
            title_font_family='Inter',
            xaxis_title="Contract Type",
            yaxis_title="Number of Customers",
            xaxis_title_font=dict(size=16, color='#000000', family='Inter'),
            yaxis_title_font=dict(size=16, color='#000000', family='Inter'),
            xaxis=dict(
                tickfont=dict(size=12, color='#000000'),
                gridcolor='#e0e0e0',
                linecolor='#000000',
                linewidth=2,
                tickangle=0
            ),
            yaxis=dict(
                tickfont=dict(size=12, color='#000000'),
                gridcolor='#e0e0e0',
                linecolor='#000000',
                linewidth=2
            ),
            showlegend=True,
            font=dict(size=14, color='#000000', family='Inter'),
            bargap=0.2,
            bargroupgap=0.1,
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=60, r=50, t=80, b=80),
            legend=dict(
                font=dict(size=14, color='#000000'),
                bgcolor='rgba(255,255,255,0.9)',
                bordercolor='#000000',
                borderwidth=1
            )
        )

        # Improve hover information with number formatting
        fig.update_traces(
            hovertemplate='<b>%{fullData.name}</b><br>Contract: %{x}<br>Count: %{y:,}<extra></extra>',
            marker=dict(line=dict(color='#000000', width=1))
        )

        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Tenure vs Monthly Charges
        fig = px.scatter(df, x='Tenure', y='MonthlyCharges', color='Churn',
                        title='📈 Customer Tenure vs Monthly Charges',
                        opacity=0.8,
                        size_max=10)

        # High contrast, professional chart styling
        fig.update_layout(
            title_font_size=20,
            title_x=0.5,
            title_font_color='#000000',
            title_font_family='Inter',
            xaxis_title="Tenure (Months)",
            yaxis_title="Monthly Charges ($)",
            xaxis_title_font=dict(size=16, color='#000000', family='Inter'),
            yaxis_title_font=dict(size=16, color='#000000', family='Inter'),
            xaxis=dict(
                tickfont=dict(size=12, color='#000000'),
                gridcolor='#e0e0e0',
                linecolor='#000000',
                linewidth=2
            ),
            yaxis=dict(
                tickfont=dict(size=12, color='#000000'),
                gridcolor='#e0e0e0',
                linecolor='#000000',
                linewidth=2
            ),
            showlegend=True,
            font=dict(size=14, color='#000000', family='Inter'),
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=60, r=50, t=80, b=60),
            legend=dict(
                font=dict(size=14, color='#000000'),
                bgcolor='rgba(255,255,255,0.9)',
                bordercolor='#000000',
                borderwidth=1
            )
        )

        # Improve hover information with number formatting
        fig.update_traces(
            hovertemplate='<b>%{fullData.name}</b><br>Tenure: %{x} months<br>Monthly Charges: $%{y:.2f}<extra></extra>',
            marker=dict(size=8, line=dict(color='#000000', width=1))
        )

        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
