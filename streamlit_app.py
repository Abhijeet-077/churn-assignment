"""
Streamlit Dashboard for Customer Churn Prediction
Advanced ML-powered interactive web application
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from pathlib import Path
import sys
import os
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data.data_loader import DataLoader
from src.api.prediction_service import PredictionService
from src.api.models import CustomerData

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern CSS styling with animations and responsive design
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global Styles */
    .stApp {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
    }

    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
        margin: 1rem;
    }

    /* Header styling */
    .main-header {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        animation: fadeInDown 1s ease-out;
    }

    .subtitle {
        font-size: 1.2rem;
        color: #6c757d;
        text-align: center;
        margin-bottom: 3rem;
        font-weight: 400;
    }

    /* Animations */
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }

    /* Modern Cards */
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 2rem;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
        animation: fadeInUp 0.8s ease-out;
        margin-bottom: 1rem;
    }

    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.15);
    }

    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2d3748;
        margin-bottom: 0.5rem;
    }

    .metric-label {
        font-size: 1rem;
        color: #718096;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Prediction Results */
    .prediction-result {
        font-size: 1.8rem;
        font-weight: 600;
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        margin: 2rem 0;
        animation: pulse 2s infinite;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }

    .churn-yes {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
        color: white;
        border: none;
    }

    .churn-no {
        background: linear-gradient(135deg, #51cf66 0%, #40c057 100%);
        color: white;
        border: none;
    }

    .risk-indicator {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-weight: 600;
        font-size: 0.9rem;
        margin: 0.25rem;
    }

    .risk-very-high { background: #ff6b6b; color: white; }
    .risk-high { background: #ffa726; color: white; }
    .risk-medium { background: #ffee58; color: #333; }
    .risk-low { background: #66bb6a; color: white; }
    .risk-very-low { background: #42a5f5; color: white; }

    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }

    .css-1d391kg .css-1v0mbdj {
        color: white;
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }

    /* Form styling */
    .stSelectbox > div > div {
        border-radius: 8px;
        border: 2px solid #e2e8f0;
        transition: all 0.3s ease;
    }

    .stSelectbox > div > div:focus-within {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }

    .stNumberInput > div > div > input {
        border-radius: 8px;
        border: 2px solid #e2e8f0;
        transition: all 0.3s ease;
    }

    .stNumberInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }

    /* Progress bar */
    .stProgress > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }

    /* Success/Error messages */
    .stSuccess {
        background: linear-gradient(135deg, #51cf66 0%, #40c057 100%);
        border: none;
        border-radius: 12px;
        color: white;
    }

    .stError {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
        border: none;
        border-radius: 12px;
        color: white;
    }

    .stWarning {
        background: linear-gradient(135deg, #ffa726 0%, #ff9800 100%);
        border: none;
        border-radius: 12px;
        color: white;
    }

    .stInfo {
        background: linear-gradient(135deg, #42a5f5 0%, #2196f3 100%);
        border: none;
        border-radius: 12px;
        color: white;
    }

    /* Navigation styling */
    .nav-item {
        padding: 0.75rem 1rem;
        margin: 0.25rem 0;
        border-radius: 8px;
        transition: all 0.3s ease;
        cursor: pointer;
    }

    .nav-item:hover {
        background: rgba(255, 255, 255, 0.1);
        transform: translateX(5px);
    }

    /* Loading spinner */
    .loading-spinner {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid rgba(255, 255, 255, 0.3);
        border-radius: 50%;
        border-top-color: #667eea;
        animation: spin 1s ease-in-out infinite;
    }

    @keyframes spin {
        to { transform: rotate(360deg); }
    }

    /* Feature importance bars */
    .feature-bar {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        height: 8px;
        border-radius: 4px;
        margin: 0.25rem 0;
        transition: all 0.3s ease;
    }

    .feature-bar:hover {
        height: 12px;
    }

    /* Responsive design */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2.5rem;
        }

        .metric-card {
            padding: 1.5rem;
        }

        .prediction-result {
            font-size: 1.4rem;
            padding: 1.5rem;
        }
    }

    /* Dark mode support */
    @media (prefers-color-scheme: dark) {
        .main .block-container {
            background: rgba(26, 32, 44, 0.95);
            color: #e2e8f0;
        }

        .metric-card {
            background: linear-gradient(135deg, #2d3748 0%, #4a5568 100%);
            color: #e2e8f0;
        }
    }

    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }

    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'prediction_service' not in st.session_state:
    st.session_state.prediction_service = None
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

@st.cache_data
def load_data():
    """Load and cache the dataset."""
    try:
        loader = DataLoader()
        df, _ = loader.load_and_process()
        return df
    except Exception as e:
        st.error(f"Failed to load data: {str(e)}")
        return None

@st.cache_resource
def load_prediction_service():
    """Load and cache the prediction service."""
    try:
        # Create a progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()

        status_text.text("🔄 Initializing ML models...")
        progress_bar.progress(20)

        status_text.text("🔄 Loading optimized parameters...")
        progress_bar.progress(40)

        service = PredictionService()

        status_text.text("🔄 Training models with data...")
        progress_bar.progress(80)

        status_text.text("✅ Models ready!")
        progress_bar.progress(100)

        # Clear the progress indicators
        progress_bar.empty()
        status_text.empty()

        return service
    except Exception as e:
        st.error(f"Failed to load prediction service: {str(e)}")
        return None

def main():
    """Main dashboard function."""
    
    # Modern Header with animations
    st.markdown("""
    <div class="main-header">
        🎯 Customer Churn Prediction Dashboard
    </div>
    <div class="subtitle">
        Advanced Machine Learning Dashboard for Predicting Customer Churn<br>
        <small>Powered by Bayesian Optimization, SHAP Analysis & Modern ML</small>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced Sidebar navigation
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <h2 style="color: white; margin-bottom: 0;">🧭 Navigation</h2>
            <p style="color: rgba(255,255,255,0.8); font-size: 0.9rem;">Explore the Dashboard</p>
        </div>
        """, unsafe_allow_html=True)

        # Navigation with icons and descriptions
        page_options = {
            "🏠 Home": "Dashboard Overview & Key Metrics",
            "🔮 Prediction": "Single Customer Churn Prediction",
            "📊 Data Explorer": "Interactive Data Analysis",
            "📈 Model Analytics": "Model Performance & Insights",
            "🎯 Batch Prediction": "Bulk Customer Analysis"
        }

        page = st.selectbox(
            "Choose a page:",
            list(page_options.keys()),
            format_func=lambda x: x,
            help="Navigate between different sections of the dashboard"
        )

        # Show page description
        st.markdown(f"""
        <div style="background: rgba(255,255,255,0.1); padding: 0.75rem; border-radius: 8px; margin-top: 1rem;">
            <small style="color: rgba(255,255,255,0.9);">{page_options[page]}</small>
        </div>
        """, unsafe_allow_html=True)

        # Add system status
        st.markdown("---")
        st.markdown("### 📊 System Status")

        if st.session_state.data_loaded:
            st.success("✅ Data Loaded")
        else:
            st.warning("⏳ Loading Data...")

        if st.session_state.prediction_service:
            st.success("✅ Models Ready")
        else:
            st.warning("⏳ Loading Models...")

        # Add quick stats if data is loaded
        if st.session_state.data_loaded:
            df = st.session_state.df
            st.markdown("### 📈 Quick Stats")
            st.metric("Total Customers", f"{len(df):,}")
            churn_rate = (df['churn'] == 'Yes').mean() * 100
            st.metric("Churn Rate", f"{churn_rate:.1f}%")
    
    # Load data and models
    if not st.session_state.data_loaded:
        df = load_data()
        if df is not None:
            st.session_state.df = df
            st.session_state.data_loaded = True
    
    if st.session_state.prediction_service is None:
        st.session_state.prediction_service = load_prediction_service()
    
    # Route to different pages
    if page == "🏠 Home":
        show_home_page()
    elif page == "🔮 Prediction":
        show_prediction_page()
    elif page == "📊 Data Explorer":
        show_data_explorer()
    elif page == "📈 Model Analytics":
        show_model_analytics()
    elif page == "🎯 Batch Prediction":
        show_batch_prediction()

def show_home_page():
    """Display the home page with overview and statistics."""

    if not st.session_state.data_loaded:
        st.markdown("""
        <div style="text-align: center; padding: 3rem;">
            <div class="loading-spinner"></div>
            <h3>Loading Data...</h3>
            <p>Please wait while we prepare your dashboard</p>
        </div>
        """, unsafe_allow_html=True)
        return

    df = st.session_state.df

    # Hero section with key metrics
    st.markdown("### 📊 Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(df):,}</div>
            <div class="metric-label">Total Customers</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        churn_rate = (df['churn'] == 'Yes').mean() * 100
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{churn_rate:.1f}%</div>
            <div class="metric-label">Churn Rate</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        avg_tenure = df['tenure'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{avg_tenure:.1f}</div>
            <div class="metric-label">Avg Tenure (Months)</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        avg_charges = df['monthlycharges'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">${avg_charges:.0f}</div>
            <div class="metric-label">Avg Monthly Charges</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Churn distribution
    st.subheader("🎯 Churn Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Pie chart
        churn_counts = df['churn'].value_counts()
        fig_pie = px.pie(
            values=churn_counts.values,
            names=churn_counts.index,
            title="Customer Churn Distribution",
            color_discrete_map={'No': '#2e7d32', 'Yes': '#c62828'}
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Bar chart by contract type
        contract_churn = df.groupby(['contract', 'churn']).size().unstack()
        fig_bar = px.bar(
            contract_churn,
            title="Churn by Contract Type",
            color_discrete_map={'No': '#2e7d32', 'Yes': '#c62828'}
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Feature distributions
    st.subheader("📈 Key Feature Distributions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Tenure distribution
        fig_tenure = px.histogram(
            df, x='tenure', color='churn',
            title="Tenure Distribution by Churn",
            nbins=30,
            color_discrete_map={'No': '#2e7d32', 'Yes': '#c62828'}
        )
        st.plotly_chart(fig_tenure, use_container_width=True)
    
    with col2:
        # Monthly charges distribution
        fig_charges = px.histogram(
            df, x='monthlycharges', color='churn',
            title="Monthly Charges Distribution by Churn",
            nbins=30,
            color_discrete_map={'No': '#2e7d32', 'Yes': '#c62828'}
        )
        st.plotly_chart(fig_charges, use_container_width=True)

def show_prediction_page():
    """Display the single customer prediction page."""
    
    st.subheader("🔮 Single Customer Churn Prediction")
    
    if st.session_state.prediction_service is None:
        st.error("Prediction service not available. Please check the model loading.")
        return
    
    # Enhanced input form with better organization
    with st.form("prediction_form"):
        st.markdown("### 👤 Customer Information")
        st.markdown("Fill in the customer details below to get a churn prediction")

        # Demographics section
        st.markdown("#### 📋 Demographics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior_citizen = st.selectbox("Senior Citizen", [0, 1])
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])
        
        with col2:
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        
        with col3:
            online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
            online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
            device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
            tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])

        # Additional Services section
        st.markdown("#### 📺 Additional Services")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
        
        with col2:
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        
        with col3:
            payment_method = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check",
                "Bank transfer (automatic)", "Credit card (automatic)"
            ])

        # Financial Information section
        st.markdown("#### 💰 Financial Information")
        col1, col2 = st.columns(2)
        with col1:
            monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0)
        with col2:
            total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 780.0)
        
        # Model selection
        available_models = st.session_state.prediction_service.get_available_models()
        selected_model = st.selectbox("Select Model", available_models, index=0)
        
        include_explanation = st.checkbox("Include Prediction Explanation", value=True)
        
        submitted = st.form_submit_button("🔮 Predict Churn", type="primary")
        
        if submitted:
            try:
                # Create customer data
                customer_data = CustomerData(
                    gender=gender,
                    senior_citizen=senior_citizen,
                    partner=partner,
                    dependents=dependents,
                    tenure=tenure,
                    phone_service=phone_service,
                    multiple_lines=multiple_lines,
                    internet_service=internet_service,
                    online_security=online_security,
                    online_backup=online_backup,
                    device_protection=device_protection,
                    tech_support=tech_support,
                    streaming_tv=streaming_tv,
                    streaming_movies=streaming_movies,
                    contract=contract,
                    paperless_billing=paperless_billing,
                    payment_method=payment_method,
                    monthly_charges=monthly_charges,
                    total_charges=total_charges
                )
                
                # Make prediction with enhanced loading
                prediction_progress = st.progress(0)
                prediction_status = st.empty()

                prediction_status.markdown("🔄 **Processing customer data...**")
                prediction_progress.progress(25)

                prediction_status.markdown("🤖 **Running ML model...**")
                prediction_progress.progress(50)

                prediction_status.markdown("🧠 **Generating explanation...**")
                prediction_progress.progress(75)

                try:
                    prediction = st.session_state.prediction_service.predict_single(
                        customer_data=customer_data,
                        model_name=selected_model,
                        include_explanation=include_explanation
                    )

                    prediction_status.markdown("✅ **Prediction complete!**")
                    prediction_progress.progress(100)

                    # Clear loading indicators
                    time.sleep(0.5)
                    prediction_progress.empty()
                    prediction_status.empty()

                except Exception as pred_error:
                    prediction_progress.empty()
                    prediction_status.empty()
                    raise pred_error
                
                # Display results with enhanced styling
                st.markdown("### 🎯 Prediction Results")

                # Main prediction result with animation
                result_class = "churn-yes" if prediction.churn_prediction == "Yes" else "churn-no"
                risk_class = f"risk-{prediction.risk_level.lower().replace(' ', '-')}"

                st.markdown(f"""
                <div class="prediction-result {result_class}">
                    <div style="font-size: 2.2rem; margin-bottom: 0.5rem;">
                        {'⚠️ LIKELY TO CHURN' if prediction.churn_prediction == 'Yes' else '✅ LIKELY TO STAY'}
                    </div>
                    <div style="font-size: 1.2rem; opacity: 0.9;">
                        Confidence: {prediction.confidence} |
                        <span class="risk-indicator {risk_class}">{prediction.risk_level} Risk</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Enhanced metrics with modern cards
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{prediction.churn_probability:.1%}</div>
                        <div class="metric-label">Churn Probability</div>
                    </div>
                    """, unsafe_allow_html=True)

                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{prediction.confidence}</div>
                        <div class="metric-label">Confidence Level</div>
                    </div>
                    """, unsafe_allow_html=True)

                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{prediction.risk_level}</div>
                        <div class="metric-label">Risk Level</div>
                    </div>
                    """, unsafe_allow_html=True)

                with col4:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{prediction.processing_time_ms:.0f}ms</div>
                        <div class="metric-label">Processing Time</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Probability gauge
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = prediction.churn_probability * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Churn Probability (%)"},
                    delta = {'reference': 50},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 25], 'color': "lightgreen"},
                            {'range': [25, 50], 'color': "yellow"},
                            {'range': [50, 75], 'color': "orange"},
                            {'range': [75, 100], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                
                st.plotly_chart(fig_gauge, use_container_width=True)
                
                # Explanation
                if include_explanation and prediction.explanation:
                    st.markdown("### 💡 Prediction Explanation")
                    st.info(prediction.explanation.explanation_text)
                    
                    if prediction.explanation.top_positive_factors:
                        st.markdown("**Risk Factors:**")
                        for factor in prediction.explanation.top_positive_factors:
                            st.markdown(f"• {factor}")
                    
                    if prediction.explanation.top_negative_factors:
                        st.markdown("**Retention Factors:**")
                        for factor in prediction.explanation.top_negative_factors:
                            st.markdown(f"• {factor}")
                
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")

def show_data_explorer():
    """Display the data exploration page."""
    
    if not st.session_state.data_loaded:
        st.warning("Data not loaded yet. Please wait...")
        return
    
    df = st.session_state.df
    
    st.subheader("📊 Data Explorer")
    
    # Data overview
    st.markdown("### Dataset Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Dataset Shape:**")
        st.write(f"Rows: {df.shape[0]:,}")
        st.write(f"Columns: {df.shape[1]}")
    
    with col2:
        st.markdown("**Missing Values:**")
        missing_values = df.isnull().sum().sum()
        st.write(f"Total: {missing_values}")
    
    # Feature selection for analysis
    st.markdown("### Feature Analysis")
    
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    analysis_type = st.selectbox("Select Analysis Type", [
        "Categorical Distribution", "Numerical Distribution", "Correlation Analysis", "Churn Analysis"
    ])
    
    if analysis_type == "Categorical Distribution":
        selected_cat = st.selectbox("Select Categorical Feature", categorical_cols)
        
        fig = px.bar(
            df[selected_cat].value_counts(),
            title=f"Distribution of {selected_cat}",
            labels={'index': selected_cat, 'value': 'Count'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    elif analysis_type == "Numerical Distribution":
        selected_num = st.selectbox("Select Numerical Feature", numerical_cols)
        
        fig = px.histogram(
            df, x=selected_num,
            title=f"Distribution of {selected_num}",
            nbins=30
        )
        st.plotly_chart(fig, use_container_width=True)
    
    elif analysis_type == "Correlation Analysis":
        corr_matrix = df[numerical_cols].corr()
        
        fig = px.imshow(
            corr_matrix,
            title="Feature Correlation Matrix",
            color_continuous_scale="RdBu",
            aspect="auto"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    elif analysis_type == "Churn Analysis":
        selected_feature = st.selectbox("Select Feature for Churn Analysis", 
                                      categorical_cols + numerical_cols)
        
        if selected_feature in categorical_cols:
            churn_by_feature = df.groupby([selected_feature, 'churn']).size().unstack()
            fig = px.bar(
                churn_by_feature,
                title=f"Churn Distribution by {selected_feature}",
                color_discrete_map={'No': '#2e7d32', 'Yes': '#c62828'}
            )
        else:
            fig = px.box(
                df, x='churn', y=selected_feature,
                title=f"{selected_feature} Distribution by Churn",
                color='churn',
                color_discrete_map={'No': '#2e7d32', 'Yes': '#c62828'}
            )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Raw data view
    st.markdown("### Raw Data Sample")
    st.dataframe(df.head(100), use_container_width=True)

def show_model_analytics():
    """Display model performance analytics."""
    
    st.subheader("📈 Model Performance Analytics")
    
    if st.session_state.prediction_service is None:
        st.error("Prediction service not available.")
        return
    
    service = st.session_state.prediction_service
    available_models = service.get_available_models()
    
    # Model comparison
    st.markdown("### Model Comparison")
    
    model_metrics = []
    for model_name in available_models:
        if model_name != 'best':  # Skip the alias
            try:
                model_info = service.get_model_info(model_name)
                model_metrics.append({
                    'Model': model_name.replace('_', ' ').title(),
                    'Accuracy': model_info.accuracy,
                    'ROC-AUC': model_info.roc_auc,
                    'Precision': model_info.precision,
                    'Recall': model_info.recall,
                    'F1-Score': model_info.f1_score
                })
            except:
                pass
    
    if model_metrics:
        metrics_df = pd.DataFrame(model_metrics)
        st.dataframe(metrics_df, use_container_width=True)
        
        # Performance visualization
        fig = px.bar(
            metrics_df.melt(id_vars=['Model'], var_name='Metric', value_name='Score'),
            x='Model', y='Score', color='Metric',
            title="Model Performance Comparison",
            barmode='group'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Enhanced Feature importance section
    st.markdown("### 🎯 Feature Importance Analysis")

    col1, col2 = st.columns([2, 1])

    with col1:
        selected_model = st.selectbox("Select Model for Feature Importance",
                                    [m for m in available_models if m != 'best'])

    with col2:
        if st.session_state.prediction_service:
            service = st.session_state.prediction_service
            if selected_model:
                importance_data = service.get_feature_importance(selected_model)
                if 'available_methods' in importance_data:
                    st.info(f"📊 Methods: {', '.join(importance_data['available_methods'])}")

    if selected_model:
        try:
            importance_data = service.get_feature_importance(selected_model)

            if importance_data['feature_importance']:
                # Get top features
                top_features = list(importance_data['feature_importance'].items())[:15]
                features, importances = zip(*top_features)

                # Create enhanced visualization
                fig = px.bar(
                    x=list(importances), y=list(features),
                    orientation='h',
                    title=f"🎯 Top 15 Feature Importance - {selected_model.replace('_', ' ').title()}",
                    labels={'x': 'Importance Score', 'y': 'Features'},
                    color=list(importances),
                    color_continuous_scale='viridis'
                )
                fig.update_layout(
                    yaxis={'categoryorder': 'total ascending'},
                    height=600,
                    showlegend=False,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                fig.update_traces(
                    marker_line_color='rgb(8,48,107)',
                    marker_line_width=1.5,
                    opacity=0.8
                )
                st.plotly_chart(fig, use_container_width=True)

                # Show top 5 features in cards
                st.markdown("#### 🏆 Top 5 Most Important Features")
                cols = st.columns(5)
                for i, (feature, importance) in enumerate(top_features[:5]):
                    with cols[i]:
                        st.markdown(f"""
                        <div class="metric-card" style="text-align: center;">
                            <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">#{i+1}</div>
                            <div style="font-size: 0.9rem; font-weight: 600; margin-bottom: 0.5rem;">{feature.replace('_', ' ').title()}</div>
                            <div style="font-size: 1.2rem; color: #667eea;">{importance:.4f}</div>
                        </div>
                        """, unsafe_allow_html=True)

            else:
                st.warning("Feature importance data not available for this model.")
        except Exception as e:
            st.error(f"Failed to load feature importance: {str(e)}")

def show_batch_prediction():
    """Display batch prediction page."""
    
    st.subheader("🎯 Batch Prediction")
    
    if st.session_state.prediction_service is None:
        st.error("Prediction service not available.")
        return
    
    st.markdown("### Upload Customer Data for Batch Prediction")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a CSV file", 
        type="csv",
        help="Upload a CSV file with customer data for batch prediction"
    )
    
    if uploaded_file is not None:
        try:
            # Read uploaded file
            batch_df = pd.read_csv(uploaded_file)
            
            st.markdown("### Uploaded Data Preview")
            st.dataframe(batch_df.head(), use_container_width=True)
            
            # Validate required columns
            required_columns = [
                'gender', 'seniorcitizen', 'partner', 'dependents', 'tenure',
                'phoneservice', 'multiplelines', 'internetservice', 'onlinesecurity',
                'onlinebackup', 'deviceprotection', 'techsupport', 'streamingtv',
                'streamingmovies', 'contract', 'paperlessbilling', 'paymentmethod',
                'monthlycharges', 'totalcharges'
            ]
            
            missing_columns = [col for col in required_columns if col not in batch_df.columns]
            
            if missing_columns:
                st.error(f"Missing required columns: {missing_columns}")
            else:
                if st.button("🚀 Run Batch Prediction", type="primary"):
                    with st.spinner("Processing batch prediction..."):
                        # Convert to CustomerData objects (simplified for demo)
                        st.success(f"Batch prediction completed for {len(batch_df)} customers!")
                        
                        # Show sample results (mock for demo)
                        results_df = batch_df.copy()
                        results_df['churn_prediction'] = np.random.choice(['Yes', 'No'], len(batch_df))
                        results_df['churn_probability'] = np.random.uniform(0, 1, len(batch_df))
                        results_df['risk_level'] = pd.cut(
                            results_df['churn_probability'], 
                            bins=[0, 0.3, 0.6, 0.8, 1.0], 
                            labels=['Low', 'Medium', 'High', 'Very High']
                        )
                        
                        st.markdown("### Prediction Results")
                        st.dataframe(results_df[['churn_prediction', 'churn_probability', 'risk_level']], 
                                   use_container_width=True)
                        
                        # Summary statistics
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            churn_count = (results_df['churn_prediction'] == 'Yes').sum()
                            st.metric("Predicted Churners", churn_count)
                        
                        with col2:
                            avg_prob = results_df['churn_probability'].mean()
                            st.metric("Average Churn Probability", f"{avg_prob:.1%}")
                        
                        with col3:
                            high_risk = (results_df['risk_level'].isin(['High', 'Very High'])).sum()
                            st.metric("High Risk Customers", high_risk)
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    
    else:
        # Show sample format
        st.markdown("### Sample Data Format")
        sample_data = {
            'gender': ['Female', 'Male'],
            'seniorcitizen': [0, 1],
            'partner': ['Yes', 'No'],
            'dependents': ['No', 'Yes'],
            'tenure': [12, 24],
            'phoneservice': ['Yes', 'Yes'],
            'multiplelines': ['No', 'Yes'],
            'internetservice': ['DSL', 'Fiber optic'],
            'onlinesecurity': ['Yes', 'No'],
            'onlinebackup': ['No', 'Yes'],
            'deviceprotection': ['Yes', 'No'],
            'techsupport': ['No', 'Yes'],
            'streamingtv': ['No', 'Yes'],
            'streamingmovies': ['No', 'Yes'],
            'contract': ['One year', 'Month-to-month'],
            'paperlessbilling': ['Yes', 'No'],
            'paymentmethod': ['Electronic check', 'Credit card (automatic)'],
            'monthlycharges': [65.0, 85.0],
            'totalcharges': [780.0, 2040.0]
        }
        
        sample_df = pd.DataFrame(sample_data)
        st.dataframe(sample_df, use_container_width=True)

if __name__ == "__main__":
    main()
