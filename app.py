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

# Page configuration
st.set_page_config(
    page_title="🎮 Cyberpunk Churn Prediction",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cyberpunk CSS Theme
st.markdown("""
<style>
    /* Import Cyberpunk Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;600;700;800;900&family=Rajdhani:wght@300;400;500;600;700&display=swap');
    
    /* Global Dark Theme */
    .stApp {
        font-family: 'Rajdhani', sans-serif;
        background: #0d1117;
        color: #ffffff;
    }
    
    /* Main Container */
    .main .block-container {
        background: #1a1a1a;
        border-radius: 20px;
        border: 2px solid #00ffff;
        box-shadow: 0 0 30px rgba(0, 255, 255, 0.3);
        padding: 2rem;
        margin: 1rem;
    }
    
    /* Cyberpunk Header */
    .main-header {
        font-family: 'Orbitron', monospace;
        font-size: 3.5rem;
        font-weight: 900;
        color: #00ffff;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 0 0 10px #00ffff, 0 0 20px #00ffff, 0 0 30px #00ffff;
        letter-spacing: 2px;
        animation: neonGlow 2s ease-in-out infinite alternate;
    }
    
    @keyframes neonGlow {
        0% { text-shadow: 0 0 10px #00ffff, 0 0 20px #00ffff, 0 0 30px #00ffff; }
        100% { text-shadow: 0 0 20px #00ffff, 0 0 30px #00ffff, 0 0 40px #00ffff, 0 0 50px #ff00ff; }
    }
    
    /* Metric Cards */
    .metric-card {
        background: #2d2d2d;
        padding: 2rem;
        border-radius: 16px;
        border: 2px solid #00ffff;
        box-shadow: 0 0 20px rgba(0, 255, 255, 0.3);
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        border-color: #ff00ff;
        box-shadow: 0 0 30px rgba(255, 0, 255, 0.5);
        transform: translateY(-5px);
    }
    
    .metric-value {
        font-family: 'Orbitron', monospace;
        font-size: 2.5rem;
        font-weight: 700;
        color: #00ffff;
        text-shadow: 0 0 10px #00ffff;
    }
    
    .metric-label {
        font-size: 1rem;
        color: #e6e6e6;
        text-transform: uppercase;
        letter-spacing: 1px;
        text-shadow: 0 0 5px #ff00ff;
    }
    
    /* Prediction Results */
    .prediction-result {
        font-family: 'Orbitron', monospace;
        font-size: 1.8rem;
        font-weight: 600;
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        margin: 2rem 0;
        border: 3px solid;
        position: relative;
        overflow: hidden;
    }
    
    .churn-yes {
        background: #1a1a1a;
        color: #ff0040;
        border-color: #ff0040;
        box-shadow: 0 0 30px rgba(255, 0, 64, 0.5);
        text-shadow: 0 0 10px #ff0040;
    }
    
    .churn-no {
        background: #1a1a1a;
        color: #00ff00;
        border-color: #00ff00;
        box-shadow: 0 0 30px rgba(0, 255, 0, 0.5);
        text-shadow: 0 0 10px #00ff00;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: #0d1117;
        border-right: 2px solid #00ffff;
        box-shadow: 2px 0 20px rgba(0, 255, 255, 0.3);
    }
    
    /* Buttons */
    .stButton > button {
        background: #1a1a1a;
        color: #00ffff;
        border: 2px solid #00ffff;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-family: 'Rajdhani', sans-serif;
        font-weight: 600;
        text-shadow: 0 0 5px #00ffff;
        text-transform: uppercase;
        letter-spacing: 1px;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: #2d2d2d;
        border-color: #ff00ff;
        color: #ff00ff;
        box-shadow: 0 0 25px rgba(255, 0, 255, 0.5);
        text-shadow: 0 0 10px #ff00ff;
        transform: translateY(-2px);
    }
    
    /* Text styling */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        text-shadow: 0 0 5px #00ffff;
        font-family: 'Orbitron', monospace !important;
    }
    
    /* Success/Error messages */
    .stSuccess {
        background: #1a1a1a;
        border: 2px solid #00ff00;
        color: #00ff00;
        box-shadow: 0 0 15px rgba(0, 255, 0, 0.3);
        text-shadow: 0 0 5px #00ff00;
    }
    
    .stError {
        background: #1a1a1a;
        border: 2px solid #ff0040;
        color: #ff0040;
        box-shadow: 0 0 15px rgba(255, 0, 64, 0.3);
        text-shadow: 0 0 5px #ff0040;
    }
    
    .stWarning {
        background: #1a1a1a;
        border: 2px solid #ffff00;
        color: #ffff00;
        box-shadow: 0 0 15px rgba(255, 255, 0, 0.3);
        text-shadow: 0 0 5px #ffff00;
    }
    
    .stInfo {
        background: #1a1a1a;
        border: 2px solid #00ffff;
        color: #00ffff;
        box-shadow: 0 0 15px rgba(0, 255, 255, 0.3);
        text-shadow: 0 0 5px #00ffff;
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
        🎮 CYBERPUNK CHURN PREDICTION
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 3rem; color: #e6e6e6; font-size: 1.2rem;">
        Advanced Machine Learning Dashboard with Cyberpunk Aesthetics<br>
        <small style="color: #ff00ff;">Powered by AI • Real-time Predictions • Neon Interface</small>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <h2 style="color: white; margin-bottom: 0;">🧭 NAVIGATION</h2>
            <p style="color: rgba(255,255,255,0.8); font-size: 0.9rem;">Explore the Cyberpunk Dashboard</p>
        </div>
        """, unsafe_allow_html=True)
        
        page = st.selectbox(
            "Choose Module:",
            ["🏠 Dashboard", "🎯 Prediction", "📊 Data Explorer", "🎮 Theme Demo"],
            help="Navigate between different sections"
        )
        
        st.markdown("---")
        st.markdown("### 📊 System Status")
        st.success("✅ Cyberpunk Theme Active")
        st.success("✅ Dashboard Online")
        st.info("🎮 Gaming Mode Enabled")
    
    # Main content based on page selection
    if page == "🏠 Dashboard":
        show_dashboard()
    elif page == "🎯 Prediction":
        show_prediction()
    elif page == "📊 Data Explorer":
        show_data_explorer()
    elif page == "🎮 Theme Demo":
        show_theme_demo()

def show_dashboard():
    """Show the main dashboard."""
    
    st.markdown("### 📊 Cyberpunk Dashboard Overview")
    
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
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{total_customers:,}</div>
            <div class="metric-label">Total Customers</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{churn_rate:.1f}%</div>
            <div class="metric-label">Churn Rate</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{avg_tenure:.1f}</div>
            <div class="metric-label">Avg Tenure</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">${avg_charges:.0f}</div>
            <div class="metric-label">Avg Charges</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Charts
    st.markdown("### 📈 Cyberpunk Data Visualization")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Churn distribution
        churn_counts = df['Churn'].value_counts()
        fig = px.pie(values=churn_counts.values, names=churn_counts.index,
                     title='🎮 Churn Distribution',
                     color_discrete_sequence=['#00ffff', '#ff00ff'])
        
        fig.update_layout(
            plot_bgcolor='#1a1a1a',
            paper_bgcolor='#1a1a1a',
            font_color='#ffffff',
            title_font_color='#00ffff'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Monthly charges distribution
        fig = px.histogram(df, x='MonthlyCharges', color='Churn',
                          title='🎮 Monthly Charges by Churn',
                          color_discrete_sequence=['#00ffff', '#ff00ff'])
        
        fig.update_layout(
            plot_bgcolor='#1a1a1a',
            paper_bgcolor='#1a1a1a',
            font_color='#ffffff',
            title_font_color='#00ffff'
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
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner = st.selectbox("Partner", ["Yes", "No"])
        
        with col2:
            dependents = st.selectbox("Dependents", ["No", "Yes"])
            tenure = st.slider("Tenure (months)", 0, 72, 32)
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        
        with col3:
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        
        col1, col2 = st.columns(2)
        
        with col1:
            payment_method = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check", 
                "Bank transfer (automatic)", "Credit card (automatic)"
            ])
        
        with col2:
            monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 64.76)
        
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
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Contract type analysis
        contract_churn = df.groupby(['Contract', 'Churn']).size().unstack()
        fig = px.bar(contract_churn, title='🎮 Churn by Contract Type',
                     color_discrete_sequence=['#00ffff', '#ff00ff'])
        
        fig.update_layout(
            plot_bgcolor='#1a1a1a',
            paper_bgcolor='#1a1a1a',
            font_color='#ffffff',
            title_font_color='#00ffff'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Tenure vs Monthly Charges
        fig = px.scatter(df, x='Tenure', y='MonthlyCharges', color='Churn',
                        title='🎮 Tenure vs Monthly Charges',
                        color_discrete_sequence=['#00ffff', '#ff00ff'])
        
        fig.update_layout(
            plot_bgcolor='#1a1a1a',
            paper_bgcolor='#1a1a1a',
            font_color='#ffffff',
            title_font_color='#00ffff'
        )
        
        st.plotly_chart(fig, use_container_width=True)

def show_theme_demo():
    """Show theme demonstration."""
    
    st.markdown("### 🎮 Cyberpunk Theme Demonstration")
    
    st.success("✅ Success message with neon green glow")
    st.error("❌ Error message with neon red glow")
    st.warning("⚠️ Warning message with neon yellow glow")
    st.info("ℹ️ Info message with neon cyan glow")
    
    st.markdown("#### 🎨 Color Palette")
    
    colors = [
        ("Neon Cyan", "#00ffff", "Primary accent color"),
        ("Neon Purple", "#ff00ff", "Secondary highlights"),
        ("Electric Green", "#00ff00", "Success states"),
        ("Hot Pink", "#ff0040", "Error states"),
        ("Electric Yellow", "#ffff00", "Warning states")
    ]
    
    for name, hex_code, description in colors:
        st.markdown(f"""
        <div style="background: {hex_code}; color: #000; padding: 1rem; margin: 0.5rem 0; border-radius: 8px; font-weight: bold;">
            {name} ({hex_code}) - {description}
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("#### 🎬 Interactive Elements")
    
    if st.button("🚀 Cyberpunk Button"):
        st.balloons()
        st.success("🎮 Cyberpunk button activated!")
    
    st.markdown("#### 📊 Sample Chart")
    
    # Sample data for chart
    sample_data = pd.DataFrame({
        'Month': pd.date_range('2024-01-01', periods=12, freq='M'),
        'Churn_Rate': np.random.uniform(20, 35, 12),
        'Revenue': np.random.uniform(50000, 80000, 12)
    })
    
    fig = px.line(sample_data, x='Month', y='Churn_Rate',
                  title='🎮 Monthly Churn Rate Trend',
                  color_discrete_sequence=['#00ffff'])
    
    fig.update_layout(
        plot_bgcolor='#1a1a1a',
        paper_bgcolor='#1a1a1a',
        font_color='#ffffff',
        title_font_color='#00ffff'
    )
    
    fig.update_traces(line=dict(width=3, color='#00ffff'))
    
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
