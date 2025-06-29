# Customer Churn Prediction - Advanced ML Solution

A comprehensive, production-ready machine learning solution for customer churn prediction featuring advanced ML techniques, interactive dashboards, and production APIs.

## 🚀 Features

### Advanced Machine Learning
- **Multiple Models**: Logistic Regression, XGBoost, LightGBM, CatBoost, Random Forest
- **Ensemble Methods**: Stacking and blending techniques
- **Bayesian Optimization**: Using Optuna for hyperparameter tuning
- **Advanced Preprocessing**: Multiple imputation, outlier detection, feature scaling
- **Feature Engineering**: Automated feature creation and selection
- **Model Interpretability**: SHAP analysis and feature importance

### Production-Ready Components
- **FastAPI REST API**: Complete with validation, versioning, and documentation
- **Interactive Dashboard**: Multi-page Streamlit application
- **MLOps Integration**: Model monitoring, drift detection, automated retraining
- **Comprehensive Testing**: Unit tests with pytest
- **Documentation**: Auto-generated with Sphinx

### Data Analysis & Visualization
- **Automated EDA**: Using pandas-profiling and sweetviz
- **Advanced Visualizations**: Interactive plots with Plotly
- **Statistical Analysis**: Comprehensive model evaluation metrics
- **Business Insights**: Actionable recommendations

## 📁 Project Structure

```
churn-prediction/
├── src/
│   ├── data/           # Data processing modules
│   ├── features/       # Feature engineering
│   ├── models/         # ML models and training
│   ├── api/           # FastAPI application
│   ├── dashboard/     # Streamlit dashboard
│   └── utils/         # Utilities and helpers
├── data/
│   ├── raw/           # Raw data files
│   ├── processed/     # Processed data
│   └── external/      # External data sources
├── notebooks/         # Jupyter notebooks for exploration
├── tests/            # Unit tests
├── docs/             # Documentation
├── config/           # Configuration files
├── deployment/       # Deployment scripts
└── logs/            # Application logs
```

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip or conda package manager

### Setup Environment

1. **Clone the repository**
```bash
git clone <repository-url>
cd churn-prediction
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Install the package in development mode**
```bash
pip install -e .
```

## 🚀 Quick Start

### 1. Data Preparation
```bash
python -m src.data.data_loader
```

### 2. Train Models
```bash
python -m src.models.train
```

### 3. Start API Server
```bash
python -m src.api.main
```

### 4. Launch Dashboard
```bash
python -m src.dashboard.main
```

## 📊 Usage Examples

### API Usage
```python
import requests

# Predict churn for a customer
data = {
    "tenure": 12,
    "MonthlyCharges": 65.5,
    "TotalCharges": 786.0,
    "Contract": "Month-to-month",
    "PaymentMethod": "Electronic check"
}

response = requests.post("http://localhost:8000/predict", json=data)
prediction = response.json()
```

### Dashboard Features
- **Data Exploration**: Interactive visualizations and statistics
- **Model Comparison**: Performance metrics across all models
- **Prediction Interface**: Real-time churn prediction
- **SHAP Analysis**: Model interpretability and feature importance

## 🧪 Testing

Run the test suite:
```bash
pytest tests/ -v --cov=src
```

## 📈 Model Performance

The solution achieves:
- **ROC-AUC**: > 0.85 on test set
- **Precision**: > 0.80 for churn prediction
- **Recall**: > 0.75 for churn identification
- **F1-Score**: > 0.77 overall performance

## 🔧 Configuration

All configuration is managed through `config/config.yaml`:
- Model parameters and hyperparameter ranges
- Data processing settings
- API and dashboard configurations
- Logging and monitoring settings

## 📚 Documentation

Generate documentation:
```bash
cd docs
make html
```

## 🐳 Docker Deployment

Build and run with Docker:
```bash
docker build -t churn-prediction .
docker run -p 8000:8000 -p 8501:8501 churn-prediction
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Telco Customer Churn dataset from IBM
- Open source ML libraries and frameworks
- Community contributions and feedback
