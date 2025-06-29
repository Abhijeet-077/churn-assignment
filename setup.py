"""
Setup script for the churn prediction project.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="churn-prediction",
    version="1.0.0",
    author="Churn Prediction Team",
    author_email="team@churnprediction.com",
    description="Advanced machine learning solution for customer churn prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/churn-prediction",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas>=2.1.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "xgboost>=2.0.0",
        "lightgbm>=4.1.0",
        "catboost>=1.2.0",
        "optuna>=3.4.0",
        "shap>=0.43.0",
        "fastapi>=0.104.0",
        "streamlit>=1.28.0",
        "plotly>=5.17.0",
        "pydantic>=2.5.0",
        "pyyaml>=6.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.11.0",
            "isort>=5.12.0",
            "flake8>=6.1.0",
            "mypy>=1.7.0",
        ],
        "docs": [
            "sphinx>=7.2.0",
            "sphinx-rtd-theme>=1.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "churn-train=src.models.train:main",
            "churn-api=src.api.main:main",
            "churn-dashboard=src.dashboard.main:main",
        ],
    },
)
