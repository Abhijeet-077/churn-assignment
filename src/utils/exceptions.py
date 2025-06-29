"""
Custom exceptions for the churn prediction project.
"""


class ChurnPredictionError(Exception):
    """Base exception for churn prediction project."""
    pass


class DataError(ChurnPredictionError):
    """Exception raised for data-related errors."""
    pass


class DataValidationError(DataError):
    """Exception raised when data validation fails."""
    pass


class DataLoadingError(DataError):
    """Exception raised when data loading fails."""
    pass


class DataPreprocessingError(DataError):
    """Exception raised during data preprocessing."""
    pass


class FeatureEngineeringError(ChurnPredictionError):
    """Exception raised during feature engineering."""
    pass


class ModelError(ChurnPredictionError):
    """Exception raised for model-related errors."""
    pass


class ModelTrainingError(ModelError):
    """Exception raised during model training."""
    pass


class ModelPredictionError(ModelError):
    """Exception raised during model prediction."""
    pass


class ModelValidationError(ModelError):
    """Exception raised when model validation fails."""
    pass


class ConfigurationError(ChurnPredictionError):
    """Exception raised for configuration-related errors."""
    pass


class APIError(ChurnPredictionError):
    """Exception raised for API-related errors."""
    pass


class DashboardError(ChurnPredictionError):
    """Exception raised for dashboard-related errors."""
    pass
