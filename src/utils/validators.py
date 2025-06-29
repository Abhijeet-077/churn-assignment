"""
Data validation utilities for the churn prediction project.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from .exceptions import DataValidationError
from .logger import get_logger

logger = get_logger(__name__)


class DataValidator:
    """Class for validating data quality and integrity."""
    
    def __init__(self):
        self.validation_results = {}
    
    def validate_dataframe(
        self, 
        df: pd.DataFrame, 
        required_columns: Optional[List[str]] = None,
        min_rows: int = 1,
        max_missing_ratio: float = 0.5
    ) -> Dict[str, Any]:
        """
        Validate a pandas DataFrame.
        
        Args:
            df: DataFrame to validate
            required_columns: List of required column names
            min_rows: Minimum number of rows required
            max_missing_ratio: Maximum ratio of missing values allowed per column
            
        Returns:
            Dictionary with validation results
            
        Raises:
            DataValidationError: If validation fails
        """
        results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'info': {}
        }
        
        # Check if DataFrame is empty
        if df.empty:
            results['is_valid'] = False
            results['errors'].append("DataFrame is empty")
            return results
        
        # Check minimum rows
        if len(df) < min_rows:
            results['is_valid'] = False
            results['errors'].append(f"DataFrame has {len(df)} rows, minimum required: {min_rows}")
        
        # Check required columns
        if required_columns:
            missing_columns = set(required_columns) - set(df.columns)
            if missing_columns:
                results['is_valid'] = False
                results['errors'].append(f"Missing required columns: {list(missing_columns)}")
        
        # Check for excessive missing values
        missing_ratios = df.isnull().sum() / len(df)
        high_missing_cols = missing_ratios[missing_ratios > max_missing_ratio].index.tolist()
        
        if high_missing_cols:
            results['warnings'].append(
                f"Columns with high missing value ratio (>{max_missing_ratio}): {high_missing_cols}"
            )
        
        # Check for duplicate rows
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            results['warnings'].append(f"Found {duplicate_count} duplicate rows")
        
        # Check data types
        results['info']['shape'] = df.shape
        results['info']['dtypes'] = df.dtypes.to_dict()
        results['info']['missing_values'] = df.isnull().sum().to_dict()
        results['info']['duplicate_rows'] = duplicate_count
        
        # Log results
        if results['is_valid']:
            logger.info(f"DataFrame validation passed. Shape: {df.shape}")
        else:
            logger.error(f"DataFrame validation failed: {results['errors']}")
        
        if results['warnings']:
            logger.warning(f"DataFrame validation warnings: {results['warnings']}")
        
        self.validation_results['dataframe'] = results
        return results
    
    def validate_target_column(
        self, 
        df: pd.DataFrame, 
        target_column: str,
        expected_values: Optional[List[Any]] = None
    ) -> Dict[str, Any]:
        """
        Validate target column for classification.
        
        Args:
            df: DataFrame containing target column
            target_column: Name of target column
            expected_values: Expected unique values in target column
            
        Returns:
            Dictionary with validation results
        """
        results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'info': {}
        }
        
        # Check if target column exists
        if target_column not in df.columns:
            results['is_valid'] = False
            results['errors'].append(f"Target column '{target_column}' not found")
            return results
        
        target_series = df[target_column]
        
        # Check for missing values in target
        missing_count = target_series.isnull().sum()
        if missing_count > 0:
            results['is_valid'] = False
            results['errors'].append(f"Target column has {missing_count} missing values")
        
        # Get unique values
        unique_values = target_series.dropna().unique()
        results['info']['unique_values'] = unique_values.tolist()
        results['info']['value_counts'] = target_series.value_counts().to_dict()
        
        # Check expected values
        if expected_values:
            unexpected_values = set(unique_values) - set(expected_values)
            if unexpected_values:
                results['warnings'].append(f"Unexpected values in target: {list(unexpected_values)}")
        
        # Check class balance
        value_counts = target_series.value_counts()
        if len(value_counts) > 1:
            min_class_ratio = value_counts.min() / value_counts.sum()
            if min_class_ratio < 0.05:  # Less than 5%
                results['warnings'].append(f"Severe class imbalance detected. Minimum class ratio: {min_class_ratio:.3f}")
            elif min_class_ratio < 0.2:  # Less than 20%
                results['warnings'].append(f"Class imbalance detected. Minimum class ratio: {min_class_ratio:.3f}")
        
        logger.info(f"Target column validation completed. Unique values: {len(unique_values)}")
        
        self.validation_results['target'] = results
        return results
    
    def validate_feature_columns(
        self, 
        df: pd.DataFrame, 
        numerical_features: List[str],
        categorical_features: List[str]
    ) -> Dict[str, Any]:
        """
        Validate feature columns.
        
        Args:
            df: DataFrame containing features
            numerical_features: List of numerical feature names
            categorical_features: List of categorical feature names
            
        Returns:
            Dictionary with validation results
        """
        results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'info': {}
        }
        
        all_features = numerical_features + categorical_features
        
        # Check if all features exist
        missing_features = set(all_features) - set(df.columns)
        if missing_features:
            results['is_valid'] = False
            results['errors'].append(f"Missing feature columns: {list(missing_features)}")
        
        # Validate numerical features
        for feature in numerical_features:
            if feature in df.columns:
                series = df[feature]
                
                # Check if actually numerical
                if not pd.api.types.is_numeric_dtype(series):
                    results['warnings'].append(f"Feature '{feature}' is not numeric: {series.dtype}")
                
                # Check for infinite values
                if np.isinf(series).any():
                    results['warnings'].append(f"Feature '{feature}' contains infinite values")
                
                # Check for extreme outliers (beyond 5 standard deviations)
                if pd.api.types.is_numeric_dtype(series):
                    z_scores = np.abs((series - series.mean()) / series.std())
                    extreme_outliers = (z_scores > 5).sum()
                    if extreme_outliers > 0:
                        results['warnings'].append(
                            f"Feature '{feature}' has {extreme_outliers} extreme outliers (>5 std)"
                        )
        
        # Validate categorical features
        for feature in categorical_features:
            if feature in df.columns:
                series = df[feature]
                unique_count = series.nunique()
                
                # Check for high cardinality
                if unique_count > 50:
                    results['warnings'].append(
                        f"Feature '{feature}' has high cardinality: {unique_count} unique values"
                    )
                
                # Check for single value
                if unique_count == 1:
                    results['warnings'].append(f"Feature '{feature}' has only one unique value")
        
        results['info']['numerical_features'] = len(numerical_features)
        results['info']['categorical_features'] = len(categorical_features)
        
        logger.info(f"Feature validation completed. Numerical: {len(numerical_features)}, Categorical: {len(categorical_features)}")
        
        self.validation_results['features'] = results
        return results
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of all validation results."""
        return self.validation_results


def validate_model_input(data: Dict[str, Any], required_features: List[str]) -> bool:
    """
    Validate input data for model prediction.
    
    Args:
        data: Input data dictionary
        required_features: List of required feature names
        
    Returns:
        True if valid, False otherwise
        
    Raises:
        DataValidationError: If validation fails
    """
    missing_features = set(required_features) - set(data.keys())
    if missing_features:
        raise DataValidationError(f"Missing required features: {list(missing_features)}")
    
    # Check for None values
    none_features = [k for k, v in data.items() if v is None and k in required_features]
    if none_features:
        raise DataValidationError(f"Features cannot be None: {none_features}")
    
    return True
