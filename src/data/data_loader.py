"""
Data loading and initial processing module for churn prediction.
"""

import pandas as pd
import numpy as np
import requests
from pathlib import Path
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from ..utils.config import config
from ..utils.logger import get_logger
from ..utils.exceptions import DataLoadingError, DataValidationError
from ..utils.validators import DataValidator

logger = get_logger(__name__)


class DataLoader:
    """Class for loading and initial processing of churn data."""
    
    def __init__(self):
        self.config = config
        self.validator = DataValidator()
        self.raw_data_path = Path(self.config.project_root) / "data" / "raw"
        self.processed_data_path = Path(self.config.project_root) / "data" / "processed"
        
        # Ensure directories exist
        self.raw_data_path.mkdir(parents=True, exist_ok=True)
        self.processed_data_path.mkdir(parents=True, exist_ok=True)
    
    def download_dataset(self, force_download: bool = False) -> Path:
        """
        Download the Telco Customer Churn dataset.
        
        Args:
            force_download: Whether to force re-download if file exists
            
        Returns:
            Path to downloaded dataset
            
        Raises:
            DataLoadingError: If download fails
        """
        dataset_url = self.config.get('data.dataset_url')
        filename = "telco_customer_churn.csv"
        file_path = self.raw_data_path / filename
        
        if file_path.exists() and not force_download:
            logger.info(f"Dataset already exists at {file_path}")
            return file_path
        
        try:
            logger.info(f"Downloading dataset from {dataset_url}")
            response = requests.get(dataset_url, timeout=30)
            response.raise_for_status()
            
            with open(file_path, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"Dataset downloaded successfully to {file_path}")
            return file_path
            
        except requests.RequestException as e:
            raise DataLoadingError(f"Failed to download dataset: {str(e)}")
        except IOError as e:
            raise DataLoadingError(f"Failed to save dataset: {str(e)}")
    
    def load_raw_data(self, file_path: Optional[Path] = None) -> pd.DataFrame:
        """
        Load raw data from CSV file.
        
        Args:
            file_path: Path to CSV file. If None, downloads dataset
            
        Returns:
            Raw DataFrame
            
        Raises:
            DataLoadingError: If loading fails
        """
        if file_path is None:
            file_path = self.download_dataset()
        
        try:
            logger.info(f"Loading data from {file_path}")
            df = pd.read_csv(file_path)
            logger.info(f"Data loaded successfully. Shape: {df.shape}")
            return df
            
        except Exception as e:
            raise DataLoadingError(f"Failed to load data from {file_path}: {str(e)}")
    
    def initial_data_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform initial data cleaning.
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        logger.info("Starting initial data cleaning")
        df_clean = df.copy()
        
        # Convert TotalCharges to numeric (it's often stored as string)
        if 'TotalCharges' in df_clean.columns:
            # Replace empty strings with NaN
            df_clean['TotalCharges'] = df_clean['TotalCharges'].replace(' ', np.nan)
            df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')
        
        # Standardize column names (remove spaces, make lowercase)
        df_clean.columns = df_clean.columns.str.replace(' ', '_').str.lower()
        
        # Standardize categorical values
        categorical_columns = df_clean.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if col != 'customerid':  # Don't modify customer ID
                df_clean[col] = df_clean[col].str.strip()
        
        # Convert binary categorical variables to numeric
        binary_mappings = {
            'yes': 1, 'no': 0,
            'male': 1, 'female': 0,
            'true': 1, 'false': 0
        }

        for col in categorical_columns:
            if col != 'customerid':
                # Convert to lowercase for comparison
                df_clean[col] = df_clean[col].astype(str).str.lower()
                unique_values = df_clean[col].dropna().unique()

                # Handle binary mappings
                if len(unique_values) == 2 and all(val in binary_mappings for val in unique_values):
                    df_clean[col] = df_clean[col].map(binary_mappings)

        # Handle SeniorCitizen column (should be binary)
        if 'seniorcitizen' in df_clean.columns:
            df_clean['seniorcitizen'] = df_clean['seniorcitizen'].astype(int)
        
        logger.info(f"Initial cleaning completed. Shape: {df_clean.shape}")
        return df_clean
    
    def validate_data(self, df: pd.DataFrame) -> dict:
        """
        Validate the loaded data.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Validation results
        """
        logger.info("Validating data quality")
        
        # Get expected columns from config
        numerical_features = self.config.get('features.numerical_features', [])
        categorical_features = self.config.get('features.categorical_features', [])
        target_column = self.config.get('data.target_column', 'churn')
        
        # Validate DataFrame structure
        results = self.validator.validate_dataframe(
            df, 
            required_columns=numerical_features + categorical_features + [target_column],
            min_rows=1000  # Minimum 1000 rows for meaningful analysis
        )
        
        # Validate target column
        target_results = self.validator.validate_target_column(
            df, 
            target_column.lower(),
            expected_values=[0, 1, 'yes', 'no']
        )
        
        # Validate features
        feature_results = self.validator.validate_feature_columns(
            df,
            [col.lower() for col in numerical_features],
            [col.lower() for col in categorical_features]
        )
        
        # Combine results
        combined_results = {
            'dataframe': results,
            'target': target_results,
            'features': feature_results,
            'overall_valid': all([
                results['is_valid'],
                target_results['is_valid'],
                feature_results['is_valid']
            ])
        }
        
        if combined_results['overall_valid']:
            logger.info("Data validation passed successfully")
        else:
            logger.warning("Data validation completed with issues")
        
        return combined_results
    
    def get_data_info(self, df: pd.DataFrame) -> dict:
        """
        Get comprehensive information about the dataset.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with dataset information
        """
        info = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
            'duplicate_rows': df.duplicated().sum(),
            'memory_usage': df.memory_usage(deep=True).sum(),
        }
        
        # Numerical columns statistics
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            info['numerical_stats'] = df[numerical_cols].describe().to_dict()
        
        # Categorical columns statistics
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            info['categorical_stats'] = {}
            for col in categorical_cols:
                info['categorical_stats'][col] = {
                    'unique_count': df[col].nunique(),
                    'unique_values': df[col].value_counts().head(10).to_dict()
                }
        
        return info
    
    def save_processed_data(self, df: pd.DataFrame, filename: str = "processed_data.csv"):
        """
        Save processed data to CSV file.
        
        Args:
            df: DataFrame to save
            filename: Name of output file
        """
        output_path = self.processed_data_path / filename
        df.to_csv(output_path, index=False)
        logger.info(f"Processed data saved to {output_path}")
    
    def load_and_process(self) -> Tuple[pd.DataFrame, dict]:
        """
        Complete data loading and processing pipeline.
        
        Returns:
            Tuple of (processed DataFrame, validation results)
        """
        # Load raw data
        raw_df = self.load_raw_data()
        
        # Initial cleaning
        clean_df = self.initial_data_cleaning(raw_df)
        
        # Validate data
        validation_results = self.validate_data(clean_df)
        
        # Get data info
        data_info = self.get_data_info(clean_df)
        logger.info(f"Dataset info: {data_info['shape']} shape, {data_info['missing_values']} missing values")
        
        # Save processed data
        self.save_processed_data(clean_df)
        
        return clean_df, validation_results


def main():
    """Main function for data loading."""
    loader = DataLoader()
    df, validation = loader.load_and_process()
    
    print(f"Data loaded successfully!")
    print(f"Shape: {df.shape}")
    print(f"Validation passed: {validation['overall_valid']}")
    
    return df, validation


if __name__ == "__main__":
    main()
