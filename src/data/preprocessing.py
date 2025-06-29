"""
Advanced data preprocessing pipeline for churn prediction.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

# Try to import category encoders, fallback if not available
try:
    import category_encoders as ce
    CATEGORY_ENCODERS_AVAILABLE = True
except ImportError:
    CATEGORY_ENCODERS_AVAILABLE = False
    ce = None
from typing import Dict, List, Tuple, Optional, Any
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from ..utils.config import config
from ..utils.logger import get_logger
from ..utils.exceptions import DataPreprocessingError
from ..utils.validators import DataValidator

logger = get_logger(__name__)


class AdvancedPreprocessor:
    """Advanced preprocessing pipeline with multiple strategies."""
    
    def __init__(self):
        self.config = config
        self.validator = DataValidator()
        self.preprocessing_config = self.config.preprocessing_config
        
        # Initialize components
        self.imputers = {}
        self.encoders = {}
        self.scalers = {}
        self.outlier_detectors = {}
        self.resampler = None
        
        # Store fitted transformers
        self.fitted_transformers = {}
        self.feature_names = []
        self.target_name = None
        
        # Paths for saving transformers
        self.artifacts_path = Path(self.config.project_root) / "artifacts" / "preprocessing"
        self.artifacts_path.mkdir(parents=True, exist_ok=True)
    
    def detect_outliers(self, df: pd.DataFrame, method: str = 'iqr') -> pd.DataFrame:
        """
        Detect outliers using various methods.
        
        Args:
            df: DataFrame to analyze
            method: Outlier detection method ('iqr', 'zscore', 'isolation_forest')
            
        Returns:
            DataFrame with outlier indicators
        """
        logger.info(f"Detecting outliers using {method} method")
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        outlier_df = df.copy()
        
        if method == 'iqr':
            for col in numerical_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                outlier_df[f'{col}_outlier_iqr'] = outlier_mask
        
        elif method == 'zscore':
            threshold = self.preprocessing_config.get('outlier_detection', {}).get('zscore_threshold', 3)
            for col in numerical_cols:
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outlier_mask = z_scores > threshold
                outlier_df[f'{col}_outlier_zscore'] = outlier_mask
        
        elif method == 'isolation_forest':
            contamination = self.preprocessing_config.get('outlier_detection', {}).get('isolation_contamination', 0.1)
            
            if len(numerical_cols) > 0:
                iso_forest = IsolationForest(contamination=contamination, random_state=42)
                outlier_predictions = iso_forest.fit_predict(df[numerical_cols].fillna(df[numerical_cols].median()))
                outlier_df['outlier_isolation_forest'] = outlier_predictions == -1
                
                self.outlier_detectors['isolation_forest'] = iso_forest
        
        return outlier_df
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'median') -> pd.DataFrame:
        """
        Handle missing values using various imputation strategies.
        
        Args:
            df: DataFrame with missing values
            strategy: Imputation strategy ('mean', 'median', 'mode', 'knn', 'iterative')
            
        Returns:
            DataFrame with imputed values
        """
        logger.info(f"Handling missing values using {strategy} strategy")
        
        df_imputed = df.copy()
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        # Handle numerical columns
        if len(numerical_cols) > 0:
            if strategy in ['mean', 'median']:
                imputer = SimpleImputer(strategy=strategy)
                df_imputed[numerical_cols] = imputer.fit_transform(df[numerical_cols])
                self.imputers[f'numerical_{strategy}'] = imputer
                
            elif strategy == 'knn':
                n_neighbors = self.preprocessing_config.get('imputation', {}).get('knn_neighbors', 5)
                imputer = KNNImputer(n_neighbors=n_neighbors)
                df_imputed[numerical_cols] = imputer.fit_transform(df[numerical_cols])
                self.imputers['numerical_knn'] = imputer
                
            elif strategy == 'iterative':
                imputer = IterativeImputer(random_state=42, max_iter=10)
                df_imputed[numerical_cols] = imputer.fit_transform(df[numerical_cols])
                self.imputers['numerical_iterative'] = imputer
        
        # Handle categorical columns
        if len(categorical_cols) > 0:
            categorical_strategy = self.preprocessing_config.get('imputation', {}).get('categorical_strategy', 'mode')
            
            if categorical_strategy == 'mode':
                imputer = SimpleImputer(strategy='most_frequent')
                df_imputed[categorical_cols] = imputer.fit_transform(df[categorical_cols])
                self.imputers['categorical_mode'] = imputer
            elif categorical_strategy == 'constant':
                imputer = SimpleImputer(strategy='constant', fill_value='Unknown')
                df_imputed[categorical_cols] = imputer.fit_transform(df[categorical_cols])
                self.imputers['categorical_constant'] = imputer
        
        logger.info(f"Missing value imputation completed. Remaining missing values: {df_imputed.isnull().sum().sum()}")
        return df_imputed
    
    def encode_categorical_features(self, df: pd.DataFrame, target_col: Optional[str] = None) -> pd.DataFrame:
        """
        Encode categorical features using various encoding techniques.

        Args:
            df: DataFrame with categorical features
            target_col: Target column for target encoding

        Returns:
            DataFrame with encoded features
        """
        logger.info("Encoding categorical features")

        df_encoded = df.copy()

        # First, handle basic binary mappings for common categorical values
        binary_mappings = {
            'yes': 1, 'no': 0,
            'male': 1, 'female': 0,
            'true': 1, 'false': 0
        }

        # Apply binary mappings to all object columns
        for col in df_encoded.select_dtypes(include=['object']).columns:
            if col not in ['customerid', target_col]:
                # Convert to lowercase for comparison
                df_encoded[col] = df_encoded[col].astype(str).str.lower()
                unique_vals = df_encoded[col].unique()

                # Check if it's a simple binary mapping
                if len(unique_vals) == 2 and all(val in binary_mappings for val in unique_vals):
                    df_encoded[col] = df_encoded[col].map(binary_mappings)

        # Get remaining categorical columns after binary encoding
        categorical_cols = df_encoded.select_dtypes(include=['object']).columns
        categorical_cols = [col for col in categorical_cols if col not in ['customerid', target_col]]
        
        encoding_config = self.preprocessing_config.get('encoding', {})
        
        # Target encoding (if category encoders available)
        target_encoding_features = encoding_config.get('target_encoding_features', [])
        if CATEGORY_ENCODERS_AVAILABLE and target_col and len(target_encoding_features) > 0:
            target_encoder = ce.TargetEncoder(cols=target_encoding_features)
            df_encoded = target_encoder.fit_transform(df_encoded, df_encoded[target_col])
            self.encoders['target'] = target_encoder
        elif target_col and len(target_encoding_features) > 0:
            logger.warning("Category encoders not available. Skipping target encoding.")

        # Binary encoding (if category encoders available)
        binary_encoding_features = encoding_config.get('binary_encoding_features', [])
        if CATEGORY_ENCODERS_AVAILABLE and len(binary_encoding_features) > 0:
            binary_encoder = ce.BinaryEncoder(cols=binary_encoding_features)
            df_encoded = binary_encoder.fit_transform(df_encoded)
            self.encoders['binary'] = binary_encoder
        elif len(binary_encoding_features) > 0:
            logger.warning("Category encoders not available. Skipping binary encoding.")
        
        # Frequency encoding
        frequency_encoding_features = encoding_config.get('frequency_encoding_features', [])
        for col in frequency_encoding_features:
            if col in df_encoded.columns:
                freq_map = df_encoded[col].value_counts().to_dict()
                df_encoded[f'{col}_frequency'] = df_encoded[col].map(freq_map)
                self.encoders[f'frequency_{col}'] = freq_map
        
        # One-hot encoding for ALL remaining categorical features
        remaining_categorical = [col for col in categorical_cols
                               if col not in target_encoding_features + binary_encoding_features + frequency_encoding_features]

        if len(remaining_categorical) > 0:
            logger.info(f"Applying one-hot encoding to columns: {remaining_categorical}")
            df_encoded = pd.get_dummies(df_encoded, columns=remaining_categorical, prefix=remaining_categorical, drop_first=True)

        # Ensure all columns are numeric (final check)
        for col in df_encoded.columns:
            if df_encoded[col].dtype == 'object':
                logger.warning(f"Column {col} is still object type, converting to category codes")
                df_encoded[col] = pd.Categorical(df_encoded[col]).codes
        
        logger.info(f"Categorical encoding completed. New shape: {df_encoded.shape}")
        return df_encoded
    
    def scale_features(self, df: pd.DataFrame, method: str = 'standard') -> pd.DataFrame:
        """
        Scale numerical features using various scaling methods.
        
        Args:
            df: DataFrame with numerical features
            method: Scaling method ('standard', 'robust', 'minmax')
            
        Returns:
            DataFrame with scaled features
        """
        logger.info(f"Scaling features using {method} method")
        
        df_scaled = df.copy()
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        # Remove target column from scaling
        target_col = self.config.get('data.target_column', 'churn')
        numerical_cols = [col for col in numerical_cols if col != target_col]
        
        if len(numerical_cols) == 0:
            logger.warning("No numerical columns found for scaling")
            return df_scaled
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise DataPreprocessingError(f"Unknown scaling method: {method}")
        
        df_scaled[numerical_cols] = scaler.fit_transform(df[numerical_cols])
        self.scalers[method] = scaler
        
        logger.info(f"Feature scaling completed using {method} method")
        return df_scaled

    def handle_class_imbalance(self, X: pd.DataFrame, y: pd.Series, method: str = 'smote') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Handle class imbalance using various resampling techniques.

        Args:
            X: Feature DataFrame
            y: Target Series
            method: Resampling method ('smote', 'adasyn', 'random_oversample', 'random_undersample')

        Returns:
            Tuple of (resampled X, resampled y)
        """
        logger.info(f"Handling class imbalance using {method} method")

        original_counts = y.value_counts()
        logger.info(f"Original class distribution: {original_counts.to_dict()}")

        sampling_strategy = self.preprocessing_config.get('resampling', {}).get('sampling_strategy', 'auto')

        if method == 'smote':
            resampler = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
        elif method == 'adasyn':
            resampler = ADASYN(sampling_strategy=sampling_strategy, random_state=42)
        elif method == 'random_oversample':
            resampler = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=42)
        elif method == 'random_undersample':
            resampler = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)
        else:
            raise DataPreprocessingError(f"Unknown resampling method: {method}")

        X_resampled, y_resampled = resampler.fit_resample(X, y)
        self.resampler = resampler

        new_counts = pd.Series(y_resampled).value_counts()
        logger.info(f"New class distribution: {new_counts.to_dict()}")

        return pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled)

    def fit_transform(self, df: pd.DataFrame, target_col: str = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Fit and transform the complete preprocessing pipeline.

        Args:
            df: Input DataFrame
            target_col: Target column name

        Returns:
            Tuple of (processed features, target)
        """
        logger.info("Starting complete preprocessing pipeline")

        if target_col is None:
            target_col = self.config.get('data.target_column', 'churn')

        self.target_name = target_col

        # Validate input data
        validation_results = self.validator.validate_dataframe(df, min_rows=100)
        if not validation_results['is_valid']:
            raise DataPreprocessingError(f"Input data validation failed: {validation_results['errors']}")

        # Separate features and target
        if target_col not in df.columns:
            raise DataPreprocessingError(f"Target column '{target_col}' not found in DataFrame")

        X = df.drop(columns=[target_col, 'customerid'] if 'customerid' in df.columns else [target_col])
        y = df[target_col]

        # Store original feature names
        self.feature_names = X.columns.tolist()

        # Step 1: Handle missing values
        imputation_strategy = self.preprocessing_config.get('imputation', {}).get('numerical_strategy', 'median')
        X = self.handle_missing_values(X, strategy=imputation_strategy)

        # Step 2: Detect and handle outliers
        outlier_methods = self.preprocessing_config.get('outlier_detection', {}).get('methods', ['iqr'])
        for method in outlier_methods:
            X_with_outliers = self.detect_outliers(X, method=method)
            # For now, just log outlier information (could implement removal/capping later)
            outlier_cols = [col for col in X_with_outliers.columns if 'outlier' in col]
            if outlier_cols:
                outlier_counts = X_with_outliers[outlier_cols].sum()
                logger.info(f"Outliers detected ({method}): {outlier_counts.to_dict()}")

        # Step 3: Encode categorical features
        X = self.encode_categorical_features(X, target_col=target_col)

        # Step 4: Scale features
        scaling_method = self.preprocessing_config.get('scaling', {}).get('method', 'standard')
        X = self.scale_features(X, method=scaling_method)

        # Update feature names after encoding
        self.feature_names = X.columns.tolist()

        # Step 5: Handle class imbalance
        resampling_method = self.preprocessing_config.get('resampling', {}).get('method', 'smote')
        if resampling_method and resampling_method != 'none':
            X, y = self.handle_class_imbalance(X, y, method=resampling_method)

        logger.info(f"Preprocessing pipeline completed. Final shape: {X.shape}")
        return X, y

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted transformers.

        Args:
            df: New DataFrame to transform

        Returns:
            Transformed DataFrame
        """
        logger.info("Transforming new data using fitted transformers")

        if not self.fitted_transformers:
            raise DataPreprocessingError("No fitted transformers found. Call fit_transform first.")

        X = df.drop(columns=[self.target_name, 'customerid'] if 'customerid' in df.columns else [self.target_name])

        # Apply transformations in the same order as fit_transform
        # This is a simplified version - in practice, you'd store and apply each transformer
        logger.warning("Transform method needs fitted transformers. Implement based on your specific needs.")

        return X

    def save_transformers(self, filepath: Optional[str] = None):
        """
        Save fitted transformers to disk.

        Args:
            filepath: Path to save transformers. If None, uses default path.
        """
        if filepath is None:
            filepath = self.artifacts_path / "transformers.joblib"

        transformers_dict = {
            'imputers': self.imputers,
            'encoders': self.encoders,
            'scalers': self.scalers,
            'outlier_detectors': self.outlier_detectors,
            'resampler': self.resampler,
            'feature_names': self.feature_names,
            'target_name': self.target_name
        }

        joblib.dump(transformers_dict, filepath)
        logger.info(f"Transformers saved to {filepath}")

    def load_transformers(self, filepath: Optional[str] = None):
        """
        Load fitted transformers from disk.

        Args:
            filepath: Path to load transformers from. If None, uses default path.
        """
        if filepath is None:
            filepath = self.artifacts_path / "transformers.joblib"

        if not Path(filepath).exists():
            raise DataPreprocessingError(f"Transformers file not found: {filepath}")

        transformers_dict = joblib.load(filepath)

        self.imputers = transformers_dict.get('imputers', {})
        self.encoders = transformers_dict.get('encoders', {})
        self.scalers = transformers_dict.get('scalers', {})
        self.outlier_detectors = transformers_dict.get('outlier_detectors', {})
        self.resampler = transformers_dict.get('resampler')
        self.feature_names = transformers_dict.get('feature_names', [])
        self.target_name = transformers_dict.get('target_name')

        logger.info(f"Transformers loaded from {filepath}")

    def get_preprocessing_summary(self) -> Dict[str, Any]:
        """
        Get summary of preprocessing steps applied.

        Returns:
            Dictionary with preprocessing summary
        """
        summary = {
            'imputers': list(self.imputers.keys()),
            'encoders': list(self.encoders.keys()),
            'scalers': list(self.scalers.keys()),
            'outlier_detectors': list(self.outlier_detectors.keys()),
            'resampler': type(self.resampler).__name__ if self.resampler else None,
            'feature_count': len(self.feature_names),
            'feature_names': self.feature_names,
            'target_name': self.target_name
        }

        return summary


def main():
    """Main function for preprocessing."""
    from .data_loader import DataLoader

    # Load data
    loader = DataLoader()
    df, _ = loader.load_and_process()

    # Preprocess data
    preprocessor = AdvancedPreprocessor()
    X, y = preprocessor.fit_transform(df)

    # Save transformers
    preprocessor.save_transformers()

    # Get summary
    summary = preprocessor.get_preprocessing_summary()

    print("Preprocessing completed!")
    print(f"Final shape: {X.shape}")
    print(f"Summary: {summary}")

    # Save processed data
    processed_data = X.copy()
    processed_data[preprocessor.target_name] = y

    output_path = Path(config.project_root) / "data" / "processed" / "preprocessed_data.csv"
    processed_data.to_csv(output_path, index=False)
    logger.info(f"Preprocessed data saved to {output_path}")

    return X, y, summary


if __name__ == "__main__":
    main()
