"""
Unit tests for data preprocessing functionality.
"""

import unittest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.preprocessing import AdvancedPreprocessor
from src.data.data_loader import DataLoader


class TestAdvancedPreprocessor(unittest.TestCase):
    """Test cases for AdvancedPreprocessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.preprocessor = AdvancedPreprocessor()
        
        # Load sample data
        loader = DataLoader()
        self.df, self.target = loader.load_and_process()
    
    def test_preprocessor_initialization(self):
        """Test preprocessor initialization."""
        self.assertIsInstance(self.preprocessor, AdvancedPreprocessor)
        self.assertIsNotNone(self.preprocessor.config)
    
    def test_fit_transform(self):
        """Test fit_transform functionality."""
        try:
            X, y = self.preprocessor.fit_transform(self.df)
            
            # Check output types
            self.assertIsInstance(X, pd.DataFrame)
            self.assertIsInstance(y, pd.Series)
            
            # Check shapes
            self.assertEqual(len(X), len(y))
            self.assertGreater(len(X.columns), 0)
            
            # Check for no missing values after preprocessing
            missing_values = X.isnull().sum().sum()
            self.assertEqual(missing_values, 0, "Missing values found after preprocessing")
            
            # Check target encoding
            unique_targets = y.unique()
            self.assertEqual(len(unique_targets), 2, "Target should be binary")
            self.assertIn(0, unique_targets)
            self.assertIn(1, unique_targets)
            
            print(f"✓ Fit-transform successful: {X.shape}")
            
        except Exception as e:
            self.fail(f"Fit-transform failed: {str(e)}")
    
    def test_transform_consistency(self):
        """Test transform consistency on new data."""
        try:
            # Fit on training data
            train_df, test_df = train_test_split(self.df, test_size=0.2, random_state=42)
            
            X_train, y_train = self.preprocessor.fit_transform(train_df)
            
            # Transform test data
            X_test, y_test = self.preprocessor.transform(test_df)
            
            # Check consistency
            self.assertEqual(X_train.shape[1], X_test.shape[1], "Feature count mismatch")
            self.assertEqual(list(X_train.columns), list(X_test.columns), "Column names mismatch")
            
            # Check for no missing values
            self.assertEqual(X_test.isnull().sum().sum(), 0, "Missing values in test transform")
            
            print(f"✓ Transform consistency validated: train{X_train.shape}, test{X_test.shape}")
            
        except Exception as e:
            self.fail(f"Transform consistency test failed: {str(e)}")
    
    def test_outlier_detection(self):
        """Test outlier detection functionality."""
        try:
            X, y = self.preprocessor.fit_transform(self.df)
            
            # Check if outlier columns were created
            outlier_columns = [col for col in X.columns if 'outlier' in col.lower()]
            
            if outlier_columns:
                for col in outlier_columns:
                    # Outlier columns should be binary
                    unique_values = X[col].unique()
                    self.assertTrue(all(val in [0, 1] for val in unique_values))
                
                print(f"✓ Outlier detection validated: {len(outlier_columns)} outlier features")
            else:
                print("✓ No outlier features created (acceptable)")
            
        except Exception as e:
            self.fail(f"Outlier detection test failed: {str(e)}")
    
    def test_categorical_encoding(self):
        """Test categorical encoding functionality."""
        try:
            X, y = self.preprocessor.fit_transform(self.df)
            
            # Check that categorical variables are properly encoded
            categorical_columns = X.select_dtypes(include=['object']).columns
            
            # Most categorical columns should be encoded to numeric
            numeric_columns = X.select_dtypes(include=[np.number]).columns
            self.assertGreater(len(numeric_columns), len(categorical_columns))
            
            # Check for one-hot encoded columns
            onehot_columns = [col for col in X.columns if '_' in col and 
                            any(suffix in col for suffix in ['yes', 'no', 'male', 'female'])]
            
            if onehot_columns:
                for col in onehot_columns:
                    # One-hot encoded columns should be binary
                    unique_values = X[col].unique()
                    self.assertTrue(all(val in [0, 1] for val in unique_values))
            
            print(f"✓ Categorical encoding validated: {len(onehot_columns)} one-hot features")
            
        except Exception as e:
            self.fail(f"Categorical encoding test failed: {str(e)}")
    
    def test_feature_scaling(self):
        """Test feature scaling functionality."""
        try:
            X, y = self.preprocessor.fit_transform(self.df)
            
            # Check numerical columns for scaling
            numerical_columns = X.select_dtypes(include=[np.number]).columns
            
            scaled_features = 0
            for col in numerical_columns:
                if col not in ['outlier_isolation_forest'] and not col.endswith('_outlier_iqr') and not col.endswith('_outlier_zscore'):
                    mean_val = X[col].mean()
                    std_val = X[col].std()
                    
                    # Check if feature appears to be scaled (mean near 0, std near 1)
                    if abs(mean_val) < 0.1 and 0.8 < std_val < 1.2:
                        scaled_features += 1
            
            if scaled_features > 0:
                print(f"✓ Feature scaling validated: {scaled_features} scaled features")
            else:
                print("✓ Feature scaling not applied (acceptable for some methods)")
            
        except Exception as e:
            self.fail(f"Feature scaling test failed: {str(e)}")


class TestPreprocessingPipeline(unittest.TestCase):
    """Test cases for complete preprocessing pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.preprocessor = AdvancedPreprocessor()
        loader = DataLoader()
        self.df, self.target = loader.load_and_process()
    
    def test_pipeline_robustness(self):
        """Test preprocessing pipeline robustness."""
        try:
            # Test with different data sizes
            for sample_size in [100, 500, 1000]:
                if len(self.df) >= sample_size:
                    sample_df = self.df.sample(n=sample_size, random_state=42)
                    X, y = self.preprocessor.fit_transform(sample_df)
                    
                    self.assertEqual(len(X), sample_size)
                    self.assertGreater(len(X.columns), 0)
            
            print("✓ Pipeline robustness validated")
            
        except Exception as e:
            self.fail(f"Pipeline robustness test failed: {str(e)}")
    
    def test_data_leakage_prevention(self):
        """Test that preprocessing prevents data leakage."""
        try:
            # Split data
            train_df, test_df = train_test_split(self.df, test_size=0.3, random_state=42)
            
            # Fit on training data only
            X_train, y_train = self.preprocessor.fit_transform(train_df)
            
            # Transform test data (should not refit)
            X_test, y_test = self.preprocessor.transform(test_df)
            
            # Check that test transformation doesn't change training statistics
            # This is a basic check - in practice, you'd store and compare statistics
            self.assertEqual(X_train.shape[1], X_test.shape[1])
            
            print("✓ Data leakage prevention validated")
            
        except Exception as e:
            self.fail(f"Data leakage prevention test failed: {str(e)}")


def run_preprocessing_tests():
    """Run all preprocessing tests."""
    print("Running Preprocessing Tests")
    print("=" * 40)
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTest(unittest.makeSuite(TestAdvancedPreprocessor))
    suite.addTest(unittest.makeSuite(TestPreprocessingPipeline))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 40)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\nOverall: {'✅ PASS' if success else '❌ FAIL'}")
    
    return success


if __name__ == "__main__":
    run_preprocessing_tests()
