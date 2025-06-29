"""
Unit tests for data loading functionality.
"""

import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.data_loader import DataLoader
from src.utils.exceptions import DataError


class TestDataLoader(unittest.TestCase):
    """Test cases for DataLoader class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.loader = DataLoader()
    
    def test_data_loader_initialization(self):
        """Test DataLoader initialization."""
        self.assertIsInstance(self.loader, DataLoader)
        self.assertIsNotNone(self.loader.config)
    
    def test_load_and_process(self):
        """Test data loading and processing."""
        try:
            df, target = self.loader.load_and_process()
            
            # Check data types
            self.assertIsInstance(df, pd.DataFrame)
            self.assertIsInstance(target, pd.Series)
            
            # Check data shape
            self.assertGreater(len(df), 0)
            self.assertGreater(len(df.columns), 0)
            self.assertEqual(len(df), len(target))
            
            # Check required columns exist
            required_columns = ['gender', 'seniorcitizen', 'partner', 'dependents', 'tenure']
            for col in required_columns:
                self.assertIn(col, df.columns)
            
            # Check target values
            unique_targets = target.unique()
            self.assertIn(0, unique_targets)
            self.assertIn(1, unique_targets)
            
            print(f"✓ Data loaded successfully: {df.shape}")
            
        except Exception as e:
            self.fail(f"Data loading failed: {str(e)}")
    
    def test_data_validation(self):
        """Test data validation functionality."""
        try:
            df, target = self.loader.load_and_process()
            
            # Check for missing values in critical columns
            critical_columns = ['gender', 'seniorcitizen', 'partner', 'dependents']
            for col in critical_columns:
                if col in df.columns:
                    missing_count = df[col].isnull().sum()
                    self.assertEqual(missing_count, 0, f"Missing values found in {col}")
            
            # Check data types
            self.assertTrue(df['tenure'].dtype in [np.int64, np.float64])
            self.assertTrue(df['monthlycharges'].dtype in [np.float64, np.int64])
            
            print("✓ Data validation passed")
            
        except Exception as e:
            self.fail(f"Data validation failed: {str(e)}")
    
    def test_data_quality_checks(self):
        """Test data quality checks."""
        try:
            df, target = self.loader.load_and_process()
            
            # Check for reasonable value ranges
            if 'tenure' in df.columns:
                self.assertTrue(df['tenure'].min() >= 0)
                self.assertTrue(df['tenure'].max() <= 100)  # Reasonable max tenure
            
            if 'monthlycharges' in df.columns:
                self.assertTrue(df['monthlycharges'].min() >= 0)
                self.assertTrue(df['monthlycharges'].max() <= 1000)  # Reasonable max charges
            
            # Check target distribution
            target_dist = target.value_counts(normalize=True)
            self.assertTrue(0.1 <= target_dist.min() <= 0.9)  # No extreme class imbalance
            
            print("✓ Data quality checks passed")
            
        except Exception as e:
            self.fail(f"Data quality checks failed: {str(e)}")


class TestDataProcessing(unittest.TestCase):
    """Test cases for data processing functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.loader = DataLoader()
    
    def test_categorical_encoding(self):
        """Test categorical variable encoding."""
        try:
            df, target = self.loader.load_and_process()
            
            # Check that categorical variables are properly encoded
            categorical_columns = df.select_dtypes(include=['object']).columns
            
            # Should have some categorical columns
            self.assertGreater(len(categorical_columns), 0)
            
            # Check for valid categorical values
            for col in categorical_columns:
                unique_values = df[col].unique()
                self.assertGreater(len(unique_values), 0)
                self.assertLess(len(unique_values), len(df))  # Not all unique
            
            print(f"✓ Categorical encoding validated for {len(categorical_columns)} columns")
            
        except Exception as e:
            self.fail(f"Categorical encoding test failed: {str(e)}")
    
    def test_numerical_features(self):
        """Test numerical feature processing."""
        try:
            df, target = self.loader.load_and_process()
            
            # Check numerical columns
            numerical_columns = df.select_dtypes(include=[np.number]).columns
            self.assertGreater(len(numerical_columns), 0)
            
            # Check for reasonable distributions
            for col in numerical_columns:
                if col in ['tenure', 'monthlycharges', 'totalcharges']:
                    # Check for no infinite values
                    self.assertFalse(np.isinf(df[col]).any())
                    
                    # Check for reasonable variance
                    if df[col].std() > 0:
                        cv = df[col].std() / df[col].mean()
                        self.assertLess(cv, 10)  # Coefficient of variation not too high
            
            print(f"✓ Numerical features validated for {len(numerical_columns)} columns")
            
        except Exception as e:
            self.fail(f"Numerical features test failed: {str(e)}")


def run_data_tests():
    """Run all data loading tests."""
    print("Running Data Loading Tests")
    print("=" * 40)
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTest(unittest.makeSuite(TestDataLoader))
    suite.addTest(unittest.makeSuite(TestDataProcessing))
    
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
    run_data_tests()
