"""
Unit tests for model training and evaluation functionality.
"""

import unittest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.model_trainer import ModelTrainer
from src.models.model_evaluator import ModelEvaluator
from src.data.data_loader import DataLoader
from src.data.preprocessing import AdvancedPreprocessor
from src.features.feature_engineering import AdvancedFeatureEngineer


class TestModelTrainer(unittest.TestCase):
    """Test cases for ModelTrainer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.trainer = ModelTrainer()
        
        # Prepare sample data
        loader = DataLoader()
        df, _ = loader.load_and_process()
        
        preprocessor = AdvancedPreprocessor()
        X, y = preprocessor.fit_transform(df)
        
        engineer = AdvancedFeatureEngineer()
        self.X, _ = engineer.fit_transform(X, y)
        self.y = y
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
    
    def test_trainer_initialization(self):
        """Test trainer initialization."""
        self.assertIsInstance(self.trainer, ModelTrainer)
        self.assertIsNotNone(self.trainer.config)
    
    def test_train_single_model(self):
        """Test training a single model."""
        try:
            # Train logistic regression
            model = self.trainer.train_logistic_regression(self.X_train, self.y_train)
            
            # Check model is fitted
            self.assertTrue(hasattr(model, 'coef_'))
            
            # Test prediction
            y_pred = model.predict(self.X_test)
            self.assertEqual(len(y_pred), len(self.y_test))
            
            # Check prediction values are valid
            unique_preds = np.unique(y_pred)
            self.assertTrue(all(pred in [0, 1] for pred in unique_preds))
            
            # Test probability prediction
            y_proba = model.predict_proba(self.X_test)
            self.assertEqual(y_proba.shape, (len(self.y_test), 2))
            self.assertTrue(np.allclose(y_proba.sum(axis=1), 1.0))
            
            print("✓ Single model training validated")
            
        except Exception as e:
            self.fail(f"Single model training failed: {str(e)}")
    
    def test_train_multiple_models(self):
        """Test training multiple models."""
        try:
            models = self.trainer.train_all_models(self.X_train, self.y_train)
            
            # Check that models were trained
            self.assertIsInstance(models, dict)
            self.assertGreater(len(models), 0)
            
            # Test each model
            for model_name, model in models.items():
                # Check model can predict
                y_pred = model.predict(self.X_test)
                self.assertEqual(len(y_pred), len(self.y_test))
                
                # Check prediction quality
                accuracy = accuracy_score(self.y_test, y_pred)
                self.assertGreater(accuracy, 0.5)  # Better than random
                
                print(f"✓ {model_name}: accuracy = {accuracy:.3f}")
            
            print(f"✓ Multiple model training validated: {len(models)} models")
            
        except Exception as e:
            self.fail(f"Multiple model training failed: {str(e)}")
    
    def test_model_persistence(self):
        """Test model saving and loading."""
        try:
            # Train a model
            model = self.trainer.train_logistic_regression(self.X_train, self.y_train)
            
            # Save model
            model_path = "test_model.joblib"
            self.trainer.save_model(model, model_path)
            
            # Load model
            loaded_model = self.trainer.load_model(model_path)
            
            # Test loaded model
            y_pred_original = model.predict(self.X_test)
            y_pred_loaded = loaded_model.predict(self.X_test)
            
            # Predictions should be identical
            np.testing.assert_array_equal(y_pred_original, y_pred_loaded)
            
            # Clean up
            import os
            if os.path.exists(model_path):
                os.remove(model_path)
            
            print("✓ Model persistence validated")
            
        except Exception as e:
            self.fail(f"Model persistence test failed: {str(e)}")


class TestModelEvaluator(unittest.TestCase):
    """Test cases for ModelEvaluator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.evaluator = ModelEvaluator()
        
        # Prepare sample data and models
        loader = DataLoader()
        df, _ = loader.load_and_process()
        
        preprocessor = AdvancedPreprocessor()
        X, y = preprocessor.fit_transform(df)
        
        engineer = AdvancedFeatureEngineer()
        self.X, _ = engineer.fit_transform(X, y)
        self.y = y
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        # Train a simple model for testing
        trainer = ModelTrainer()
        self.model = trainer.train_logistic_regression(self.X_train, self.y_train)
    
    def test_evaluator_initialization(self):
        """Test evaluator initialization."""
        self.assertIsInstance(self.evaluator, ModelEvaluator)
        self.assertIsNotNone(self.evaluator.config)
    
    def test_single_model_evaluation(self):
        """Test evaluation of a single model."""
        try:
            metrics = self.evaluator.evaluate_model(
                self.model, self.X_test, self.y_test, "test_model"
            )
            
            # Check metrics structure
            self.assertIsInstance(metrics, dict)
            
            # Check required metrics exist
            required_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
            for metric in required_metrics:
                self.assertIn(metric, metrics)
                self.assertIsInstance(metrics[metric], (int, float))
                self.assertTrue(0 <= metrics[metric] <= 1)
            
            # Check confusion matrix
            self.assertIn('confusion_matrix', metrics)
            cm = metrics['confusion_matrix']
            self.assertEqual(cm.shape, (2, 2))
            
            print(f"✓ Single model evaluation: ROC-AUC = {metrics['roc_auc']:.3f}")
            
        except Exception as e:
            self.fail(f"Single model evaluation failed: {str(e)}")
    
    def test_cross_validation(self):
        """Test cross-validation functionality."""
        try:
            cv_scores = self.evaluator.cross_validate_model(
                self.model, self.X_train, self.y_train, cv=3
            )
            
            # Check CV scores structure
            self.assertIsInstance(cv_scores, dict)
            
            # Check that scores are reasonable
            for metric, scores in cv_scores.items():
                if isinstance(scores, (list, np.ndarray)):
                    self.assertEqual(len(scores), 3)  # 3-fold CV
                    self.assertTrue(all(0 <= score <= 1 for score in scores))
            
            print(f"✓ Cross-validation validated: {len(cv_scores)} metrics")
            
        except Exception as e:
            self.fail(f"Cross-validation test failed: {str(e)}")
    
    def test_model_comparison(self):
        """Test model comparison functionality."""
        try:
            # Train multiple models
            trainer = ModelTrainer()
            models = {
                'logistic_regression': trainer.train_logistic_regression(self.X_train, self.y_train),
                'random_forest': trainer.train_random_forest(self.X_train, self.y_train)
            }
            
            # Compare models
            comparison = self.evaluator.compare_models(models, self.X_test, self.y_test)
            
            # Check comparison structure
            self.assertIsInstance(comparison, dict)
            self.assertEqual(len(comparison), len(models))
            
            # Check each model's metrics
            for model_name, metrics in comparison.items():
                self.assertIn(model_name, models.keys())
                self.assertIn('roc_auc', metrics)
                self.assertTrue(0 <= metrics['roc_auc'] <= 1)
            
            print(f"✓ Model comparison validated: {len(comparison)} models")
            
        except Exception as e:
            self.fail(f"Model comparison test failed: {str(e)}")


class TestModelPerformance(unittest.TestCase):
    """Test cases for model performance requirements."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Prepare data
        loader = DataLoader()
        df, _ = loader.load_and_process()
        
        preprocessor = AdvancedPreprocessor()
        X, y = preprocessor.fit_transform(df)
        
        engineer = AdvancedFeatureEngineer()
        self.X, _ = engineer.fit_transform(X, y)
        self.y = y
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
    
    def test_minimum_performance_requirements(self):
        """Test that models meet minimum performance requirements."""
        try:
            trainer = ModelTrainer()
            evaluator = ModelEvaluator()
            
            # Train models
            models = trainer.train_all_models(self.X_train, self.y_train)
            
            # Evaluate each model
            for model_name, model in models.items():
                metrics = evaluator.evaluate_model(model, self.X_test, self.y_test, model_name)
                
                # Minimum performance requirements
                self.assertGreater(metrics['accuracy'], 0.6, f"{model_name} accuracy too low")
                self.assertGreater(metrics['roc_auc'], 0.6, f"{model_name} ROC-AUC too low")
                self.assertGreater(metrics['precision'], 0.3, f"{model_name} precision too low")
                self.assertGreater(metrics['recall'], 0.3, f"{model_name} recall too low")
                
                print(f"✓ {model_name}: meets performance requirements")
            
            print("✓ All models meet minimum performance requirements")
            
        except Exception as e:
            self.fail(f"Performance requirements test failed: {str(e)}")


def run_model_tests():
    """Run all model tests."""
    print("Running Model Tests")
    print("=" * 40)
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTest(unittest.makeSuite(TestModelTrainer))
    suite.addTest(unittest.makeSuite(TestModelEvaluator))
    suite.addTest(unittest.makeSuite(TestModelPerformance))
    
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
    run_model_tests()
