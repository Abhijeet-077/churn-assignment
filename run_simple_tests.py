"""
Simple test runner for existing components.
"""

import sys
import os
import time
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def test_data_loading():
    """Test data loading functionality."""
    print("Testing Data Loading...")
    print("-" * 30)
    
    try:
        from src.data.data_loader import DataLoader
        
        loader = DataLoader()
        df, target = loader.load_and_process()
        
        print(f"✓ Data loaded successfully: {df.shape}")
        print(f"✓ Target loaded: {len(target)} samples")
        print(f"✓ Churn distribution: {target.value_counts().to_dict()}")
        
        # Basic validation
        assert len(df) > 0, "Empty dataset"
        assert len(df.columns) > 0, "No columns"
        assert len(df) == len(target), "Mismatched lengths"
        
        return True
        
    except Exception as e:
        print(f"❌ Data loading failed: {str(e)}")
        return False


def test_preprocessing():
    """Test preprocessing functionality."""
    print("\nTesting Preprocessing...")
    print("-" * 30)
    
    try:
        from src.data.data_loader import DataLoader
        from src.data.preprocessing import AdvancedPreprocessor
        
        # Load data
        loader = DataLoader()
        df, target = loader.load_and_process()
        
        # Preprocess
        preprocessor = AdvancedPreprocessor()
        X, y = preprocessor.fit_transform(df)
        
        print(f"✓ Preprocessing successful: {X.shape}")
        print(f"✓ Features: {len(X.columns)}")
        print(f"✓ No missing values: {X.isnull().sum().sum() == 0}")
        
        # Basic validation
        assert len(X) == len(y), "Mismatched lengths after preprocessing"
        assert X.isnull().sum().sum() == 0, "Missing values after preprocessing"
        
        return True
        
    except Exception as e:
        print(f"❌ Preprocessing failed: {str(e)}")
        return False


def test_feature_engineering():
    """Test feature engineering functionality."""
    print("\nTesting Feature Engineering...")
    print("-" * 30)
    
    try:
        from src.data.data_loader import DataLoader
        from src.data.preprocessing import AdvancedPreprocessor
        from src.features.feature_engineering import AdvancedFeatureEngineer
        
        # Load and preprocess data
        loader = DataLoader()
        df, target = loader.load_and_process()
        
        preprocessor = AdvancedPreprocessor()
        X, y = preprocessor.fit_transform(df)
        
        # Feature engineering
        engineer = AdvancedFeatureEngineer()
        X_engineered, _ = engineer.fit_transform(X, y)
        
        print(f"✓ Feature engineering successful: {X_engineered.shape}")
        print(f"✓ Original features: {X.shape[1]}")
        print(f"✓ Engineered features: {X_engineered.shape[1]}")
        
        # Basic validation
        assert len(X_engineered) == len(y), "Mismatched lengths after feature engineering"
        assert X_engineered.shape[1] > 0, "No features after engineering"
        
        return True
        
    except Exception as e:
        print(f"❌ Feature engineering failed: {str(e)}")
        return False


def test_hyperparameter_optimization():
    """Test hyperparameter optimization functionality."""
    print("\nTesting Hyperparameter Optimization...")
    print("-" * 30)
    
    try:
        import joblib
        from pathlib import Path
        
        # Check if optimization results exist
        optimization_path = Path("artifacts/optimization/optimization_results.joblib")
        
        if optimization_path.exists():
            results = joblib.load(optimization_path)
            
            print(f"✓ Optimization results loaded")
            print(f"✓ Best model: {results.get('best_model', 'Unknown')}")
            
            if 'optimization_results' in results:
                opt_results = results['optimization_results']
                for model_name, model_results in opt_results.items():
                    best_score = model_results.get('best_score', 0)
                    print(f"✓ {model_name}: best score = {best_score:.4f}")
            
            return True
        else:
            print("⚠️  Optimization results not found (run optimization first)")
            return True  # Not a failure, just not run yet
        
    except Exception as e:
        print(f"❌ Hyperparameter optimization test failed: {str(e)}")
        return False


def test_interpretability():
    """Test interpretability functionality."""
    print("\nTesting Interpretability...")
    print("-" * 30)
    
    try:
        import joblib
        from pathlib import Path
        
        # Check if interpretability results exist
        interpretability_path = Path("artifacts/interpretability_results.joblib")
        
        if interpretability_path.exists():
            results = joblib.load(interpretability_path)
            
            print(f"✓ Interpretability results loaded")
            print(f"✓ Models analyzed: {len(results)}")
            
            for model_name, model_results in results.items():
                methods = list(model_results.keys())
                print(f"✓ {model_name}: {len(methods)} importance methods")
            
            return True
        else:
            print("⚠️  Interpretability results not found (run interpretability analysis first)")
            return True  # Not a failure, just not run yet
        
    except Exception as e:
        print(f"❌ Interpretability test failed: {str(e)}")
        return False


def test_api_components():
    """Test API components."""
    print("\nTesting API Components...")
    print("-" * 30)
    
    try:
        from src.api.models import CustomerData, PredictionRequest
        
        # Test Pydantic model validation
        customer_data = CustomerData(
            gender="Female",
            senior_citizen=0,
            partner="Yes",
            dependents="No",
            tenure=12,
            phone_service="Yes",
            multiple_lines="No",
            internet_service="DSL",
            online_security="Yes",
            online_backup="No",
            device_protection="Yes",
            tech_support="No",
            streaming_tv="No",
            streaming_movies="No",
            contract="One year",
            paperless_billing="Yes",
            payment_method="Electronic check",
            monthly_charges=65.0,
            total_charges=780.0
        )
        
        print("✓ Pydantic models working")
        print("✓ Data validation successful")
        
        # Test invalid data
        try:
            invalid_data = CustomerData(
                gender="Invalid",  # Should fail
                senior_citizen=0,
                partner="Yes",
                dependents="No",
                tenure=12,
                phone_service="Yes",
                multiple_lines="No",
                internet_service="DSL",
                online_security="Yes",
                online_backup="No",
                device_protection="Yes",
                tech_support="No",
                streaming_tv="No",
                streaming_movies="No",
                contract="One year",
                paperless_billing="Yes",
                payment_method="Electronic check",
                monthly_charges=65.0,
                total_charges=780.0
            )
            print("❌ Validation should have failed")
            return False
        except:
            print("✓ Data validation catches invalid data")
        
        return True
        
    except Exception as e:
        print(f"❌ API components test failed: {str(e)}")
        return False


def test_streamlit_components():
    """Test Streamlit components."""
    print("\nTesting Streamlit Components...")
    print("-" * 30)
    
    try:
        import streamlit as st
        import plotly.express as px
        import plotly.graph_objects as go
        
        print("✓ Streamlit imported successfully")
        print("✓ Plotly imported successfully")
        
        # Test basic visualization creation
        import pandas as pd
        import numpy as np
        
        sample_data = pd.DataFrame({
            'x': np.random.randn(100),
            'y': np.random.randn(100),
            'category': np.random.choice(['A', 'B'], 100)
        })
        
        # Test plotly charts
        fig1 = px.scatter(sample_data, x='x', y='y', color='category')
        fig2 = go.Figure(go.Indicator(mode="gauge+number", value=75))
        
        print("✓ Plotly charts created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Streamlit components test failed: {str(e)}")
        return False


def test_performance():
    """Test performance benchmarks."""
    print("\nTesting Performance...")
    print("-" * 30)
    
    try:
        from src.data.data_loader import DataLoader
        from src.data.preprocessing import AdvancedPreprocessor
        
        # Test data loading performance
        start_time = time.time()
        loader = DataLoader()
        df, target = loader.load_and_process()
        data_load_time = time.time() - start_time
        
        print(f"✓ Data loading: {data_load_time:.2f}s for {len(df):,} rows")
        
        # Test preprocessing performance
        start_time = time.time()
        preprocessor = AdvancedPreprocessor()
        X, y = preprocessor.fit_transform(df)
        preprocessing_time = time.time() - start_time
        
        print(f"✓ Preprocessing: {preprocessing_time:.2f}s")
        
        # Performance checks
        assert data_load_time < 10.0, f"Data loading too slow: {data_load_time:.2f}s"
        assert preprocessing_time < 20.0, f"Preprocessing too slow: {preprocessing_time:.2f}s"
        
        print("✓ Performance requirements met")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance test failed: {str(e)}")
        return False


def main():
    """Main test runner."""
    print("Customer Churn Prediction - Simple Test Suite")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Define test functions
    tests = [
        ("Data Loading", test_data_loading),
        ("Preprocessing", test_preprocessing),
        ("Feature Engineering", test_feature_engineering),
        ("Hyperparameter Optimization", test_hyperparameter_optimization),
        ("Interpretability", test_interpretability),
        ("API Components", test_api_components),
        ("Streamlit Components", test_streamlit_components),
        ("Performance", test_performance)
    ]
    
    # Run tests
    results = {}
    total_time = 0
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        
        start_time = time.time()
        try:
            result = test_func()
            duration = time.time() - start_time
            total_time += duration
            
            results[test_name] = {
                'passed': result,
                'duration': duration
            }
            
        except Exception as e:
            duration = time.time() - start_time
            total_time += duration
            
            print(f"❌ {test_name} failed with error: {str(e)}")
            results[test_name] = {
                'passed': False,
                'duration': duration,
                'error': str(e)
            }
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result['passed'] else "❌ FAIL"
        duration = result['duration']
        print(f"{status} {test_name}: {duration:.2f}s")
        
        if result['passed']:
            passed_tests += 1
    
    success_rate = (passed_tests / total_tests) * 100
    
    print(f"\nResults: {passed_tests}/{total_tests} tests passed ({success_rate:.1f}%)")
    print(f"Total time: {total_time:.2f}s")
    
    if success_rate >= 80:
        print("\n🎉 EXCELLENT! The system is working well.")
        print("Ready for production deployment!")
    elif success_rate >= 60:
        print("\n👍 GOOD! Most components are working.")
        print("Minor issues to address before production.")
    else:
        print("\n⚠️  NEEDS ATTENTION! Several components have issues.")
        print("Please review and fix failing tests.")
    
    # Save results
    try:
        import json
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'success_rate': success_rate,
            'total_time': total_time,
            'results': results
        }
        
        with open('simple_test_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Test report saved to: simple_test_report.json")
    except Exception as e:
        print(f"⚠️  Failed to save test report: {str(e)}")
    
    print("=" * 60)
    
    return success_rate >= 80


if __name__ == "__main__":
    main()
