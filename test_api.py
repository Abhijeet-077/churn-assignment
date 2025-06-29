"""
Test script for the FastAPI application.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.api.models import CustomerData, PredictionRequest
from src.api.prediction_service import PredictionService
import json

def test_prediction_service():
    """Test the prediction service directly."""
    print("Testing Prediction Service...")
    print("=" * 50)
    
    try:
        # Initialize service
        print("1. Initializing prediction service...")
        service = PredictionService()
        print(f"   ✓ Service initialized")
        print(f"   ✓ Available models: {service.get_available_models()}")
        
        # Test customer data
        print("\n2. Creating test customer data...")
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
        print("   ✓ Customer data created")
        
        # Test prediction
        print("\n3. Making prediction...")
        prediction = service.predict_single(
            customer_data=customer_data,
            model_name='best',
            include_explanation=True
        )
        
        print(f"   ✓ Prediction completed!")
        print(f"   - Churn Prediction: {prediction.churn_prediction}")
        print(f"   - Churn Probability: {prediction.churn_probability:.3f}")
        print(f"   - Confidence: {prediction.confidence}")
        print(f"   - Risk Level: {prediction.risk_level}")
        print(f"   - Model Used: {prediction.model_used}")
        print(f"   - Processing Time: {prediction.processing_time_ms:.2f}ms")
        
        if prediction.explanation:
            print(f"   - Explanation: {prediction.explanation.explanation_text}")
        
        # Test model info
        print("\n4. Getting model information...")
        model_info = service.get_model_info('best')
        print(f"   ✓ Model: {model_info.model_name}")
        print(f"   - Type: {model_info.model_type}")
        print(f"   - Features: {model_info.feature_count}")
        
        # Test feature importance
        print("\n5. Getting feature importance...")
        importance = service.get_feature_importance('best')
        print(f"   ✓ Top 5 features: {importance['top_features'][:5]}")
        
        print("\n" + "=" * 50)
        print("✅ All tests passed! Prediction service is working correctly.")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_batch_prediction():
    """Test batch prediction."""
    print("\nTesting Batch Prediction...")
    print("=" * 30)
    
    try:
        service = PredictionService()
        
        # Create multiple customers
        customers = []
        for i in range(3):
            customer = CustomerData(
                gender="Female" if i % 2 == 0 else "Male",
                senior_citizen=i % 2,
                partner="Yes",
                dependents="No" if i % 2 == 0 else "Yes",
                tenure=12 + i * 6,
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
                monthly_charges=65.0 + i * 10,
                total_charges=780.0 + i * 120
            )
            customers.append(customer)
        
        # Make batch prediction
        result = service.predict_batch(customers, 'best')
        
        print(f"✓ Batch prediction completed")
        print(f"  - Total customers: {result['summary']['total_customers']}")
        print(f"  - Successful predictions: {result['summary']['successful_predictions']}")
        print(f"  - Predicted churners: {result['summary']['predicted_churners']}")
        print(f"  - High risk customers: {result['summary']['high_risk_customers']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Batch test failed: {str(e)}")
        return False

def main():
    """Main test function."""
    print("Customer Churn Prediction API - Test Suite")
    print("=" * 60)
    
    # Test prediction service
    service_test = test_prediction_service()
    
    # Test batch prediction
    batch_test = test_batch_prediction()
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY:")
    print(f"Prediction Service: {'✅ PASS' if service_test else '❌ FAIL'}")
    print(f"Batch Prediction: {'✅ PASS' if batch_test else '❌ FAIL'}")
    
    if service_test and batch_test:
        print("\n🎉 All tests passed! The API is ready to run.")
        print("\nTo start the API server, run:")
        print("python -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    main()
