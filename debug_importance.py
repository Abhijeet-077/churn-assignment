"""
Debug script to investigate feature importance data structure.
"""

import joblib
import json
from pathlib import Path

def debug_importance_structure():
    """Debug the interpretability results structure."""
    
    try:
        # Load interpretability results
        results_path = Path("artifacts/interpretability_results.joblib")
        
        if not results_path.exists():
            print("❌ Interpretability results file not found!")
            return
        
        results = joblib.load(results_path)
        
        print("🔍 Debugging Feature Importance Data Structure")
        print("=" * 50)
        
        print(f"📊 Top-level keys: {list(results.keys())}")
        print(f"📊 Number of models: {len(results)}")
        
        # Examine each model's structure
        for model_name, model_data in results.items():
            print(f"\n🤖 Model: {model_name}")
            print(f"   Keys: {list(model_data.keys())}")
            
            # Check each importance method
            for method_name, method_data in model_data.items():
                print(f"   📈 {method_name}:")
                
                if isinstance(method_data, dict):
                    print(f"      Type: dict")
                    print(f"      Keys: {list(method_data.keys())}")
                    
                    # Check for nested structure
                    for key, value in method_data.items():
                        if isinstance(value, dict):
                            print(f"         {key}: dict with keys {list(value.keys())}")
                        else:
                            print(f"         {key}: {type(value).__name__}")
                else:
                    print(f"      Type: {type(method_data).__name__}")
        
        # Show sample data
        print("\n📋 Sample Data Structure:")
        sample_model = list(results.keys())[0]
        sample_data = results[sample_model]
        
        for method_name, method_data in sample_data.items():
            print(f"\n{method_name}:")
            if isinstance(method_data, dict):
                for key, value in list(method_data.items())[:3]:  # Show first 3 items
                    if isinstance(value, dict):
                        print(f"  {key}: {dict(list(value.items())[:3])}")
                    else:
                        print(f"  {key}: {value}")
            
        return results
        
    except Exception as e:
        print(f"❌ Error debugging importance structure: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    debug_importance_structure()
