"""
Run model interpretability analysis with robust error handling.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_optimized_models():
    """Load optimized models and data."""
    from src.data.data_loader import DataLoader
    from src.data.preprocessing import AdvancedPreprocessor
    from src.features.feature_engineering import AdvancedFeatureEngineer
    
    # Load and preprocess data
    loader = DataLoader()
    df, _ = loader.load_and_process()
    
    preprocessor = AdvancedPreprocessor()
    X, y = preprocessor.fit_transform(df)
    
    # Feature engineering
    engineer = AdvancedFeatureEngineer()
    X_engineered, _ = engineer.fit_transform(X, y)
    
    # Load optimization results
    optimization_path = Path("artifacts/optimization/optimization_results.joblib")
    optimization_results = joblib.load(optimization_path)
    
    # Create optimized models
    models = {}
    
    # Logistic Regression
    lr_params = optimization_results['optimization_results']['logistic_regression']['best_params']
    models['logistic_regression'] = LogisticRegression(**lr_params, random_state=42)
    
    # Random Forest
    rf_params = optimization_results['optimization_results']['random_forest']['best_params']
    models['random_forest'] = RandomForestClassifier(**rf_params, random_state=42)
    
    # Gradient Boosting
    gb_params = optimization_results['optimization_results']['gradient_boosting']['best_params']
    models['gradient_boosting'] = GradientBoostingClassifier(**gb_params, random_state=42)
    
    # Fit models
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_engineered, y)
    
    return models, X_engineered, y

def create_feature_importance_plot(importance_dict, title, save_path):
    """Create feature importance plot."""
    # Get top 15 features
    top_features = dict(list(importance_dict.items())[:15])
    
    plt.figure(figsize=(12, 8))
    features = list(top_features.keys())
    importances = list(top_features.values())
    
    # Create horizontal bar plot
    y_pos = np.arange(len(features))
    bars = plt.barh(y_pos, importances, color='skyblue', alpha=0.8)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width + max(importances) * 0.01, bar.get_y() + bar.get_height()/2, 
                f'{width:.4f}', ha='left', va='center', fontsize=10)
    
    plt.yticks(y_pos, features)
    plt.xlabel('Importance Score', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()  # Highest importance at top
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved: {save_path}")

def analyze_feature_importance(models, X, y):
    """Analyze feature importance using multiple methods."""
    results = {}
    plots_dir = Path("plots/interpretability")
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    for model_name, model in models.items():
        print(f"\n=== Analyzing {model_name} ===")
        model_results = {}
        
        # 1. Built-in feature importance (for tree-based models)
        if hasattr(model, 'feature_importances_'):
            importance_dict = dict(zip(X.columns, model.feature_importances_))
            importance_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
            model_results['feature_importance'] = importance_dict
            
            # Create plot
            plot_path = plots_dir / f"{model_name}_feature_importance.png"
            create_feature_importance_plot(
                importance_dict, 
                f"{model_name.replace('_', ' ').title()} - Feature Importance",
                plot_path
            )
            
            print(f"Top 5 features: {list(importance_dict.keys())[:5]}")
        
        # 2. Permutation importance
        print("Calculating permutation importance...")
        perm_importance = permutation_importance(
            model, X, y, n_repeats=5, random_state=42, scoring='roc_auc'
        )
        
        perm_dict = dict(zip(X.columns, perm_importance.importances_mean))
        perm_dict = dict(sorted(perm_dict.items(), key=lambda x: x[1], reverse=True))
        model_results['permutation_importance'] = perm_dict
        
        # Create plot
        plot_path = plots_dir / f"{model_name}_permutation_importance.png"
        create_feature_importance_plot(
            perm_dict,
            f"{model_name.replace('_', ' ').title()} - Permutation Importance",
            plot_path
        )
        
        print(f"Top 5 permutation features: {list(perm_dict.keys())[:5]}")
        
        # 3. Try SHAP analysis (if available)
        try:
            import shap
            print("Calculating SHAP values...")
            
            # Use smaller sample for faster computation
            X_sample = X.sample(min(500, len(X)), random_state=42)
            
            if model_name in ['random_forest', 'gradient_boosting']:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_sample)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]  # Take positive class
            else:
                explainer = shap.Explainer(model, X_sample)
                shap_values = explainer(X_sample).values
            
            # Calculate mean absolute SHAP values
            mean_shap = np.abs(shap_values).mean(axis=0)
            shap_dict = dict(zip(X_sample.columns, mean_shap))
            shap_dict = dict(sorted(shap_dict.items(), key=lambda x: x[1], reverse=True))
            model_results['shap_importance'] = shap_dict
            
            # Create SHAP summary plot
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, X_sample, show=False, max_display=15)
            plt.title(f"{model_name.replace('_', ' ').title()} - SHAP Summary")
            plot_path = plots_dir / f"{model_name}_shap_summary.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"SHAP analysis completed. Top 5 SHAP features: {list(shap_dict.keys())[:5]}")
            
        except ImportError:
            print("SHAP not available, skipping SHAP analysis")
        except Exception as e:
            print(f"SHAP analysis failed: {str(e)}")
        
        results[model_name] = model_results
    
    return results

def create_comparison_plots(results):
    """Create comparison plots across different importance methods."""
    plots_dir = Path("plots/interpretability")
    
    for model_name, model_results in results.items():
        # Create comparison DataFrame
        comparison_data = []
        
        # Get all unique features
        all_features = set()
        for method_name, importance_dict in model_results.items():
            if isinstance(importance_dict, dict):
                all_features.update(importance_dict.keys())
        
        # Create comparison data
        for feature in all_features:
            row = {'Feature': feature}
            for method_name, importance_dict in model_results.items():
                if isinstance(importance_dict, dict):
                    row[method_name.replace('_', ' ').title()] = importance_dict.get(feature, 0)
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Save comparison CSV
        csv_path = plots_dir / f"{model_name}_importance_comparison.csv"
        comparison_df.to_csv(csv_path, index=False)
        print(f"Comparison saved: {csv_path}")
        
        # Create correlation plot if multiple methods available
        if len(comparison_df.columns) > 2:
            numeric_cols = comparison_df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                plt.figure(figsize=(10, 8))
                correlation_matrix = comparison_df[numeric_cols].corr()
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
                plt.title(f"{model_name.replace('_', ' ').title()} - Importance Methods Correlation")
                plot_path = plots_dir / f"{model_name}_importance_correlation.png"
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Correlation plot saved: {plot_path}")

def main():
    """Main function for interpretability analysis."""
    print("Starting Model Interpretability Analysis...")
    print("=" * 50)
    
    # Load models and data
    models, X, y = load_optimized_models()
    
    # Analyze feature importance
    results = analyze_feature_importance(models, X, y)
    
    # Create comparison plots
    create_comparison_plots(results)
    
    # Save results
    results_path = Path("artifacts/interpretability_results.joblib")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(results, results_path)
    
    print("\n" + "=" * 50)
    print("Model Interpretability Analysis Completed!")
    print(f"Results saved to: {results_path}")
    print(f"Plots saved in: plots/interpretability/")
    
    # Print summary
    print("\nSummary:")
    for model_name, model_results in results.items():
        print(f"\n{model_name.replace('_', ' ').title()}:")
        for method_name in model_results.keys():
            print(f"  ✓ {method_name.replace('_', ' ').title()}")

if __name__ == "__main__":
    main()
