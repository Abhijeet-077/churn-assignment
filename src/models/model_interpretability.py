"""
Model interpretability and explainability using SHAP and other techniques.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, accuracy_score
import joblib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Try to import SHAP, fallback if not available
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    shap = None

from ..utils.config import config
from ..utils.logger import get_logger
from ..utils.exceptions import ModelError

logger = get_logger(__name__)


class ModelInterpreter:
    """Model interpretability and explainability suite."""
    
    def __init__(self):
        self.config = config
        
        # Store interpretation results
        self.shap_values = {}
        self.feature_importance = {}
        self.permutation_importance = {}
        
        # Paths for saving results
        self.plots_path = Path(self.config.project_root) / "plots" / "interpretability"
        self.plots_path.mkdir(parents=True, exist_ok=True)
    
    def calculate_feature_importance(self, model, X: pd.DataFrame, feature_names: List[str]) -> Dict[str, float]:
        """Calculate feature importance from tree-based models."""
        logger.info("Calculating feature importance")
        
        if hasattr(model, 'feature_importances_'):
            importance_dict = dict(zip(feature_names, model.feature_importances_))
            
            # Sort by importance
            sorted_importance = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
            
            logger.info(f"Top 5 important features: {list(sorted_importance.keys())[:5]}")
            return sorted_importance
        else:
            logger.warning("Model does not have feature_importances_ attribute")
            return {}
    
    def calculate_permutation_importance(self, model, X: pd.DataFrame, y: pd.Series, 
                                       feature_names: List[str], n_repeats: int = 10) -> Dict[str, Dict]:
        """Calculate permutation importance."""
        logger.info("Calculating permutation importance")
        
        try:
            perm_importance = permutation_importance(
                model, X, y, 
                n_repeats=n_repeats, 
                random_state=42,
                scoring='roc_auc'
            )
            
            importance_dict = {
                'importances_mean': dict(zip(feature_names, perm_importance.importances_mean)),
                'importances_std': dict(zip(feature_names, perm_importance.importances_std))
            }
            
            # Sort by mean importance
            sorted_mean = dict(sorted(
                importance_dict['importances_mean'].items(), 
                key=lambda x: x[1], 
                reverse=True
            ))
            
            logger.info(f"Top 5 permutation important features: {list(sorted_mean.keys())[:5]}")
            return importance_dict
            
        except Exception as e:
            logger.error(f"Failed to calculate permutation importance: {str(e)}")
            return {}
    
    def calculate_shap_values(self, model, X: pd.DataFrame, model_type: str = 'tree') -> np.ndarray:
        """Calculate SHAP values for model interpretability."""
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available. Skipping SHAP analysis.")
            return None

        logger.info(f"Calculating SHAP values for {model_type} model")

        try:
            # Use a smaller sample for faster computation
            X_sample = X.sample(min(1000, len(X)), random_state=42)

            # Choose appropriate explainer based on model type
            if model_type in ['tree', 'random_forest', 'gradient_boosting', 'xgboost', 'lightgbm']:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_sample)
                # For binary classification, take the positive class
                if isinstance(shap_values, list) and len(shap_values) == 2:
                    shap_values = shap_values[1]
            elif model_type == 'linear':
                # For linear models, use a simpler approach
                explainer = shap.Explainer(model, X_sample)
                shap_values = explainer(X_sample).values
            else:
                # Use KernelExplainer as fallback (slower but works with any model)
                explainer = shap.KernelExplainer(model.predict_proba, X_sample.sample(100, random_state=42))
                shap_values = explainer.shap_values(X_sample)
                if isinstance(shap_values, list) and len(shap_values) == 2:
                    shap_values = shap_values[1]

            logger.info(f"SHAP values calculated. Shape: {shap_values.shape}")
            return shap_values, X_sample

        except Exception as e:
            logger.error(f"Failed to calculate SHAP values: {str(e)}")
            return None, None
    
    def create_feature_importance_plot(self, importance_dict: Dict[str, float], 
                                     title: str = "Feature Importance", top_n: int = 15):
        """Create feature importance plot."""
        logger.info(f"Creating feature importance plot: {title}")
        
        # Get top N features
        top_features = dict(list(importance_dict.items())[:top_n])
        
        plt.figure(figsize=(10, 8))
        features = list(top_features.keys())
        importances = list(top_features.values())
        
        # Create horizontal bar plot
        y_pos = np.arange(len(features))
        plt.barh(y_pos, importances, color='skyblue')
        plt.yticks(y_pos, features)
        plt.xlabel('Importance')
        plt.title(title)
        plt.gca().invert_yaxis()  # Highest importance at top
        
        plt.tight_layout()
        plot_path = self.plots_path / f"{title.lower().replace(' ', '_')}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Feature importance plot saved to {plot_path}")
        return str(plot_path)
    
    def create_shap_summary_plot(self, shap_values: np.ndarray, X: pd.DataFrame,
                               title: str = "SHAP Summary Plot"):
        """Create SHAP summary plot."""
        if not SHAP_AVAILABLE or shap_values is None:
            logger.warning("SHAP not available or SHAP values not calculated")
            return None

        logger.info("Creating SHAP summary plot")

        try:
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, X, show=False, max_display=15)
            plt.title(title, fontsize=14)
            plt.tight_layout()

            plot_path = self.plots_path / f"{title.lower().replace(' ', '_')}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"SHAP summary plot saved to {plot_path}")
            return str(plot_path)

        except Exception as e:
            logger.error(f"Failed to create SHAP summary plot: {str(e)}")
            # Create a simple feature importance plot as fallback
            try:
                mean_shap = np.abs(shap_values).mean(axis=0)
                feature_importance = dict(zip(X.columns, mean_shap))
                return self.create_feature_importance_plot(feature_importance, title)
            except:
                return None
    
    def create_shap_waterfall_plot(self, shap_values: np.ndarray, X: pd.DataFrame, 
                                 instance_idx: int = 0, title: str = "SHAP Waterfall Plot"):
        """Create SHAP waterfall plot for a single instance."""
        if not SHAP_AVAILABLE or shap_values is None:
            logger.warning("SHAP not available or SHAP values not calculated")
            return None
        
        logger.info(f"Creating SHAP waterfall plot for instance {instance_idx}")
        
        try:
            plt.figure(figsize=(10, 8))
            
            # Create explanation object
            explainer = shap.Explainer(lambda x: shap_values, X)
            explanation = explainer(X.iloc[[instance_idx]])
            
            shap.waterfall_plot(explanation[0], show=False)
            plt.title(title)
            plt.tight_layout()
            
            plot_path = self.plots_path / f"{title.lower().replace(' ', '_')}_instance_{instance_idx}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"SHAP waterfall plot saved to {plot_path}")
            return str(plot_path)
            
        except Exception as e:
            logger.error(f"Failed to create SHAP waterfall plot: {str(e)}")
            return None
    
    def analyze_model_interpretability(self, model, X: pd.DataFrame, y: pd.Series, 
                                     model_name: str, model_type: str = 'tree') -> Dict[str, Any]:
        """Complete model interpretability analysis."""
        logger.info(f"Starting interpretability analysis for {model_name}")
        
        results = {
            'model_name': model_name,
            'model_type': model_type,
            'feature_names': X.columns.tolist()
        }
        
        # 1. Feature importance (for tree-based models)
        feature_importance = self.calculate_feature_importance(model, X, X.columns.tolist())
        if feature_importance:
            results['feature_importance'] = feature_importance
            self.feature_importance[model_name] = feature_importance
            
            # Create feature importance plot
            importance_plot_path = self.create_feature_importance_plot(
                feature_importance, 
                f"{model_name} Feature Importance"
            )
            results['feature_importance_plot'] = importance_plot_path
        
        # 2. Permutation importance
        perm_importance = self.calculate_permutation_importance(model, X, y, X.columns.tolist())
        if perm_importance:
            results['permutation_importance'] = perm_importance
            self.permutation_importance[model_name] = perm_importance
            
            # Create permutation importance plot
            perm_plot_path = self.create_feature_importance_plot(
                perm_importance['importances_mean'],
                f"{model_name} Permutation Importance"
            )
            results['permutation_importance_plot'] = perm_plot_path
        
        # 3. SHAP analysis
        shap_result = self.calculate_shap_values(model, X, model_type)
        if shap_result is not None and len(shap_result) == 2:
            shap_values, X_sample = shap_result
            if shap_values is not None:
                results['shap_available'] = True
                self.shap_values[model_name] = shap_values

                # Create SHAP summary plot
                shap_summary_path = self.create_shap_summary_plot(
                    shap_values, X_sample, f"{model_name} SHAP Summary"
                )
                results['shap_summary_plot'] = shap_summary_path

                # Skip waterfall plot for now due to compatibility issues
                # shap_waterfall_path = self.create_shap_waterfall_plot(
                #     shap_values, X_sample, 0, f"{model_name} SHAP Waterfall"
                # )
                # results['shap_waterfall_plot'] = shap_waterfall_path

                # Calculate mean absolute SHAP values for feature ranking
                try:
                    mean_shap_importance = np.abs(shap_values).mean(axis=0)
                    shap_feature_importance = dict(zip(X_sample.columns, mean_shap_importance))
                    shap_feature_importance = dict(sorted(
                        shap_feature_importance.items(),
                        key=lambda x: x[1],
                        reverse=True
                    ))
                    results['shap_feature_importance'] = shap_feature_importance
                except Exception as e:
                    logger.warning(f"Failed to calculate SHAP feature importance: {str(e)}")
            else:
                results['shap_available'] = False
        else:
            results['shap_available'] = False
        
        logger.info(f"Interpretability analysis completed for {model_name}")
        return results
    
    def compare_feature_importance_methods(self, model_name: str) -> pd.DataFrame:
        """Compare different feature importance methods."""
        logger.info(f"Comparing feature importance methods for {model_name}")
        
        comparison_data = []
        
        # Get feature importance from different methods
        methods = {}
        
        if model_name in self.feature_importance:
            methods['Tree_Importance'] = self.feature_importance[model_name]
        
        if model_name in self.permutation_importance:
            methods['Permutation_Importance'] = self.permutation_importance[model_name]['importances_mean']
        
        if model_name in self.shap_values:
            shap_vals = self.shap_values[model_name]
            mean_shap = np.abs(shap_vals).mean(axis=0)
            # Assuming feature names are stored somewhere accessible
            feature_names = list(self.feature_importance.get(model_name, {}).keys())
            if len(feature_names) == len(mean_shap):
                methods['SHAP_Importance'] = dict(zip(feature_names, mean_shap))
        
        # Create comparison DataFrame
        all_features = set()
        for method_dict in methods.values():
            all_features.update(method_dict.keys())
        
        for feature in all_features:
            row = {'Feature': feature}
            for method_name, method_dict in methods.items():
                row[method_name] = method_dict.get(feature, 0)
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Save comparison
        comparison_path = self.plots_path / f"{model_name}_importance_comparison.csv"
        comparison_df.to_csv(comparison_path, index=False)
        
        logger.info(f"Feature importance comparison saved to {comparison_path}")
        return comparison_df
    
    def save_interpretability_results(self, filepath: Optional[str] = None):
        """Save all interpretability results."""
        if filepath is None:
            filepath = Path(self.config.project_root) / "artifacts" / "interpretability_results.joblib"
        
        results_dict = {
            'shap_values': self.shap_values,
            'feature_importance': self.feature_importance,
            'permutation_importance': self.permutation_importance
        }
        
        joblib.dump(results_dict, filepath)
        logger.info(f"Interpretability results saved to {filepath}")


def main():
    """Main function for model interpretability analysis."""
    from .hyperparameter_optimization import BayesianOptimizer
    from ..features.feature_engineering import AdvancedFeatureEngineer
    from ..data.preprocessing import AdvancedPreprocessor
    from ..data.data_loader import DataLoader
    
    # Load and preprocess data
    loader = DataLoader()
    df, _ = loader.load_and_process()
    
    preprocessor = AdvancedPreprocessor()
    X, y = preprocessor.fit_transform(df)
    
    # Feature engineering
    engineer = AdvancedFeatureEngineer()
    X_engineered, _ = engineer.fit_transform(X, y)
    
    # Load optimized models
    optimizer = BayesianOptimizer()
    optimizer.load_optimization_results()
    
    # Create optimized models
    models_to_analyze = {}
    for model_name in optimizer.best_params.keys():
        try:
            model = optimizer.create_optimized_model(model_name)
            model.fit(X_engineered, y)
            models_to_analyze[model_name] = model
        except Exception as e:
            logger.error(f"Failed to create {model_name}: {str(e)}")
    
    # Analyze interpretability
    interpreter = ModelInterpreter()
    
    for model_name, model in models_to_analyze.items():
        model_type = 'tree' if 'forest' in model_name or 'boosting' in model_name else 'linear'
        
        results = interpreter.analyze_model_interpretability(
            model, X_engineered, y, model_name, model_type
        )
        
        print(f"\nInterpretability Analysis for {model_name}:")
        print(f"- Feature importance available: {'Yes' if 'feature_importance' in results else 'No'}")
        print(f"- Permutation importance available: {'Yes' if 'permutation_importance' in results else 'No'}")
        print(f"- SHAP analysis available: {'Yes' if results.get('shap_available', False) else 'No'}")
        
        # Compare importance methods
        comparison_df = interpreter.compare_feature_importance_methods(model_name)
        print(f"- Feature importance comparison saved")
    
    # Save results
    interpreter.save_interpretability_results()
    
    print("\nModel Interpretability Analysis completed!")
    print(f"Plots saved in: {interpreter.plots_path}")
    
    return interpreter


if __name__ == "__main__":
    main()
