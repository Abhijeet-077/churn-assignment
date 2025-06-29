"""
Bayesian hyperparameter optimization for churn prediction models.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import make_scorer, roc_auc_score
import xgboost as xgb
import lightgbm as lgb
from typing import Dict, List, Tuple, Optional, Any
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Try to import Optuna, fallback to sklearn methods if not available
try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    optuna = None

from ..utils.config import config
from ..utils.logger import get_logger
from ..utils.exceptions import ModelError

logger = get_logger(__name__)


class BayesianOptimizer:
    """Bayesian hyperparameter optimization using Optuna or sklearn methods."""
    
    def __init__(self):
        self.config = config
        self.optimization_config = self.config.get('optimization', {})
        
        # Store optimization results
        self.optimization_results = {}
        self.best_params = {}
        self.best_scores = {}
        
        # Paths for saving results
        self.artifacts_path = Path(self.config.project_root) / "artifacts" / "optimization"
        self.artifacts_path.mkdir(parents=True, exist_ok=True)
    
    def create_parameter_space(self, model_name: str) -> Dict[str, Any]:
        """Create parameter space for different models."""
        model_config = self.config.model_config.get(model_name, {})
        params = model_config.get('params', {})
        
        if model_name == 'logistic_regression':
            return {
                'C': params.get('C', [0.001, 0.01, 0.1, 1, 10, 100]),
                'penalty': params.get('penalty', ['l1', 'l2']),
                'solver': params.get('solver', ['liblinear', 'saga']),
                'max_iter': [1000]
            }
        
        elif model_name == 'random_forest':
            return {
                'n_estimators': params.get('n_estimators', [100, 200, 300]),
                'max_depth': params.get('max_depth', [10, 20, 30, None]),
                'min_samples_split': params.get('min_samples_split', [2, 5, 10]),
                'min_samples_leaf': params.get('min_samples_leaf', [1, 2, 4]),
                'max_features': params.get('max_features', ['sqrt', 'log2', None])
            }
        
        elif model_name == 'gradient_boosting':
            return {
                'n_estimators': params.get('n_estimators', [100, 200, 300]),
                'max_depth': params.get('max_depth', [3, 5, 7]),
                'learning_rate': params.get('learning_rate', [0.01, 0.05, 0.1, 0.2]),
                'subsample': params.get('subsample', [0.8, 0.9, 1.0])
            }
        
        elif model_name == 'xgboost':
            return {
                'n_estimators': params.get('n_estimators', [100, 200, 300]),
                'max_depth': params.get('max_depth', [3, 4, 5, 6]),
                'learning_rate': params.get('learning_rate', [0.01, 0.05, 0.1, 0.2]),
                'subsample': params.get('subsample', [0.8, 0.9, 1.0]),
                'colsample_bytree': params.get('colsample_bytree', [0.8, 0.9, 1.0])
            }
        
        else:
            return {}
    
    def optimize_with_optuna(self, model, X: pd.DataFrame, y: pd.Series, 
                           model_name: str, n_trials: int = 100) -> Dict[str, Any]:
        """Optimize hyperparameters using Optuna."""
        if not OPTUNA_AVAILABLE:
            logger.warning("Optuna not available. Falling back to RandomizedSearchCV.")
            return self.optimize_with_sklearn(model, X, y, model_name)
        
        logger.info(f"Starting Optuna optimization for {model_name} with {n_trials} trials")
        
        def objective(trial):
            # Define parameter suggestions based on model type
            if model_name == 'logistic_regression':
                params = {
                    'C': trial.suggest_float('C', 0.001, 100, log=True),
                    'penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),
                    'solver': trial.suggest_categorical('solver', ['liblinear', 'saga']),
                    'max_iter': 1000,
                    'random_state': 42
                }
            
            elif model_name == 'random_forest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                    'max_depth': trial.suggest_int('max_depth', 5, 50),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                    'random_state': 42,
                    'n_jobs': -1
                }
            
            elif model_name == 'gradient_boosting':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'random_state': 42
                }
            
            else:
                raise ValueError(f"Unsupported model for Optuna optimization: {model_name}")
            
            # Create model with suggested parameters
            if model_name == 'logistic_regression':
                model_instance = LogisticRegression(**params)
            elif model_name == 'random_forest':
                model_instance = RandomForestClassifier(**params)
            elif model_name == 'gradient_boosting':
                model_instance = GradientBoostingClassifier(**params)
            
            # Perform cross-validation
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_score(model_instance, X, y, cv=cv, scoring='roc_auc')
            
            return scores.mean()
        
        # Create study and optimize
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42)
        )
        
        study.optimize(objective, n_trials=n_trials, timeout=3600)  # 1 hour timeout
        
        results = {
            'best_params': study.best_params,
            'best_score': study.best_value,
            'n_trials': len(study.trials),
            'optimization_method': 'optuna'
        }
        
        logger.info(f"Optuna optimization completed. Best score: {study.best_value:.4f}")
        return results
    
    def optimize_with_sklearn(self, model, X: pd.DataFrame, y: pd.Series, 
                            model_name: str, n_iter: int = 50) -> Dict[str, Any]:
        """Optimize hyperparameters using sklearn RandomizedSearchCV."""
        logger.info(f"Starting sklearn optimization for {model_name} with {n_iter} iterations")
        
        param_space = self.create_parameter_space(model_name)
        
        if not param_space:
            logger.warning(f"No parameter space defined for {model_name}")
            return {'best_params': {}, 'best_score': 0, 'optimization_method': 'none'}
        
        # Use RandomizedSearchCV for efficiency
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_space,
            n_iter=n_iter,
            cv=cv,
            scoring='roc_auc',
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        
        search.fit(X, y)
        
        results = {
            'best_params': search.best_params_,
            'best_score': search.best_score_,
            'n_iterations': n_iter,
            'optimization_method': 'randomized_search'
        }
        
        logger.info(f"Sklearn optimization completed. Best score: {search.best_score_:.4f}")
        return results
    
    def optimize_model(self, model, X: pd.DataFrame, y: pd.Series, 
                      model_name: str) -> Dict[str, Any]:
        """Optimize a single model using the best available method."""
        n_trials = self.optimization_config.get('n_trials', 100)
        
        if OPTUNA_AVAILABLE:
            results = self.optimize_with_optuna(model, X, y, model_name, n_trials)
        else:
            results = self.optimize_with_sklearn(model, X, y, model_name, n_trials)
        
        # Store results
        self.optimization_results[model_name] = results
        self.best_params[model_name] = results['best_params']
        self.best_scores[model_name] = results['best_score']
        
        return results
    
    def optimize_all_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Optimize all enabled models."""
        logger.info("Starting hyperparameter optimization for all models")
        
        models_to_optimize = {
            'logistic_regression': LogisticRegression(random_state=42),
            'random_forest': RandomForestClassifier(random_state=42, n_jobs=-1),
            'gradient_boosting': GradientBoostingClassifier(random_state=42)
        }
        
        all_results = {}
        
        for model_name, model in models_to_optimize.items():
            if self.config.model_config.get(model_name, {}).get('enabled', True):
                try:
                    logger.info(f"Optimizing {model_name}")
                    results = self.optimize_model(model, X, y, model_name)
                    all_results[model_name] = results
                except Exception as e:
                    logger.error(f"Failed to optimize {model_name}: {str(e)}")
                    all_results[model_name] = {'error': str(e)}
        
        # Find best overall model
        best_model = None
        best_score = 0
        
        for model_name, results in all_results.items():
            if 'best_score' in results and results['best_score'] > best_score:
                best_score = results['best_score']
                best_model = model_name
        
        summary = {
            'optimization_results': all_results,
            'best_model': best_model,
            'best_score': best_score,
            'optimization_method': 'optuna' if OPTUNA_AVAILABLE else 'sklearn'
        }
        
        logger.info(f"Optimization completed. Best model: {best_model} with score: {best_score:.4f}")
        return summary
    
    def create_optimized_model(self, model_name: str) -> Any:
        """Create a model with optimized parameters."""
        if model_name not in self.best_params:
            raise ModelError(f"No optimized parameters found for {model_name}")
        
        params = self.best_params[model_name]
        
        if model_name == 'logistic_regression':
            return LogisticRegression(**params, random_state=42)
        elif model_name == 'random_forest':
            return RandomForestClassifier(**params, random_state=42, n_jobs=-1)
        elif model_name == 'gradient_boosting':
            return GradientBoostingClassifier(**params, random_state=42)
        else:
            raise ModelError(f"Unsupported model: {model_name}")
    
    def save_optimization_results(self, filepath: Optional[str] = None):
        """Save optimization results."""
        if filepath is None:
            filepath = self.artifacts_path / "optimization_results.joblib"
        
        results_dict = {
            'optimization_results': self.optimization_results,
            'best_params': self.best_params,
            'best_scores': self.best_scores
        }
        
        joblib.dump(results_dict, filepath)
        logger.info(f"Optimization results saved to {filepath}")
    
    def load_optimization_results(self, filepath: Optional[str] = None):
        """Load optimization results."""
        if filepath is None:
            filepath = self.artifacts_path / "optimization_results.joblib"
        
        if not Path(filepath).exists():
            raise ModelError(f"Optimization results file not found: {filepath}")
        
        results_dict = joblib.load(filepath)
        
        self.optimization_results = results_dict.get('optimization_results', {})
        self.best_params = results_dict.get('best_params', {})
        self.best_scores = results_dict.get('best_scores', {})
        
        logger.info(f"Optimization results loaded from {filepath}")
    
    def get_optimization_summary(self) -> pd.DataFrame:
        """Get summary of optimization results."""
        summary_data = []
        
        for model_name, results in self.optimization_results.items():
            if 'best_score' in results:
                summary_data.append({
                    'Model': model_name,
                    'Best_Score': f"{results['best_score']:.4f}",
                    'Optimization_Method': results.get('optimization_method', 'unknown'),
                    'Trials/Iterations': results.get('n_trials', results.get('n_iterations', 'unknown'))
                })
        
        return pd.DataFrame(summary_data).sort_values('Best_Score', ascending=False)


def main():
    """Main function for hyperparameter optimization."""
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
    
    # Hyperparameter optimization
    optimizer = BayesianOptimizer()
    results = optimizer.optimize_all_models(X_engineered, y)
    
    # Save results
    optimizer.save_optimization_results()
    
    # Print results
    print("Hyperparameter Optimization completed!")
    print(f"Results: {results}")
    
    summary = optimizer.get_optimization_summary()
    print("\nOptimization Summary:")
    print(summary.to_string(index=False))
    
    return optimizer, results


if __name__ == "__main__":
    main()
