"""
Advanced machine learning models for churn prediction.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import lightgbm as lgb
from typing import Dict, List, Tuple, Optional, Any
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from ..utils.config import config
from ..utils.logger import get_logger
from ..utils.exceptions import ModelError

logger = get_logger(__name__)


class AdvancedModelSuite:
    """Suite of advanced machine learning models for churn prediction."""
    
    def __init__(self):
        self.config = config
        self.model_config = self.config.model_config
        
        # Store fitted models
        self.models = {}
        self.model_scores = {}
        self.best_model = None
        self.best_score = 0
        
        # Paths for saving models
        self.models_path = Path(self.config.project_root) / "models"
        self.models_path.mkdir(parents=True, exist_ok=True)
    
    def create_logistic_regression(self) -> LogisticRegression:
        """Create Logistic Regression model with regularization."""
        logger.info("Creating Logistic Regression model")
        
        lr_config = self.model_config.get('logistic_regression', {})
        if not lr_config.get('enabled', True):
            return None
        
        # Use default parameters for now (will be optimized later)
        model = LogisticRegression(
            C=1.0,
            penalty='l2',
            solver='liblinear',
            max_iter=1000,
            random_state=42
        )
        
        return model
    
    def create_xgboost(self) -> xgb.XGBClassifier:
        """Create XGBoost model with early stopping."""
        logger.info("Creating XGBoost model")
        
        xgb_config = self.model_config.get('xgboost', {})
        if not xgb_config.get('enabled', True):
            return None
        
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            eval_metric='logloss',
            early_stopping_rounds=10
        )
        
        return model
    
    def create_random_forest(self) -> RandomForestClassifier:
        """Create Random Forest model."""
        logger.info("Creating Random Forest model")
        
        rf_config = self.model_config.get('random_forest', {})
        if not rf_config.get('enabled', True):
            return None
        
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        
        return model
    
    def create_lightgbm(self) -> lgb.LGBMClassifier:
        """Create LightGBM model."""
        logger.info("Creating LightGBM model")
        
        lgb_config = self.model_config.get('lightgbm', {})
        if not lgb_config.get('enabled', True):
            return None
        
        model = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=7,
            learning_rate=0.1,
            num_leaves=50,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        
        return model
    
    def create_gradient_boosting(self) -> GradientBoostingClassifier:
        """Create Gradient Boosting model."""
        logger.info("Creating Gradient Boosting model")
        
        model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )
        
        return model
    
    def create_voting_ensemble(self, base_models: List) -> VotingClassifier:
        """Create Voting Classifier ensemble."""
        logger.info("Creating Voting Classifier ensemble")
        
        # Filter out None models
        valid_models = [(name, model) for name, model in base_models if model is not None]
        
        if len(valid_models) < 2:
            logger.warning("Not enough models for ensemble. Need at least 2 models.")
            return None
        
        ensemble = VotingClassifier(
            estimators=valid_models,
            voting='soft'
        )
        
        return ensemble
    
    def create_stacking_ensemble(self, base_models: List, meta_learner=None) -> StackingClassifier:
        """Create Stacking Classifier ensemble."""
        logger.info("Creating Stacking Classifier ensemble")
        
        # Filter out None models
        valid_models = [(name, model) for name, model in base_models if model is not None]
        
        if len(valid_models) < 2:
            logger.warning("Not enough models for stacking. Need at least 2 models.")
            return None
        
        if meta_learner is None:
            meta_learner = LogisticRegression(random_state=42)
        
        ensemble = StackingClassifier(
            estimators=valid_models,
            final_estimator=meta_learner,
            cv=5
        )
        
        return ensemble
    
    def evaluate_model(self, model, X: pd.DataFrame, y: pd.Series, cv_folds: int = 5) -> Dict[str, float]:
        """Evaluate model using cross-validation."""
        logger.info(f"Evaluating {model.__class__.__name__}")
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        # Calculate multiple metrics
        scoring_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        scores = {}
        
        for metric in scoring_metrics:
            cv_scores = cross_val_score(model, X, y, cv=cv, scoring=metric)
            scores[f'{metric}_mean'] = cv_scores.mean()
            scores[f'{metric}_std'] = cv_scores.std()
        
        logger.info(f"Model evaluation completed. ROC-AUC: {scores['roc_auc_mean']:.4f} ± {scores['roc_auc_std']:.4f}")
        return scores
    
    def fit_all_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Fit all models and evaluate them."""
        logger.info("Fitting all models")
        
        # Create individual models
        models_to_fit = {
            'logistic_regression': self.create_logistic_regression(),
            'xgboost': self.create_xgboost(),
            'random_forest': self.create_random_forest(),
            'lightgbm': self.create_lightgbm(),
            'gradient_boosting': self.create_gradient_boosting()
        }
        
        # Fit and evaluate individual models
        for name, model in models_to_fit.items():
            if model is not None:
                try:
                    logger.info(f"Fitting {name}")
                    
                    # Special handling for XGBoost with early stopping
                    if name == 'xgboost':
                        # Split data for early stopping
                        from sklearn.model_selection import train_test_split
                        X_train, X_val, y_train, y_val = train_test_split(
                            X, y, test_size=0.2, random_state=42, stratify=y
                        )
                        model.fit(
                            X_train, y_train,
                            eval_set=[(X_val, y_val)],
                            verbose=False
                        )
                    else:
                        model.fit(X, y)
                    
                    # Evaluate model
                    scores = self.evaluate_model(model, X, y)
                    
                    self.models[name] = model
                    self.model_scores[name] = scores
                    
                    # Track best model
                    if scores['roc_auc_mean'] > self.best_score:
                        self.best_score = scores['roc_auc_mean']
                        self.best_model = name
                    
                except Exception as e:
                    logger.error(f"Failed to fit {name}: {str(e)}")
        
        # Create ensemble models
        base_models = [(name, model) for name, model in self.models.items()]
        
        # Voting ensemble
        voting_ensemble = self.create_voting_ensemble(base_models)
        if voting_ensemble is not None:
            try:
                logger.info("Fitting Voting Ensemble")
                voting_ensemble.fit(X, y)
                scores = self.evaluate_model(voting_ensemble, X, y)
                
                self.models['voting_ensemble'] = voting_ensemble
                self.model_scores['voting_ensemble'] = scores
                
                if scores['roc_auc_mean'] > self.best_score:
                    self.best_score = scores['roc_auc_mean']
                    self.best_model = 'voting_ensemble'
                    
            except Exception as e:
                logger.error(f"Failed to fit voting ensemble: {str(e)}")
        
        # Stacking ensemble
        stacking_ensemble = self.create_stacking_ensemble(base_models)
        if stacking_ensemble is not None:
            try:
                logger.info("Fitting Stacking Ensemble")
                stacking_ensemble.fit(X, y)
                scores = self.evaluate_model(stacking_ensemble, X, y)
                
                self.models['stacking_ensemble'] = stacking_ensemble
                self.model_scores['stacking_ensemble'] = scores
                
                if scores['roc_auc_mean'] > self.best_score:
                    self.best_score = scores['roc_auc_mean']
                    self.best_model = 'stacking_ensemble'
                    
            except Exception as e:
                logger.error(f"Failed to fit stacking ensemble: {str(e)}")
        
        logger.info(f"All models fitted. Best model: {self.best_model} with ROC-AUC: {self.best_score:.4f}")
        
        return {
            'models': list(self.models.keys()),
            'best_model': self.best_model,
            'best_score': self.best_score,
            'all_scores': self.model_scores
        }
    
    def save_models(self, filepath: Optional[str] = None):
        """Save all fitted models."""
        if filepath is None:
            filepath = self.models_path / "trained_models.joblib"
        
        models_dict = {
            'models': self.models,
            'model_scores': self.model_scores,
            'best_model': self.best_model,
            'best_score': self.best_score
        }
        
        joblib.dump(models_dict, filepath)
        logger.info(f"Models saved to {filepath}")
    
    def load_models(self, filepath: Optional[str] = None):
        """Load fitted models."""
        if filepath is None:
            filepath = self.models_path / "trained_models.joblib"
        
        if not Path(filepath).exists():
            raise ModelError(f"Models file not found: {filepath}")
        
        models_dict = joblib.load(filepath)
        
        self.models = models_dict.get('models', {})
        self.model_scores = models_dict.get('model_scores', {})
        self.best_model = models_dict.get('best_model')
        self.best_score = models_dict.get('best_score', 0)
        
        logger.info(f"Models loaded from {filepath}")
    
    def get_model_summary(self) -> pd.DataFrame:
        """Get summary of all model performances."""
        summary_data = []
        
        for model_name, scores in self.model_scores.items():
            summary_data.append({
                'Model': model_name,
                'ROC-AUC': f"{scores['roc_auc_mean']:.4f} ± {scores['roc_auc_std']:.4f}",
                'Accuracy': f"{scores['accuracy_mean']:.4f} ± {scores['accuracy_std']:.4f}",
                'Precision': f"{scores['precision_mean']:.4f} ± {scores['precision_std']:.4f}",
                'Recall': f"{scores['recall_mean']:.4f} ± {scores['recall_std']:.4f}",
                'F1-Score': f"{scores['f1_mean']:.4f} ± {scores['f1_std']:.4f}"
            })
        
        return pd.DataFrame(summary_data).sort_values('ROC-AUC', ascending=False)


def main():
    """Main function for model training."""
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
    
    # Train models
    model_suite = AdvancedModelSuite()
    results = model_suite.fit_all_models(X_engineered, y)
    
    # Save models
    model_suite.save_models()
    
    # Print results
    print("Model Training completed!")
    print(f"Results: {results}")
    
    summary = model_suite.get_model_summary()
    print("\nModel Performance Summary:")
    print(summary.to_string(index=False))
    
    return model_suite, results


if __name__ == "__main__":
    main()
