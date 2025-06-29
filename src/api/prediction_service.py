"""
Prediction service for handling model inference and explanations.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import time
from datetime import datetime
import logging

from .models import (
    CustomerData, PredictionResponse, DetailedPredictionResponse,
    ExplanationResponse, ModelInfo
)
from ..utils.config import config
from ..utils.logger import get_logger

logger = get_logger(__name__)


class PredictionService:
    """Service for handling model predictions and explanations."""
    
    def __init__(self):
        self.config = config
        self.models = {}
        self.preprocessors = {}
        self.feature_engineers = {}
        self.model_info = {}
        self.feature_importance = {}
        self.startup_time = time.time()
        
        # Load models and preprocessors
        self._load_models()
        self._load_preprocessors()
        self._train_models()  # Train models with data
        self._load_model_metadata()
        
    def _load_models(self):
        """Load trained models."""
        logger.info("Loading trained models...")
        
        try:
            # Load optimization results to get best models
            optimization_path = Path(self.config.project_root) / "artifacts" / "optimization" / "optimization_results.joblib"
            
            if optimization_path.exists():
                optimization_results = joblib.load(optimization_path)
                
                # Load best models
                from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
                from sklearn.linear_model import LogisticRegression
                
                # Create models with best parameters
                for model_name, results in optimization_results['optimization_results'].items():
                    try:
                        best_params = results['best_params']
                        
                        if model_name == 'logistic_regression':
                            model = LogisticRegression(**best_params, random_state=42)
                        elif model_name == 'random_forest':
                            model = RandomForestClassifier(**best_params, random_state=42)
                        elif model_name == 'gradient_boosting':
                            model = GradientBoostingClassifier(**best_params, random_state=42)
                        else:
                            continue
                        
                        # For now, we'll create the model with best params
                        # In production, you would load the fitted model
                        logger.info(f"Created model {model_name} with optimized parameters")
                        
                        self.models[model_name] = model
                        logger.info(f"Loaded model: {model_name}")
                        
                    except Exception as e:
                        logger.error(f"Failed to load model {model_name}: {str(e)}")
                
                # Set best model
                best_model_name = optimization_results.get('best_model', 'gradient_boosting')
                if best_model_name in self.models:
                    self.models['best'] = self.models[best_model_name]
                    logger.info(f"Best model set to: {best_model_name}")
                
            else:
                logger.warning("Optimization results not found, loading default models")
                self._load_default_models()
                
        except Exception as e:
            logger.error(f"Failed to load models: {str(e)}")
            self._load_default_models()
    
    def _load_default_models(self):
        """Load default models if optimization results not available."""
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        
        # Default models with reasonable parameters
        self.models = {
            'logistic_regression': LogisticRegression(random_state=42),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingClassifier(random_state=42)
        }
        self.models['best'] = self.models['gradient_boosting']
        logger.info("Loaded default models")
    
    def _load_preprocessors(self):
        """Load preprocessors and feature engineers."""
        logger.info("Loading preprocessors...")
        
        try:
            # Load preprocessor
            preprocessor_path = Path(self.config.project_root) / "artifacts" / "preprocessor.joblib"
            if preprocessor_path.exists():
                self.preprocessors['main'] = joblib.load(preprocessor_path)
                logger.info("Loaded main preprocessor")
            
            # Load feature engineer
            engineer_path = Path(self.config.project_root) / "artifacts" / "feature_engineer.joblib"
            if engineer_path.exists():
                self.feature_engineers['main'] = joblib.load(engineer_path)
                logger.info("Loaded feature engineer")

        except Exception as e:
            logger.error(f"Failed to load preprocessors: {str(e)}")

    def _train_models(self):
        """Train models with data."""
        logger.info("Training models with data...")

        try:
            # Load and preprocess data for training
            from ..data.data_loader import DataLoader
            from ..data.preprocessing import AdvancedPreprocessor
            from ..features.feature_engineering import AdvancedFeatureEngineer

            # Load data
            loader = DataLoader()
            df, _ = loader.load_and_process()

            # Preprocess
            if 'main' in self.preprocessors:
                preprocessor = self.preprocessors['main']
                X, y = preprocessor.transform(df)
            else:
                preprocessor = AdvancedPreprocessor()
                X, y = preprocessor.fit_transform(df)
                self.preprocessors['main'] = preprocessor

            # Feature engineering
            if 'main' in self.feature_engineers:
                engineer = self.feature_engineers['main']
                X_engineered, _ = engineer.transform(X, y)
            else:
                engineer = AdvancedFeatureEngineer()
                X_engineered, _ = engineer.fit_transform(X, y)
                self.feature_engineers['main'] = engineer

            # Train all models
            for model_name, model in self.models.items():
                if model_name != 'best':  # Skip the 'best' alias
                    logger.info(f"Training {model_name}...")
                    model.fit(X_engineered, y)
                    logger.info(f"✓ {model_name} trained successfully")

            logger.info("All models trained successfully")

        except Exception as e:
            logger.error(f"Failed to train models: {str(e)}")
            # Create simple fallback models
            self._create_fallback_models()

    def _create_fallback_models(self):
        """Create simple fallback models with dummy data."""
        logger.warning("Creating fallback models with dummy data")

        try:
            from sklearn.datasets import make_classification

            # Create dummy data
            X_dummy, y_dummy = make_classification(
                n_samples=1000, n_features=20, n_classes=2, random_state=42
            )

            # Train models on dummy data
            for model_name, model in self.models.items():
                if model_name != 'best':
                    model.fit(X_dummy, y_dummy)
                    logger.info(f"✓ {model_name} trained with dummy data")

        except Exception as e:
            logger.error(f"Failed to create fallback models: {str(e)}")
    
    def _load_model_metadata(self):
        """Load model metadata and performance metrics."""
        logger.info("Loading model metadata...")
        
        try:
            # Load evaluation results
            evaluation_path = Path(self.config.project_root) / "artifacts" / "evaluation_results.joblib"
            if evaluation_path.exists():
                evaluation_results = joblib.load(evaluation_path)
                
                for model_name, metrics in evaluation_results.items():
                    if isinstance(metrics, dict) and 'test_metrics' in metrics:
                        test_metrics = metrics['test_metrics']
                        self.model_info[model_name] = ModelInfo(
                            model_name=model_name,
                            model_type=self._get_model_type(model_name),
                            accuracy=test_metrics.get('accuracy', 0.0),
                            roc_auc=test_metrics.get('roc_auc', 0.0),
                            precision=test_metrics.get('precision', 0.0),
                            recall=test_metrics.get('recall', 0.0),
                            f1_score=test_metrics.get('f1_score', 0.0),
                            training_date=datetime.now().isoformat(),
                            feature_count=len(self._get_feature_names())
                        )
                
                logger.info(f"Loaded metadata for {len(self.model_info)} models")
            
            # Load feature importance
            importance_path = Path(self.config.project_root) / "artifacts" / "interpretability_results.joblib"
            if importance_path.exists():
                importance_results = joblib.load(importance_path)
                self.feature_importance = importance_results
                logger.info("Loaded feature importance data")
                
        except Exception as e:
            logger.error(f"Failed to load model metadata: {str(e)}")
    
    def _get_model_type(self, model_name: str) -> str:
        """Get model type string."""
        type_mapping = {
            'logistic_regression': 'Linear Model',
            'random_forest': 'Ensemble (Random Forest)',
            'gradient_boosting': 'Ensemble (Gradient Boosting)',
            'best': 'Best Performing Model'
        }
        return type_mapping.get(model_name, 'Unknown')
    
    def _get_feature_names(self) -> List[str]:
        """Get feature names from feature engineer."""
        try:
            if 'main' in self.feature_engineers:
                engineer = self.feature_engineers['main']
                if hasattr(engineer, 'selected_features_'):
                    return engineer.selected_features_
                elif hasattr(engineer, 'feature_names_'):
                    return engineer.feature_names_
            return []
        except:
            return []
    
    def _preprocess_customer_data(self, customer_data: CustomerData) -> pd.DataFrame:
        """Preprocess customer data for prediction."""
        # Convert to DataFrame
        data_dict = {
            'customerid': 'temp_id',
            'gender': customer_data.gender.value,
            'seniorcitizen': customer_data.senior_citizen,
            'partner': customer_data.partner.value,
            'dependents': customer_data.dependents.value,
            'tenure': customer_data.tenure,
            'phoneservice': customer_data.phone_service.value,
            'multiplelines': customer_data.multiple_lines,
            'internetservice': customer_data.internet_service.value,
            'onlinesecurity': customer_data.online_security,
            'onlinebackup': customer_data.online_backup,
            'deviceprotection': customer_data.device_protection,
            'techsupport': customer_data.tech_support,
            'streamingtv': customer_data.streaming_tv,
            'streamingmovies': customer_data.streaming_movies,
            'contract': customer_data.contract.value,
            'paperlessbilling': customer_data.paperless_billing.value,
            'paymentmethod': customer_data.payment_method.value,
            'monthlycharges': customer_data.monthly_charges,
            'totalcharges': customer_data.total_charges,
            'churn': 'No'  # Dummy target for preprocessing
        }
        
        df = pd.DataFrame([data_dict])
        
        # Apply preprocessing if available
        if 'main' in self.preprocessors:
            try:
                preprocessor = self.preprocessors['main']
                X, _ = preprocessor.transform(df)
                
                # Apply feature engineering if available
                if 'main' in self.feature_engineers:
                    engineer = self.feature_engineers['main']
                    X_engineered, _ = engineer.transform(X, None)
                    return X_engineered
                else:
                    return X
            except Exception as e:
                logger.error(f"Preprocessing failed: {str(e)}")
                # Fallback to basic preprocessing
                return self._basic_preprocessing(df)
        else:
            return self._basic_preprocessing(df)
    
    def _basic_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic preprocessing fallback."""
        # Remove non-feature columns
        feature_cols = [col for col in df.columns if col not in ['customerid', 'churn']]
        X = df[feature_cols].copy()
        
        # Basic encoding for categorical variables
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            X[col] = pd.Categorical(X[col]).codes
        
        return X
    
    def predict_single(self, customer_data: CustomerData, model_name: str = 'best', 
                      include_explanation: bool = False) -> DetailedPredictionResponse:
        """Make prediction for a single customer."""
        start_time = time.time()
        
        try:
            # Preprocess data
            X = self._preprocess_customer_data(customer_data)
            
            # Get model
            if model_name not in self.models:
                model_name = 'best'
            
            model = self.models[model_name]
            
            # Make prediction
            prediction_proba = model.predict_proba(X)[0]
            churn_probability = prediction_proba[1]  # Probability of churn (class 1)
            churn_prediction = "Yes" if churn_probability > 0.5 else "No"
            
            # Determine confidence and risk level
            confidence = self._get_confidence_level(churn_probability)
            risk_level = self._get_risk_level(churn_probability)
            
            # Create base response
            response = DetailedPredictionResponse(
                churn_probability=float(churn_probability),
                churn_prediction=churn_prediction,
                confidence=confidence,
                risk_level=risk_level,
                model_used=model_name,
                feature_values=customer_data.dict(),
                processing_time_ms=(time.time() - start_time) * 1000
            )
            
            # Add explanation if requested
            if include_explanation:
                response.explanation = self._generate_explanation(
                    customer_data, X, churn_probability, model_name
                )
            
            return response
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise ValueError(f"Prediction failed: {str(e)}")
    
    def _get_confidence_level(self, probability: float) -> str:
        """Get confidence level based on probability."""
        if probability < 0.3 or probability > 0.7:
            return "High"
        elif probability < 0.4 or probability > 0.6:
            return "Medium"
        else:
            return "Low"
    
    def _get_risk_level(self, probability: float) -> str:
        """Get risk level based on churn probability."""
        if probability >= 0.8:
            return "Very High"
        elif probability >= 0.6:
            return "High"
        elif probability >= 0.4:
            return "Medium"
        elif probability >= 0.2:
            return "Low"
        else:
            return "Very Low"
    
    def _generate_explanation(self, customer_data: CustomerData, X: pd.DataFrame, 
                            probability: float, model_name: str) -> ExplanationResponse:
        """Generate prediction explanation."""
        try:
            # Get feature importance for the model
            feature_importance = {}

            if model_name in self.feature_importance:
                importance_data = self.feature_importance[model_name]
                if 'feature_importance' in importance_data:
                    feature_importance = importance_data['feature_importance']
                elif 'permutation_importance' in importance_data:
                    feature_importance = importance_data['permutation_importance']  # Direct dict
                elif 'shap_importance' in importance_data:
                    feature_importance = importance_data['shap_importance']
            
            # Get top contributing factors
            top_positive = []
            top_negative = []
            
            # Simple heuristic-based explanation
            if customer_data.contract.value == "Month-to-month":
                top_positive.append("Month-to-month contract (higher churn risk)")
            
            if customer_data.tenure < 12:
                top_positive.append("Low tenure (new customers more likely to churn)")
            else:
                top_negative.append("High tenure (loyal customer)")
            
            if customer_data.monthly_charges > 70:
                top_positive.append("High monthly charges")
            
            if customer_data.total_charges < customer_data.monthly_charges * 6:
                top_positive.append("Low total charges (new customer)")
            
            # Generate explanation text
            risk_level = self._get_risk_level(probability)
            explanation_text = f"This customer has a {risk_level.lower()} risk of churning " \
                             f"with a probability of {probability:.1%}. "
            
            if probability > 0.5:
                explanation_text += "Key risk factors include: " + ", ".join(top_positive[:3])
            else:
                explanation_text += "Retention factors include: " + ", ".join(top_negative[:3])
            
            return ExplanationResponse(
                feature_contributions=feature_importance,
                top_positive_factors=top_positive,
                top_negative_factors=top_negative,
                explanation_text=explanation_text
            )
            
        except Exception as e:
            logger.error(f"Explanation generation failed: {str(e)}")
            return ExplanationResponse(
                feature_contributions={},
                top_positive_factors=[],
                top_negative_factors=[],
                explanation_text="Explanation not available"
            )
    
    def predict_batch(self, customers: List[CustomerData], model_name: str = 'best') -> Dict[str, Any]:
        """Make batch predictions."""
        predictions = []
        
        for i, customer_data in enumerate(customers):
            try:
                prediction = self.predict_single(customer_data, model_name, include_explanation=False)
                prediction.customer_id = f"customer_{i+1}"
                predictions.append(prediction)
            except Exception as e:
                logger.error(f"Failed to predict for customer {i+1}: {str(e)}")
                # Add error prediction
                predictions.append(PredictionResponse(
                    customer_id=f"customer_{i+1}",
                    churn_probability=0.0,
                    churn_prediction="Error",
                    confidence="Low",
                    risk_level="Unknown",
                    model_used=model_name
                ))
        
        # Calculate summary
        successful_predictions = [p for p in predictions if p.churn_prediction != "Error"]
        churn_count = sum(1 for p in successful_predictions if p.churn_prediction == "Yes")
        
        summary = {
            "total_customers": len(customers),
            "successful_predictions": len(successful_predictions),
            "failed_predictions": len(customers) - len(successful_predictions),
            "predicted_churners": churn_count,
            "predicted_non_churners": len(successful_predictions) - churn_count,
            "average_churn_probability": np.mean([p.churn_probability for p in successful_predictions]) if successful_predictions else 0.0,
            "high_risk_customers": sum(1 for p in successful_predictions if p.risk_level in ["High", "Very High"])
        }
        
        return {
            "predictions": predictions,
            "summary": summary
        }
    
    def get_model_info(self, model_name: str = 'best') -> ModelInfo:
        """Get model information."""
        if model_name in self.model_info:
            return self.model_info[model_name]
        else:
            # Return basic info
            return ModelInfo(
                model_name=model_name,
                model_type=self._get_model_type(model_name),
                accuracy=0.0,
                roc_auc=0.0,
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                training_date=datetime.now().isoformat(),
                feature_count=len(self._get_feature_names())
            )
    
    def get_feature_importance(self, model_name: str = 'best') -> Dict[str, Any]:
        """Get feature importance for a model."""
        if model_name in self.feature_importance:
            importance_data = self.feature_importance[model_name]

            # Get the best available importance method
            # The data is stored as direct dictionaries (feature_name -> importance_value)
            if 'feature_importance' in importance_data:
                importance = importance_data['feature_importance']
            elif 'permutation_importance' in importance_data:
                importance = importance_data['permutation_importance']  # Direct dict, no nested structure
            elif 'shap_importance' in importance_data:
                importance = importance_data['shap_importance']
            else:
                importance = {}

            # Sort by importance value (descending)
            if importance:
                sorted_importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
                top_features = list(sorted_importance.keys())[:10]
            else:
                sorted_importance = {}
                top_features = []

            return {
                "model_name": model_name,
                "feature_importance": sorted_importance,
                "top_features": top_features,
                "available_methods": list(importance_data.keys()),
                "total_features": len(sorted_importance)
            }
        else:
            return {
                "model_name": model_name,
                "feature_importance": {},
                "top_features": [],
                "available_methods": [],
                "total_features": 0
            }
    
    def get_available_models(self) -> List[str]:
        """Get list of available models."""
        return list(self.models.keys())
    
    def get_uptime(self) -> float:
        """Get service uptime in seconds."""
        return time.time() - self.startup_time
