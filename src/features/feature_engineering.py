"""
Advanced feature engineering module for churn prediction.
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, RFE, SelectFromModel, f_classif, chi2
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import PolynomialFeatures, KBinsDiscretizer
from sklearn.model_selection import cross_val_score
from typing import Dict, List, Tuple, Optional, Any
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from ..utils.config import config
from ..utils.logger import get_logger
from ..utils.exceptions import FeatureEngineeringError

logger = get_logger(__name__)


class AdvancedFeatureEngineer:
    """Advanced feature engineering and selection pipeline."""
    
    def __init__(self):
        self.config = config
        self.feature_config = self.config.feature_config
        
        # Store fitted transformers
        self.feature_selectors = {}
        self.feature_transformers = {}
        self.engineered_features = []
        self.selected_features = []
        
        # Paths for saving artifacts
        self.artifacts_path = Path(self.config.project_root) / "artifacts" / "features"
        self.artifacts_path.mkdir(parents=True, exist_ok=True)
    
    def create_interaction_features(self, df: pd.DataFrame, max_interactions: int = 10) -> pd.DataFrame:
        """
        Create interaction features between numerical variables.
        
        Args:
            df: Input DataFrame
            max_interactions: Maximum number of interaction features to create
            
        Returns:
            DataFrame with interaction features
        """
        logger.info("Creating interaction features")
        
        df_interactions = df.copy()
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target column if present
        target_col = self.config.get('data.target_column', 'churn')
        if target_col in numerical_cols:
            numerical_cols.remove(target_col)
        
        interaction_count = 0
        interactions_created = []
        
        # Create pairwise interactions
        for i, col1 in enumerate(numerical_cols):
            for col2 in numerical_cols[i+1:]:
                if interaction_count >= max_interactions:
                    break
                
                # Multiplication interaction
                interaction_name = f"{col1}_x_{col2}"
                df_interactions[interaction_name] = df[col1] * df[col2]
                interactions_created.append(interaction_name)
                interaction_count += 1
                
                if interaction_count >= max_interactions:
                    break
            
            if interaction_count >= max_interactions:
                break
        
        self.engineered_features.extend(interactions_created)
        logger.info(f"Created {len(interactions_created)} interaction features: {interactions_created}")
        
        return df_interactions
    
    def create_polynomial_features(self, df: pd.DataFrame, degree: int = 2, max_features: int = 5) -> pd.DataFrame:
        """
        Create polynomial features for selected numerical columns.
        
        Args:
            df: Input DataFrame
            degree: Polynomial degree
            max_features: Maximum number of features to apply polynomial transformation
            
        Returns:
            DataFrame with polynomial features
        """
        logger.info(f"Creating polynomial features with degree {degree}")
        
        df_poly = df.copy()
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target column if present
        target_col = self.config.get('data.target_column', 'churn')
        if target_col in numerical_cols:
            numerical_cols.remove(target_col)
        
        # Select top features based on variance
        selected_cols = numerical_cols[:max_features]
        
        poly_features_created = []
        
        for col in selected_cols:
            # Create squared terms
            if degree >= 2:
                poly_name = f"{col}_squared"
                df_poly[poly_name] = df[col] ** 2
                poly_features_created.append(poly_name)
            
            # Create cubic terms
            if degree >= 3:
                poly_name = f"{col}_cubed"
                df_poly[poly_name] = df[col] ** 3
                poly_features_created.append(poly_name)
        
        self.engineered_features.extend(poly_features_created)
        logger.info(f"Created {len(poly_features_created)} polynomial features: {poly_features_created}")
        
        return df_poly
    
    def create_binned_features(self, df: pd.DataFrame, n_bins: int = 5) -> pd.DataFrame:
        """
        Create binned features for numerical variables.
        
        Args:
            df: Input DataFrame
            n_bins: Number of bins
            
        Returns:
            DataFrame with binned features
        """
        logger.info(f"Creating binned features with {n_bins} bins")
        
        df_binned = df.copy()
        binning_features = self.feature_config.get('binning_features', [])
        
        binned_features_created = []
        
        for col in binning_features:
            if col in df.columns:
                # Create bins using quantile-based strategy
                discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
                
                try:
                    binned_values = discretizer.fit_transform(df[[col]])
                    binned_col_name = f"{col}_binned"
                    df_binned[binned_col_name] = binned_values.flatten()
                    binned_features_created.append(binned_col_name)
                    
                    # Store the discretizer
                    self.feature_transformers[f'discretizer_{col}'] = discretizer
                    
                except Exception as e:
                    logger.warning(f"Failed to create bins for {col}: {str(e)}")
        
        self.engineered_features.extend(binned_features_created)
        logger.info(f"Created {len(binned_features_created)} binned features: {binned_features_created}")
        
        return df_binned
    
    def create_ratio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create ratio features between related numerical variables.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with ratio features
        """
        logger.info("Creating ratio features")
        
        df_ratios = df.copy()
        ratio_features_created = []
        
        # Create meaningful ratios for telecom data
        if 'monthlycharges' in df.columns and 'tenure' in df.columns:
            # Total charges per month of tenure
            df_ratios['charges_per_tenure'] = df['monthlycharges'] / (df['tenure'] + 1)  # +1 to avoid division by zero
            ratio_features_created.append('charges_per_tenure')
        
        if 'totalcharges' in df.columns and 'monthlycharges' in df.columns:
            # Ratio of total to monthly charges
            df_ratios['total_to_monthly_ratio'] = df['totalcharges'] / (df['monthlycharges'] + 0.01)  # +0.01 to avoid division by zero
            ratio_features_created.append('total_to_monthly_ratio')
        
        if 'totalcharges' in df.columns and 'tenure' in df.columns:
            # Average charges per month
            df_ratios['avg_charges_per_month'] = df['totalcharges'] / (df['tenure'] + 1)
            ratio_features_created.append('avg_charges_per_month')
        
        self.engineered_features.extend(ratio_features_created)
        logger.info(f"Created {len(ratio_features_created)} ratio features: {ratio_features_created}")
        
        return df_ratios
    
    def create_aggregate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create aggregate features from categorical variables.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with aggregate features
        """
        logger.info("Creating aggregate features")
        
        df_agg = df.copy()
        agg_features_created = []
        
        # Count of services (for telecom data)
        service_columns = [col for col in df.columns if any(service in col.lower() 
                          for service in ['streaming', 'online', 'device', 'tech', 'phone'])]
        
        if service_columns:
            # Count of active services
            df_agg['total_services'] = df[service_columns].sum(axis=1)
            agg_features_created.append('total_services')
        
        # Binary feature combinations
        binary_cols = [col for col in df.columns if df[col].nunique() == 2 and col != self.config.get('data.target_column', 'churn')]
        
        if len(binary_cols) > 1:
            # Sum of binary features
            df_agg['binary_features_sum'] = df[binary_cols].sum(axis=1)
            agg_features_created.append('binary_features_sum')
        
        self.engineered_features.extend(agg_features_created)
        logger.info(f"Created {len(agg_features_created)} aggregate features: {agg_features_created}")

        return df_agg

    def select_features_univariate(self, X: pd.DataFrame, y: pd.Series, k: int = 20) -> Tuple[pd.DataFrame, List[str]]:
        """Select features using univariate statistical tests."""
        logger.info(f"Selecting top {k} features using univariate tests")

        selector = SelectKBest(score_func=f_classif, k=k)
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()].tolist()

        self.feature_selectors['univariate'] = selector
        logger.info(f"Selected {len(selected_features)} features using univariate tests")
        return pd.DataFrame(X_selected, columns=selected_features, index=X.index), selected_features

    def select_features_rfe(self, X: pd.DataFrame, y: pd.Series, n_features: int = 15) -> Tuple[pd.DataFrame, List[str]]:
        """Select features using Recursive Feature Elimination."""
        logger.info(f"Selecting {n_features} features using RFE")

        estimator = LogisticRegression(random_state=42, max_iter=1000)
        selector = RFE(estimator=estimator, n_features_to_select=n_features)
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()].tolist()

        self.feature_selectors['rfe'] = selector
        logger.info(f"Selected {len(selected_features)} features using RFE")
        return pd.DataFrame(X_selected, columns=selected_features, index=X.index), selected_features

    def apply_dimensionality_reduction(self, X: pd.DataFrame, method: str = 'pca', n_components: int = 10) -> Tuple[pd.DataFrame, Any]:
        """Apply dimensionality reduction techniques."""
        logger.info(f"Applying {method.upper()} dimensionality reduction with {n_components} components")

        if method.lower() == 'pca':
            reducer = PCA(n_components=n_components, random_state=42)
        elif method.lower() == 'lda':
            reducer = LinearDiscriminantAnalysis(n_components=n_components)
        else:
            raise FeatureEngineeringError(f"Unknown dimensionality reduction method: {method}")

        X_reduced = reducer.fit_transform(X)
        component_names = [f"{method}_{i+1}" for i in range(n_components)]
        X_reduced_df = pd.DataFrame(X_reduced, columns=component_names, index=X.index)

        self.feature_transformers[f'{method}_reducer'] = reducer
        logger.info(f"Reduced features from {X.shape[1]} to {n_components} using {method.upper()}")
        return X_reduced_df, reducer

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Complete feature engineering pipeline."""
        logger.info("Starting feature engineering pipeline")

        # Step 1: Create engineered features
        X_engineered = self.create_interaction_features(X, max_interactions=5)
        X_engineered = self.create_polynomial_features(X_engineered, degree=2, max_features=3)
        X_engineered = self.create_binned_features(X_engineered)
        X_engineered = self.create_ratio_features(X_engineered)
        X_engineered = self.create_aggregate_features(X_engineered)

        # Step 2: Feature selection
        X_univariate, univariate_features = self.select_features_univariate(X_engineered, y, k=20)
        X_rfe, rfe_features = self.select_features_rfe(X_engineered, y, n_features=15)

        # Step 3: Combine selected features (union of both methods)
        combined_features = list(set(univariate_features + rfe_features))
        X_selected = X_engineered[combined_features]

        self.selected_features = combined_features

        # Step 4: Optional dimensionality reduction
        X_pca, pca_reducer = self.apply_dimensionality_reduction(X_selected, method='pca', n_components=10)

        results = {
            'original_features': X.shape[1],
            'engineered_features': X_engineered.shape[1],
            'selected_features': len(combined_features),
            'final_features': X_selected.shape[1],
            'pca_components': X_pca.shape[1],
            'feature_names': combined_features,
            'engineered_feature_names': self.engineered_features
        }

        logger.info(f"Feature engineering completed: {results}")
        return X_selected, results

    def save_transformers(self, filepath: Optional[str] = None):
        """Save fitted transformers."""
        if filepath is None:
            filepath = self.artifacts_path / "feature_transformers.joblib"

        transformers_dict = {
            'feature_selectors': self.feature_selectors,
            'feature_transformers': self.feature_transformers,
            'engineered_features': self.engineered_features,
            'selected_features': self.selected_features
        }

        joblib.dump(transformers_dict, filepath)
        logger.info(f"Feature transformers saved to {filepath}")


def main():
    """Main function for feature engineering."""
    from ..data.preprocessing import AdvancedPreprocessor
    from ..data.data_loader import DataLoader

    # Load and preprocess data
    loader = DataLoader()
    df, _ = loader.load_and_process()

    preprocessor = AdvancedPreprocessor()
    X, y = preprocessor.fit_transform(df)

    # Feature engineering
    engineer = AdvancedFeatureEngineer()
    X_engineered, results = engineer.fit_transform(X, y)

    # Save transformers
    engineer.save_transformers()

    print("Feature Engineering completed!")
    print(f"Results: {results}")

    # Save engineered features
    engineered_data = X_engineered.copy()
    engineered_data['churn'] = y

    output_path = Path(config.project_root) / "data" / "processed" / "engineered_features.csv"
    engineered_data.to_csv(output_path, index=False)
    logger.info(f"Engineered features saved to {output_path}")

    return X_engineered, results


if __name__ == "__main__":
    main()

    def select_features_univariate(self, X: pd.DataFrame, y: pd.Series, k: int = 20) -> Tuple[pd.DataFrame, List[str]]:
        """
        Select features using univariate statistical tests.

        Args:
            X: Feature DataFrame
            y: Target Series
            k: Number of features to select

        Returns:
            Tuple of (selected features DataFrame, selected feature names)
        """
        logger.info(f"Selecting top {k} features using univariate tests")

        # Use f_classif for numerical features and chi2 for non-negative features
        selector = SelectKBest(score_func=f_classif, k=k)

        try:
            X_selected = selector.fit_transform(X, y)
            selected_features = X.columns[selector.get_support()].tolist()

            self.feature_selectors['univariate'] = selector

            logger.info(f"Selected {len(selected_features)} features using univariate tests")
            return pd.DataFrame(X_selected, columns=selected_features, index=X.index), selected_features

        except Exception as e:
            logger.error(f"Univariate feature selection failed: {str(e)}")
            return X, X.columns.tolist()

    def select_features_rfe(self, X: pd.DataFrame, y: pd.Series, n_features: int = 15) -> Tuple[pd.DataFrame, List[str]]:
        """
        Select features using Recursive Feature Elimination.

        Args:
            X: Feature DataFrame
            y: Target Series
            n_features: Number of features to select

        Returns:
            Tuple of (selected features DataFrame, selected feature names)
        """
        logger.info(f"Selecting {n_features} features using RFE")

        # Use LogisticRegression as the estimator for RFE
        estimator = LogisticRegression(random_state=42, max_iter=1000)
        selector = RFE(estimator=estimator, n_features_to_select=n_features)

        try:
            X_selected = selector.fit_transform(X, y)
            selected_features = X.columns[selector.get_support()].tolist()

            self.feature_selectors['rfe'] = selector

            logger.info(f"Selected {len(selected_features)} features using RFE")
            return pd.DataFrame(X_selected, columns=selected_features, index=X.index), selected_features

        except Exception as e:
            logger.error(f"RFE feature selection failed: {str(e)}")
            return X, X.columns.tolist()

    def select_features_lasso(self, X: pd.DataFrame, y: pd.Series, alpha: Optional[float] = None) -> Tuple[pd.DataFrame, List[str]]:
        """
        Select features using LASSO regularization.

        Args:
            X: Feature DataFrame
            y: Target Series
            alpha: Regularization strength. If None, uses cross-validation

        Returns:
            Tuple of (selected features DataFrame, selected feature names)
        """
        logger.info("Selecting features using LASSO regularization")

        if alpha is None:
            # Use cross-validation to find optimal alpha
            lasso = LassoCV(cv=5, random_state=42, max_iter=2000)
        else:
            lasso = LassoCV(alphas=[alpha], cv=5, random_state=42, max_iter=2000)

        try:
            selector = SelectFromModel(lasso)
            X_selected = selector.fit_transform(X, y)
            selected_features = X.columns[selector.get_support()].tolist()

            self.feature_selectors['lasso'] = selector

            logger.info(f"Selected {len(selected_features)} features using LASSO")
            return pd.DataFrame(X_selected, columns=selected_features, index=X.index), selected_features

        except Exception as e:
            logger.error(f"LASSO feature selection failed: {str(e)}")
            return X, X.columns.tolist()

    def select_features_tree_based(self, X: pd.DataFrame, y: pd.Series, n_features: int = 20) -> Tuple[pd.DataFrame, List[str]]:
        """
        Select features using tree-based feature importance.

        Args:
            X: Feature DataFrame
            y: Target Series
            n_features: Number of features to select

        Returns:
            Tuple of (selected features DataFrame, selected feature names)
        """
        logger.info(f"Selecting {n_features} features using tree-based importance")

        # Use RandomForest for feature importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42)

        try:
            rf.fit(X, y)

            # Get feature importances
            importances = rf.feature_importances_
            feature_importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance': importances
            }).sort_values('importance', ascending=False)

            # Select top features
            selected_features = feature_importance_df.head(n_features)['feature'].tolist()
            X_selected = X[selected_features]

            self.feature_selectors['tree_based'] = {
                'model': rf,
                'feature_importance': feature_importance_df
            }

            logger.info(f"Selected {len(selected_features)} features using tree-based importance")
            return X_selected, selected_features

        except Exception as e:
            logger.error(f"Tree-based feature selection failed: {str(e)}")
            return X, X.columns.tolist()

    def apply_dimensionality_reduction(self, X: pd.DataFrame, method: str = 'pca', n_components: int = 10) -> Tuple[pd.DataFrame, Any]:
        """
        Apply dimensionality reduction techniques.

        Args:
            X: Feature DataFrame
            method: Reduction method ('pca' or 'lda')
            n_components: Number of components

        Returns:
            Tuple of (transformed DataFrame, fitted transformer)
        """
        logger.info(f"Applying {method.upper()} dimensionality reduction with {n_components} components")

        if method.lower() == 'pca':
            reducer = PCA(n_components=n_components, random_state=42)
        elif method.lower() == 'lda':
            reducer = LinearDiscriminantAnalysis(n_components=n_components)
        else:
            raise FeatureEngineeringError(f"Unknown dimensionality reduction method: {method}")

        try:
            X_reduced = reducer.fit_transform(X)

            # Create column names for reduced features
            component_names = [f"{method}_{i+1}" for i in range(n_components)]
            X_reduced_df = pd.DataFrame(X_reduced, columns=component_names, index=X.index)

            self.feature_transformers[f'{method}_reducer'] = reducer

            logger.info(f"Reduced features from {X.shape[1]} to {n_components} using {method.upper()}")
            return X_reduced_df, reducer

        except Exception as e:
            logger.error(f"{method.upper()} dimensionality reduction failed: {str(e)}")
            return X, None
