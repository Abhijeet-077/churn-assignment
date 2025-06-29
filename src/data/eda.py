"""
Exploratory Data Analysis module for churn prediction.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

# EDA Libraries
try:
    from ydata_profiling import ProfileReport
    PROFILING_AVAILABLE = True
except ImportError:
    try:
        from pandas_profiling import ProfileReport
        PROFILING_AVAILABLE = True
    except ImportError:
        PROFILING_AVAILABLE = False

try:
    import sweetviz as sv
    SWEETVIZ_AVAILABLE = True
except ImportError:
    SWEETVIZ_AVAILABLE = False

from ..utils.config import config
from ..utils.logger import get_logger
from ..utils.exceptions import DataError

logger = get_logger(__name__)


class EDAAnalyzer:
    """Class for comprehensive exploratory data analysis."""
    
    def __init__(self, output_dir: Optional[str] = None):
        self.config = config
        if output_dir is None:
            self.output_dir = Path(self.config.project_root) / "reports" / "eda"
        else:
            self.output_dir = Path(output_dir)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def generate_automated_report(self, df: pd.DataFrame, title: str = "Churn Data Analysis") -> str:
        """
        Generate automated EDA report using pandas-profiling.
        
        Args:
            df: DataFrame to analyze
            title: Report title
            
        Returns:
            Path to generated report
        """
        if not PROFILING_AVAILABLE:
            logger.warning("pandas-profiling/ydata-profiling not available. Skipping automated report.")
            return None
        
        try:
            logger.info("Generating automated EDA report with pandas-profiling")
            
            profile = ProfileReport(
                df,
                title=title,
                explorative=True,
                dark_mode=True,
                minimal=False
            )
            
            report_path = self.output_dir / "automated_eda_report.html"
            profile.to_file(report_path)
            
            logger.info(f"Automated EDA report saved to {report_path}")
            return str(report_path)
            
        except Exception as e:
            logger.error(f"Failed to generate automated report: {str(e)}")
            return None
    
    def generate_sweetviz_report(self, df: pd.DataFrame, target_col: str = "churn") -> str:
        """
        Generate EDA report using Sweetviz.
        
        Args:
            df: DataFrame to analyze
            target_col: Target column name
            
        Returns:
            Path to generated report
        """
        if not SWEETVIZ_AVAILABLE:
            logger.warning("Sweetviz not available. Skipping Sweetviz report.")
            return None
        
        try:
            logger.info("Generating EDA report with Sweetviz")
            
            report = sv.analyze(df, target_feat=target_col)
            report_path = self.output_dir / "sweetviz_report.html"
            report.show_html(str(report_path))
            
            logger.info(f"Sweetviz report saved to {report_path}")
            return str(report_path)
            
        except Exception as e:
            logger.error(f"Failed to generate Sweetviz report: {str(e)}")
            return None
    
    def basic_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Generate basic statistics for the dataset.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with basic statistics
        """
        logger.info("Calculating basic statistics")
        
        stats = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'data_types': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'missing_percentage': (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
            'duplicate_rows': df.duplicated().sum(),
            'memory_usage_mb': round(df.memory_usage(deep=True).sum() / 1024**2, 2)
        }
        
        # Numerical statistics
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            stats['numerical_summary'] = df[numerical_cols].describe().to_dict()
        
        # Categorical statistics
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            stats['categorical_summary'] = {}
            for col in categorical_cols:
                stats['categorical_summary'][col] = {
                    'unique_count': df[col].nunique(),
                    'most_frequent': df[col].mode().iloc[0] if not df[col].empty else None,
                    'value_counts': df[col].value_counts().head(10).to_dict()
                }
        
        # Save statistics
        stats_path = self.output_dir / "basic_statistics.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        
        logger.info(f"Basic statistics saved to {stats_path}")
        return stats
    
    def target_analysis(self, df: pd.DataFrame, target_col: str = "churn") -> Dict:
        """
        Analyze target variable distribution and characteristics.
        
        Args:
            df: DataFrame to analyze
            target_col: Target column name
            
        Returns:
            Dictionary with target analysis results
        """
        logger.info(f"Analyzing target variable: {target_col}")
        
        if target_col not in df.columns:
            raise DataError(f"Target column '{target_col}' not found in DataFrame")
        
        target_series = df[target_col]
        
        analysis = {
            'value_counts': target_series.value_counts().to_dict(),
            'value_percentages': (target_series.value_counts(normalize=True) * 100).round(2).to_dict(),
            'missing_values': target_series.isnull().sum(),
            'unique_values': target_series.nunique()
        }
        
        # Calculate class imbalance ratio
        if analysis['unique_values'] == 2:
            counts = target_series.value_counts()
            majority_class = counts.max()
            minority_class = counts.min()
            analysis['imbalance_ratio'] = round(majority_class / minority_class, 2)
            analysis['minority_class_percentage'] = round((minority_class / len(target_series)) * 100, 2)
        
        # Create target distribution plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Count plot
        target_series.value_counts().plot(kind='bar', ax=ax1, color='skyblue')
        ax1.set_title(f'{target_col.title()} Distribution (Counts)')
        ax1.set_xlabel(target_col.title())
        ax1.set_ylabel('Count')
        
        # Percentage plot
        (target_series.value_counts(normalize=True) * 100).plot(kind='bar', ax=ax2, color='lightcoral')
        ax2.set_title(f'{target_col.title()} Distribution (Percentage)')
        ax2.set_xlabel(target_col.title())
        ax2.set_ylabel('Percentage')
        
        plt.tight_layout()
        plot_path = self.output_dir / f"{target_col}_distribution.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        analysis['plot_path'] = str(plot_path)
        
        logger.info(f"Target analysis completed. Imbalance ratio: {analysis.get('imbalance_ratio', 'N/A')}")
        return analysis
    
    def correlation_analysis(self, df: pd.DataFrame) -> Dict:
        """
        Perform correlation analysis on numerical features.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with correlation analysis results
        """
        logger.info("Performing correlation analysis")
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) < 2:
            logger.warning("Not enough numerical columns for correlation analysis")
            return {'message': 'Insufficient numerical columns'}
        
        # Calculate correlation matrix
        corr_matrix = df[numerical_cols].corr()
        
        # Find highly correlated pairs
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:  # High correlation threshold
                    high_corr_pairs.append({
                        'feature1': corr_matrix.columns[i],
                        'feature2': corr_matrix.columns[j],
                        'correlation': round(corr_val, 3)
                    })
        
        # Create correlation heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(
            corr_matrix, 
            mask=mask, 
            annot=True, 
            cmap='coolwarm', 
            center=0,
            square=True,
            fmt='.2f'
        )
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        
        heatmap_path = self.output_dir / "correlation_heatmap.png"
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        analysis = {
            'correlation_matrix': corr_matrix.to_dict(),
            'high_correlation_pairs': high_corr_pairs,
            'heatmap_path': str(heatmap_path)
        }
        
        logger.info(f"Correlation analysis completed. Found {len(high_corr_pairs)} highly correlated pairs")
        return analysis

    def feature_target_analysis(self, df: pd.DataFrame, target_col: str = "churn") -> Dict:
        """
        Analyze relationship between features and target variable.

        Args:
            df: DataFrame to analyze
            target_col: Target column name

        Returns:
            Dictionary with feature-target analysis results
        """
        logger.info("Analyzing feature-target relationships")

        if target_col not in df.columns:
            raise DataError(f"Target column '{target_col}' not found in DataFrame")

        analysis = {}

        # Numerical features vs target
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        numerical_cols = [col for col in numerical_cols if col != target_col]

        if len(numerical_cols) > 0:
            analysis['numerical_features'] = {}

            # Create subplots for numerical features
            n_cols = min(3, len(numerical_cols))
            n_rows = (len(numerical_cols) + n_cols - 1) // n_cols

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
            if n_rows == 1:
                axes = [axes] if n_cols == 1 else axes
            else:
                axes = axes.flatten()

            for i, col in enumerate(numerical_cols):
                if i < len(axes):
                    # Box plot for numerical feature vs target
                    df.boxplot(column=col, by=target_col, ax=axes[i])
                    axes[i].set_title(f'{col} by {target_col}')
                    axes[i].set_xlabel(target_col)
                    axes[i].set_ylabel(col)

                # Calculate statistics
                stats_by_target = df.groupby(target_col)[col].agg(['mean', 'median', 'std']).to_dict()
                analysis['numerical_features'][col] = stats_by_target

            # Hide unused subplots
            for i in range(len(numerical_cols), len(axes)):
                axes[i].set_visible(False)

            plt.tight_layout()
            numerical_plot_path = self.output_dir / "numerical_features_vs_target.png"
            plt.savefig(numerical_plot_path, dpi=300, bbox_inches='tight')
            plt.close()

            analysis['numerical_plot_path'] = str(numerical_plot_path)

        # Categorical features vs target
        categorical_cols = df.select_dtypes(include=['object']).columns
        categorical_cols = [col for col in categorical_cols if col != target_col and col != 'customerid']

        if len(categorical_cols) > 0:
            analysis['categorical_features'] = {}

            for col in categorical_cols:
                # Cross-tabulation
                crosstab = pd.crosstab(df[col], df[target_col], normalize='index') * 100
                analysis['categorical_features'][col] = {
                    'crosstab_percentage': crosstab.to_dict(),
                    'value_counts_by_target': df.groupby(target_col)[col].value_counts().to_dict()
                }

        return analysis

    def outlier_analysis(self, df: pd.DataFrame) -> Dict:
        """
        Detect and analyze outliers in numerical features.

        Args:
            df: DataFrame to analyze

        Returns:
            Dictionary with outlier analysis results
        """
        logger.info("Performing outlier analysis")

        numerical_cols = df.select_dtypes(include=[np.number]).columns

        if len(numerical_cols) == 0:
            logger.warning("No numerical columns found for outlier analysis")
            return {'message': 'No numerical columns available'}

        analysis = {}

        for col in numerical_cols:
            series = df[col].dropna()

            # IQR method
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            iqr_outliers = series[(series < lower_bound) | (series > upper_bound)]

            # Z-score method
            z_scores = np.abs((series - series.mean()) / series.std())
            zscore_outliers = series[z_scores > 3]

            analysis[col] = {
                'iqr_outliers_count': len(iqr_outliers),
                'iqr_outliers_percentage': round((len(iqr_outliers) / len(series)) * 100, 2),
                'zscore_outliers_count': len(zscore_outliers),
                'zscore_outliers_percentage': round((len(zscore_outliers) / len(series)) * 100, 2),
                'bounds': {
                    'iqr_lower': lower_bound,
                    'iqr_upper': upper_bound,
                    'mean': series.mean(),
                    'std': series.std()
                }
            }

        # Create outlier visualization
        n_cols = min(3, len(numerical_cols))
        n_rows = (len(numerical_cols) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()

        for i, col in enumerate(numerical_cols):
            if i < len(axes):
                df.boxplot(column=col, ax=axes[i])
                axes[i].set_title(f'Outliers in {col}')
                axes[i].set_ylabel(col)

        # Hide unused subplots
        for i in range(len(numerical_cols), len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        outlier_plot_path = self.output_dir / "outlier_analysis.png"
        plt.savefig(outlier_plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        analysis['plot_path'] = str(outlier_plot_path)

        logger.info("Outlier analysis completed")
        return analysis

    def comprehensive_eda(self, df: pd.DataFrame, target_col: str = "churn") -> Dict:
        """
        Perform comprehensive EDA analysis.

        Args:
            df: DataFrame to analyze
            target_col: Target column name

        Returns:
            Dictionary with all analysis results
        """
        logger.info("Starting comprehensive EDA analysis")

        results = {}

        # Basic statistics
        results['basic_statistics'] = self.basic_statistics(df)

        # Target analysis
        results['target_analysis'] = self.target_analysis(df, target_col)

        # Correlation analysis
        results['correlation_analysis'] = self.correlation_analysis(df)

        # Feature-target analysis
        results['feature_target_analysis'] = self.feature_target_analysis(df, target_col)

        # Outlier analysis
        results['outlier_analysis'] = self.outlier_analysis(df)

        # Generate automated reports
        results['automated_report_path'] = self.generate_automated_report(df)
        results['sweetviz_report_path'] = self.generate_sweetviz_report(df, target_col)

        # Save comprehensive results (convert complex types to strings)
        def clean_for_json(obj):
            """Recursively clean object for JSON serialization."""
            if isinstance(obj, dict):
                return {str(k): clean_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_for_json(item) for item in obj]
            elif isinstance(obj, tuple):
                return list(obj)
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif pd.isna(obj):
                return None
            else:
                return obj

        cleaned_results = clean_for_json(results)

        results_path = self.output_dir / "comprehensive_eda_results.json"
        with open(results_path, 'w') as f:
            json.dump(cleaned_results, f, indent=2, default=str)

        logger.info(f"Comprehensive EDA completed. Results saved to {results_path}")
        return results


def main():
    """Main function for EDA analysis."""
    from .data_loader import DataLoader

    # Load data
    loader = DataLoader()
    df, _ = loader.load_and_process()

    # Perform EDA
    analyzer = EDAAnalyzer()
    results = analyzer.comprehensive_eda(df)

    print("EDA Analysis completed!")
    print(f"Reports generated in: {analyzer.output_dir}")

    return results


if __name__ == "__main__":
    main()
