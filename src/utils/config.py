"""
Configuration management utilities for the churn prediction project.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class Config:
    """Configuration class to hold all project settings."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration from YAML file.
        
        Args:
            config_path: Path to configuration file. If None, uses default.
        """
        if config_path is None:
            # Get project root directory
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config" / "config.yaml"
        
        self.config_path = Path(config_path)
        self._config = self._load_config()
        
        # Set up paths
        self.project_root = self.config_path.parent.parent
        self._setup_paths()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
            return config
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing configuration file: {e}")
    
    def _setup_paths(self):
        """Set up all project paths."""
        # Create directories if they don't exist
        paths_to_create = [
            self.project_root / "data" / "raw",
            self.project_root / "data" / "processed",
            self.project_root / "data" / "external",
            self.project_root / "models",
            self.project_root / "artifacts",
            self.project_root / "reports",
            self.project_root / "plots",
            self.project_root / "logs"
        ]
        
        for path in paths_to_create:
            path.mkdir(parents=True, exist_ok=True)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'data.test_size')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any):
        """
        Set configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'data.test_size')
            value: Value to set
        """
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self, path: Optional[str] = None):
        """
        Save configuration to YAML file.
        
        Args:
            path: Path to save configuration. If None, uses original path.
        """
        save_path = Path(path) if path else self.config_path
        
        with open(save_path, 'w', encoding='utf-8') as file:
            yaml.dump(self._config, file, default_flow_style=False, indent=2)
    
    @property
    def data_config(self) -> Dict[str, Any]:
        """Get data configuration."""
        return self._config.get('data', {})
    
    @property
    def model_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return self._config.get('models', {})
    
    @property
    def preprocessing_config(self) -> Dict[str, Any]:
        """Get preprocessing configuration."""
        return self._config.get('preprocessing', {})
    
    @property
    def feature_config(self) -> Dict[str, Any]:
        """Get feature engineering configuration."""
        return self._config.get('features', {})
    
    @property
    def evaluation_config(self) -> Dict[str, Any]:
        """Get evaluation configuration."""
        return self._config.get('evaluation', {})
    
    @property
    def api_config(self) -> Dict[str, Any]:
        """Get API configuration."""
        return self._config.get('api', {})
    
    @property
    def dashboard_config(self) -> Dict[str, Any]:
        """Get dashboard configuration."""
        return self._config.get('dashboard', {})
    
    @property
    def logging_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return self._config.get('logging', {})
    
    @property
    def mlops_config(self) -> Dict[str, Any]:
        """Get MLOps configuration."""
        return self._config.get('mlops', {})


# Global configuration instance
config = Config()
