"""
Utility Functions for Retention Intelligence System
====================================================
Handles configuration loading, logging setup, and common utilities.
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime


class Config:
    """Configuration manager for the retention system."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to config.yaml file. If None, uses default location.
        """
        if config_path is None:
            # Default to config/config.yaml relative to project root
            project_root = Path(__file__).parent.parent
            config_path = project_root / "config" / "config.yaml"
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._setup_directories()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing configuration file: {e}")
    
    def _setup_directories(self):
        """Create necessary directories if they don't exist."""
        dirs = [
            self.get('paths.processed_data'),
            self.get('paths.features'),
            self.get('paths.models'),
            self.get('paths.outputs'),
            self.get('paths.logs'),
        ]
        
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key: Configuration key in dot notation (e.g., 'data_cleaning.min_unit_price')
            default: Default value if key not found
            
        Returns:
            Configuration value
            
        Example:
            >>> config.get('churn.churn_window_days')
            90
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access."""
        return self.get(key)


def setup_logging(config: Config) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        config: Configuration object
        
    Returns:
        Configured logger
    """
    log_level = getattr(logging, config.get('logging.level', 'INFO'))
    log_format = config.get('logging.format')
    log_file = config.get('logging.log_file')
    
    # Create logger
    logger = logging.getLogger('retention_system')
    logger.setLevel(log_level)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Console handler with UTF-8 encoding for Windows
    if config.get('logging.console_logging', True):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_formatter = logging.Formatter(log_format)
        console_handler.setFormatter(console_formatter)
        # Set UTF-8 encoding
        import sys
        if hasattr(sys.stdout, 'reconfigure'):
            try:
                sys.stdout.reconfigure(encoding='utf-8')
            except:
                pass
        logger.addHandler(console_handler)
    
    # File handler with UTF-8 encoding
    if config.get('logging.file_logging', True):
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(log_level)
        file_formatter = logging.Formatter(log_format)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


def set_random_seed(seed: int):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    # If using other libraries, set their seeds too
    # random.seed(seed)
    # torch.manual_seed(seed)


def get_project_root() -> Path:
    """Get project root directory."""
    return Path(__file__).parent.parent


def format_currency(value: float, currency: str = "$") -> str:
    """
    Format value as currency.
    
    Args:
        value: Numeric value
        currency: Currency symbol
        
    Returns:
        Formatted string
    """
    return f"{currency}{value:,.2f}"


def format_percentage(value: float, decimals: int = 2) -> str:
    """
    Format value as percentage.
    
    Args:
        value: Numeric value (0-1)
        decimals: Number of decimal places
        
    Returns:
        Formatted string
    """
    return f"{value * 100:.{decimals}f}%"


def print_section_header(title: str, width: int = 80):
    """
    Print a formatted section header.
    
    Args:
        title: Section title
        width: Total width of header
    """
    print("\n" + "=" * width)
    print(f"{title.upper():^{width}}")
    print("=" * width + "\n")


def print_metrics_table(metrics: Dict[str, float], title: str = "Metrics"):
    """
    Print metrics in a formatted table.
    
    Args:
        metrics: Dictionary of metric names and values
        title: Table title
    """
    print_section_header(title, width=60)
    
    max_name_len = max(len(name) for name in metrics.keys())
    
    for name, value in metrics.items():
        if isinstance(value, float):
            print(f"{name:<{max_name_len}} : {value:>10.4f}")
        else:
            print(f"{name:<{max_name_len}} : {value:>10}")
    
    print()


def save_dataframe(
    df: pd.DataFrame,
    path: str,
    format: str = 'parquet',
    compression: str = 'snappy',
    logger: Optional[logging.Logger] = None
):
    """
    Save DataFrame to disk with specified format.
    
    Args:
        df: DataFrame to save
        path: Output path (without extension)
        format: File format ('parquet', 'csv', 'feather')
        compression: Compression method
        logger: Logger instance
    """
    if logger:
        logger.info(f"Saving DataFrame to {path}.{format}")
    
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'parquet':
        df.to_parquet(f"{path}.parquet", compression=compression, index=False)
    elif format == 'csv':
        df.to_csv(f"{path}.csv", index=False)
    elif format == 'feather':
        df.to_feather(f"{path}.feather")
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    if logger:
        logger.info(f"Successfully saved {len(df):,} rows")


def load_dataframe(
    path: str,
    format: str = 'parquet',
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """
    Load DataFrame from disk.
    
    Args:
        path: Input path (without extension)
        format: File format ('parquet', 'csv', 'feather')
        logger: Logger instance
        
    Returns:
        Loaded DataFrame
    """
    if logger:
        logger.info(f"Loading DataFrame from {path}.{format}")
    
    if format == 'parquet':
        df = pd.read_parquet(f"{path}.parquet")
    elif format == 'csv':
        df = pd.read_csv(f"{path}.csv")
    elif format == 'feather':
        df = pd.read_feather(f"{path}.feather")
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    if logger:
        logger.info(f"Successfully loaded {len(df):,} rows, {len(df.columns)} columns")
    
    return df


def create_snapshot_dates(
    min_date: pd.Timestamp,
    max_date: pd.Timestamp,
    frequency: str = 'M',
    observation_window_days: int = 180
) -> list:
    """
    Create snapshot dates for temporal validation.
    
    Args:
        min_date: Minimum date in dataset
        max_date: Maximum date in dataset
        frequency: Snapshot frequency ('M' = monthly, 'W' = weekly)
        observation_window_days: Days of history needed before first snapshot
        
    Returns:
        List of snapshot dates
    """
    # First snapshot must have enough history
    first_snapshot = min_date + pd.Timedelta(days=observation_window_days)
    
    # Generate snapshots
    snapshots = pd.date_range(
        start=first_snapshot,
        end=max_date,
        freq=frequency
    )
    
    return snapshots.tolist()


class Timer:
    """Context manager for timing code execution."""
    
    def __init__(self, name: str = "Operation", logger: Optional[logging.Logger] = None):
        """
        Initialize timer.
        
        Args:
            name: Name of operation being timed
            logger: Logger instance
        """
        self.name = name
        self.logger = logger
        self.start_time = None
        
    def __enter__(self):
        """Start timer."""
        self.start_time = datetime.now()
        if self.logger:
            self.logger.info(f"Starting: {self.name}")
        else:
            print(f"Starting: {self.name}")
        return self
    
    def __exit__(self, *args):
        """Stop timer and log duration."""
        duration = (datetime.now() - self.start_time).total_seconds()
        
        if duration < 60:
            time_str = f"{duration:.2f} seconds"
        elif duration < 3600:
            time_str = f"{duration / 60:.2f} minutes"
        else:
            time_str = f"{duration / 3600:.2f} hours"
        
        message = f"Completed: {self.name} in {time_str}"
        
        if self.logger:
            self.logger.info(message)
        else:
            print(message)


def validate_dataframe_schema(
    df: pd.DataFrame,
    required_columns: list,
    logger: Optional[logging.Logger] = None
) -> bool:
    """
    Validate that DataFrame has required columns.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        logger: Logger instance
        
    Returns:
        True if valid, raises ValueError if not
    """
    missing_columns = set(required_columns) - set(df.columns)
    
    if missing_columns:
        error_msg = f"Missing required columns: {missing_columns}"
        if logger:
            logger.error(error_msg)
        raise ValueError(error_msg)
    
    if logger:
        logger.info("DataFrame schema validation passed")
    
    return True


# Example usage and testing
if __name__ == "__main__":
    # Load configuration
    config = Config()
    
    # Setup logging
    logger = setup_logging(config)
    
    # Test utilities
    logger.info("Configuration and logging setup complete")
    print_section_header("Test Section")
    print_metrics_table({
        'Accuracy': 0.8542,
        'Precision': 0.7823,
        'Recall': 0.8901,
        'F1-Score': 0.8330
    })
    
    # Test timer
    with Timer("Test operation", logger):
        import time
        time.sleep(1)
    
    print("\n✓ Utilities module tested successfully")
