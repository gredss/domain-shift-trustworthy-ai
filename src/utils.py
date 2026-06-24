"""
Utility Module for IndoBERT Clickbait Detection System

This module provides common helper functions, logging utilities, file I/O operations,
and reproducibility helpers used across the entire system.
"""

import os
import json
import pickle
import random
import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union, Callable
from pathlib import Path
from datetime import datetime
import hashlib


class FileManager:
    """Handles file I/O operations with error handling."""
    
    @staticmethod
    def ensure_directory(directory: str) -> None:
        """
        Ensure directory exists, create if it doesn't.
        
        Args:
            directory: Path to directory
        """
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def save_json(data: Dict[str, Any], filepath: str, indent: int = 2) -> None:
        """
        Save dictionary to JSON file.
        
        Args:
            data: Dictionary to save
            filepath: Path to output file
            indent: JSON indentation level
        """
        FileManager.ensure_directory(os.path.dirname(filepath))
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
    
    @staticmethod
    def load_json(filepath: str) -> Dict[str, Any]:
        """
        Load dictionary from JSON file.
        
        Args:
            filepath: Path to JSON file
            
        Returns:
            Dictionary with loaded data
            
        Raises:
            FileNotFoundError: If file doesn't exist
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    @staticmethod
    def save_pickle(data: Any, filepath: str) -> None:
        """
        Save object to pickle file.
        
        Args:
            data: Object to save
            filepath: Path to output file
        """
        FileManager.ensure_directory(os.path.dirname(filepath))
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    @staticmethod
    def load_pickle(filepath: str) -> Any:
        """
        Load object from pickle file.
        
        Args:
            filepath: Path to pickle file
            
        Returns:
            Loaded object
            
        Raises:
            FileNotFoundError: If file doesn't exist
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    @staticmethod
    def save_csv(df: pd.DataFrame, filepath: str, index: bool = False) -> None:
        """
        Save DataFrame to CSV file.
        
        Args:
            df: DataFrame to save
            filepath: Path to output file
            index: Whether to save index
        """
        FileManager.ensure_directory(os.path.dirname(filepath))
        df.to_csv(filepath, index=index, encoding='utf-8')
    
    @staticmethod
    def load_csv(filepath: str) -> pd.DataFrame:
        """
        Load DataFrame from CSV file.
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            Loaded DataFrame
            
        Raises:
            FileNotFoundError: If file doesn't exist
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        return pd.read_csv(filepath, encoding='utf-8')
    
    @staticmethod
    def save_text(text: str, filepath: str) -> None:
        """
        Save text to file.
        
        Args:
            text: Text content to save
            filepath: Path to output file
        """
        FileManager.ensure_directory(os.path.dirname(filepath))
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(text)
    
    @staticmethod
    def load_text(filepath: str) -> str:
        """
        Load text from file.
        
        Args:
            filepath: Path to text file
            
        Returns:
            Text content
            
        Raises:
            FileNotFoundError: If file doesn't exist
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    
    @staticmethod
    def file_exists(filepath: str) -> bool:
        """
        Check if file exists.
        
        Args:
            filepath: Path to file
            
        Returns:
            True if file exists, False otherwise
        """
        return os.path.exists(filepath)
    
    @staticmethod
    def get_file_size(filepath: str) -> int:
        """
        Get file size in bytes.
        
        Args:
            filepath: Path to file
            
        Returns:
            File size in bytes
            
        Raises:
            FileNotFoundError: If file doesn't exist
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        return os.path.getsize(filepath)
    
    @staticmethod
    def list_files(directory: str, extension: Optional[str] = None) -> List[str]:
        """
        List files in directory.
        
        Args:
            directory: Path to directory
            extension: Optional file extension filter (e.g., '.json')
            
        Returns:
            List of file paths
        """
        if not os.path.exists(directory):
            return []
        
        files = []
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            if os.path.isfile(filepath):
                if extension is None or filename.endswith(extension):
                    files.append(filepath)
        
        return sorted(files)


class LoggerManager:
    """Manages logging configuration and logger instances."""
    
    _loggers: Dict[str, logging.Logger] = {}
    
    @staticmethod
    def setup_logger(
        name: str,
        log_file: Optional[str] = None,
        level: int = logging.INFO,
        format_string: Optional[str] = None
    ) -> logging.Logger:
        """
        Setup and configure a logger.
        
        Args:
            name: Logger name
            log_file: Optional log file path
            level: Logging level
            format_string: Optional custom format string
            
        Returns:
            Configured logger instance
        """
        if name in LoggerManager._loggers:
            return LoggerManager._loggers[name]
        
        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.handlers.clear()
        
        # Default format
        if format_string is None:
            format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        formatter = logging.Formatter(format_string)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler
        if log_file:
            FileManager.ensure_directory(os.path.dirname(log_file))
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        LoggerManager._loggers[name] = logger
        
        return logger
    
    @staticmethod
    def get_logger(name: str) -> logging.Logger:
        """
        Get existing logger or create new one.
        
        Args:
            name: Logger name
            
        Returns:
            Logger instance
        """
        if name not in LoggerManager._loggers:
            return LoggerManager.setup_logger(name)
        
        return LoggerManager._loggers[name]


class ReproducibilityHelper:
    """Ensures reproducibility across the system."""
    
    @staticmethod
    def set_seed(seed: int = 42) -> None:
        """
        Set random seeds for reproducibility.
        
        Args:
            seed: Random seed value
        """
        random.seed(seed)
        np.random.seed(seed)
        
        try:
            import torch
            torch.manual_seed(seed)
            
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
        except ImportError:
            pass
    
    @staticmethod
    def get_random_state() -> Dict[str, Any]:
        """
        Get current random state for all libraries.
        
        Returns:
            Dictionary with random states
        """
        state = {
            'python': random.getstate(),
            'numpy': np.random.get_state()
        }
        
        try:
            import torch
            state['torch'] = torch.get_rng_state()
            if torch.cuda.is_available():
                state['torch_cuda'] = torch.cuda.get_rng_state_all()
        except ImportError:
            pass
        
        return state
    
    @staticmethod
    def set_random_state(state: Dict[str, Any]) -> None:
        """
        Restore random state for all libraries.
        
        Args:
            state: Dictionary with random states
        """
        if 'python' in state:
            random.setstate(state['python'])
        
        if 'numpy' in state:
            np.random.set_state(state['numpy'])
        
        try:
            import torch
            if 'torch' in state:
                torch.set_rng_state(state['torch'])
            if 'torch_cuda' in state and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(state['torch_cuda'])
        except ImportError:
            pass


class Timer:
    """Simple timer for performance measurement."""
    
    def __init__(self):
        """Initialize timer."""
        self.start_time = None
        self.end_time = None
    
    def start(self) -> None:
        """Start the timer."""
        self.start_time = datetime.now()
    
    def stop(self) -> float:
        """
        Stop the timer and return elapsed time.
        
        Returns:
            Elapsed time in seconds
        """
        self.end_time = datetime.now()
        return self.elapsed()
    
    def elapsed(self) -> float:
        """
        Get elapsed time.
        
        Returns:
            Elapsed time in seconds
        """
        if self.start_time is None:
            return 0.0
        
        end = self.end_time if self.end_time else datetime.now()
        delta = end - self.start_time
        return delta.total_seconds()
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, *args):
        """Context manager exit."""
        self.stop()


class DataValidator:
    """Validates data integrity and quality."""
    
    @staticmethod
    def validate_dataframe(
        df: pd.DataFrame,
        required_columns: List[str],
        allow_null: bool = False
    ) -> bool:
        """
        Validate DataFrame structure and content.
        
        Args:
            df: DataFrame to validate
            required_columns: List of required column names
            allow_null: Whether to allow null values
            
        Returns:
            True if valid, False otherwise
        """
        # Check required columns
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            return False
        
        # Check for null values
        if not allow_null:
            if df[required_columns].isnull().any().any():
                return False
        
        return True
    
    @staticmethod
    def validate_labels(labels: np.ndarray, num_classes: int = 2) -> bool:
        """
        Validate label array.
        
        Args:
            labels: Array of labels
            num_classes: Expected number of classes
            
        Returns:
            True if valid, False otherwise
        """
        unique_labels = np.unique(labels)
        
        # Check if labels are in valid range
        if not all(0 <= label < num_classes for label in unique_labels):
            return False
        
        return True
    
    @staticmethod
    def check_data_balance(labels: np.ndarray, threshold: float = 0.3) -> bool:
        """
        Check if data is reasonably balanced.
        
        Args:
            labels: Array of labels
            threshold: Maximum acceptable deviation from 50-50 split
            
        Returns:
            True if balanced within threshold, False otherwise
        """
        label_ratio = np.mean(labels)
        deviation = abs(label_ratio - 0.5)
        
        return deviation <= threshold


class HashHelper:
    """Provides hashing utilities for data integrity."""
    
    @staticmethod
    def hash_string(text: str, algorithm: str = 'sha256') -> str:
        """
        Generate hash of string.
        
        Args:
            text: Text to hash
            algorithm: Hash algorithm ('md5', 'sha256', etc.)
            
        Returns:
            Hexadecimal hash string
        """
        hash_obj = hashlib.new(algorithm)
        hash_obj.update(text.encode('utf-8'))
        return hash_obj.hexdigest()
    
    @staticmethod
    def hash_file(filepath: str, algorithm: str = 'sha256') -> str:
        """
        Generate hash of file contents.
        
        Args:
            filepath: Path to file
            algorithm: Hash algorithm
            
        Returns:
            Hexadecimal hash string
            
        Raises:
            FileNotFoundError: If file doesn't exist
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        hash_obj = hashlib.new(algorithm)
        
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                hash_obj.update(chunk)
        
        return hash_obj.hexdigest()
    
    @staticmethod
    def hash_dataframe(df: pd.DataFrame, algorithm: str = 'sha256') -> str:
        """
        Generate hash of DataFrame contents.
        
        Args:
            df: DataFrame to hash
            algorithm: Hash algorithm
            
        Returns:
            Hexadecimal hash string
        """
        hash_obj = hashlib.new(algorithm)
        hash_obj.update(pd.util.hash_pandas_object(df, index=True).values)
        return hash_obj.hexdigest()


class MetricsFormatter:
    """Formats metrics for display and export."""
    
    @staticmethod
    def format_metrics(
        metrics: Dict[str, float],
        precision: int = 4
    ) -> Dict[str, str]:
        """
        Format metrics dictionary for display.
        
        Args:
            metrics: Dictionary of metric values
            precision: Number of decimal places
            
        Returns:
            Dictionary with formatted metric strings
        """
        formatted = {}
        
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                formatted[key] = f"{value:.{precision}f}"
            else:
                formatted[key] = str(value)
        
        return formatted
    
    @staticmethod
    def metrics_to_table(
        metrics_dict: Dict[str, Dict[str, float]],
        row_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Convert metrics dictionary to DataFrame table.
        
        Args:
            metrics_dict: Dictionary mapping names to metric dictionaries
            row_names: Optional custom row names
            
        Returns:
            DataFrame with metrics as table
        """
        if row_names is None:
            row_names = list(metrics_dict.keys())
        
        df = pd.DataFrame(metrics_dict).T
        df.index = row_names
        
        return df
    
    @staticmethod
    def format_percentage(value: float, precision: int = 2) -> str:
        """
        Format value as percentage.
        
        Args:
            value: Value to format (0-1 range)
            precision: Number of decimal places
            
        Returns:
            Formatted percentage string
        """
        return f"{value * 100:.{precision}f}%"


class ProgressTracker:
    """Tracks progress of long-running operations."""
    
    def __init__(self, total: int, description: str = "Processing"):
        """
        Initialize progress tracker.
        
        Args:
            total: Total number of items
            description: Description of operation
        """
        self.total = total
        self.current = 0
        self.description = description
        self.start_time = datetime.now()
    
    def update(self, n: int = 1) -> None:
        """
        Update progress.
        
        Args:
            n: Number of items completed
        """
        self.current += n
    
    def get_progress(self) -> float:
        """
        Get current progress as percentage.
        
        Returns:
            Progress percentage (0-100)
        """
        if self.total == 0:
            return 100.0
        
        return (self.current / self.total) * 100
    
    def get_eta(self) -> Optional[float]:
        """
        Get estimated time to completion.
        
        Returns:
            Estimated seconds remaining, or None if not enough data
        """
        if self.current == 0:
            return None
        
        elapsed = (datetime.now() - self.start_time).total_seconds()
        rate = self.current / elapsed
        remaining = self.total - self.current
        
        return remaining / rate if rate > 0 else None
    
    def __str__(self) -> str:
        """String representation of progress."""
        progress = self.get_progress()
        return f"{self.description}: {self.current}/{self.total} ({progress:.1f}%)"


class ConfigValidator:
    """Validates configuration settings."""
    
    @staticmethod
    def validate_range(
        value: float,
        min_val: float,
        max_val: float,
        name: str
    ) -> bool:
        """
        Validate value is within range.
        
        Args:
            value: Value to validate
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            name: Parameter name for error messages
            
        Returns:
            True if valid, False otherwise
        """
        if not (min_val <= value <= max_val):
            logging.warning(
                f"{name} value {value} is outside valid range [{min_val}, {max_val}]"
            )
            return False
        
        return True
    
    @staticmethod
    def validate_positive(value: float, name: str) -> bool:
        """
        Validate value is positive.
        
        Args:
            value: Value to validate
            name: Parameter name for error messages
            
        Returns:
            True if valid, False otherwise
        """
        if value <= 0:
            logging.warning(f"{name} must be positive, got {value}")
            return False
        
        return True
    
    @staticmethod
    def validate_probability(value: float, name: str) -> bool:
        """
        Validate value is a valid probability (0-1).
        
        Args:
            value: Value to validate
            name: Parameter name for error messages
            
        Returns:
            True if valid, False otherwise
        """
        return ConfigValidator.validate_range(value, 0.0, 1.0, name)


def get_timestamp(format_string: str = "%Y%m%d_%H%M%S") -> str:
    """
    Get current timestamp as formatted string.
    
    Args:
        format_string: strftime format string
        
    Returns:
        Formatted timestamp string
    """
    return datetime.now().strftime(format_string)


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero.
    
    Args:
        numerator: Numerator value
        denominator: Denominator value
        default: Default value if division by zero
        
    Returns:
        Division result or default value
    """
    if denominator == 0:
        return default
    
    return numerator / denominator


def flatten_dict(
    d: Dict[str, Any],
    parent_key: str = '',
    sep: str = '_'
) -> Dict[str, Any]:
    """
    Flatten nested dictionary.
    
    Args:
        d: Dictionary to flatten
        parent_key: Parent key for recursion
        sep: Separator for nested keys
        
    Returns:
        Flattened dictionary
    """
    items = []
    
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    
    return dict(items)


def batch_iterator(
    items: List[Any],
    batch_size: int
) -> List[List[Any]]:
    """
    Create batches from list of items.
    
    Args:
        items: List of items to batch
        batch_size: Size of each batch
        
    Returns:
        List of batches
    """
    batches = []
    
    for i in range(0, len(items), batch_size):
        batches.append(items[i:i + batch_size])
    
    return batches


def retry_on_failure(
    func: Callable,
    max_retries: int = 3,
    delay: float = 1.0
) -> Callable:
    """
    Decorator to retry function on failure.
    
    Args:
        func: Function to wrap
        max_retries: Maximum number of retry attempts
        delay: Delay between retries in seconds
        
    Returns:
        Wrapped function
    """
    import time
    from functools import wraps
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                
                logging.warning(
                    f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}. "
                    f"Retrying in {delay}s..."
                )
                time.sleep(delay)
        
        return None
    
    return wrapper


# Convenience instances
file_manager = FileManager()
logger_manager = LoggerManager()
reproducibility = ReproducibilityHelper()
data_validator = DataValidator()
hash_helper = HashHelper()
metrics_formatter = MetricsFormatter()
config_validator = ConfigValidator()
