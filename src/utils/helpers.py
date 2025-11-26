"""
Utility Functions
=================
Helper functions used across the project.

Author: Your Name
Date: 2024
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import yaml


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_str: Optional[str] = None
) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional path to log file
        format_str: Optional custom format string
        
    Returns:
        Configured logger
    """
    format_str = format_str or "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_str,
        handlers=handlers
    )
    
    return logging.getLogger(__name__)


def load_config(config_path: str = "config/config.yaml") -> dict:
    """
    Load YAML configuration file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def save_config(config: dict, config_path: str):
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
    """
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def get_project_root() -> Path:
    """
    Get the project root directory.
    
    Returns:
        Path to project root
    """
    return Path(__file__).parent.parent.parent


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if not.
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def generate_timestamp() -> str:
    """
    Generate timestamp string for file naming.
    
    Returns:
        Timestamp string (YYYYMMDD_HHMMSS)
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def format_metrics(metrics: Dict[str, float], precision: int = 4) -> str:
    """
    Format metrics dictionary as string.
    
    Args:
        metrics: Dictionary of metric names and values
        precision: Decimal precision
        
    Returns:
        Formatted string
    """
    lines = [f"{name}: {value:.{precision}f}" for name, value in metrics.items()]
    return "\n".join(lines)


def calculate_business_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    cost_false_negative: float = 100,  # Cost of missing a churner
    cost_false_positive: float = 10,   # Cost of false alarm
    cost_retention: float = 25         # Cost of retention campaign
) -> Dict[str, float]:
    """
    Calculate business-relevant metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities
        cost_false_negative: Cost of missing a churning customer
        cost_false_positive: Cost of targeting non-churner
        cost_retention: Cost of retention campaign per customer
        
    Returns:
        Dictionary of business metrics
    """
    from sklearn.metrics import confusion_matrix
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Cost without model (no intervention)
    baseline_cost = y_true.sum() * cost_false_negative
    
    # Cost with model
    model_cost = (
        fn * cost_false_negative +  # Missed churners
        fp * cost_false_positive +   # False alarms
        tp * cost_retention          # Successful targeting
    )
    
    # Savings
    savings = baseline_cost - model_cost
    savings_rate = savings / baseline_cost if baseline_cost > 0 else 0
    
    # Expected value per prediction
    n_predictions = len(y_true)
    expected_value = savings / n_predictions
    
    return {
        "baseline_cost": baseline_cost,
        "model_cost": model_cost,
        "savings": savings,
        "savings_rate": savings_rate,
        "expected_value_per_prediction": expected_value,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn)
    }


def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    metric: str = "f1"
) -> float:
    """
    Find optimal classification threshold.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        metric: Metric to optimize ('f1', 'precision', 'recall', 'profit')
        
    Returns:
        Optimal threshold
    """
    from sklearn.metrics import f1_score, precision_score, recall_score
    
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_threshold = 0.5
    best_score = 0
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        
        if metric == "f1":
            score = f1_score(y_true, y_pred)
        elif metric == "precision":
            score = precision_score(y_true, y_pred)
        elif metric == "recall":
            score = recall_score(y_true, y_pred)
        elif metric == "profit":
            business = calculate_business_metrics(y_true, y_pred, y_proba)
            score = business["savings"]
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    return best_threshold


def memory_usage(df: pd.DataFrame) -> str:
    """
    Get memory usage of DataFrame as formatted string.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Formatted memory usage string
    """
    bytes_used = df.memory_usage(deep=True).sum()
    
    if bytes_used < 1024:
        return f"{bytes_used} B"
    elif bytes_used < 1024**2:
        return f"{bytes_used/1024:.2f} KB"
    elif bytes_used < 1024**3:
        return f"{bytes_used/1024**2:.2f} MB"
    else:
        return f"{bytes_used/1024**3:.2f} GB"


def reduce_memory_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Reduce DataFrame memory usage by downcasting numeric types.
    
    Args:
        df: DataFrame to optimize
        verbose: Whether to print memory savings
        
    Returns:
        Optimized DataFrame
    """
    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min >= 0:
                    if c_max < 255:
                        df[col] = df[col].astype(np.uint8)
                    elif c_max < 65535:
                        df[col] = df[col].astype(np.uint16)
                    elif c_max < 4294967295:
                        df[col] = df[col].astype(np.uint32)
                else:
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                        
            elif str(col_type)[:5] == 'float':
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
    
    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    
    if verbose:
        print(f"Memory usage: {start_mem:.2f} MB → {end_mem:.2f} MB ({100*(start_mem-end_mem)/start_mem:.1f}% reduction)")
    
    return df


class Timer:
    """Context manager for timing code blocks."""
    
    def __init__(self, name: str = "Block"):
        self.name = name
        self.start = None
        self.end = None
        self.elapsed = None
    
    def __enter__(self):
        self.start = datetime.now()
        return self
    
    def __exit__(self, *args):
        self.end = datetime.now()
        self.elapsed = (self.end - self.start).total_seconds()
        print(f"⏱️ {self.name}: {self.elapsed:.2f}s")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    # Test utilities
    print("🔧 Testing utilities...")
    
    # Test logging
    logger = setup_logging(level="INFO")
    logger.info("Logging configured successfully")
    
    # Test config loading
    try:
        config = load_config()
        print(f"✅ Config loaded: {config['project']['name']}")
    except FileNotFoundError:
        print("⚠️ Config file not found (expected in config/config.yaml)")
    
    # Test timer
    with Timer("Test timer"):
        import time
        time.sleep(0.5)
    
    print("\n✅ All utilities working!")
