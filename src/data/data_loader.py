"""
Data Loader Module
==================
Handles data acquisition from various sources including Kaggle, local files,
and databases. Implements caching and validation.

Author: Your Name
Date: 2024
"""

import logging
import os
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DataLoader:
    """
    Professional data loader with support for multiple sources and caching.
    
    Attributes:
        config (dict): Configuration dictionary
        data_dir (Path): Path to data directory
    
    Example:
        >>> loader = DataLoader(config_path="config/config.yaml")
        >>> df = loader.load_data()
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize DataLoader with configuration.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config = self._load_config(config_path)
        self.project_root = Path(__file__).parent.parent.parent
        self.data_dir = self.project_root / "data"
        
        # Create directories if they don't exist
        (self.data_dir / "raw").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "processed").mkdir(parents=True, exist_ok=True)
        
        logger.info(f"DataLoader initialized with config from {config_path}")
    
    def _load_config(self, config_path: str) -> dict:
        """Load YAML configuration file."""
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    
    def load_data(
        self,
        source: Optional[str] = None,
        force_download: bool = False
    ) -> pd.DataFrame:
        """
        Load data from configured source.
        
        Args:
            source: Override config source (kaggle dataset ID or file path)
            force_download: Force re-download even if file exists
            
        Returns:
            pd.DataFrame: Loaded dataset
        """
        source = source or self.config["data"]["source"]
        raw_path = self.project_root / self.config["data"]["raw_path"]
        
        # Check if data already exists
        if raw_path.exists() and not force_download:
            logger.info(f"Loading existing data from {raw_path}")
            return pd.read_csv(raw_path)
        
        # Download from Kaggle if source looks like a dataset ID
        if "/" in source and not Path(source).exists():
            df = self._download_from_kaggle(source)
        else:
            df = pd.read_csv(source)
        
        # Save to raw directory
        df.to_csv(raw_path, index=False)
        logger.info(f"Data saved to {raw_path}")
        
        return df
    
    def _download_from_kaggle(self, dataset_id: str) -> pd.DataFrame:
        """
        Download dataset from Kaggle.
        
        Args:
            dataset_id: Kaggle dataset identifier (user/dataset-name)
            
        Returns:
            pd.DataFrame: Downloaded dataset
        """
        try:
            import kaggle
            
            logger.info(f"Downloading dataset from Kaggle: {dataset_id}")
            
            download_path = self.data_dir / "raw"
            kaggle.api.dataset_download_files(
                dataset_id,
                path=download_path,
                unzip=True
            )
            
            # Find the CSV file
            csv_files = list(download_path.glob("*.csv"))
            if not csv_files:
                raise FileNotFoundError("No CSV files found in downloaded data")
            
            return pd.read_csv(csv_files[0])
            
        except ImportError:
            logger.warning("Kaggle package not installed. Using sample data.")
            return self._create_sample_data()
    
    def _create_sample_data(self) -> pd.DataFrame:
        """
        Create sample data for testing when Kaggle is not available.
        Uses realistic telco churn data structure.
        """
        import numpy as np
        
        np.random.seed(42)
        n_samples = 7043  # Same size as original dataset
        
        logger.info("Creating sample data for demonstration")
        
        data = {
            "customerID": [f"CUST-{i:05d}" for i in range(n_samples)],
            "gender": np.random.choice(["Male", "Female"], n_samples),
            "SeniorCitizen": np.random.choice([0, 1], n_samples, p=[0.84, 0.16]),
            "Partner": np.random.choice(["Yes", "No"], n_samples),
            "Dependents": np.random.choice(["Yes", "No"], n_samples, p=[0.3, 0.7]),
            "tenure": np.random.randint(0, 73, n_samples),
            "PhoneService": np.random.choice(["Yes", "No"], n_samples, p=[0.9, 0.1]),
            "MultipleLines": np.random.choice(
                ["Yes", "No", "No phone service"], n_samples, p=[0.42, 0.48, 0.1]
            ),
            "InternetService": np.random.choice(
                ["DSL", "Fiber optic", "No"], n_samples, p=[0.34, 0.44, 0.22]
            ),
            "OnlineSecurity": np.random.choice(
                ["Yes", "No", "No internet service"], n_samples
            ),
            "OnlineBackup": np.random.choice(
                ["Yes", "No", "No internet service"], n_samples
            ),
            "DeviceProtection": np.random.choice(
                ["Yes", "No", "No internet service"], n_samples
            ),
            "TechSupport": np.random.choice(
                ["Yes", "No", "No internet service"], n_samples
            ),
            "StreamingTV": np.random.choice(
                ["Yes", "No", "No internet service"], n_samples
            ),
            "StreamingMovies": np.random.choice(
                ["Yes", "No", "No internet service"], n_samples
            ),
            "Contract": np.random.choice(
                ["Month-to-month", "One year", "Two year"],
                n_samples,
                p=[0.55, 0.21, 0.24]
            ),
            "PaperlessBilling": np.random.choice(["Yes", "No"], n_samples),
            "PaymentMethod": np.random.choice(
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)"
                ],
                n_samples
            ),
            "MonthlyCharges": np.round(np.random.uniform(18, 118, n_samples), 2),
        }
        
        # Calculate TotalCharges based on tenure and monthly charges
        data["TotalCharges"] = np.round(
            data["tenure"] * data["MonthlyCharges"] + np.random.uniform(-50, 50, n_samples),
            2
        )
        data["TotalCharges"] = np.maximum(data["TotalCharges"], 0)
        
        # Generate Churn with realistic correlations
        churn_prob = np.zeros(n_samples)
        churn_prob += (data["Contract"] == "Month-to-month") * 0.3
        churn_prob += (np.array(data["tenure"]) < 12) * 0.2
        churn_prob += (np.array(data["MonthlyCharges"]) > 70) * 0.1
        churn_prob += (data["InternetService"] == "Fiber optic") * 0.1
        churn_prob = np.clip(churn_prob, 0.05, 0.85)
        
        data["Churn"] = np.random.binomial(1, churn_prob)
        data["Churn"] = ["Yes" if x == 1 else "No" for x in data["Churn"]]
        
        return pd.DataFrame(data)
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Validate loaded data against expected schema.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            bool: True if validation passes
        """
        required_columns = (
            self.config["features"]["numerical"] +
            self.config["features"]["categorical"] +
            [self.config["data"]["target_column"]]
        )
        
        missing = set(required_columns) - set(df.columns)
        if missing:
            logger.error(f"Missing required columns: {missing}")
            return False
        
        logger.info("Data validation passed")
        return True
    
    def get_data_info(self, df: pd.DataFrame) -> dict:
        """
        Get summary information about the dataset.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            dict: Summary statistics
        """
        target_col = self.config["data"]["target_column"]
        
        info = {
            "n_samples": len(df),
            "n_features": len(df.columns) - 1,
            "target_distribution": df[target_col].value_counts().to_dict(),
            "missing_values": df.isnull().sum().sum(),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024**2
        }
        
        logger.info(f"Dataset info: {info}")
        return info


# =============================================================================
# CLI Interface
# =============================================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Load and validate data")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download of data"
    )
    
    args = parser.parse_args()
    
    # Load data
    loader = DataLoader(config_path=args.config)
    df = loader.load_data(force_download=args.force_download)
    
    # Validate and show info
    loader.validate_data(df)
    loader.get_data_info(df)
    
    print(f"\n✅ Data loaded successfully: {df.shape}")
    print(f"📁 Saved to: {loader.config['data']['raw_path']}")
