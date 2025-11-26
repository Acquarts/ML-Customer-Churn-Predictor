"""
Data Preprocessing Module
=========================
Handles data cleaning, transformation, and splitting for ML pipeline.
Implements sklearn-compatible transformers for production use.

Author: Your Name
Date: 2024
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import yaml
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DataCleaner(BaseEstimator, TransformerMixin):
    """
    Custom transformer for data cleaning operations.
    Sklearn-compatible for use in pipelines.
    """
    
    def __init__(self, target_column: str = "Churn"):
        self.target_column = target_column
        self.columns_to_drop: List[str] = []
    
    def fit(self, X: pd.DataFrame, y=None):
        """Fit the cleaner - identify columns to drop."""
        # Identify columns with too many missing values
        missing_threshold = 0.5
        missing_ratio = X.isnull().sum() / len(X)
        self.columns_to_drop = missing_ratio[missing_ratio > missing_threshold].index.tolist()
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply cleaning transformations."""
        df = X.copy()
        
        # Drop high-missing columns
        df = df.drop(columns=self.columns_to_drop, errors='ignore')
        
        # Convert TotalCharges to numeric (handle empty strings)
        if 'TotalCharges' in df.columns:
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
            # Fill missing TotalCharges with tenure * MonthlyCharges
            if 'tenure' in df.columns and 'MonthlyCharges' in df.columns:
                mask = df['TotalCharges'].isnull()
                df.loc[mask, 'TotalCharges'] = (
                    df.loc[mask, 'tenure'] * df.loc[mask, 'MonthlyCharges']
                )
        
        # Convert SeniorCitizen to categorical
        if 'SeniorCitizen' in df.columns:
            df['SeniorCitizen'] = df['SeniorCitizen'].map({0: 'No', 1: 'Yes'})
        
        logger.info(f"Data cleaned: {df.shape}")
        return df


class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Select features based on configuration.
    """
    
    def __init__(
        self,
        numerical_features: List[str],
        categorical_features: List[str],
        id_column: Optional[str] = None
    ):
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.id_column = id_column
        self.feature_columns: List[str] = []
    
    def fit(self, X: pd.DataFrame, y=None):
        """Determine available features."""
        available = set(X.columns)
        
        self.feature_columns = [
            col for col in self.numerical_features + self.categorical_features
            if col in available
        ]
        
        if self.id_column and self.id_column in available:
            self.feature_columns = [
                col for col in self.feature_columns if col != self.id_column
            ]
        
        logger.info(f"Selected {len(self.feature_columns)} features")
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Select only configured features."""
        return X[self.feature_columns].copy()


class DataPreprocessor:
    """
    Main preprocessing class that orchestrates the entire pipeline.
    
    Example:
        >>> preprocessor = DataPreprocessor(config_path="config/config.yaml")
        >>> X_train, X_test, y_train, y_test = preprocessor.fit_transform(df)
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = self._load_config(config_path)
        self.project_root = Path(__file__).parent.parent.parent
        
        self.numerical_features = self.config["features"]["numerical"]
        self.categorical_features = self.config["features"]["categorical"]
        self.target_column = self.config["data"]["target_column"]
        self.id_column = self.config["data"]["id_column"]
        
        self.preprocessor: Optional[ColumnTransformer] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self.cleaner: Optional[DataCleaner] = None
        
        logger.info("DataPreprocessor initialized")
    
    def _load_config(self, config_path: str) -> dict:
        """Load YAML configuration."""
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    
    def _create_preprocessor(self) -> ColumnTransformer:
        """Create sklearn ColumnTransformer for feature preprocessing."""
        
        # Numerical pipeline
        numerical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Categorical pipeline
        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(
                handle_unknown='ignore',
                sparse_output=False,
                drop='if_binary'
            ))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_pipeline, self.numerical_features),
                ('cat', categorical_pipeline, self.categorical_features)
            ],
            remainder='drop'
        )
        
        return preprocessor
    
    def fit_transform(
        self,
        df: pd.DataFrame,
        test_size: Optional[float] = None,
        random_state: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Fit preprocessor and transform data.
        
        Args:
            df: Input DataFrame
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        test_size = test_size or self.config["data"]["test_size"]
        random_state = random_state or self.config["data"]["random_state"]
        
        # Clean data
        self.cleaner = DataCleaner(target_column=self.target_column)
        df_clean = self.cleaner.fit_transform(df)
        
        # Separate features and target
        X = df_clean.drop(columns=[self.target_column, self.id_column], errors='ignore')
        y = df_clean[self.target_column]
        
        # Encode target
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Filter features to only those present in data
        available_num = [f for f in self.numerical_features if f in X.columns]
        available_cat = [f for f in self.categorical_features if f in X.columns]
        
        self.numerical_features = available_num
        self.categorical_features = available_cat
        
        # Train/test split
        stratify = y_encoded if self.config["data"]["stratify"] else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify
        )
        
        # Create and fit preprocessor
        self.preprocessor = self._create_preprocessor()
        X_train_processed = self.preprocessor.fit_transform(X_train)
        X_test_processed = self.preprocessor.transform(X_test)
        
        logger.info(f"Train shape: {X_train_processed.shape}")
        logger.info(f"Test shape: {X_test_processed.shape}")
        logger.info(f"Target distribution - Train: {np.bincount(y_train)}")
        logger.info(f"Target distribution - Test: {np.bincount(y_test)}")
        
        return X_train_processed, X_test_processed, y_train, y_test
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform new data using fitted preprocessor.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Transformed features as numpy array
        """
        if self.preprocessor is None:
            raise ValueError("Preprocessor not fitted. Call fit_transform first.")
        
        df_clean = self.cleaner.transform(df)
        X = df_clean.drop(columns=[self.target_column, self.id_column], errors='ignore')
        
        return self.preprocessor.transform(X)
    
    def get_feature_names(self) -> List[str]:
        """Get feature names after preprocessing."""
        if self.preprocessor is None:
            raise ValueError("Preprocessor not fitted.")
        
        return self.preprocessor.get_feature_names_out().tolist()
    
    def save(self, path: Optional[str] = None):
        """Save preprocessor artifacts."""
        path = path or str(self.project_root / "models" / "preprocessor.pkl")
        
        artifacts = {
            'preprocessor': self.preprocessor,
            'label_encoder': self.label_encoder,
            'cleaner': self.cleaner,
            'numerical_features': self.numerical_features,
            'categorical_features': self.categorical_features
        }
        
        joblib.dump(artifacts, path)
        logger.info(f"Preprocessor saved to {path}")
    
    def load(self, path: Optional[str] = None):
        """Load preprocessor artifacts."""
        path = path or str(self.project_root / "models" / "preprocessor.pkl")
        
        artifacts = joblib.load(path)
        
        self.preprocessor = artifacts['preprocessor']
        self.label_encoder = artifacts['label_encoder']
        self.cleaner = artifacts['cleaner']
        self.numerical_features = artifacts['numerical_features']
        self.categorical_features = artifacts['categorical_features']
        
        logger.info(f"Preprocessor loaded from {path}")


# =============================================================================
# CLI Interface
# =============================================================================
if __name__ == "__main__":
    import argparse
    from data_loader import DataLoader
    
    parser = argparse.ArgumentParser(description="Preprocess data")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to config file"
    )
    
    args = parser.parse_args()
    
    # Load data
    loader = DataLoader(config_path=args.config)
    df = loader.load_data()
    
    # Preprocess
    preprocessor = DataPreprocessor(config_path=args.config)
    X_train, X_test, y_train, y_test = preprocessor.fit_transform(df)
    
    # Save preprocessor
    preprocessor.save()
    
    print(f"\n✅ Preprocessing complete!")
    print(f"   Train samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    print(f"   Features: {X_train.shape[1]}")
