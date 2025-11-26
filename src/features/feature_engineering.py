"""
Feature Engineering Module
==========================
Advanced feature creation, selection, and transformation.
Implements domain-specific features for churn prediction.

Author: Your Name
Date: 2024
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import (
    SelectKBest,
    f_classif,
    mutual_info_classif,
    RFE
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ChurnFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Domain-specific feature engineering for telecom churn prediction.
    Creates meaningful business features from raw data.
    """
    
    def __init__(self, create_interactions: bool = True):
        self.create_interactions = create_interactions
        self.feature_names_: List[str] = []
    
    def fit(self, X: pd.DataFrame, y=None):
        """Fit - no fitting needed for this transformer."""
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Create engineered features.
        
        Args:
            X: Input DataFrame
            
        Returns:
            DataFrame with additional engineered features
        """
        df = X.copy()
        
        # =====================================================================
        # Tenure-based features
        # =====================================================================
        if 'tenure' in df.columns:
            # Tenure bins (customer lifecycle stage)
            df['tenure_group'] = pd.cut(
                df['tenure'],
                bins=[0, 12, 24, 48, 72],
                labels=['0-1yr', '1-2yr', '2-4yr', '4yr+'],
                include_lowest=True
            )
            
            # New customer flag
            df['is_new_customer'] = (df['tenure'] <= 6).astype(int)
            
            # Loyal customer flag
            df['is_loyal_customer'] = (df['tenure'] >= 36).astype(int)
            
            logger.debug("Created tenure-based features")
        
        # =====================================================================
        # Financial features
        # =====================================================================
        if 'MonthlyCharges' in df.columns and 'tenure' in df.columns:
            # Average charges per month of tenure
            df['avg_charges_per_tenure'] = np.where(
                df['tenure'] > 0,
                df['TotalCharges'] / df['tenure'],
                df['MonthlyCharges']
            )
            
            # Charge increase indicator (high charges relative to tenure)
            avg_charge = df['MonthlyCharges'].mean()
            df['high_charges'] = (df['MonthlyCharges'] > avg_charge).astype(int)
            
        if 'TotalCharges' in df.columns and 'MonthlyCharges' in df.columns:
            # Expected vs actual total charges
            df['expected_total'] = df['tenure'] * df['MonthlyCharges']
            df['charge_deviation'] = df['TotalCharges'] - df['expected_total']
            
            # Price sensitivity indicator
            df['charge_to_tenure_ratio'] = np.where(
                df['tenure'] > 0,
                df['MonthlyCharges'] / df['tenure'],
                df['MonthlyCharges']
            )
        
        # =====================================================================
        # Service-based features
        # =====================================================================
        service_columns = [
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies'
        ]
        available_services = [col for col in service_columns if col in df.columns]
        
        if available_services:
            # Count of additional services
            df['num_services'] = df[available_services].apply(
                lambda row: (row == 'Yes').sum(), axis=1
            )
            
            # Has any protection service
            protection_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection']
            avail_protection = [c for c in protection_cols if c in df.columns]
            if avail_protection:
                df['has_protection'] = df[avail_protection].apply(
                    lambda row: (row == 'Yes').any(), axis=1
                ).astype(int)
            
            # Has streaming services
            streaming_cols = ['StreamingTV', 'StreamingMovies']
            avail_streaming = [c for c in streaming_cols if c in df.columns]
            if avail_streaming:
                df['has_streaming'] = df[avail_streaming].apply(
                    lambda row: (row == 'Yes').any(), axis=1
                ).astype(int)
            
            logger.debug(f"Created service features from {len(available_services)} services")
        
        # =====================================================================
        # Contract and payment features
        # =====================================================================
        if 'Contract' in df.columns:
            # Contract security (long-term contracts are more secure)
            df['contract_length_months'] = df['Contract'].map({
                'Month-to-month': 1,
                'One year': 12,
                'Two year': 24
            })
            
            # High churn risk contract
            df['is_month_to_month'] = (df['Contract'] == 'Month-to-month').astype(int)
        
        if 'PaymentMethod' in df.columns:
            # Automatic payment (usually indicates lower churn risk)
            df['has_auto_payment'] = df['PaymentMethod'].str.contains(
                'automatic', case=False, na=False
            ).astype(int)
            
            # Electronic check (often associated with higher churn)
            df['uses_electronic_check'] = (
                df['PaymentMethod'] == 'Electronic check'
            ).astype(int)
        
        # =====================================================================
        # Interaction features
        # =====================================================================
        if self.create_interactions:
            # High risk combination: month-to-month + electronic check
            if 'is_month_to_month' in df.columns and 'uses_electronic_check' in df.columns:
                df['high_risk_combo'] = (
                    df['is_month_to_month'] & df['uses_electronic_check']
                ).astype(int)
            
            # Value score: services per dollar
            if 'num_services' in df.columns and 'MonthlyCharges' in df.columns:
                df['value_score'] = np.where(
                    df['MonthlyCharges'] > 0,
                    df['num_services'] / df['MonthlyCharges'] * 100,
                    0
                )
            
            # Tenure-contract alignment
            if 'tenure' in df.columns and 'contract_length_months' in df.columns:
                df['tenure_contract_ratio'] = np.where(
                    df['contract_length_months'] > 0,
                    df['tenure'] / df['contract_length_months'],
                    df['tenure']
                )
        
        # Store feature names
        self.feature_names_ = df.columns.tolist()
        
        logger.info(f"Feature engineering complete: {len(df.columns)} features")
        return df
    
    def get_feature_names_out(self) -> List[str]:
        """Return feature names."""
        return self.feature_names_


class FeatureSelector:
    """
    Feature selection using multiple methods.
    Helps identify most predictive features.
    """
    
    def __init__(
        self,
        method: str = "mutual_info",
        n_features: int = 20
    ):
        """
        Initialize feature selector.
        
        Args:
            method: Selection method ('mutual_info', 'f_classif', 'rfe')
            n_features: Number of features to select
        """
        self.method = method
        self.n_features = n_features
        self.selector = None
        self.selected_features_: List[str] = []
        self.feature_scores_: Optional[np.ndarray] = None
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str]
    ) -> 'FeatureSelector':
        """
        Fit the feature selector.
        
        Args:
            X: Feature matrix
            y: Target vector
            feature_names: List of feature names
        """
        if self.method == "mutual_info":
            self.selector = SelectKBest(
                score_func=mutual_info_classif,
                k=min(self.n_features, X.shape[1])
            )
        elif self.method == "f_classif":
            self.selector = SelectKBest(
                score_func=f_classif,
                k=min(self.n_features, X.shape[1])
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        self.selector.fit(X, y)
        self.feature_scores_ = self.selector.scores_
        
        # Get selected feature names
        mask = self.selector.get_support()
        self.selected_features_ = [
            name for name, selected in zip(feature_names, mask) if selected
        ]
        
        logger.info(f"Selected {len(self.selected_features_)} features using {self.method}")
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data to selected features."""
        if self.selector is None:
            raise ValueError("Selector not fitted. Call fit first.")
        return self.selector.transform(X)
    
    def fit_transform(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str]
    ) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(X, y, feature_names)
        return self.transform(X)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance scores.
        
        Returns:
            DataFrame with feature names and scores
        """
        if self.feature_scores_ is None:
            raise ValueError("Selector not fitted.")
        
        return pd.DataFrame({
            'feature': self.selected_features_,
            'score': self.feature_scores_[self.selector.get_support()]
        }).sort_values('score', ascending=False)


def create_feature_report(
    df: pd.DataFrame,
    target_column: str = "Churn"
) -> pd.DataFrame:
    """
    Create a report of feature statistics and correlations with target.
    
    Args:
        df: DataFrame with features and target
        target_column: Name of target column
        
    Returns:
        DataFrame with feature statistics
    """
    # Encode target if needed
    if df[target_column].dtype == 'object':
        target = (df[target_column] == 'Yes').astype(int)
    else:
        target = df[target_column]
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_column in numerical_cols:
        numerical_cols.remove(target_column)
    
    report_data = []
    
    for col in numerical_cols:
        if col == target_column:
            continue
            
        report_data.append({
            'feature': col,
            'dtype': str(df[col].dtype),
            'missing': df[col].isnull().sum(),
            'missing_pct': df[col].isnull().mean() * 100,
            'mean': df[col].mean(),
            'std': df[col].std(),
            'min': df[col].min(),
            'max': df[col].max(),
            'correlation_with_target': df[col].corr(target)
        })
    
    return pd.DataFrame(report_data).sort_values(
        'correlation_with_target',
        key=abs,
        ascending=False
    )


# =============================================================================
# CLI Interface
# =============================================================================
if __name__ == "__main__":
    import sys
    sys.path.append(str(Path(__file__).parent.parent))

    from src.data.data_loader import DataLoader
    
    # Load data
    loader = DataLoader()
    df = loader.load_data()
    
    # Apply feature engineering
    engineer = ChurnFeatureEngineer()
    df_engineered = engineer.fit_transform(df)
    
    print(f"\n✅ Feature engineering complete!")
    print(f"   Original features: {len(df.columns)}")
    print(f"   Engineered features: {len(df_engineered.columns)}")
    print(f"\n📊 New features created:")
    
    new_features = set(df_engineered.columns) - set(df.columns)
    for feat in sorted(new_features):
        print(f"   - {feat}")
