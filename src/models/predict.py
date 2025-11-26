"""
Model Prediction Module
=======================
Handles model loading and inference for production use.
Provides clean API for single and batch predictions.

Author: Your Name
Date: 2024
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ChurnPredictor:
    """
    Production-ready churn predictor.
    Handles data preprocessing and model inference.
    
    Example:
        >>> predictor = ChurnPredictor()
        >>> result = predictor.predict(customer_data)
        >>> print(f"Churn probability: {result['probability']:.2%}")
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        preprocessor_path: Optional[str] = None,
        config_path: str = "config/config.yaml"
    ):
        """
        Initialize predictor with model and preprocessor.
        
        Args:
            model_path: Path to saved model
            preprocessor_path: Path to saved preprocessor
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.project_root = Path(__file__).parent.parent.parent
        
        # Default paths
        model_path = model_path or str(
            self.project_root / "models" / "best_model.pkl"
        )
        preprocessor_path = preprocessor_path or str(
            self.project_root / "models" / "preprocessor.pkl"
        )
        
        # Load model and preprocessor
        self.model = self._load_model(model_path)
        self.preprocessor_artifacts = self._load_preprocessor(preprocessor_path)
        
        # Load metadata
        metadata_path = model_path.replace('.pkl', '_metadata.yaml')
        self.metadata = self._load_metadata(metadata_path)
        
        logger.info("ChurnPredictor initialized successfully")
    
    def _load_config(self, config_path: str) -> dict:
        """Load YAML configuration."""
        try:
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config not found at {config_path}, using defaults")
            return {}
    
    def _load_model(self, path: str) -> Any:
        """Load trained model."""
        try:
            model = joblib.load(path)
            logger.info(f"Model loaded from {path}")
            return model
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Model not found at {path}. "
                "Please train the model first using train.py"
            )
    
    def _load_preprocessor(self, path: str) -> dict:
        """Load preprocessor artifacts."""
        try:
            artifacts = joblib.load(path)
            logger.info(f"Preprocessor loaded from {path}")
            return artifacts
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Preprocessor not found at {path}. "
                "Please train the model first using train.py"
            )
    
    def _load_metadata(self, path: str) -> dict:
        """Load model metadata."""
        try:
            with open(path, 'r') as f:
                # Try safe_load first
                try:
                    return yaml.safe_load(f)
                except yaml.constructor.ConstructorError:
                    # If safe_load fails due to numpy objects, use unsafe load
                    logger.warning("Metadata contains unsafe YAML objects, using full_load")
                    f.seek(0)
                    metadata = yaml.full_load(f)

                    # Convert numpy scalars to native Python types
                    if 'metrics' in metadata:
                        for key, value in metadata['metrics'].items():
                            if hasattr(value, 'item'):
                                metadata['metrics'][key] = value.item()

                    return metadata
        except FileNotFoundError:
            logger.warning("Model metadata not found")
            return {}
        except Exception as e:
            logger.warning(f"Error loading metadata: {e}")
            return {}
    
    def _preprocess(self, data: pd.DataFrame) -> np.ndarray:
        """
        Preprocess input data using saved preprocessor.
        
        Args:
            data: Raw input data
            
        Returns:
            Preprocessed feature array
        """
        preprocessor = self.preprocessor_artifacts['preprocessor']
        cleaner = self.preprocessor_artifacts['cleaner']
        
        # Clean data
        data_clean = cleaner.transform(data)
        
        # Remove target and ID columns if present
        target_col = self.config.get("data", {}).get("target_column", "Churn")
        id_col = self.config.get("data", {}).get("id_column", "customerID")
        
        cols_to_drop = [target_col, id_col]
        data_clean = data_clean.drop(
            columns=[c for c in cols_to_drop if c in data_clean.columns],
            errors='ignore'
        )
        
        # Transform
        return preprocessor.transform(data_clean)
    
    def predict(
        self,
        data: Union[pd.DataFrame, Dict],
        return_proba: bool = True
    ) -> Dict[str, Any]:
        """
        Make prediction for customer data.
        
        Args:
            data: Customer data (DataFrame or dict for single customer)
            return_proba: Whether to return probabilities
            
        Returns:
            Dictionary with prediction results
        """
        # Convert dict to DataFrame if needed
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        
        # Preprocess
        X = self._preprocess(data)
        
        # Predict
        prediction = self.model.predict(X)
        
        result = {
            "prediction": int(prediction[0]),
            "churn_label": "Yes" if prediction[0] == 1 else "No"
        }
        
        if return_proba and hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(X)
            result["probability"] = float(proba[0, 1])
            result["confidence"] = float(max(proba[0]))
            
            # Risk level
            if result["probability"] >= 0.7:
                result["risk_level"] = "High"
            elif result["probability"] >= 0.4:
                result["risk_level"] = "Medium"
            else:
                result["risk_level"] = "Low"
        
        return result
    
    def predict_batch(
        self,
        data: pd.DataFrame,
        return_proba: bool = True
    ) -> pd.DataFrame:
        """
        Make predictions for multiple customers.
        
        Args:
            data: DataFrame with customer data
            return_proba: Whether to return probabilities
            
        Returns:
            DataFrame with predictions
        """
        # Store original index
        original_index = data.index
        
        # Preprocess
        X = self._preprocess(data)
        
        # Predict
        predictions = self.model.predict(X)
        
        results = pd.DataFrame({
            "prediction": predictions,
            "churn_label": ["Yes" if p == 1 else "No" for p in predictions]
        }, index=original_index)
        
        if return_proba and hasattr(self.model, 'predict_proba'):
            probas = self.model.predict_proba(X)
            results["probability"] = probas[:, 1]
            results["confidence"] = probas.max(axis=1)
            
            # Risk levels
            results["risk_level"] = pd.cut(
                results["probability"],
                bins=[0, 0.4, 0.7, 1.0],
                labels=["Low", "Medium", "High"],
                include_lowest=True
            )
        
        return results
    
    def get_risk_factors(
        self,
        data: Union[pd.DataFrame, Dict],
        top_n: int = 5
    ) -> List[Dict]:
        """
        Get top risk factors for a customer's churn prediction.
        
        Args:
            data: Customer data
            top_n: Number of top factors to return
            
        Returns:
            List of risk factors with their contributions
        """
        # This would use SHAP values in a full implementation
        # For now, return feature importance based risk factors
        
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        
        risk_factors = []
        
        # Contract type
        if 'Contract' in data.columns:
            if data['Contract'].iloc[0] == 'Month-to-month':
                risk_factors.append({
                    "factor": "Month-to-month contract",
                    "impact": "High",
                    "recommendation": "Consider offering contract upgrade incentives"
                })
        
        # Tenure
        if 'tenure' in data.columns:
            tenure = data['tenure'].iloc[0]
            if tenure < 12:
                risk_factors.append({
                    "factor": f"New customer (tenure: {tenure} months)",
                    "impact": "High",
                    "recommendation": "Implement early engagement program"
                })
        
        # Payment method
        if 'PaymentMethod' in data.columns:
            if data['PaymentMethod'].iloc[0] == 'Electronic check':
                risk_factors.append({
                    "factor": "Electronic check payment",
                    "impact": "Medium",
                    "recommendation": "Encourage automatic payment setup"
                })
        
        # Monthly charges
        if 'MonthlyCharges' in data.columns:
            charges = data['MonthlyCharges'].iloc[0]
            if charges > 70:
                risk_factors.append({
                    "factor": f"High monthly charges (${charges:.2f})",
                    "impact": "Medium",
                    "recommendation": "Review for potential discount eligibility"
                })
        
        # Internet service
        if 'InternetService' in data.columns:
            if data['InternetService'].iloc[0] == 'Fiber optic':
                # Check for lack of support services
                support_services = ['OnlineSecurity', 'TechSupport']
                missing_support = [
                    s for s in support_services
                    if s in data.columns and data[s].iloc[0] == 'No'
                ]
                if missing_support:
                    risk_factors.append({
                        "factor": "Fiber optic without support services",
                        "impact": "Medium",
                        "recommendation": "Bundle security and support services"
                    })
        
        return risk_factors[:top_n]
    
    def get_model_info(self) -> Dict:
        """Get model information and metadata."""
        return {
            "model_name": self.metadata.get("model_name", "Unknown"),
            "training_date": self.metadata.get("training_date", "Unknown"),
            "metrics": self.metadata.get("metrics", {}),
            "n_features": len(self.metadata.get("feature_names", []))
        }


def create_sample_customer() -> Dict:
    """Create a sample customer for testing."""
    return {
        "customerID": "TEST-001",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 89.95,
        "TotalCharges": 1079.40
    }


# =============================================================================
# CLI Interface
# =============================================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Make churn predictions")
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Run prediction on sample customer"
    )
    
    args = parser.parse_args()
    
    if args.sample:
        print("🔮 Churn Predictor Demo")
        print("="*50)
        
        try:
            predictor = ChurnPredictor()
            
            # Sample customer
            customer = create_sample_customer()
            print("\n📋 Customer Profile:")
            for key, value in customer.items():
                print(f"   {key}: {value}")
            
            # Make prediction
            result = predictor.predict(customer)
            
            print("\n🎯 Prediction Result:")
            print(f"   Churn: {result['churn_label']}")
            print(f"   Probability: {result['probability']:.1%}")
            print(f"   Risk Level: {result['risk_level']}")
            print(f"   Confidence: {result['confidence']:.1%}")
            
            # Risk factors
            print("\n⚠️  Risk Factors:")
            for factor in predictor.get_risk_factors(customer):
                print(f"   • {factor['factor']} ({factor['impact']} impact)")
                print(f"     → {factor['recommendation']}")
            
        except FileNotFoundError as e:
            print(f"\n❌ Error: {e}")
            print("   Please train the model first: python src/models/train.py")
