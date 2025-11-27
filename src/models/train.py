"""
Model Training Module
=====================
Handles model training, hyperparameter optimization, and evaluation.
Supports multiple algorithms with MLflow tracking.

Author: Your Name
Date: 2024
"""

import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    average_precision_score
)
from sklearn.model_selection import cross_val_score, StratifiedKFold

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Try to import optional dependencies
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not available")

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logger.warning("LightGBM not available")

try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("MLflow not available")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not available")


class ModelTrainer:
    """
    Professional model trainer with support for multiple algorithms,
    hyperparameter tuning, and experiment tracking.
    
    Example:
        >>> trainer = ModelTrainer(config_path="config/config.yaml")
        >>> best_model = trainer.train(X_train, y_train, X_test, y_test)
        >>> trainer.save_model()
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize trainer with configuration."""
        self.config = self._load_config(config_path)
        self.project_root = Path(__file__).parent.parent.parent
        
        self.models: Dict[str, Any] = {}
        self.results: Dict[str, Dict] = {}
        self.best_model: Optional[Any] = None
        self.best_model_name: str = ""
        self.feature_names: List[str] = []
        
        # Initialize MLflow if available and enabled
        if MLFLOW_AVAILABLE and self.config["training"]["mlflow"]["enabled"]:
            mlflow.set_experiment(self.config["training"]["mlflow"]["experiment_name"])
        
        logger.info("ModelTrainer initialized")
    
    def _load_config(self, config_path: str) -> dict:
        """Load YAML configuration."""
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    
    def _get_models(self) -> Dict[str, Any]:
        """
        Get configured models with default hyperparameters.
        
        Returns:
            Dictionary of model name to model instance
        """
        models = {}
        
        # Logistic Regression
        models["logistic_regression"] = LogisticRegression(
            max_iter=1000,
            random_state=42,
            solver='saga',
            penalty='l2',
            class_weight='balanced'
        )
        
        # Random Forest
        models["random_forest"] = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        # Gradient Boosting
        models["gradient_boosting"] = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        
        # XGBoost
        if XGBOOST_AVAILABLE:
            models["xgboost"] = XGBClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=3,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            )
        
        # LightGBM
        if LIGHTGBM_AVAILABLE:
            models["lightgbm"] = LGBMClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                num_leaves=31,
                class_weight='balanced',
                random_state=42,
                verbose=-1
            )
        
        return models
    
    def _evaluate_model(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate model on test data.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of metric name to value
        """
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
            "average_precision": average_precision_score(y_test, y_proba)
        }
        
        return metrics
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> Any:
        """
        Train all configured models and select the best one.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            feature_names: List of feature names
            
        Returns:
            Best trained model
        """
        self.feature_names = feature_names or []
        models_to_train = self.config["models"]["train_models"]
        primary_metric = self.config["models"]["primary_metric"]
        
        logger.info(f"Training {len(models_to_train)} models...")
        logger.info(f"Primary metric: {primary_metric}")
        
        all_models = self._get_models()
        best_score = -np.inf
        
        for model_name in models_to_train:
            if model_name not in all_models:
                logger.warning(f"Model {model_name} not available, skipping")
                continue
            
            model = all_models[model_name]
            logger.info(f"\n{'='*50}")
            logger.info(f"Training: {model_name}")
            logger.info(f"{'='*50}")
            
            # Cross-validation
            cv = StratifiedKFold(
                n_splits=self.config["training"]["cv_folds"],
                shuffle=True,
                random_state=42
            )
            cv_scores = cross_val_score(
                model, X_train, y_train,
                cv=cv,
                scoring='roc_auc',
                n_jobs=-1
            )
            
            logger.info(f"CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
            
            # Train on full training set
            model.fit(X_train, y_train)
            
            # Evaluate
            metrics = self._evaluate_model(model, X_test, y_test)
            metrics["cv_score_mean"] = cv_scores.mean()
            metrics["cv_score_std"] = cv_scores.std()
            
            # Store results
            self.models[model_name] = model
            self.results[model_name] = metrics
            
            # Log metrics
            for metric_name, value in metrics.items():
                logger.info(f"  {metric_name}: {value:.4f}")
            
            # Track with MLflow
            if MLFLOW_AVAILABLE and self.config["training"]["mlflow"]["enabled"]:
                with mlflow.start_run(run_name=model_name):
                    mlflow.log_params(model.get_params())
                    mlflow.log_metrics(metrics)
                    mlflow.sklearn.log_model(model, "model")
            
            # Check if best
            if metrics[primary_metric] > best_score:
                best_score = metrics[primary_metric]
                self.best_model = model
                self.best_model_name = model_name
        
        logger.info(f"\n{'='*50}")
        logger.info(f"🏆 Best Model: {self.best_model_name}")
        logger.info(f"   {primary_metric}: {best_score:.4f}")
        logger.info(f"{'='*50}")
        
        return self.best_model
    
    def get_results_summary(self) -> pd.DataFrame:
        """
        Get summary of all model results.
        
        Returns:
            DataFrame with model comparison
        """
        return pd.DataFrame(self.results).T.round(4)
    
    def get_feature_importance(
        self,
        model: Optional[Any] = None,
        top_n: int = 20
    ) -> pd.DataFrame:
        """
        Get feature importance from model.
        
        Args:
            model: Model to analyze (default: best model)
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importances
        """
        model = model or self.best_model
        
        if model is None:
            raise ValueError("No model trained yet")
        
        # Get feature importance based on model type
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
        else:
            raise ValueError("Model doesn't support feature importance")
        
        # Handle case where feature_names might be empty or wrong length
        if len(self.feature_names) != len(importances):
            feature_names = [f"feature_{i}" for i in range(len(importances))]
        else:
            feature_names = self.feature_names
        
        df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return df.head(top_n)
    
    def explain_with_shap(
        self,
        X: np.ndarray,
        sample_size: int = 100
    ) -> Optional[Any]:
        """
        Generate SHAP explanations for model predictions.
        
        Args:
            X: Feature matrix to explain
            sample_size: Number of samples for explanation
            
        Returns:
            SHAP values object
        """
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available")
            return None
        
        if self.best_model is None:
            raise ValueError("No model trained yet")
        
        # Sample data for efficiency
        if len(X) > sample_size:
            indices = np.random.choice(len(X), sample_size, replace=False)
            X_sample = X[indices]
        else:
            X_sample = X
        
        logger.info(f"Computing SHAP values for {len(X_sample)} samples...")
        
        # Use appropriate explainer based on model type
        if hasattr(self.best_model, 'feature_importances_'):
            explainer = shap.TreeExplainer(self.best_model)
        else:
            explainer = shap.LinearExplainer(
                self.best_model,
                X_sample,
                feature_names=self.feature_names
            )
        
        shap_values = explainer.shap_values(X_sample)
        
        return shap_values
    
    def save_model(
        self,
        model: Optional[Any] = None,
        path: Optional[str] = None
    ):
        """
        Save trained model to disk.
        
        Args:
            model: Model to save (default: best model)
            path: Save path (default: from config)
        """
        model = model or self.best_model
        
        if model is None:
            raise ValueError("No model to save")
        
        save_dir = self.project_root / self.config["training"]["save_path"]
        save_dir.mkdir(parents=True, exist_ok=True)
        
        path = path or str(save_dir / self.config["training"]["model_name"])
        
        # Save model
        joblib.dump(model, path)
        logger.info(f"Model saved to {path}")
        
        # Save metadata
        # Convert numpy values to native Python types for safe YAML serialization
        metrics_dict = self.results.get(self.best_model_name, {})
        safe_metrics = {}
        for key, value in metrics_dict.items():
            if hasattr(value, 'item'):  # NumPy scalar
                safe_metrics[key] = float(value.item())
            elif isinstance(value, (np.integer, np.floating)):
                safe_metrics[key] = float(value)
            else:
                safe_metrics[key] = value

        metadata = {
            "model_name": self.best_model_name,
            "training_date": datetime.now().isoformat(),
            "metrics": safe_metrics,
            "feature_names": self.feature_names
        }

        metadata_path = path.replace('.pkl', '_metadata.yaml')
        with open(metadata_path, 'w') as f:
            yaml.dump(metadata, f, default_flow_style=False)
        
        logger.info(f"Metadata saved to {metadata_path}")
    
    def load_model(self, path: Optional[str] = None) -> Any:
        """
        Load trained model from disk.
        
        Args:
            path: Model path (default: from config)
            
        Returns:
            Loaded model
        """
        save_dir = self.project_root / self.config["training"]["save_path"]
        path = path or str(save_dir / self.config["training"]["model_name"])
        
        self.best_model = joblib.load(path)
        logger.info(f"Model loaded from {path}")
        
        # Load metadata if available
        metadata_path = path.replace('.pkl', '_metadata.yaml')
        if Path(metadata_path).exists():
            with open(metadata_path, 'r') as f:
                metadata = yaml.safe_load(f)
                self.best_model_name = metadata.get("model_name", "unknown")
                self.feature_names = metadata.get("feature_names", [])
        
        return self.best_model


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: List[str] = ['No Churn', 'Churn']
):
    """Print formatted classification report."""
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(y_true, y_pred, target_names=target_names))
    
    print("\nCONFUSION MATRIX:")
    print("-"*40)
    cm = confusion_matrix(y_true, y_pred)
    print(f"                  Predicted")
    print(f"                  No Churn  Churn")
    print(f"Actual No Churn     {cm[0,0]:5d}  {cm[0,1]:5d}")
    print(f"Actual Churn        {cm[1,0]:5d}  {cm[1,1]:5d}")


# =============================================================================
# CLI Interface
# =============================================================================
if __name__ == "__main__":
    import sys
    import argparse
    
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from src.data.data_loader import DataLoader
    from src.data.preprocessing import DataPreprocessor
    
    parser = argparse.ArgumentParser(description="Train churn prediction models")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to config file"
    )
    
    args = parser.parse_args()
    
    # Load and preprocess data
    print("Loading data...")
    loader = DataLoader(config_path=args.config)
    df = loader.load_data()

    print("Preprocessing data...")
    preprocessor = DataPreprocessor(config_path=args.config)
    X_train, X_test, y_train, y_test = preprocessor.fit_transform(df)
    feature_names = preprocessor.get_feature_names()

    # Train models
    print("\nTraining models...")
    trainer = ModelTrainer(config_path=args.config)
    best_model = trainer.train(
        X_train, y_train,
        X_test, y_test,
        feature_names=feature_names
    )

    # Print results
    print("\nResults Summary:")
    print(trainer.get_results_summary())

    print("\nFeature Importance (Top 10):")
    print(trainer.get_feature_importance(top_n=10))

    # Print classification report
    y_pred = best_model.predict(X_test)
    print_classification_report(y_test, y_pred)

    # Save model and preprocessor
    trainer.save_model()
    preprocessor.save()

    print("\nTraining complete!")
    print(f"   Best model: {trainer.best_model_name}")
    print(f"   Model saved to: models/best_model.pkl")
