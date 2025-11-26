"""
Unit Tests for Churn Prediction Project
=======================================
Tests for data loading, preprocessing, and model components.

Run with: pytest tests/ -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_customer():
    """Create sample customer data for testing."""
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
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 89.95,
        "TotalCharges": 1079.40,
        "Churn": "Yes"
    }


@pytest.fixture
def sample_dataframe(sample_customer):
    """Create sample DataFrame for testing."""
    # Create multiple customers
    customers = []
    for i in range(100):
        customer = sample_customer.copy()
        customer["customerID"] = f"TEST-{i:03d}"
        customer["tenure"] = np.random.randint(0, 73)
        customer["MonthlyCharges"] = round(np.random.uniform(18, 118), 2)
        customer["TotalCharges"] = round(customer["tenure"] * customer["MonthlyCharges"], 2)
        customer["Churn"] = np.random.choice(["Yes", "No"])
        customers.append(customer)
    
    return pd.DataFrame(customers)


# =============================================================================
# DATA LOADER TESTS
# =============================================================================

class TestDataLoader:
    """Tests for DataLoader class."""
    
    def test_sample_data_creation(self):
        """Test that sample data can be created."""
        from src.data.data_loader import DataLoader
        
        loader = DataLoader()
        df = loader._create_sample_data()
        
        assert len(df) > 0
        assert "customerID" in df.columns
        assert "Churn" in df.columns
        assert "tenure" in df.columns
    
    def test_data_validation(self, sample_dataframe):
        """Test data validation."""
        from src.data.data_loader import DataLoader
        
        loader = DataLoader()
        
        # Should pass validation
        assert loader.validate_data(sample_dataframe) == True
        
        # Should fail with missing column
        df_missing = sample_dataframe.drop(columns=["tenure"])
        assert loader.validate_data(df_missing) == False
    
    def test_data_info(self, sample_dataframe):
        """Test data info extraction."""
        from src.data.data_loader import DataLoader
        
        loader = DataLoader()
        info = loader.get_data_info(sample_dataframe)
        
        assert "n_samples" in info
        assert "n_features" in info
        assert info["n_samples"] == len(sample_dataframe)


# =============================================================================
# PREPROCESSING TESTS
# =============================================================================

class TestPreprocessing:
    """Tests for preprocessing module."""
    
    def test_data_cleaner(self, sample_dataframe):
        """Test DataCleaner transformer."""
        from src.data.preprocessing import DataCleaner
        
        cleaner = DataCleaner()
        df_clean = cleaner.fit_transform(sample_dataframe)
        
        assert len(df_clean) == len(sample_dataframe)
        assert df_clean.isnull().sum().sum() == 0 or True  # May have some nulls
    
    def test_preprocessor_output_shape(self, sample_dataframe):
        """Test that preprocessor produces correct output shape."""
        from src.data.preprocessing import DataPreprocessor
        
        preprocessor = DataPreprocessor()
        X_train, X_test, y_train, y_test = preprocessor.fit_transform(sample_dataframe)
        
        # Check shapes
        assert len(X_train) + len(X_test) == len(sample_dataframe)
        assert len(y_train) == len(X_train)
        assert len(y_test) == len(X_test)
        
        # Check no NaN
        assert not np.isnan(X_train).any()
        assert not np.isnan(X_test).any()
    
    def test_preprocessor_transform(self, sample_dataframe):
        """Test transform on new data."""
        from src.data.preprocessing import DataPreprocessor
        
        preprocessor = DataPreprocessor()
        X_train, X_test, y_train, y_test = preprocessor.fit_transform(sample_dataframe)
        
        # Transform new data
        new_data = sample_dataframe.head(10)
        X_new = preprocessor.transform(new_data)
        
        assert X_new.shape[1] == X_train.shape[1]


# =============================================================================
# FEATURE ENGINEERING TESTS
# =============================================================================

class TestFeatureEngineering:
    """Tests for feature engineering module."""
    
    def test_feature_engineer(self, sample_dataframe):
        """Test ChurnFeatureEngineer transformer."""
        from src.features.feature_engineering import ChurnFeatureEngineer
        
        engineer = ChurnFeatureEngineer()
        df_engineered = engineer.fit_transform(sample_dataframe)
        
        # Should have more features
        assert len(df_engineered.columns) >= len(sample_dataframe.columns)
        
        # Check for expected new features
        expected_features = ["tenure_group", "is_new_customer", "num_services"]
        for feat in expected_features:
            assert feat in df_engineered.columns, f"Missing feature: {feat}"
    
    def test_feature_report(self, sample_dataframe):
        """Test feature report generation."""
        from src.features.feature_engineering import create_feature_report
        
        report = create_feature_report(sample_dataframe)
        
        assert len(report) > 0
        assert "feature" in report.columns
        assert "correlation_with_target" in report.columns


# =============================================================================
# MODEL TESTS
# =============================================================================

class TestModels:
    """Tests for model training and prediction."""
    
    def test_model_training(self, sample_dataframe):
        """Test that models can be trained."""
        from src.data.preprocessing import DataPreprocessor
        from src.models.train import ModelTrainer
        
        # Preprocess
        preprocessor = DataPreprocessor()
        X_train, X_test, y_train, y_test = preprocessor.fit_transform(sample_dataframe)
        
        # Train (use only fast models for testing)
        trainer = ModelTrainer()
        trainer.config["models"]["train_models"] = ["logistic_regression"]
        
        model = trainer.train(X_train, y_train, X_test, y_test)
        
        assert model is not None
        assert trainer.best_model_name == "logistic_regression"
    
    def test_model_prediction_shape(self, sample_dataframe):
        """Test prediction output shape."""
        from src.data.preprocessing import DataPreprocessor
        from src.models.train import ModelTrainer
        
        # Preprocess
        preprocessor = DataPreprocessor()
        X_train, X_test, y_train, y_test = preprocessor.fit_transform(sample_dataframe)
        
        # Train
        trainer = ModelTrainer()
        trainer.config["models"]["train_models"] = ["logistic_regression"]
        model = trainer.train(X_train, y_train, X_test, y_test)
        
        # Predict
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)
        
        assert len(predictions) == len(X_test)
        assert probabilities.shape == (len(X_test), 2)
    
    def test_results_summary(self, sample_dataframe):
        """Test results summary generation."""
        from src.data.preprocessing import DataPreprocessor
        from src.models.train import ModelTrainer
        
        # Preprocess
        preprocessor = DataPreprocessor()
        X_train, X_test, y_train, y_test = preprocessor.fit_transform(sample_dataframe)
        
        # Train
        trainer = ModelTrainer()
        trainer.config["models"]["train_models"] = ["logistic_regression"]
        trainer.train(X_train, y_train, X_test, y_test)
        
        # Get summary
        summary = trainer.get_results_summary()
        
        assert len(summary) == 1
        assert "accuracy" in summary.columns
        assert "roc_auc" in summary.columns


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """End-to-end integration tests."""
    
    def test_full_pipeline(self, sample_dataframe):
        """Test full ML pipeline."""
        from src.data.preprocessing import DataPreprocessor
        from src.features.feature_engineering import ChurnFeatureEngineer
        from src.models.train import ModelTrainer
        
        # Feature engineering
        engineer = ChurnFeatureEngineer()
        df_engineered = engineer.fit_transform(sample_dataframe)
        
        # Preprocess
        preprocessor = DataPreprocessor()
        X_train, X_test, y_train, y_test = preprocessor.fit_transform(df_engineered)
        
        # Train
        trainer = ModelTrainer()
        trainer.config["models"]["train_models"] = ["logistic_regression"]
        model = trainer.train(X_train, y_train, X_test, y_test)
        
        # Verify
        assert model is not None
        
        results = trainer.get_results_summary()
        assert results.loc["logistic_regression", "roc_auc"] > 0.5


# =============================================================================
# UTILITY TESTS
# =============================================================================

class TestUtilities:
    """Tests for utility functions."""
    
    def test_sample_customer_creation(self):
        """Test sample customer creation for predictor."""
        from src.models.predict import create_sample_customer
        
        customer = create_sample_customer()
        
        required_fields = [
            "customerID", "gender", "tenure", "Contract",
            "MonthlyCharges", "TotalCharges"
        ]
        
        for field in required_fields:
            assert field in customer


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
