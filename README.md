# 🔮 Customer Churn Prediction System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> **End-to-end Machine Learning project** for predicting customer churn in a telecommunications company. Built following industry best practices and MLOps principles.

![App Preview](reports/figures/app_preview.png)

---

## 📋 Table of Contents

- [Business Problem](#-business-problem)
- [Project Structure](#-project-structure)
- [Methodology](#-methodology)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Performance](#-model-performance)
- [Deployment](#-deployment)
- [Contributing](#-contributing)

---

## 🎯 Business Problem

### Context
Customer churn is a critical metric for subscription-based businesses. Acquiring new customers costs **5-25x more** than retaining existing ones. This project develops a predictive system to identify at-risk customers before they leave.

### Objectives
1. **Predict** which customers are likely to churn in the next month
2. **Identify** key factors driving customer attrition
3. **Enable** proactive retention strategies through actionable insights
4. **Deploy** a real-time prediction interface for business users

### Business Impact
- 📉 Reduce churn rate by targeting high-risk customers
- 💰 Optimize marketing spend on retention campaigns
- 📊 Data-driven decision making for customer success teams

---

## 📁 Project Structure

```
churn-ml-project/
│
├── 📂 config/                    # Configuration files
│   └── config.yaml               # Project parameters
│
├── 📂 data/
│   ├── raw/                      # Original immutable data
│   ├── processed/                # Cleaned, transformed data
│   └── external/                 # External data sources
│
├── 📂 notebooks/                 # Jupyter notebooks (numbered)
│   ├── 01_eda.ipynb             # Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling.ipynb        # Model training & evaluation
│
├── 📂 src/                       # Source code modules
│   ├── data/                    # Data loading & preprocessing
│   ├── features/                # Feature engineering
│   ├── models/                  # Training & prediction
│   └── utils/                   # Helper functions
│
├── 📂 models/                    # Trained model artifacts
│
├── 📂 reports/                   # Generated analysis & figures
│   └── figures/
│
├── 📂 tests/                     # Unit tests
│
├── 📂 app/                       # Streamlit application
│   └── streamlit_app.py
│
├── requirements.txt
├── setup.py
└── README.md
```

---

## 🔬 Methodology

This project follows the **CRISP-DM** methodology:

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   1. Business        2. Data           3. Data                  │
│   Understanding  →   Understanding  →  Preparation              │
│        │                                    │                   │
│        │                                    ▼                   │
│        │              6. Deployment  ← 5. Evaluation            │
│        │                    │              │                    │
│        │                    │              ▼                    │
│        └────────────────────┴──────── 4. Modeling               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Tech Stack

| Category | Technologies |
|----------|-------------|
| **Data Processing** | Pandas, NumPy, Polars |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **ML Framework** | Scikit-learn, XGBoost, LightGBM |
| **Interpretability** | SHAP, Feature Importance |
| **Experiment Tracking** | MLflow |
| **Deployment** | Streamlit, Docker |
| **Code Quality** | Black, Pytest, Pre-commit |

---

## ⚙️ Installation

### Prerequisites
- Python 3.10 or higher
- pip or conda

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/churn-ml-project.git
cd churn-ml-project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

---

## 🚀 Usage

### 1. Data Pipeline

```bash
# Download and prepare data
python src/data/data_loader.py

# Run preprocessing
python src/data/preprocessing.py
```

### 2. Training

```bash
# Train models with default config
python src/models/train.py

# Train with custom parameters
python src/models/train.py --config config/config.yaml
```

### 3. Prediction

```python
from src.models.predict import ChurnPredictor

predictor = ChurnPredictor(model_path="models/best_model.pkl")
prediction = predictor.predict(customer_data)
```

### 4. Run Streamlit App

```bash
streamlit run app/streamlit_app.py
```

---

## 📊 Model Performance

### Metrics Comparison

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.79 | 0.65 | 0.54 | 0.59 | 0.82 |
| Random Forest | 0.84 | 0.73 | 0.62 | 0.67 | 0.87 |
| **XGBoost** | **0.86** | **0.76** | **0.68** | **0.72** | **0.89** |
| LightGBM | 0.85 | 0.75 | 0.66 | 0.70 | 0.88 |

### Feature Importance (Top 10)

```
Contract Type          ████████████████████ 0.18
Tenure                 ███████████████████  0.17
Monthly Charges        █████████████████    0.15
Total Charges          ████████████████     0.14
Internet Service       █████████████        0.11
Payment Method         ████████████         0.10
Online Security        ███████████          0.09
Tech Support           ██████████           0.08
Senior Citizen         █████████            0.07
Multiple Lines         ████████             0.06
```

---

## 🌐 Deployment

### Local Deployment

```bash
streamlit run app/streamlit_app.py
```

### Docker Deployment

```bash
# Build image
docker build -t churn-predictor .

# Run container
docker run -p 8501:8501 churn-predictor
```

### Streamlit Cloud

1. Push code to GitHub
2. Connect repository to [Streamlit Cloud](https://streamlit.io/cloud)
3. Deploy!

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/
```

---

## 📝 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) first.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 👤 Author

**Adrian Zambrana Acquaroni**
- LinkedIn: [adrianzambranaacquaroni](https://www.linkedin.com/in/adrianzambranaacquaroni/)
- GitHub: [@Acquarts](https://github.com/Acquarts)

---

<p align="center">
  Made with ❤️ and ☕
</p>
