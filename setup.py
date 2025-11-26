"""
Setup configuration for Churn Prediction package.

Install in development mode:
    pip install -e .

Install with extras:
    pip install -e ".[dev]"
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Core dependencies
INSTALL_REQUIRES = [
    "pandas>=2.0.0",
    "numpy>=1.24.0",
    "scikit-learn>=1.3.0",
    "xgboost>=2.0.0",
    "lightgbm>=4.0.0",
    "matplotlib>=3.8.0",
    "seaborn>=0.13.0",
    "plotly>=5.18.0",
    "streamlit>=1.28.0",
    "pyyaml>=6.0.0",
    "joblib>=1.3.0",
    "tqdm>=4.66.0",
]

# Development dependencies
DEV_REQUIRES = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "black>=23.11.0",
    "isort>=5.12.0",
    "flake8>=6.1.0",
    "mypy>=1.7.0",
    "pre-commit>=3.6.0",
    "jupyter>=1.0.0",
    "ipykernel>=6.27.0",
]

# Optional dependencies
EXTRAS_REQUIRE = {
    "dev": DEV_REQUIRES,
    "mlflow": ["mlflow>=2.8.0"],
    "shap": ["shap>=0.43.0"],
    "all": DEV_REQUIRES + ["mlflow>=2.8.0", "shap>=0.43.0"],
}

setup(
    name="churn-prediction",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="End-to-end ML system for customer churn prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/churn-ml-project",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/churn-ml-project/issues",
        "Documentation": "https://github.com/yourusername/churn-ml-project#readme",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    entry_points={
        "console_scripts": [
            "churn-train=src.models.train:main",
            "churn-predict=src.models.predict:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
