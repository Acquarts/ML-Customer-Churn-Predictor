# =============================================================================
# MAKEFILE - Churn Prediction Project
# =============================================================================
# Run 'make help' to see available commands

.PHONY: help install install-dev clean lint format test train app docker

# Default target
.DEFAULT_GOAL := help

# =============================================================================
# VARIABLES
# =============================================================================
PYTHON := python
PIP := pip
STREAMLIT := streamlit
PYTEST := pytest
BLACK := black
ISORT := isort
FLAKE8 := flake8

# Directories
SRC_DIR := src
TEST_DIR := tests
APP_DIR := app
DATA_DIR := data
MODELS_DIR := models

# =============================================================================
# HELP
# =============================================================================
help: ## Show this help message
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════════╗"
	@echo "║           🔮 Customer Churn Prediction - Makefile                ║"
	@echo "╚══════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""

# =============================================================================
# INSTALLATION
# =============================================================================
install: ## Install production dependencies
	$(PIP) install -r requirements.txt

install-dev: ## Install development dependencies
	$(PIP) install -r requirements.txt
	$(PIP) install -e ".[dev]"
	pre-commit install
	@echo "✅ Development environment ready!"

create-env: ## Create virtual environment
	$(PYTHON) -m venv venv
	@echo "✅ Virtual environment created. Activate with: source venv/bin/activate"

# =============================================================================
# CODE QUALITY
# =============================================================================
lint: ## Run linting checks
	@echo "🔍 Running linters..."
	$(FLAKE8) $(SRC_DIR) --max-line-length=88 --extend-ignore=E203,E501,W503
	@echo "✅ Linting complete!"

format: ## Format code with black and isort
	@echo "🎨 Formatting code..."
	$(ISORT) $(SRC_DIR) $(TEST_DIR) --profile=black
	$(BLACK) $(SRC_DIR) $(TEST_DIR) --line-length=88
	@echo "✅ Formatting complete!"

type-check: ## Run type checking with mypy
	mypy $(SRC_DIR) --ignore-missing-imports

quality: lint format type-check ## Run all code quality checks

# =============================================================================
# TESTING
# =============================================================================
test: ## Run tests
	@echo "🧪 Running tests..."
	$(PYTEST) $(TEST_DIR) -v

test-cov: ## Run tests with coverage
	@echo "🧪 Running tests with coverage..."
	$(PYTEST) $(TEST_DIR) -v --cov=$(SRC_DIR) --cov-report=html --cov-report=term-missing
	@echo "📊 Coverage report: htmlcov/index.html"

# =============================================================================
# DATA & TRAINING
# =============================================================================
data: ## Download and prepare data
	@echo "📥 Loading data..."
	$(PYTHON) $(SRC_DIR)/data/data_loader.py --config config/config.yaml
	@echo "✅ Data ready!"

train: ## Train the model
	@echo "🚀 Training model..."
	$(PYTHON) $(SRC_DIR)/models/train.py --config config/config.yaml
	@echo "✅ Training complete!"

predict: ## Run prediction demo
	$(PYTHON) $(SRC_DIR)/models/predict.py --sample

pipeline: data train ## Run full ML pipeline (data + training)

# =============================================================================
# APPLICATION
# =============================================================================
app: ## Run Streamlit application
	@echo "🌐 Starting Streamlit app..."
	$(STREAMLIT) run $(APP_DIR)/streamlit_app.py

app-dev: ## Run Streamlit in development mode
	$(STREAMLIT) run $(APP_DIR)/streamlit_app.py --server.runOnSave=true

# =============================================================================
# DOCKER
# =============================================================================
docker-build: ## Build Docker image
	@echo "🐳 Building Docker image..."
	docker build -t churn-predictor:latest .

docker-run: ## Run Docker container
	@echo "🐳 Running Docker container..."
	docker run -p 8501:8501 churn-predictor:latest

docker-push: ## Push Docker image to registry
	docker tag churn-predictor:latest $(DOCKER_REGISTRY)/churn-predictor:latest
	docker push $(DOCKER_REGISTRY)/churn-predictor:latest

# =============================================================================
# NOTEBOOKS
# =============================================================================
notebooks: ## Start Jupyter Lab
	jupyter lab --notebook-dir=notebooks

# =============================================================================
# CLEANUP
# =============================================================================
clean: ## Clean cache and temporary files
	@echo "🧹 Cleaning up..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf htmlcov/ .coverage 2>/dev/null || true
	@echo "✅ Cleanup complete!"

clean-models: ## Clean model artifacts
	@echo "🧹 Cleaning model artifacts..."
	rm -rf $(MODELS_DIR)/*.pkl $(MODELS_DIR)/*.yaml 2>/dev/null || true
	@echo "✅ Model artifacts cleaned!"

clean-data: ## Clean data files
	@echo "🧹 Cleaning data files..."
	rm -rf $(DATA_DIR)/raw/*.csv $(DATA_DIR)/processed/*.csv 2>/dev/null || true
	@echo "✅ Data files cleaned!"

clean-all: clean clean-models clean-data ## Clean everything

# =============================================================================
# DOCUMENTATION
# =============================================================================
docs: ## Generate documentation
	@echo "📚 Generating documentation..."
	pdoc --html --output-dir docs $(SRC_DIR)

# =============================================================================
# RELEASE
# =============================================================================
version: ## Show current version
	@$(PYTHON) -c "import yaml; print(yaml.safe_load(open('config/config.yaml'))['project']['version'])"

release: quality test ## Prepare for release
	@echo "🚀 Ready for release!"
	@echo "   Don't forget to:"
	@echo "   1. Update version in config/config.yaml"
	@echo "   2. Update CHANGELOG.md"
	@echo "   3. Create git tag"
