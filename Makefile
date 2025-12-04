.PHONY: install dev test build clean publish docs

# Installation
install:
	pip install -e .

dev:
	pip install -e ".[dev]"

# Testing
test:
	pytest tests/ -v

test-cov:
	pytest tests/ --cov=src/loan_risk_predictor --cov-report=html

# Code quality
format:
	black src/ tests/
	isort src/ tests/

lint:
	flake8 src/ tests/
	mypy src/

# Building
build:
	python -m build

check-build:
	twine check dist/*

# Publishing
publish-test:
	twine upload --repository testpypi dist/*

publish:
	twine upload dist/*

# Documentation
docs:
	sphinx-build -b html docs/source docs/build

# Cleanup
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf __pycache__/
	rm -rf src/__pycache__/
	rm -rf src/loan_risk_predictor/__pycache__/
	rm -rf tests/__pycache__/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/

clean-all: clean
	rm -rf model_artifacts/*.pkl
	rm -rf examples/predictions.csv
	rm -rf examples/sample_loans.csv

# Development workflow
all: format lint test build check-build

help:
	@echo "Available commands:"
	@echo "  install    - Install package in development mode"
	@echo "  dev        - Install with development dependencies"
	@echo "  test       - Run tests"
	@echo "  test-cov   - Run tests with coverage report"
	@echo "  format     - Format code with black and isort"
	@echo "  lint       - Run flake8 and mypy"
	@echo "  build      - Build package"
	@echo "  check-build - Check built package"
	@echo "  publish-test - Upload to TestPyPI"
	@echo "  publish    - Upload to PyPI"
	@echo "  clean      - Clean build artifacts"
	@echo "  clean-all  - Clean everything including predictions"
	@echo "  all        - Run format, lint, test, build, check-build"
	@echo "  help       - Show this help message"