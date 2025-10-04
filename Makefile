.PHONY: help install test lint format type-check pre-commit clean all

help:
	@echo "Available commands:"
	@echo "  make install       - Install dependencies and pre-commit hooks"
	@echo "  make test          - Run all tests with pytest"
	@echo "  make lint          - Run flake8 linter"
	@echo "  make format        - Format code with black and isort"
	@echo "  make type-check    - Run mypy type checking"
	@echo "  make pre-commit    - Run all pre-commit hooks"
	@echo "  make clean         - Remove cache and build files"
	@echo "  make all           - Run format, lint, type-check, and test"

install:
	pip install -r requirements.txt
	pip install pre-commit black flake8 mypy isort bandit
	pre-commit install

test:
	pytest tests/ -v

lint:
	flake8 src/ tests/

format:
	black src/ tests/ --line-length 100
	isort src/ tests/ --profile black --line-length 100

type-check:
	mypy src/ --ignore-missing-imports --check-untyped-defs

pre-commit:
	pre-commit run --all-files

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	rm -rf build/ dist/

all: format lint type-check test
	@echo "All checks passed successfully!"
