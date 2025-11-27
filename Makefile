.PHONY: format lint check fix help

# Default variables
PYTHON_FILES := src

help:
	@echo "Available commands:"
	@echo "  make format    - Format code with ruff and isort"
	@echo "  make lint      - Run linting checks (ruff + isort)"
	@echo "  make check     - Check code without making changes"
	@echo "  make fix       - Auto-fix what can be fixed"
	@echo "  make all       - Run format and lint"

# Format code with both tools
format:
	@echo "Formatting code with ruff..."
	ruff format $(PYTHON_FILES)
	@echo "Sorting imports with isort..."
	isort $(PYTHON_FILES)

# Run linting checks
lint:
	@echo "Running ruff check..."
	ruff check $(PYTHON_FILES)
	@echo "Running isort check..."
	isort --check-only --diff $(PYTHON_FILES)

# Check code without making changes
check: lint

# Auto-fix what can be fixed
fix:
	@echo "Fixing code with ruff..."
	ruff check --fix $(PYTHON_FILES)
	@echo "Fixing imports with isort..."
	isort $(PYTHON_FILES)

ruff-format:
	ruff format $(PYTHON_FILES)

ruff-check:
	ruff check $(PYTHON_FILES)

ruff-fix:
	ruff check --fix $(PYTHON_FILES)

isort-check:
	isort --check-only --diff $(PYTHON_FILES)

isort-fix:
	isort $(PYTHON_FILES)

format-file:
	ruff format $(file)
	isort $(file)

lint-file:
	ruff check $(file)
	isort --check-only --diff $(file)

## testing
test:
	pytest tests/test_prepare_synthetic_dataset.py

## remove python file artifacts
clean-pyc:
	find . -name '__pycache__' -exec rm -fr {} +
	find . -name '*.egg-info' -exec rm -fr {} +

## remove all artifacts
clean: clean-pyc
	rm -fr .ruff_cache
	rm -fr .pytest_cache

## all checks
all: format lint test