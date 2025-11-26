.PHONY: format lint check fix help

# Default variables
PYTHON_FILES := .
RUFF_CONFIG := ruff.toml
ISORT_CONFIG := isort.toml

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


all: format lint

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