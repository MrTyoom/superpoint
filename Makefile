.PHONY: *
.SILENT:

VENV_PATH=.venv
PYTHON=$(VENV_PATH)/bin/python3

## setup environment
setup: install
	@echo "⚫ Installing pre-commit hook"
	$(PYTHON) -m pre_commit install

## install dependencies
install: venv
	@echo "⚫ Install the repo dependencies"
	uv sync

## python venv setup with uv
venv:
	@echo "⚫ Install uv (if not already installed)"
	command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh

	@echo "⚫ Create a .venv local virtual environment (if it doesn't exist)"
	[ -d ".venv" ] || uv venv


## checking code format
pre-commit-check:
	@echo "⚫ Checking code format..."
	. .venv/bin/activate && git ls-files -- '*.py' | xargs pre-commit run --files

## remove all artifacts
clean:
	@echo "⚫ Remove all artifacts..."
	find . -name '__pycache__' -exec rm -fr {} +
	find . -type f -name '.DS_Store' -delete
	rm -fr .ruff_cache
