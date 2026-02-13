.PHONY: *
.SILENT:

VENV_PATH=.venv
PYTHON=$(VENV_PATH)/bin/python3

## setup environment
setup:
	@echo "⚫ Install the repo dependencies"
	uv sync
	@echo "⚫ Installing pre-commit hook"
	$(PYTHON) -m pre_commit install

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
