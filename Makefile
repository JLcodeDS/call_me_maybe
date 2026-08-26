PYTHON = python3 

install: 
	uv sync
	uv pip install ./llm_sdk

all: install run

clean:

fclean:

lint: flake8 . mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict: flake8 . mypy . --strict

exec: uv run $(PYTHON) -m src

debug: 

.PHONY: all clean fclean lint lint-strict run debug