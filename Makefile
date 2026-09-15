PYTHON = uv run python

all: install run

install:
	uv sync

run:
	$(PYTHON) -m src

debug:
	$(PYTHON) -m pdb -m src

lint:
	flake8 src
	mypy src --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	flake8 src
	mypy src --strict

clean:
	rm -rf __pycache__ .mypy_cache .pytest_cache src/__pycache__ data/output .venv

re: clean all

.PHONY: all install run debug clean lint lint-strict re