PYTHON = uv run python
export UV_CACHE_DIR := $(CURDIR)/.uv_cache
export HF_HOME      := $(CURDIR)/.hf_cache

all: install run

install:
	uv sync

run:
	$(PYTHON) -m src

debug:
	$(PYTHON) -m pdb -m src

lint:
	uv run flake8 src
	uv run mypy src --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 src
	uv run mypy src --strict

clean:
	rm -rf __pycache__ .mypy_cache .pytest_cache src/__pycache__ data/output .venv

re: clean all

.PHONY: all install run debug clean lint lint-strict re