PYTHON = python3 


install: pip install flake8 mypy pydantic

all: install run

clean:

fclean:

lint: flake8 . mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict: flake8 . mypy . --strict

run: $(PYTHON)

debug: 