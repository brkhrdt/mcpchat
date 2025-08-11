.PHONY: lint type test

all: lint type test

lint:
	uv run ruff format
	uv run ruff check --fix
	uv run ruff check

type:
	mypy . --check-untyped-defs

test:
	uv run pytest
