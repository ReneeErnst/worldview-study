.PHONY: deps deps-all mypy ruff-check ruff-sort ruff-format

deps:
	uv pip sync pyproject.toml

deps-all:
	uv sync --all-extras

mypy:
	mypy .

ruff-check:
	ruff .

ruff-format:
	ruff format .

ruff-sort:
	ruff check --select I --fix .
