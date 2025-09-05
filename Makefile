.PHONY: deps deps-all mypy check sort format

deps:
	uv pip sync pyproject.toml

deps-all:
	uv sync --all-extras

mypy:
	mypy .

check:
	ruff .

format:
	ruff format .

sort:
	ruff check --select I --fix .