set positional-arguments

# Help: shows all recipes with comments
help:
    @just -l

qa *args: lint type (test args)

test $NUMBA_DISABLE_JIT="1":
    uv run coverage run -m pytest
    uv run coverage xml

lint:
    uv run ruff check --fix .

format:
    uv run ruff format .

type:
    uv run mypy --ignore-missing-imports src/

# Install the development environment
environment:
	@if command -v uv > /dev/null; then \
	  echo '>>> Detected uv.'; \
	  uv sync --all-groups; \
	  uv run pre-commit install; \
	else \
	  echo '>>> Install uv first.'; \
	  exit 1; \
	fi
