set positional-arguments
export NUMBA_DISABLE_JIT:="1"

qa *args: lint type (test args)

test *args:
    uv run pytest tests/ --import-mode importlib --cov --cov-report xml --junitxml=report.xml "$@"
    coverage report

lint:
    uv run ruff check --fix .

type:
    uv run mypy --ignore-missing-imports src/
