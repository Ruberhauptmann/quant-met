set positional-arguments

qa *args: lint type (test args)

test *args:
    uv run pytest tests/ --import-mode importlib --cov --cov-report xml --junitxml=report.xml "$@"
    uv run coverage report

lint:
    uv run ruff check --fix .

type:
    uv run mypy --ignore-missing-imports src/
