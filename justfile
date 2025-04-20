set shell := ['uv', 'run', 'bash', '-euxo', 'pipefail', '-c']
set positional-arguments
export NUMBA_DISABLE_JIT:="1"

qa *args: lint type (test args)

test *args:
    pytest tests/ --import-mode importlib --cov --cov-report xml --junitxml=report.xml "$@"
    coverage report

lint *args:
    ruff check --fix "$@"

type *args:
    mypy --ignore-missing-imports src/
