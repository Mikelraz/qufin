# Claude Code Guidelines: qufin

## Project Overview

High-performance Python toolkit for quantitative analysis and modelling of financial market products. Used to develop, train, and test trading strategies and models. Research notes live as Jupyter notebooks in `notebooks/`. Library code lives in `src/qufin/`.

---

## 1. Tooling (Non-Negotiable)

| Tool | Purpose | Never use instead |
|---|---|---|
| `uv` | Package management | `pip`, `poetry`, `conda` |
| `ruff` | Linting and formatting | `flake8`, `black`, `isort` |
| `pytest` + `pytest-xdist` | Testing | `unittest` |
| `pyright` | Type checking | `mypy` |

All dependencies are declared in `pyproject.toml`. Dev dependencies are in the `[dependency-groups] dev` group.

---

## 2. Standard Commands

```bash
uv sync --all-groups          # install all deps including dev
uv add <package>              # add a runtime dependency
uv add --group dev <package>  # add a dev dependency

ruff check --fix .            # lint and auto-fix
ruff format .                 # format

pytest -n auto                # run tests across all cores
pytest -n auto --cov=src      # with coverage

python -m cProfile -s time    # profile a script
```

---

## 3. Performance Mandates

### Data
- Use `polars` for all DataFrame work. Never use `pandas` unless a third-party library forces it (convert immediately with `pl.from_pandas()`).
- Use `numpy` for array-level numerical work. Avoid Python-level loops over array data.
- Persist data with Parquet (`polars` native) or Arrow IPC. Avoid CSV for anything beyond small inputs.

### Computation
- I/O-bound tasks: `asyncio`.
- CPU-bound tasks: `multiprocessing` or `concurrent.futures.ProcessPoolExecutor`.
- Numerical hot paths: annotate with `@numba.jit(nopython=True)` or `@numba.njit`. Prefer `@numba.njit(parallel=True)` + `numba.prange` over manual multiprocessing for array loops.
- Prefer algorithms with O(log n) or better complexity. Avoid O(n²) loops on market data.
- Use `math` and `cmath` built-ins for scalar math — they are faster than `numpy` equivalents for scalars.

### Classes and Data Structures
- Use `@dataclass(slots=True)` for all data-holding classes. This reduces memory footprint and speeds up attribute access.
- Use `__slots__` explicitly on non-dataclass classes that are instantiated frequently.

---

## 4. Coding Style

- **Type hints everywhere.** All function signatures must be fully annotated. Pyright is set to strict mode.
- **PEP 634 pattern matching** over `if/elif` chains for dispatch on types or structured data.
- **No comments explaining what the code does.** Only add a comment when the *why* is non-obvious (a hidden constraint, a numerical stability trick, a domain-specific invariant).
- Line length: 100 characters (`ruff` enforces this).
- Target: Python 3.11+. Use `tomllib`, `match`, `Self`, `TypeAlias`, and other 3.10–3.11 features freely.

---

## 5. Financial Domain Conventions

- Always be explicit about time zone when handling timestamps (`zoneinfo.ZoneInfo("UTC")` as default).
- Prices and returns are `float64` unless there is a specific reason to downcast.
- Keep raw market data immutable — transform into new frames rather than mutating in place.
- Separate data fetching, feature engineering, and model logic into distinct modules.
